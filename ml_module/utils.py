import os
import glob
from skimage.util import img_as_ubyte
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
import cv2
from utils.utils import load_and_crop

"""
    From here is handcraft features stuffs
"""

def process_img(img_path):
    cropped_image, _ = load_and_crop(img_path+".bmp", input_size=128)
    image_color = img_as_ubyte(cropped_image)
    return image_color

def IncreaseDescreaseMask(Mask, Size):
    from skimage.morphology import erosion, dilation, opening, closing, white_tophat
    from skimage.morphology import disk
    selem = disk(abs(Size))
    if(Size > 0):
        result = dilation(Mask, selem)
    else:
        result = erosion(Mask, selem)  
    return result

def evaluate_mean_n_std(listClassesvalues):
    valueList = []
    [valueList.extend(data_per_class) for data_per_class in listClassesvalues]
    mean = np.mean(np.array(valueList))
    std = np.std(np.array(valueList))
    return mean, std

def get_standard_score(listClassesvalues, mean, std):
    dataReturn = []
    [dataReturn.append(((np.array(data_per_class) - mean) / std)) for data_per_class in listClassesvalues]
    return dataReturn

def intToBitArray(img) :
    row ,col = img.shape
    list = []
    for i in range(row):
        for j in range(col):
             list.append (np.binary_repr( img[i][j] ,width=8  ) )
    return list #the binary_repr() fucntion returns binary values but in 
                #string 
                #, not integer, which has it's own perk as you will notice 
def bitplane(bitImgVal , img1D ):
    bitList = [  int(   i[bitImgVal]  )    for i in img1D]
    return bitList
def GetBitImage(index, image2D):
    ImageIn1D = intToBitArray(image2D)
    Imagebit = np.array( bitplane(index, ImageIn1D ) )
    Imagebit = np.reshape(Imagebit , image2D.shape )
    return Imagebit
def GetAllBitImage(image2D):
    image2D_Bit = list()
    for i in range(8):
        image2D_Bit.append(GetBitImage(i, image2D))
    return image2D_Bit

def pick_Threshold(listClasses):
    from skimage.filters import threshold_multiotsu
    listImage = []
    multi_threshold = []
    [listImage.extend(classList) for classList in listClasses]

    for image_path in listImage:
        img = process_img(image_path)
        image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        image_dilation = IncreaseDescreaseMask(image_gray, 3)
        image_erosion = IncreaseDescreaseMask(image_gray, -3)
        image_re_dilation = IncreaseDescreaseMask(image_erosion, 6)
        image_compare = image_re_dilation - image_dilation
        multi_threshold.append(threshold_multiotsu(image_compare))
    return np.mean(np.array(multi_threshold), axis=0).tolist()
    
def GetFeature_Differential_GrayInfo_Mean(image_color, multi_thresholds):
    from skimage.filters import threshold_multiotsu
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

    image_dilation = IncreaseDescreaseMask(image_gray, 6)
    image_erosion = IncreaseDescreaseMask(image_gray, -3)
    image_re_dilation = IncreaseDescreaseMask(image_erosion, 9)
    
    # image_compare_1 = image_dilation - image_re_dilation
    image_compare = image_re_dilation - image_dilation
    threshold_compare = threshold_multiotsu(image_compare)
    image_differential = (image_compare > multi_thresholds[0]) * (image_compare < multi_thresholds[1])

    return [np.mean(image_differential), np.mean(threshold_compare)]

def GetFeature_Differential_BitInfo_Mean(image_color):
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_RGB2GRAY)

    image_dilation = IncreaseDescreaseMask(image_gray, 3)
    image_erosion = IncreaseDescreaseMask(image_gray, -3)
    image_re_dilation = IncreaseDescreaseMask(image_erosion, 6)
    
    image2D_Bit_Dilation = GetAllBitImage(image_dilation)
    image2D_Bit_reDilation = GetAllBitImage(image_re_dilation)


    image_differential00 = image2D_Bit_Dilation[0] - image2D_Bit_reDilation[0]
    image_differential01 = image2D_Bit_Dilation[1] - image2D_Bit_reDilation[1]
    image_differential02 = image2D_Bit_Dilation[2] - image2D_Bit_reDilation[2]
    # image_differential00 = image2D_Bit_Dilation[0] - image2D_Bit_reDilation[0]

    get_differential00 = image_differential00 > 0
    get_differential01 = image_differential01 > 0
    get_differential02 = image_differential02 > 0

    feature_list = [np.mean(get_differential00), np.mean(get_differential01), np.mean(get_differential02)]
    # feature = np.mean(get_differential00)
    return feature_list

def SegmentByOtsu(Image):
    from skimage.filters import threshold_multiotsu
    IM = Image.copy()
    
    if(len(IM.shape) == 3):
        IM = cv2.cvtColor(IM, cv2.COLOR_RGB2GRAY)
        
    thresh = threshold_multiotsu(IM)
    Mask0 = (IM > thresh[0])
    Mask1 = (IM > thresh[1])
    Mask0 = Mask0.astype(int)
    Mask1 = Mask1.astype(int)

    return Mask0, Mask1, thresh

def GetFeature_ContoursCount(Image):
    image_mask00, image_mask01, image_thresh = SegmentByOtsu(Image)

    result00 = cv2.Canny((image_mask00 * 255).astype(np.uint8), image_thresh[0], image_thresh[1])
    result01 = cv2.Canny((image_mask01 * 255).astype(np.uint8), image_thresh[0], image_thresh[1])
    _, hierachy00 = cv2.findContours(result00, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    _, hierachy01 = cv2.findContours(result01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    valueReturn = [len(hierachy00[0]), len(hierachy01[0])]
    return valueReturn


def ReadImageToDataFrame(Path, DictofClasses):
    ActionDF = pd.DataFrame(columns = ["Path", "Class", "Feature01", "Feature02", "Feature03"])
    listOfImage = []
    listOfClass = []
    [listOfImage.extend(DictofClasses[i]) for i in DictofClasses]
    [listOfClass.extend(len(DictofClasses[i])*[i]) for i in DictofClasses]

    progress_bar = tqdm(listOfImage)
    for iter, image_path in enumerate(progress_bar):
        # for class_name in DictofClasses:
        #     for impath in DictofClasses[class_name]:
        ActionDF.loc[iter, "Path"] = image_path
        ActionDF.loc[iter, "Class"] = listOfClass[iter]
        progress_bar.update()
    return ActionDF

def get_dataframe_one(img_path):
    one_dataframe = pd.DataFrame()
    image_color = process_img(img_path)
    
    multi_threshold = pd.read_pickle("otsu_threshold.pkl")['multi_threshold'].tolist()
    # print(multi_threshold)
    feature_gray_diff = GetFeature_Differential_GrayInfo_Mean(image_color, multi_threshold)
    feature_bit_diff = GetFeature_Differential_BitInfo_Mean(image_color)
    feature_contours = GetFeature_ContoursCount(image_color)
    DF_model = pd.DataFrame(columns=["Feature01", "Feature02", "Feature03"])
    DF_model.loc[0, "Feature01"] = feature_gray_diff
    DF_model.loc[0, "Feature02"] = feature_bit_diff
    DF_model.loc[0, "Feature03"] = feature_contours
    
    idx = 0
    Feature_Vector = []
    Feature_Vector.append(DF_model.loc[idx, "Feature01"])
    Feature_Vector.append(DF_model.loc[idx, "Feature02"])
    Feature_Vector.append(DF_model.loc[idx, "Feature03"])
    Feature_Vector = [item for sublist in Feature_Vector for item in sublist]

    FeaturesColumn = ["Feature " + str(i + 1) for i in range(len(Feature_Vector))]
    DataFrame_Model = pd.DataFrame(columns = FeaturesColumn)

    Feature_Vector = []
    Feature_Vector.append(DF_model.loc[0, "Feature01"])
    Feature_Vector.append(DF_model.loc[0, "Feature02"])
    Feature_Vector.append(DF_model.loc[0, "Feature03"])
    # Feature_Vector.append(TestDefectDF_Model.loc[idx, "Feature04"])
    # Feature_Vector.append(TestDefectDF_Model1.loc[idx, "Feature05"])
    Feature_Vector = [item for sublist in Feature_Vector for item in sublist]

    DataFrame_Model.loc[0, :] = Feature_Vector

    return DataFrame_Model

def get_dataframe(list_Path, fail_list_class, dataframe_name):
    
    list_Dataframe = []
    # random.seed(0)
    for Path in list_Path:
        Reject_ls = [image for image in glob.glob(os.path.join(Path,"*.bmp")) if load_and_crop(image)[1] in fail_list_class]
        Pass_ls = [image for image in glob.glob(os.path.join(Path,"*.bmp")) if load_and_crop(image)[1] == "Pass"]

        DictofClasses = {"Reject" : Reject_ls, "Pass": Pass_ls}
        if "train" in Path.lower():
            multi_threshold = pick_Threshold([Reject_ls])
            otsu_threshold = pd.DataFrame()
            otsu_threshold['multi_threshold'] = multi_threshold
            otsu_threshold.to_pickle("otsu_threshold.pkl")
        else:
            multi_threshold = pd.read_pickle("otsu_threshold.pkl")['multi_threshold'].tolist()
        
        list_Dataframe.append(ReadImageToDataFrame(Path, DictofClasses))

    for DataFrame_ in list_Dataframe:
        DF_model = DataFrame_.copy()
        print(f"")
        progress_bar = tqdm(DF_model.loc[:, "Path"])
        for iter, imgpath in enumerate(progress_bar):
            image_color = process_img(imgpath)
            
            feature_gray_diff = GetFeature_Differential_GrayInfo_Mean(image_color, multi_threshold)
            feature_bit_diff = GetFeature_Differential_BitInfo_Mean(image_color)
            feature_contours = GetFeature_ContoursCount(image_color)

            DF_model.loc[iter, "Feature01"] = feature_gray_diff
            DF_model.loc[iter, "Feature02"] = feature_bit_diff
            DF_model.loc[iter, "Feature03"] = feature_contours
            progress_bar.update()
        # print(DF_model)
        idx = 0
        Feature_Vector = []
        Feature_Vector.append([DF_model.loc[idx, "Path"]])
        Feature_Vector.append([DF_model.loc[idx, "Class"]])
        Feature_Vector.append(DF_model.loc[idx, "Feature01"])
        Feature_Vector.append(DF_model.loc[idx, "Feature02"])
        Feature_Vector.append(DF_model.loc[idx, "Feature03"])
        Feature_Vector = [item for sublist in Feature_Vector for item in sublist]

        ColumnNames = ["Path", "Class"]
        FeaturesColumn = ["Feature " + str(i + 1) for i in range(len(Feature_Vector) - len(ColumnNames))]
        ColumnNames = ColumnNames + FeaturesColumn
        DataFrame_Model = pd.DataFrame(columns = ColumnNames)

        for idx in DF_model.index:
            Feature_Vector = []
            Feature_Vector.append([DF_model.loc[idx, "Path"]])
            Feature_Vector.append([DF_model.loc[idx, "Class"]])
            Feature_Vector.append(DF_model.loc[idx, "Feature01"])
            Feature_Vector.append(DF_model.loc[idx, "Feature02"])
            Feature_Vector.append(DF_model.loc[idx, "Feature03"])
            # Feature_Vector.append(TestDefectDF_Model.loc[idx, "Feature04"])
            # Feature_Vector.append(TestDefectDF_Model1.loc[idx, "Feature05"])
            Feature_Vector = [item for sublist in Feature_Vector for item in sublist]

            DataFrame_Model.loc[idx, :] = Feature_Vector

        DataFrame_Model.to_pickle( dataframe_name +'DF_Model.pkl')

def get_z_score_info(DF_model):
    TempData = DF_model.copy()
    TempData = TempData.drop(["Path","Class"], axis=1)
    list_distribute = []
    [list_distribute.append(columnFeature)  for columnFeature in TempData.columns.tolist() if any(TempData[columnFeature] > 1)]
    Mean_std_DF = pd.DataFrame()
    for feature in list_distribute:
        mean, std = evaluate_mean_n_std([TempData[feature].tolist()])
        Mean_std_DF[feature] = [mean, std]

    Mean_std_DF.to_pickle("Mean_std_value.pkl")

def get_z_score(DF_z_score_info, list_DF):
    Feature_ls = DF_z_score_info.columns.tolist()
    for DF_ in list_DF:
        for feature in Feature_ls:
            mean, std = DF_z_score_info[feature].tolist()
            standardized_data = get_standard_score([DF_[feature].tolist()], mean, std)
            DF_[feature] = standardized_data[0]

def SplitDataFrameToTrainAndTest(DataFrame, TrainDataRate, TargetAtt):
    # gets a random TrainDataRate % of the entire set
    training = DataFrame.sample(frac=TrainDataRate, random_state=1)
    # gets the left out portion of the dataset
    testing = DataFrame.loc[~DataFrame.index.isin(training.index)]

    X_train = training.drop(TargetAtt, 1)
    y_train = training[[TargetAtt]]
    X_test = testing.drop(TargetAtt, 1)
    y_test = testing[[TargetAtt]]

    return X_train, y_train, X_test, y_test