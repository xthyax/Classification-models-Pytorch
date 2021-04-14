# import xgboost as xgb
from xgboost import XGBClassifier
import os
import pandas as pd
import numpy as np
import cv2


from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.metrics import accuracy_score
from .utils import get_dataframe, get_z_score_info, get_z_score, SplitDataFrameToTrainAndTest, get_dataframe_one
import numpy as np
np.set_printoptions(suppress=True)

class MlWrapper:
    def __init__(self, config, dataset_path):
        self.failClasses = config.FAILCLASS_NAME
        self.passClasses = config.PASSCLASS_NAME
        self.classes = config.CLASS_NAME
        self.dataset_path = dataset_path
        self.ensemble_model = None

    def get_handcraft_feature(self):
        Data_Path = os.path.join(self.dataset_path, "Train\\OriginImage")
        Data_Aug_Path = os.path.join(self.dataset_path, "Train\\TransformImage")
        get_dataframe([Data_Path, Data_Aug_Path], self.failClasses, "Train")

        Val_data_Path = os.path.join(self.dataset_path, "Validation")
        get_dataframe([Val_data_Path], self.failClasses, "Validation")

        Test_data_Path = os.path.join(self.dataset_path, "Test")
        get_dataframe([Test_data_Path], self.failClasses, "Test")


    def get_ensemble_model(self):
        TrainDF_Model = pd.read_pickle("TrainDF_Model.pkl")
        # if self.binary_option:
        #     TrainDF_Model['Class'] = ['Reject' if current_class in self.failClasses else current_class for current_class in TrainDF_Model['Class'].tolist()]
        get_z_score_info(TrainDF_Model)
        Mean_std_DF = pd.read_pickle("Mean_std_value.pkl")
        get_z_score(Mean_std_DF, [TrainDF_Model])

        train_data = TrainDF_Model.copy()
        # print(train_data["Path"].tolist())
        train_data = train_data.drop("Path", axis=1)
        data_train, target_train, _, _ = SplitDataFrameToTrainAndTest(train_data, 1, 'Class')
        X_train = data_train
        y_train = target_train
        # print(y_train)
        y_train = [self.classes.index(target) for target in y_train['Class'].tolist()]
        # print(y_train)
        X_train = X_train.apply(pd.to_numeric, errors = 'coerce')
        np.random.seed(1)
        Classifier_ls = [\
        BaggingClassifier(DecisionTreeClassifier(criterion='entropy', class_weight='balanced', min_samples_split=12), n_estimators=1000, max_samples=0.8, max_features=1.0, random_state=0), \
        RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=0, class_weight='balanced', min_samples_split=12, max_samples=0.8), \
        AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', class_weight='balanced', min_samples_split=12), n_estimators=2000, learning_rate=1.0, random_state=0), \
        XGBClassifier( n_estimators=200, random_state=0, learning_rate=0.3, tree_method='gpu_hist', gpu_id=0),\
        ]
        self.ensemble_model = VotingClassifier(estimators=[('bg', Classifier_ls[0]), ('rf',Classifier_ls[1]),('ada',Classifier_ls[2]),('xgb',Classifier_ls[3])], voting='soft', weights=[1,1,1,1])
        # print(TrainDF_Model.head())
        self.ensemble_model.fit(X_train, y_train)

    def evaluate_ensemble_model(self, listof_data=["Train","Validation","Test"]):
        for dataname in listof_data:
            EvaluateDF_Model = pd.read_pickle(dataname + "DF_Model.pkl")
            Mean_std_DF = pd.read_pickle("Mean_std_value.pkl")
            get_z_score(Mean_std_DF, [EvaluateDF_Model])
            evaluate_data = EvaluateDF_Model.copy()
            evaluate_data = evaluate_data.drop("Path", axis=1)
            data_evaluate, target_evaluate, _, _ = SplitDataFrameToTrainAndTest(evaluate_data, 1, 'Class')
            X_evaluate = data_evaluate
            y_evaluate = target_evaluate
            y_evaluate = [self.classes.index(target) for target in y_evaluate['Class'].tolist()]
            X_evaluate = X_evaluate.apply(pd.to_numeric, errors = 'coerce')

            y_pred = self.ensemble_model.predict(X_evaluate)
            Accuracy = accuracy_score(y_evaluate, y_pred)
            print(f"Evaluate data: {dataname + ' ' * (len(max(listof_data)) - len(dataname))}\t Accuracy_score: {Accuracy}")

    def get_feature_one(self, img_path):
        feature_dataframe = get_dataframe_one(img_path)
        Mean_std_DF = pd.read_pickle("Mean_std_value.pkl")
        get_z_score(Mean_std_DF, [feature_dataframe])
        features_data = feature_dataframe.copy()
        X_data = features_data
        X_data = X_data.apply(pd.to_numeric, errors = 'coerce')
        # print(f"X_data:\n{X_data}")
        return X_data

    def ensemble_prediction(self, input_feature):
        pred_score = self.ensemble_model.predict_proba(input_feature)
        return pred_score
    
    def measure_data_complex(self, table_path):
        table_data = pd.read_csv(table_path)
        table_data["Data_complexity"] = 0
        # o = 0
        for i in range(len(table_data)):
            image_path_target = table_data.loc[i]["Image_path":"Image_name"].tolist()
            image_path = os.path.join(image_path_target[0], image_path_target[1])
            
            image_feature = self.get_feature_one(image_path)

            prediction = self.ensemble_prediction(image_feature)

            table_data.loc[i, "Data_complexity"] = 1 - prediction[0][self.classes.index(table_data.loc[i]["Binary_label"])]

            # print(table_data.loc[i]["Image_name"])
            # print(image_path)
            # print(table_data.loc[i]["Binary_label"])
            # print(self.classes.index(table_data.loc[i]["Binary_label"]))
            # print(prediction)
            # print(f'Defect complexity :{1 - prediction[0][self.classes.index(table_data.loc[i]["Binary_label"])]}')
            # o += 1
            # if o > 6:
            #     break
        table_data.to_csv(table_path,  index=False)
