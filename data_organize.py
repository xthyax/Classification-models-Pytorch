import glob
import os
import pandas as pd
import numpy as np
import json
from classification_module import config_loader
import traceback
import sys
import argparse
import cv2
import shutil
import ast
from ml_module.ml_wrapper import MlWrapper

extension = [".bmp", ".mvsd"]

def make_augment(image_path, image_name, folder_path, augment_ls, adjust_class=False):
    image = cv2.imread(image_path)
    H, W, _ = image.shape

    for augment in augment_ls:
        # print(augment)
        with open(image_path +  ".json") as f:
            json_data = json.load(f)
        adjust_data = json_data
        # print(adjust_data['box'])
        if augment == "hor":
            new_img_name = "hor_" + image_name
            # adjust_data["filename"] = new_img_name
            augment_img = cv2.flip(image, 1)
            adjust_data["box"]["centerX"] = [W - adjust_data["box"]["centerX"][0]]
            
        elif augment =="ver":
            new_img_name = "ver_" + image_name
            # adjust_data["filename"] = new_img_name
            augment_img = cv2.flip(image, 0)
            adjust_data["box"]["centerY"] = [H - adjust_data["box"]["centerY"][0]]
        
        elif augment =="hor_ver":
            new_img_name = "hor_ver_" + image_name
            # adjust_data["filename"] = new_img_name
            augment_img = cv2.flip(image, -1)
            adjust_data["box"]["centerX"] = [W - adjust_data["box"]["centerX"][0]]
            adjust_data["box"]["centerY"] = [H - adjust_data["box"]["centerY"][0]]

        adjust_data["filename"] = new_img_name
        if adjust_class:
            adjust_data["classId"] = [adjust_class]
        else:
            pass
        new_img_path = os.path.join(folder_path, "Train", "TransformImage", new_img_name)
        cv2.imwrite(new_img_path, augment_img)
        # print(adjust_data['box'])

        with open(new_img_path +  ".json", 'w') as outfile:
            json.dump(adjust_data, outfile)

def updating_table(patch_table_path, current_table_path, config):
    patch_data = pd.read_csv(patch_table_path)
    current_data = pd.read_csv(current_table_path)
    update_data = current_data.copy()
    columns_length = len(update_data.columns)

    augment_clm_pos = list(current_data.columns).index("Augment")

    # We will look up the Image_name and change the label if necessary
    for patch_image_name in patch_data["Image_name"]:
        patch_location = np.where(patch_data["Image_name"]==patch_image_name)[0][0]

        if len(current_data.loc[current_data["Image_name"]==patch_image_name]) > 0:
            current_location = np.where(current_data["Image_name"]==patch_image_name)[0][0]

            update_data.loc[current_location][augment_clm_pos + 1:columns_length] =\
            patch_data.loc[patch_location][augment_clm_pos + 1:columns_length]

        else:
            # temp = input("Do you want this image to be put in train? (y/n)")
            # if temp.lower() =="y":
            patch_data.loc[patch_location]["Augment"] = config.AUGMENT_LIST
            # else:
            #     pass
            update_data.loc[len(update_data)] = patch_data.loc[patch_location]
    
    update_data.to_csv(current_table_path, index=False)

def create_table(data_path, config):
    dataframe = pd.DataFrame(columns=["Folder_path","Image_path","Image_name","Extension","Augment","Label","Binary_label","Origin_name","Component"])
    dataframe_ls = []
    for dirpath, dirname, filenames in os.walk(data_path):
        # print(dirpath)
        for filename in [f for f in filenames if f.endswith(tuple(extension))]:
            data_info = []
            # Since the cursor will go to OriginImage first
            if any(aug in filename for aug in config.AUGMENT_LIST):
                for augment in config.AUGMENT_LIST:
                    if filename.startswith(augment+"_"):
                        origin_name = filename.split(augment+"_")[-1].split(".")[0]
                        if len(dataframe.loc[dataframe["Image_name"]==origin_name]) > 0:
                            exists_image_loc = np.where(dataframe["Image_name"]==origin_name)[0][0]
                            # print(origin_name)
                            # print(augment)
                            dataframe.loc[exists_image_loc]["Augment"].append(augment)
                # print("End of augment")
            else:
                # continue
                # Append Path
                data_info.append(data_path)
                data_info.append(dirpath)
                # Append image name
                data_info.append(filename.split(".")[0])

                # Append file extension
                data_info.append(filename.split(".")[-1])

                # Append augment
                augment_opt = []
                data_info.append(augment_opt)

                # Append label
                with open(os.path.join(dirpath,filename + ".json")) as f:
                    json_data = json.load(f)
                label = json_data['classId'][0]
                if label in config.FAILCLASS_NAME or label in config.PASSCLASS_NAME:
                    data_info.append(label)
                else:
                    data_info.append("")

                # Append binary label
                binary_label = 'Reject' if (label in config.FAILCLASS_NAME) or (label=="Reject") else 'Pass'
                data_info.append(binary_label)
                # Apeend origin name and component name (temp)
                data_info.append(filename.split(".")[0])
                data_info.append(filename.split(".")[0])

                # dataframe_ls.append(data_info)
                dataframe.loc[len(dataframe)] = data_info

    # dataframe = pd.DataFrame( dataframe_ls, columns=["Folder_path","Image_path","Image_name","Augment","Label","Binary_label"])
    dataframe.to_csv(os.path.join(data_path, data_path.split("\\")[-1]+".csv"), index=False)

def updating_data(table_path, data_path):
    table_data = pd.read_csv(table_path)
    # print(len(table_data))
    sub_folder_ls = [b.split("\\")[-1] for b in np.unique(table_data["Image_path"])]
    for image_name in table_data["Image_name"]:
        # Update label
        # print(table_data.loc[np.where(table_data["Image_name"])[0][0]]["Folder_path"])
        if table_data.loc[np.where(table_data["Image_name"]==image_name)[0][0]]["Folder_path"] == data_path:
            data_loc = np.where(table_data["Image_name"]==image_name)[0][0]
            image_path_r_ls = table_data.loc[data_loc]["Image_path":"Image_name"].tolist()
            image_path = os.path.join(image_path_r_ls[0], image_path_r_ls[1])

            if "OriginImage" in sub_folder_ls:
                with open(image_path +  ".json") as f:
                    json_data = json.load(f)
                class_id = table_data.loc[data_loc]["Binary_label"]
                json_data["classId"] = [class_id]
            else:
                with open(image_path +  ".json") as f:
                    json_data = json.load(f)
                class_id = table_data.loc[data_loc]["Label"]
                json_data["classId"] = [class_id]

            with open(image_path +  ".json", 'w') as outfile:
                json.dump(json_data, outfile)
        # Update data
        else:
            data_loc = np.where(table_data["Image_name"]==image_name)[0][0]
            # print(f"Adding image :{image_name}")
            image_path_r_ls = table_data.loc[data_loc]["Image_path":"Image_name"].tolist()
            patch_image_path = os.path.join(image_path_r_ls[0], image_path_r_ls[1])
            
            shutil.copy(patch_image_path,os.path.join(data_path, "Train", "OriginImage",image_name))
            shutil.copy(patch_image_path + ".json",os.path.join(data_path, "Train", "OriginImage",image_name+ ".json"))
            # print(table_data.loc[data_loc]["Augment"])
            augment_ls = ast.literal_eval(table_data.loc[data_loc]["Augment"])
            # print(augment_ls)
            if table_data.loc[data_loc]["Extension"] == "bmp":
                make_augment(patch_image_path, image_name, data_path, augment_ls)
                
            else:
                pass

def create_data(table_path, data_path, patch_table_path):
    table_data = pd.read_csv(table_path)
    patch_data = pd.read_csv(patch_table_path)
    os.makedirs(os.path.join(data_path, "Train", "TransformImage"), exist_ok=True)
    for image_name in table_data["Image_name"]:
        data_loc = np.where(table_data["Image_name"]==image_name)[0][0]
        patch_data_loc = np.where(patch_data["Image_name"]==image_name)[0][0]

        # print(f"Adding image :{image_name}")

        image_path_r_ls = patch_data.loc[patch_data_loc]["Image_path":"Image_name"].tolist()
        image_path_target = table_data.loc[data_loc]["Image_path":"Image_name"].tolist()
        
        os.makedirs(image_path_target[0], exist_ok=True)
        patch_image_path = os.path.join(image_path_r_ls[0], image_path_r_ls[1])
        target_image_path = os.path.join(image_path_target[0], image_path_target[1])
        
        shutil.copy(patch_image_path, target_image_path)

        with open(patch_image_path +  ".json") as f:
            json_data = json.load(f)
        class_id = table_data.loc[data_loc]["Binary_label"]
        json_data["classId"] = [class_id]

        with open(target_image_path +  ".json", 'w') as outfile:
            json.dump(json_data, outfile)
        # shutil.copy(patch_image_path + ".json",target_image_path + ".json")
        # print(table_data.loc[data_loc]["Augment"])
        augment_ls = ast.literal_eval(table_data.loc[data_loc]["Augment"])
        # print(augment_ls)

        if table_data.loc[data_loc]["Extension"] == "bmp":
            make_augment(patch_image_path, image_name, data_path, augment_ls, adjust_class=class_id)
        else:
            pass

def update_origin_name(table_path, patch_table):
    table_data = pd.read_csv(table_path)
    patch_data = pd.read_csv(patch_table)

    table_data["Origin_name"] = table_data["Image_name"]

    # for new_img in patch_data["new_img"]:
    #     patch_location = np.where(patch_data["new_img"]==new_img)[0][0]
    for new_img in patch_data["new_mvsd"]:
        patch_location = np.where(patch_data["new_mvsd"]==new_img)[0][0]
        try:
            data_loc = np.where(table_data["Origin_name"] == new_img.split("/")[-1].split(".")[0])[0][0]
            origin_name = patch_data["original_img"][patch_location].split("/")[-1].split(".")[0]
            table_data["Origin_name"][data_loc] = origin_name

        except:
            print(f"{new_img} doesn't exist in table")

    table_data["Component"] = table_data["Origin_name"]
    
    for img_name in table_data["Component"]:
        component_location = np.where(table_data["Component"]==img_name)[0][0]
        table_data["Component"][component_location] = "_".join(img_name.split("_")[0:-1])

    table_data.to_csv(table_path, index=False)

def shift_component(table_path, base_on_complex):
    table_data = pd.read_csv(table_path)
    train_path = [b for b in np.unique(table_data["Image_path"]) if "Train" in b][0]
    # print(train_path)
    # Get path

    last_ele_idx = 0
    for class_ in np.unique(table_data['Binary_label']):
        table_data = pd.read_csv(table_path)

        list_of_total_comp = np.unique(table_data["Component"][np.where(table_data["Binary_label"]==class_)[0]]).tolist()
        
        # print(len(list_of_total_comp))
        list_of_comp = np.unique(table_data["Component"][np.where(np.array(table_data["Image_path"]==train_path) * np.array(table_data["Binary_label"]==class_))[0]]).tolist()
        list_of_idx = np.where( np.array(table_data["Image_path"]==train_path) * np.array(table_data["Data_complexity"] < 0.4) * np.array(table_data["Binary_label"]==class_) )[0].tolist()

        for path in np.unique(table_data['Image_path']):
            if path != train_path:
                # print(list_of_comp)
                # Get component
                # table_data = pd.read_csv(table_path)
                # list_of_comp = np.unique(table_data["Component"][np.where(table_data["Image_path"]==train_path) and np.where(table_data["Binary_label"]==class_)[0]]).tolist()
                # print(len(list_of_comp))
                if base_on_complex:
                    for idx in list_of_idx:
                        path_distribute = len(np.where(table_data['Binary_label'][np.where(table_data['Image_path']==path)[0]]==class_)[0])
                        
                        total_distribute = len(np.where(table_data['Binary_label']==class_)[0])
                        ratio = path_distribute/total_distribute

                        if ratio > 0.1:
                            table_data.to_csv(table_path, index=False)
                            print(path)
                            print(f"{class_}: {round(ratio,4)} % total distribute")
                            break
                        else:
                            table_data.loc[idx ,"Image_path"] = path
                            table_data.loc[idx, "Augment"] = "[]"
                            last_ele_idx = list_of_idx.index(idx)

                    list_of_idx = list_of_idx[last_ele_idx + 1:]
                    table_data.to_csv(table_path, index=False)
                    
                else:

                    for comp in list_of_comp:
                        # table_data = pd.read_csv(table_path)
                        # Checking distribute on the fly
                        path_distribute = len(np.where(table_data['Binary_label'][np.where(table_data['Image_path']==path)[0]]==class_)[0])
                        
                        total_distribute = len(np.where(table_data['Binary_label']==class_)[0])
                        ratio = path_distribute/total_distribute

                        if ratio > 0.1:
                            table_data.to_csv(table_path, index=False)
                            print(path)
                            print(f"{class_}: {round(ratio,4)} % total distribute")
                            break

                        if len(np.where(table_data["Component"]==comp)[0]) <= 3:
                            # print(len(np.where(table_data["Component"]==comp)[0]))
                            for idx in np.where(table_data["Component"]==comp)[0]:
                                
                                table_data["Image_path"][idx] = path
                                table_data["Augment"][idx] = "[]"
                                # print(table_data["Image_name"][idx])
                            # print(np.where(table_data["Component"]==comp)[0])
                            last_ele_idx = list_of_comp.index(comp)
                        table_data.to_csv(table_path, index=False)
                        
                    list_of_comp = list_of_comp[last_ele_idx + 1:]
                
            else:
                pass

    table_data.to_csv(table_path, index=False)

def change_component(table_path, base_on_complex):
    table_data = pd.read_csv(table_path)
    train_path = [b for b in np.unique(table_data["Image_path"]) if "Train" in b][0]
    if base_on_complex:
        for idx in np.where(table_data["Data_complexity"] >= 0.5)[0]:
            table_data.loc[idx, "Image_path"] = train_path
            table_data.loc[idx ,"Augment"] = "['hor','ver','hor_ver']"
    else:
        for comp in np.unique(table_data["Component"]):
            # print(comp)
            if len(np.where(table_data["Component"]==comp)[0]) > 1:
                if len(np.unique(table_data["Image_path"][np.where(table_data["Component"]==comp)[0]])) > 1:
                    for idx in np.where(table_data["Component"]==comp)[0]:
                        table_data["Image_path"][idx] = train_path
                        table_data["Augment"][idx] = "['hor','ver','hor_ver']"
                    # print(np.where(table_data["Component"]==comp)[0])
                    # print(comp)
                    # print(np.unique(table_data["Image_path"][np.where(table_data["Component"]==comp)[0]], return_counts=True))
            else:
                pass
    table_data.to_csv(table_path, index=False)

def checking_component(table_path):
    table_data = pd.read_csv(table_path)
    for comp in np.unique(table_data["Component"]):
        # print(comp)
        if len(np.where(table_data["Component"]==comp)[0]) > 1:
            # print(np.where(table_data["Component"]==comp)[0])
            if len(np.unique(table_data["Image_path"][np.where(table_data["Component"]==comp)[0]])) > 1:
                print(np.where(table_data["Component"]==comp)[0])
                print(comp)
                print(np.unique(table_data["Image_path"][np.where(table_data["Component"]==comp)[0]], return_counts=True))
        else:
            pass

def data_distribution(table_path):
    table_data = pd.read_csv(table_path)
    for path in np.unique(table_data['Image_path']):
        print("#" * 20)
        print(path)
        for class_ in np.unique(table_data['Binary_label']):
            path_distribute = len(np.where(table_data['Binary_label'][np.where(table_data['Image_path']==path)[0]]==class_)[0])
            total_distribute = len(np.where(table_data['Binary_label']==class_)[0])
            print(f"{class_}: {round(path_distribute/total_distribute,4)} % total distribute")
        print("\n")


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Manage data by using Struture data")

    parser.add_argument("command",
                        metavar="<command>",
                        help="'update_label', 'update_data', 'create_table', 'visualize_image','update_table' ")
    parser.add_argument('--config', required=False,
                        help='Path to config json file')
    parser.add_argument("--table", required=False, help="Table working with")

    parser.add_argument("--patch_table", required=False, help="Table use to update another table")

    # parser.add_argument("--current_table", required=False, help="Table need to update")

    parser.add_argument("--datapath", required=False, help="Physical path need to work with")

    parser.add_argument("--patch_datapath", required=False, help="Physical path use to update another physical path")

    parser.add_argument("--complex_base", required=False, type=bool, help="Distribute data base on defect complexity", default=True)

    args = parser.parse_args()
    
    try:
        # 
        if args.command == "create_table":
            param = config_loader.LoadConfig(args.config)
            create_table(args.datapath, param)

        # Update physical data using updated table
        if args.command == "update_data":
            updating_data(args.table, args.datapath)
            
        # Update table by table
        if args.command == "update_table":
            param = config_loader.LoadConfig(args.config)
            updating_table(args.patch_table, args.table, param)

        if args.command == "prepare_dataframe":
            param = config_loader.LoadConfig(args.config)
            model = MlWrapper(param, args.datapath)
            model.get_handcraft_feature()

        if args.command == "ensemble_model":
            param = config_loader.LoadConfig(args.config)
            model = MlWrapper(param, args.datapath)
            model.get_ensemble_model()
            model.evaluate_ensemble_model()
            model.measure_data_complex(args.table)
        
        if args.command == "update_origin_img":
            update_origin_name(args.table, args.patch_table)

        if args.command == "check_component":
            checking_component(args.table)

        if args.command == "change_component":
            change_component(args.table, args.complex_base)

        if args.command == "shift_component":
            shift_component(args.table, args.complex_base)
        
        if args.command == "create_data":
            create_data(args.table, args.datapath, args.patch_table)

        if args.command == "distribution":
            data_distribution(args.table)

    except Exception as e:
        print(traceback.format_exc())
        print(f'[ERROR] {e}', file=sys.stderr)
    finally:
        print('[INFO] End of process.')