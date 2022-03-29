import os
import glob
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from .utils import preprocess_input, load_and_crop, metadata_count
import random

class LoadingData:
    def __init__(self, input_dir, classes, failClasses, passClasses, input_size, binary_option, log, index_layer=0, loadToMemory=True, augmentation=None):
        """
        Args:
            input_dir (str): path input
            classes (list): list of class
            failClasses (list) : list fail classes
            passClasses (list) : list pass classes
            input_size (int) : desire width and height
            binary_option (bool) : Binary/Multi classification
            log (obj) : log obj for logging
            index_layer (int) : layer index
            augmentation (list): list augment option

        Use:
            Use load_data(self) function to return data dictionary
        """
        self.log = log
        self.input_dir = input_dir
        self.failClasses = failClasses
        self.passClasses = passClasses
        self.binary_option = binary_option
        self.index_layer = index_layer
        self.classes = self.load_classes(classes)
        self.input_size = input_size
        self.load_to_memory = loadToMemory
        self.augmentation = augmentation

    def load_classes(self, classes):
        if self.binary_option:
            return ['Fail', 'Pass']
        else:
            return classes

    def load_data(self):
        img_path_labels = {
            "path": [],
            "img_path": [],
            "img_label": [],
            "img_content": [],
            "transform_img_path": [],
            "transform_img_label": [],
            "transform_img_content": [],
        }

        paths_data = []
        img_path_labels["path"].append(self.input_dir)

        if "train" in self.input_dir.lower().split("\\")[-1]:
            paths_data.append(os.path.join(self.input_dir, "OriginImage"))

            if self.augmentation is not None:
                paths_data.append(os.path.join(self.input_dir, "TransformImage"))

        else:
            paths_data.append(self.input_dir)

        for path_data in paths_data:
            self.log.write_log(f"Loading data from {path_data}", message_type=0)

            list_img_path = [f for f_ in [glob.glob(e) for e in
                                          [os.path.join(path_data, "*.mvsd"), os.path.join(path_data, "*.bmp")]] for
                             f in f_]

            if len(list_img_path) == 0:
                message = f"Folder path {path_data} is empty. Please check {path_data}."
                raise Exception(message)

            for img_path in list_img_path:
                json_path = img_path + ".json"
                ## Read and preprocess image
                if self.load_to_memory:
                    try:
                        img, _ = load_and_crop(img_path, self.input_size, index_layer=self.index_layer)
                        img = img.copy()
                        img = preprocess_input(img)

                    except Exception as e:
                        message = f'Due to {e} cannot load image: {img_path}'
                        raise Exception(message)

                else:
                    img = np.zeros((self.input_size, self.index_layer))

                try:
                    with open(json_path, encoding='utf-8') as json_file:
                        json_data = json.load(json_file)

                    if self.binary_option:
                        id_image = 'Fail' if json_data['classId'][0] in self.failClasses else 'Pass'

                    else:
                        id_image = json_data['classId'][0]

                    if "transform" in path_data.lower().split("\\")[-1]:
                        img_path_labels["transform_img_path"].append(img_path)
                        img_path_labels["transform_img_label"].append(id_image)
                        img_path_labels["transform_img_content"].append(img)

                    else:
                        img_path_labels["img_path"].append(img_path)
                        img_path_labels["img_label"].append(id_image)
                        img_path_labels["img_content"].append(img)

                except Exception as e:
                    message = f"Due to {e}. Please check your JSON file at {json_path} if it's in the correct format or it's missing"
                    # raise Exception(message)
                    img_path_labels["img_path"].append(img_path)
                    img_path_labels["img_label"].append("unclassified")
                    img_path_labels["img_content"].append(img)

            metadata_count(path_data, self.classes,
                           img_path_labels["transform_img_label"] if "transform" in path_data.lower().split("\\")[
                               -1] else img_path_labels["img_label"],
                           self.log, show_table=True)

            total_image = len(img_path_labels["transform_img_label"] if "transform" in path_data.lower().split("\\")[-1] else img_path_labels["img_label"])

            self.log.write_log(f'Total images loaded: {total_image} from {path_data}', message_type=0)

        return img_path_labels

class DataGenerator(Dataset):
    def __init__(self, data_dict, classes, testing=False, augmentation=None):
        """
            Args:

                data_dict (dict) : a dictionary contain :{
                                                            "path": [],
                                                            "img_path": [],
                                                            "img_label": [],
                                                            "img_content": [],
                                                            "transform_img_path": [],
                                                            "transform_img_label": [],
                                                            "transform_img_content": [],
                                                        }
                classes (list): list of class
                testing (bool) : return  image(False)/image path(True)
                augmentation (list): list augment option
        """
        self.img_path_labels = data_dict
        self.classes = classes
        self.augmentation = augmentation
        self.testing = testing

    def __len__(self):
        return len(self.img_path_labels['img_path'])
        # return 16

    def __getitem__(self, index):
        img_path = None
        try:
            image_name = self.img_path_labels["img_path"][index].split("\\")[-1]

            if self.augmentation and torch.randint(0, 2, (1,)).bool().item():
                img_path = os.path.join(self.img_path_labels["path"][0], "TransformImage", random.choice(self.augmentation)+"_"+image_name)
                img = self.img_path_labels["transform_img_content"][self.img_path_labels["transform_img_path"].index(img_path)]
                # list_img = self.img_path_labels["transform_img_content"]
                # index_img = self.img_path_labels["transform_img_path"].index(img_path)

            else:
                img_path = self.img_path_labels["img_path"][index]
                img = self.img_path_labels["img_content"][index]
                # list_img = self.img_path_labels["img_content"]
                # index_img = index

            try:
                single_label = self.classes.index(self.img_path_labels["img_label"][index]) # Pytorch don't use one-hot label



            except:
                # If the class name not in the current class pool
                single_label = len(self.classes)

            if self.testing:
                return img_path, single_label

            else:
                if np.sum(img) == 0:
                    img, _ = load_and_crop(img_path, img.shape[0], index_layer=img.shape[1])
                    img = img.copy()
                    img = preprocess_input(img)
                    # We update the image into memory so that we won't load that image next time
                    # list_img[index_img] = img

                return img, single_label

        except Exception as e:
            if img_path is not None:
                message = f"{e} at {img_path}"
            else:
                message = f"{e}"
            raise Exception(message)

