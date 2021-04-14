import cv2
import numpy as np
import os
import glob
import argparse
import json
import shutil
from utils.utils import load_and_crop

def convert_image_to_npz(dataset):
    print("Cropping image....")
    list_path = [
        # os.path.join(dataset, "Train\\OriginImage"),
        # os.path.join(dataset, "Train\\TransformImage"),
        os.path.join(dataset, "Validation"),
        # os.path.join(dataset, "Test"),
    ]
    
    for subpath in list_path:
        target_path  = os.path.join(dataset, "Cropped_image_from_{}".format(subpath.split("\\")[-1]))
        os.makedirs(target_path, exist_ok=True)
        for image_path in glob.glob(os.path.join(subpath, "*.bmp")):
            image_name = image_path.split("\\")[-1]
            img, _ = load_and_crop(image_path,256)
            cv2.imwrite(os.path.join(target_path, image_name) ,img )
            shutil.copy(image_path +".json",os.path.join(target_path, image_name +".json"))
    print("Finish")

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="Convert data type")

    parser.add_argument("--dataset", required=False, help="Physical path need to work with")
    
    args = parser.parse_args()

    convert_image_to_npz(args.dataset)