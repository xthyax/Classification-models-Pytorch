from __future__ import division
import sys
sys.path.append("..")
import os
import glob
import json

from classification_nn_pytorch import EfficientNet
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics

from utils.data_generator import DataGenerator
from utils.custom_dataloader import FastDataLoader
# from .augmentation_setup import custom_augment
from utils.utils import FocalLoss, load_and_crop, preprocess_input, multi_threshold,\
    CustomDataParallel
from .callback import SaveModelCheckpoint
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm.autonotebook import tqdm
import itertools
import pandas as pd
import numpy as np
import random
import torch_optimizer
import shutil
import xlsxwriter
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import cv2
from datetime import datetime
import time
import ttach as tta


# torch.backends.cudnn.enabled = False
# torch.cuda.synchronize()
def seed_torch(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    np.random.seed(1)
    # pass

class EfficientNetWrapper:
    def __init__(self, config):
        self.config = config
        self.classes = config.CLASS_NAME
        self.input_size = config.INPUT_SIZE
        self.binary_option = config.BINARY
        self.failClasses = config.FAIL_CLASSNAME
        self.passClasses = config.PASS_CLASSNAME
        self.pytorch_model = None
        self.num_of_classes = len(self.classes)
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.id_class_mapping = None
        self.class_weights = None
        self.evaluate_generator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.padding_crop = round(self.input_size / 5) if round(self.input_size / 5)  % 2 == 0 else  round(self.input_size / 5) - 1
        self.padding_crop = 0
        self.tta_option = tta.Compose([
                    # tta.FiveCrops(self.input_size, self.input_size),
                    tta.Rotate90([0, 90, 180, 270]),
                    # tta.HorizontalFlip(),
                    # tta.VerticalFlip()
                ])
        
    def _build_model(self):
        try:
            model_class = {
                'B0': 'efficientnet-b0',
                'B1': 'efficientnet-b1',
                'B2': 'efficientnet-b2',
                'B3': 'efficientnet-b3',
                'B4': 'efficientnet-b4',
                'B5': 'efficientnet-b5',
                'B6': 'efficientnet-b6',
                'B7': 'efficientnet-b7',
                'B8': 'efficientnet-b8',
            }[self.config.ARCHITECTURE]
        except KeyError:
            raise ValueError('Invalid Model architecture')

        if self.config.WEIGHT_PATH and "startingmodel.pth" in self.config.WEIGHT_PATH.lower():
            base_model = EfficientNet.from_pretrained(model_class,
                                                      weights_path=self.config.WEIGHT_PATH,
                                                      advprop=False,
                                                      num_classes=len(self.classes)
                                                      , image_size=self.config.INPUT_SIZE
                                                      )

        elif self.config.WEIGHT_PATH:
            base_model = EfficientNet.from_name(model_class,
                                                num_classes=len(self.classes)
                                                , image_size=self.config.INPUT_SIZE
                                                )

            base_model.load_state_dict(torch.load(self.config.WEIGHT_PATH, map_location=self.device))

        else:
            base_model = EfficientNet.from_name(model_class,
                                                num_classes=len(self.classes)
                                                , image_size=self.config.INPUT_SIZE
                                                )

            # base_model = torch.load(self.config.WEIGHT_PATH)
        return base_model

    def load_classes(self):
        if self.binary_option:
            init_class = ['Reject','Pass']
            self.classes = init_class
            self.num_of_classes = len(init_class)
            self.id_class_mapping = {class_id: class_name for class_id, class_name in enumerate(init_class)}

        else:
            self.num_of_classes = len(self.classes)
            self.id_class_mapping = {class_id: class_name for class_id, class_name in enumerate(self.classes)}

    def check_path(self, list_Directory, list_Generator, path_to_check):
        return [list_Generator[s_value] for s_value in [value for value in [list_Directory.index(set_path) for set_path in list_Directory if path_to_check in set_path.split("\\")[-1].lower()]]]

    def prepare_data(self, test_time=False):
        self.load_classes()

        list_Directory = [
            os.path.join(self.config.DATASET_PATH, 'Train'),
            os.path.join(self.config.DATASET_PATH, 'Validation'),
            os.path.join(self.config.DATASET_PATH, 'Test'),
            os.path.join(self.config.DATASET_PATH, 'MassData_25Sept_bmp'),
            # os.path.join(self.config.DATASET_PATH, 'Part4_updated'),
            # os.path.join(self.config.DATASET_PATH, 'Part5'),
            # os.path.join(self.config.DATASET_PATH, 'Part6_Steven'),
            # os.path.join(self.config.DATASET_PATH, 'Gerd_Underkill_bmp'),
        ]
        
        # Remove empty folder from default folder list
        list_Generator = []
        for diRectory in list_Directory.copy():
            if not os.path.exists(diRectory) or len(os.listdir(diRectory)) == 0:
                list_Directory.remove(diRectory)

        # Make generator for every available directory
        for diRectory in list_Directory:
            generator = DataGenerator(diRectory, self.classes, 
                        self.failClasses, self.passClasses, self.input_size, 
                        self.binary_option, testing=test_time, 
                        augmentation=self.config.AU_LIST if "train" in diRectory.split("\\")[-1].lower() else None )
            
            list_Generator.append(generator)
        
        check_train = self.check_path(list_Directory, list_Generator, "train")
        self.train_generator = check_train[0] if len(check_train) > 0 else None
        
        check_val = self.check_path(list_Directory, list_Generator, "validation")
        self.val_generator = check_val[0] if len(check_val) > 0 else None
        
        check_test = self.check_path(list_Directory, list_Generator, "test")
        self.test_generator = check_test[0] if len(check_test) > 0 else None
            
        self.evaluate_generator =  DataGenerator(list_Directory,
        self.classes, self.failClasses, self.passClasses,
        self.input_size + self.padding_crop, self.binary_option, testing=test_time)

        # self.class_weights = compute_class_weight('balanced',self.train_generator.metadata[0], self.train_generator.metadata[1])

        return list_Generator

    def optimizer_chosen(self, model_param):
        try:
            optimizer_dict = {
                'sgd': optim.SGD(params= model_param, lr=self.config.LEARNING_RATE, momentum=0.9, nesterov=True),
                'adam': optim.Adam(params=model_param, lr=self.config.LEARNING_RATE),
                'adadelta': optim.Adadelta(params=model_param, lr=self.config.LEARNING_RATE),
                'adagrad': optim.Adagrad(params=model_param, lr=self.config.LEARNING_RATE),
                'adamax': optim.Adamax(params=model_param, lr=self.config.LEARNING_RATE),
                'adamw': optim.AdamW(params=model_param, lr=self.config.LEARNING_RATE),
                'asgd': optim.ASGD(params=model_param, lr=self.config.LEARNING_RATE),
                'rmsprop': optim.RMSprop(params=model_param, lr=self.config.LEARNING_RATE, weight_decay=1e-5, momentum=0.9),
                'radam': torch_optimizer.RAdam(params=model_param, lr=self.config.LEARNING_RATE),
                'ranger': torch_optimizer.Ranger(params=model_param, lr=self.config.LEARNING_RATE)
            }[self.config.OPTIMIZER.lower()]

            return optimizer_dict
        except KeyError:
            print("Invalid optimizers")

    def _get_data_loader(self):

        num_worker = min(8,self.config.GPU_COUNT * 2) if self.config.GPU_COUNT > 1 else 0

        seed_torch()
        trainloader = FastDataLoader(self.train_generator, pin_memory=False,
            worker_init_fn= _init_fn,
            batch_size=self.config.BATCH_SIZE * self.config.GPU_COUNT, shuffle=True, 
            num_workers=num_worker)
            # num_workers= 0)

        seed_torch()
        valloader = FastDataLoader(self.val_generator, pin_memory=False,
            worker_init_fn= _init_fn,
            batch_size=self.config.BATCH_SIZE * self.config.GPU_COUNT, shuffle=False, 
            num_workers=num_worker)
            # num_workers= 0)
        
        seed_torch()
        evalloader = FastDataLoader(self.evaluate_generator, pin_memory=False,
            worker_init_fn= _init_fn,
            batch_size=self.config.BATCH_SIZE * self.config.GPU_COUNT, shuffle=False, 
            num_workers=num_worker)
            # num_workers= 0)
            # num_workers=12)
        return trainloader, valloader, evalloader

    def _train_one_epoch(self, epoch, data_loader, criterion, optimizer):

        running_loss = AverageMeter('Loss')
        running_correct = AverageMeter('Acc')

        progress_bar = tqdm(data_loader)

        self.pytorch_model.train()

        for iter, data in enumerate(progress_bar):
            inputs, labels = data[0], data[1]

            if self.config.GPU_COUNT == 1:
                inputs = inputs.to(self.device, non_blocking=True)
                # labels = labels.to(self.device, non_blocking=True)
                
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = self.pytorch_model(inputs)

            labels = labels.to(outputs.device, non_blocking=True)
    
            # forward + backward + optimize
            _, preds = torch.max(outputs,1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item() * inputs.size(0), inputs.size(0))
            running_correct.update(torch.sum(preds == labels.data).double(), inputs.size(0))

            # Update running bar
            
            progress_bar.set_description(\
            'Epoch: {}/{}. {}: {:.5}. {}: {:.5}'.format(\
            epoch, self.config.NO_EPOCH, \
            running_loss.dict_return()['name'] ,running_loss.dict_return()['avg'], \
            running_correct.dict_return()['name'] ,running_correct.dict_return()['avg']))

            progress_bar.update()

        return running_correct, running_loss 

    def _validate_one_epoch(self, data_loader, criterion):

        running_val_loss = AverageMeter('Val_Loss')
        running_val_acc = AverageMeter('Val_Acc')

        self.pytorch_model.eval()

        with torch.no_grad():
            for val_data in data_loader:
                inputs_val , labels_val = val_data[0], val_data[1]

                if self.config.GPU_COUNT == 1:
                    inputs_val = inputs_val.to(self.device, non_blocking=True)
                    # labels_val = labels_val.to(self.device, non_blocking=True)

                outputs_val = self.pytorch_model(inputs_val)
                labels_val = labels_val.to(outputs_val.device, non_blocking=True)

                val_score, val_preds = torch.max(outputs_val, 1)

                loss = criterion(outputs_val, labels_val)

                running_val_loss.update(loss.item() * inputs_val.size(0), inputs_val.size(0))
                running_val_acc.update(torch.sum(val_preds == labels_val.data).double(), inputs_val.size(0))

                # Accurate per class - toggle if you need

                # for i in range(min(self.config.BATCH_SIZE, inputs_val.size(0))):
                #     label = labels_val[i]
                #     class_correct[label] += c[i].item()
                #     class_total[label] += 1.
    
        print(f"val Loss: {running_val_loss.dict_return()['avg']} val Acc: { running_val_acc.dict_return()['avg']}")
        
        return running_val_acc, running_val_loss
    
    def _custom_evaluate_one_epoch(self, data_loader, fail_class_index, pass_class_index):
        # Evaluate metric part
        custom_dict ={
            "y_pred": [],
            "y_gtruth": []
        }
        print("====================================")
        print("Calculating FN/FP rate.....")
        print("====================================")
        self.pytorch_model.eval()

        if self.config.GPU_COUNT > 1:
            # evaluate_model = self.load_weight(weight_path=weight_path, multi_gpus=False)
            evaluate_model = self.pytorch_model.module
            evaluate_model.eval()
            tta_model = tta.ClassificationTTAWrapper(evaluate_model, self.tta_option)
            tta_model = nn.DataParallel(tta_model)
        else:
            evaluate_model = self.pytorch_model
            tta_model = tta.ClassificationTTAWrapper(evaluate_model, self.tta_option)

        tta_model.eval()
        tta_opt = True

        with torch.no_grad():
            
            draft_img = preprocess_input(np.zeros((self.input_size, self.input_size, 3)).astype(np.float32))
            if self.config.GPU_COUNT >1:
                pass
            else:
                draft_img = draft_img.to(self.device, non_blocking=True)

            if self.tta_opt:
                tta_model(draft_img.unsqueeze(0))
            else:
                self.pytorch_model(draft_img.unsqueeze(0))
                
            start_time_eval = datetime.now()
            for eval_data in data_loader:
                if tta_opt:
                    inputs_eval , labels_eval = eval_data[0], eval_data[1]

                    if self.config.GPU_COUNT == 1:
                        inputs_eval = inputs_eval.to(self.device, non_blocking=True)
                    
                    outputs_eval = tta_model(inputs_eval)
                    labels_eval = labels_eval.to(outputs_eval.device, non_blocking=True)
                    _, eval_preds = torch.max(outputs_eval, 1)

                    custom_dict["y_pred"].extend(eval_preds.tolist())

                else:
                    inputs_eval , labels_eval = eval_data[0], eval_data[1]

                    if self.config.GPU_COUNT == 1:
                        inputs_eval = inputs_eval.to(self.device, non_blocking=True)

                    labels_eval = labels_eval.to(self.device, non_blocking=True)
                    outputs_eval = self.pytorch_model(inputs_eval)
                    _, eval_preds = torch.max(outputs_eval, 1)
                    
                    custom_dict["y_pred"].extend(eval_preds.tolist())

                custom_dict["y_gtruth"].extend(labels_eval.tolist())

        def _FP_FN_metric(y_pred, y_gth, specific_class_index):
            # Get positive groundtruth
            gth_list  = [np.array(y_gth) == class_ for class_ in specific_class_index]
            gth_list = np.sum(gth_list, axis=0)

            # Get total positive groundtruth
            total_gth = np.sum(gth_list)

            # Get positive prediction
            false_pred_list = [np.array(y_pred) == class_ for class_ in specific_class_index]

            # Invert positive to negative <-> negative to positive
            false_pred_list = np.invert(np.sum(false_pred_list, axis=0).astype('bool'))

            # Get False negative prediction
            false_pred_list = false_pred_list * gth_list
            total_false_predict = np.sum(false_pred_list)
            _ratio = (total_false_predict / total_gth) * 100

            return _ratio

        UK_ratio = _FP_FN_metric(custom_dict["y_pred"], custom_dict["y_gtruth"], fail_class_index)

        OK_ratio = _FP_FN_metric(custom_dict["y_pred"], custom_dict["y_gtruth"], pass_class_index)
        
        end_time_eval = datetime.now()
        print(f"Evaluating time : {end_time_eval-start_time_eval}")

        print(f"False Pass rate: {UK_ratio} %")
        print(f"False Reject rate: {OK_ratio} %")

        try:
            del evaluate_model
        except NameError:
            pass

        return UK_ratio, OK_ratio

    def train(self):
        train_checkpoint_dir = self.config.LOGS_PATH
        os.makedirs(train_checkpoint_dir,exist_ok=True)

        trainloader, valloader, evalloader = self._get_data_loader()
        
        if self.config.GPU_COUNT > 1:
            self.pytorch_model = self._build_model().to(self.device)
            self.pytorch_model = nn.DataParallel(self.pytorch_model)
        else:
            self.pytorch_model = self._build_model().to(self.device)
        
        criterion = nn.CrossEntropyLoss().to(self.device)
        print(f"[DEBUG] class_weight : {self.class_weights}")
        # criterion = FocalLoss().to(self.device)
        
        model_parameters = list(self.pytorch_model.parameters())

        optimizer = self.optimizer_chosen(model_parameters)

        # Init tensorboard
        writer = SummaryWriter(log_dir=train_checkpoint_dir)

        self.failClasses = ['Reject'] if self.binary_option else self.failClasses
        self.passClasses = ['Pass'] if self.binary_option else self.passClasses
        pass_class_index = [self.classes.index(class_) for class_ in self.passClasses]
        fail_class_index = [self.classes.index(class_) for class_ in self.failClasses]

        start_time = datetime.now()

        total_model_info = {}

        state_dict ={
            "lowest_OK_rate": 10,
            "model_count": 0,
            "best_val_loss": 100
        }

        # enumerate epoch
        try:
            for current_epoch in range(1, self.config.NO_EPOCH + 1):
                
                model_info = {}
                # Set determinism behavior
                seed_torch(current_epoch - 1)

                # Ignore this line for now (clean up later)
                # print(f"Epoch {current_epoch}/ {self.config.NO_EPOCH}")
                print('-' * 20)

                running_loss, running_correct = self._train_one_epoch(current_epoch, trainloader, criterion, optimizer)

                running_val_acc, running_val_loss = self._validate_one_epoch(valloader, criterion)
                
                UK_ratio, OK_ratio = self._custom_evaluate_one_epoch(evalloader, fail_class_index, pass_class_index)
                
                # Tensorboard part
                writer.add_scalars('Loss',{'Train': running_loss.dict_return()['avg'],\
                                    'Val' : running_val_loss.dict_return()['avg']}, current_epoch)

                writer.add_scalars('Acc', {'Train': running_correct.dict_return()['avg'],\
                                            'Val' : running_val_acc.dict_return()['avg']}, current_epoch)

                writer.add_scalars('Custom',{"False_Pass_rate": UK_ratio,\
                                             "False_Reject_rate": OK_ratio}, current_epoch)

                writer.flush()
                # Save model part
                if self.config.IS_SAVE_BEST_MODELS:
                    if state_dict["best_val_loss"] > running_val_loss.dict_return()['avg']:
                        model_name = SaveModelCheckpoint(self.pytorch_model, self.config.LOGS_PATH, current_epoch, state_dict["best_val_loss"], True)
                    else:
                        pass
                else:
                    model_name = SaveModelCheckpoint(self.pytorch_model, self.config.LOGS_PATH, current_epoch)

                # Add-on when have acceptable UK/OK rate or have best val_loss
                if (UK_ratio <= 0.1 and OK_ratio <= state_dict["lowest_OK_rate"]) or state_dict["best_val_loss"] > running_val_loss.dict_return()['avg']:
                    # os.path.join(*(x.split(os.path.sep)[2:]))
                    state_dict["best_val_loss"] = running_val_loss.dict_return()['avg']
                    model_info["path"] = self.config.LOGS_PATH
                    model_info["model_name"] = model_name
                    model_info["Underkill_rate"] = UK_ratio
                    model_info["Overkill_rate"] = OK_ratio
                    model_info["Val_loss"] = running_val_loss.dict_return()['avg']
                    # min_OK_rate = OK_rate
                    state_dict["model_count"] += 1
                    model_n = "model_" + str(state_dict["model_count"])
                    total_model_info[model_n] = model_info
                    # total_model_info.update( model_n : model_info)
                with open(os.path.join(self.config.LOGS_PATH , f"model_info.json"), "w") as model_json:
                    json.dump(total_model_info, model_json)

        except KeyboardInterrupt:
            if self.config.IS_SAVE_BEST_MODELS:
                pass
            else:
                SaveModelCheckpoint(self.pytorch_model, self.config.LOGS_PATH, current_epoch)

        writer.close()
        end_time = datetime.now()
        print("Training time: {}".format(end_time-start_time))
    
    def evaluate(self):
        pass

    def predict_one(self, img, TTA=True):
        # img, _ = load_and_crop(img_path, input_size=self.input_size + self.padding_crop)
        self.pytorch_model.eval()
        with torch.no_grad():
            if TTA:
                img = preprocess_input(img).to(self.device, non_blocking=True)
                tta_model = tta.ClassificationTTAWrapper(self.pytorch_model, self.tta_option)

                outputs = tta_model(img.unsqueeze(0).to(self.device, non_blocking=True))
                propability = torch.nn.Softmax(dim = 1)(outputs)

                return self.manage_prediction(propability.tolist())
            else:

                img = preprocess_input(img).to(self.device, non_blocking=True)
                outputs = self.pytorch_model(img.unsqueeze(0))

                propability = torch.nn.Softmax(dim = 1)(outputs)
                # return propability[0]
                return self.manage_prediction(propability.tolist())
                # print(f"[DEBUG] propability :{propability}")

    def manage_prediction(self, propability_prediction):
        if self.config.CLASS_THRESHOLD is None or len(self.config.CLASS_THRESHOLD) == 0:
            prob_id = np.argmax(propability_prediction, axis=-1)
        else:
            ret = multi_threshold(np.array(propability_prediction), self.config.CLASS_THRESHOLD)
            if ret is None:
                # classID = len(self.config.CLASS_THRESHOLD)
                classID = -1
                className = "Unknown"
                all_scores = propability_prediction[0]
                return classID, all_scores, className
            else:
                prob_id, _ = ret

        class_name = self.id_class_mapping[prob_id[0]]
      
        return prob_id[0], propability_prediction[0], class_name

    def load_weight(self):
        self.load_classes()
        self.pytorch_model = self._build_model().to(self.device)
        
    def confusion_matrix_evaluate(self):
        generator_list = self.prepare_data(test_time=True)

        self.pytorch_model.eval()
        
        model_file_name = self.config.WEIGHT_PATH.split("\\")[-1].split(".")[0]

        workbook = xlsxwriter.Workbook("_model_" + model_file_name +"_result.xlsx")

        cell_format = workbook.add_format()
        cell_format.set_align('center')
        cell_format.set_align('vcenter')
        cell_format.set_text_wrap()

        highlight_format = workbook.add_format()
        highlight_format.set_align('center')
        highlight_format.set_align('vcenter')
        highlight_format.set_bg_color("red")

        Header = ["image_id","Image","Label","Predict"]
        Header.extend(self.classes)
        Header.append("Underkill")
        Header.append("Overkill")

        image_size_display = int(self.input_size * 1.5) # 192 as Default

        self.failClasses = ["Reject"] if self.binary_option else self.failClasses
        self.passClasses = ["Pass"] if self.binary_option else self.passClasses

        fail_class_index = [self.classes.index(class_) for class_ in self.failClasses]
        pass_class_index = [self.classes.index(class_) for class_ in self.passClasses]

        time_statistic = pd.DataFrame(columns=['Execution_time'])

        for generator in generator_list:
            generator_loader = FastDataLoader(generator, batch_size=1, shuffle=False, num_workers=0)
            # generator_loader = DataLoader(generator, batch_size=1, shuffle=False, num_workers=0)
            print(f"Inspecting PATH: {generator.input_dir}")

            start_row = 0
            start_column = 1
            worksheet = workbook.add_worksheet(generator.input_dir[0].split("\\")[-1])
            worksheet.write_row( start_row, start_column, Header, cell_format)
            worksheet.set_column("B:B", 15)
            worksheet.set_column("C:C", int(15 * (image_size_display / 192)) )

            progress_bar = tqdm(generator_loader)
            y_gth_eval_ls = []
            y_pred_eval_ls = []

            for iter, data_eval in enumerate(progress_bar):
                with torch.no_grad():
                    Data = [0] * len(Header)
                    start_row += 1
                    worksheet.set_row(start_row, int(90 * (image_size_display / 192)) )
                    underkill_overkill_flag = 0
                    # print(data_eval[1])
                    # print(data_eval)
                    image_path , labels_eval = data_eval[0][0], \
                            data_eval[1].to(self.device, non_blocking=True)

                    image_name = image_path.split("\\")[-1]

                    img, gt_name = load_and_crop(image_path, self.input_size + self.padding_crop)

                    start_inference = time.time()
                    pred_id, all_scores, pred_name = self.predict_one(img)
                    end_inference = time.time()

                    inference_time = end_inference - start_inference
                    time_statistic = time_statistic.append({"Execution_time": inference_time}, ignore_index=True)

                    gt_id = labels_eval.tolist()[0]
                    
                    if self.binary_option:
                        gt_name = 'Reject' if gt_name in self.failClasses else 'Pass'
                    else:
                        pass

                    if gt_id in fail_class_index and (pred_id in pass_class_index or pred_id == len(self.classes)):    # Underkill
                        underkill_path = os.path.join("_Result",image_path.split("\\")[-2],"UK")
                        os.makedirs(underkill_path, exist_ok=True)
                        image_output_path = os.path.join(underkill_path,image_name)
                        img = cv2.resize(img, (image_size_display, image_size_display) )
                        cv2.imwrite(image_output_path, img)
                        shutil.copy(image_path + ".json", os.path.join(underkill_path,image_name+".json"))
                        underkill_overkill_flag = -1

                    elif gt_id in pass_class_index and pred_id in fail_class_index:     # Overkill
                        overkill_path = os.path.join("_Result",image_path.split("\\")[-2],"OK")
                        os.makedirs(overkill_path, exist_ok=True)
                        image_output_path = os.path.join(overkill_path,image_name)
                        img = cv2.resize(img, (image_size_display, image_size_display) )
                        cv2.imwrite(image_output_path, img)
                        shutil.copy(image_path + ".json", os.path.join(overkill_path,image_name+".json"))
                        underkill_overkill_flag = 1
                    
                    else:                                                               # Correct result
                        result_path = os.path.join("_Result", image_path.split("\\")[-2])
                        os.makedirs(result_path, exist_ok=True)
                        image_output_path = os.path.join(result_path,image_name)
                        img = cv2.resize(img, (image_size_display , image_size_display) )
                        cv2.imwrite(image_output_path, img)
                        shutil.copy(image_path + ".json", os.path.join(result_path,image_name + ".json"))

                    y_gth_eval_ls.extend(labels_eval.tolist())
                    y_pred_eval_ls.extend([pred_id])

                    Data[0] = image_name.split(".")[0]
                    Data[2] = gt_name
                    Data[3] = pred_name
                    Data[4:4+len(self.classes)] = all_scores
                    Data[-2] = True if underkill_overkill_flag == -1 else False
                    Data[-1] = True if underkill_overkill_flag == 1 else False
                    # print(f"[DEBUG]:\n{Data}")
                    for index, info in enumerate(Data):
                        
                        excel_format = highlight_format if (Data[index] == True and isinstance(Data[index],bool)) else cell_format

                        worksheet.insert_image(start_row, index + 1, image_output_path, {'x_scale': 0.5,'y_scale': 0.5, 'x_offset': 5, 'y_offset': 5,'object_position':1}\
                            ) if index == 1 else worksheet.write(start_row, index + 1, Data[index], excel_format)

                    progress_bar.update()

            header = [{'header': head} for head in Header]

            worksheet.add_table(0, 1, start_row, len(Header), {'columns':header})
            worksheet.freeze_panes(1,0)
            worksheet.hide_gridlines(2)

            confusion_matrix = metrics.confusion_matrix(y_gth_eval_ls, y_pred_eval_ls)
            print(f"Confusion matrix : \n{confusion_matrix}")

        workbook.close()
        time_statistic.to_csv("Execution_time.csv", index=False)

    def labelling_raw_data(self):
        generator_list = self.prepare_data(test_time=True)

        self.pytorch_model.eval()

        self.failClasses = ["Reject"] if self.binary_option else self.failClasses
        self.passClasses = ["Pass"] if self.binary_option else self.passClasses

        fail_class_index = [self.classes.index(class_) for class_ in self.failClasses]
        pass_class_index = [self.classes.index(class_) for class_ in self.passClasses]

        # result_path = [
        #     os.path.join("_Labelled","Reject"),
        #     os.path.join("_Labelled","Pass"),
        #     os.path.join("_Labelled","Unclear")
        # ]

        for generator in generator_list:
            generator_loader = FastDataLoader(generator, batch_size=1, shuffle=False, num_workers=12)
            # generator_loader = DataLoader(generator, batch_size=1, shuffle=False, num_workers=0)
            print(f"Inspecting PATH: {generator.input_dir}")
            progress_bar = tqdm(generator_loader)

            for iter, data_eval in enumerate(progress_bar):
                with torch.no_grad():
                    image_path , labels_eval = data_eval[0][0], \
                            data_eval[1].to(self.device, non_blocking=True)

                    image_name = image_path.split("\\")[-1]

                    img, gt_name = load_and_crop(image_path, self.input_size)

                    pred_id, all_scores, pred_name = self.predict_one(img)

                    # print(f"[DEBUG] image id:\t{image_name}")
                    # print(f"[DEBUG] all scores:\t{all_scores}")
                    # print(f"[DEBUG] pred_id: {pred_id} - pred_score: {pred_score} -  pred_name: {pred_name}")
                    
                    if pred_id in pass_class_index or pred_id == len(self.classes):   # Pass
                        Pass_path = os.path.join("_Labelled","Pass")
                        os.makedirs(Pass_path, exist_ok=True)

                        shutil.copy(image_path, os.path.join(Pass_path,image_name))
                        json_path = image_path + ".json"

                        # with open(json_path, encoding='utf-8') as json_file:
                        #     json_data = json.load(json_file)
                        # json_data["classId"] = ["Pass"]

                        # with open(os.path.join(Pass_path,image_name) + ".json", "w") as bmp_json:
                        #     json.dump(json_data, bmp_json)

                    elif pred_id in fail_class_index:                    # Reject
                        Reject_path = os.path.join("_Labelled","Reject")
                        os.makedirs(Reject_path, exist_ok=True)
                        
                        shutil.copy(image_path, os.path.join(Reject_path,image_name))
                        json_path = image_path + ".json"

                        # with open(json_path, encoding='utf-8') as json_file:
                        #     json_data = json.load(json_file)
                        # json_data["classId"] = ["Burr"]

                        # with open(os.path.join(Reject_path,image_name) + ".json", "w") as bmp_json:
                        #     json.dump(json_data, bmp_json)
                    else:                                                       # Unknow
                        Unclear_path = os.path.join("_Labelled","Unknow")
                        os.makedirs(Unclear_path, exist_ok=True)
                        
                        shutil.copy(image_path, os.path.join(Unclear_path,image_name))
                        json_path = image_path + ".json"

                        # with open(json_path, encoding='utf-8') as json_file:
                        #     json_data = json.load(json_file)
                        # json_data["classId"] = ["Unclear"]

                        # with open(os.path.join(Unclear_path,image_name) + ".json", "w") as bmp_json:
                        #     json.dump(json_data, bmp_json)

                    progress_bar.update()

        print("[INFO] Done")

    def cam_testing(self, img_path):
        from visualisation.core import ClassActivationMapping, GradCam
        from visualisation.core.utils import image_net_postprocessing, imshow
        img, _ = load_and_crop(img_path, input_size=self.input_size + self.padding_crop)
        self.pytorch_model.eval()
        # print(self.pytorch_model)
        # vis = ClassActivationMapping(self.pytorch_model, self.device)
        vis = GradCam(self.pytorch_model, self.device)
        img = preprocess_input(img).to(self.device, non_blocking=True)
        img_w_hm, prediction = vis(img.unsqueeze(0), None, postprocessing=image_net_postprocessing)
        
        # img_w_hm = img_w_hm.squeeze().permute(1,2,0)
        # print(img_w_hm.numpy())
        print(prediction)
        imshow(img_w_hm)
        # print(features.size())

    def checking_models(self):
        seed_torch()
        evalloader = FastDataLoader(self.evaluate_generator, pin_memory=False,\
            # worker_init_fn= torch.initial_seed(),\
            worker_init_fn= _init_fn,\
            batch_size=self.config.BATCH_SIZE * self.config.GPU_COUNT, shuffle=False, num_workers=12)
        model_path_ls = []
        
        for path_trained_model in glob.glob(os.path.join(self.config.WEIGHT_PATH, "*.pth")):
            model_path_ls.append(path_trained_model)

        fail_ls_class =  ['Reject'] if self.binary_option else self.failClasses
        pass_ls_class = ['Pass'] if self.binary_option else self.passClasses

        total_model_info = {}

        min_OK_rate = 8

        count_model = 0
        for trained_model in model_path_ls:
            self.config.WEIGHT_PATH = trained_model
        
            model_info = {
                "path": [],
                "model_name": [],
                "Underkill_rate": [],
                "Overkill_rate": []
            }

            self.load_weight()

            fail_class_index = [self.classes.index(item_class) for item_class in fail_ls_class]
            pass_class_index = [self.classes.index(item_class) for item_class in pass_ls_class]
            
            y_gth_list = []
            y_pred_list = []
            
            print("====================================")
            print(f"Testing model : {self.config.WEIGHT_PATH}.....")

            self.pytorch_model.eval()
            tta_model = tta.ClassificationTTAWrapper(self.pytorch_model, self.tta_option).eval()
            tta_opt = True
            with torch.no_grad():
                draft_img = preprocess_input(np.zeros((self.input_size, self.input_size, 3)).astype(np.float32)).to(self.device, non_blocking=True)
                tta_model(draft_img.unsqueeze(0))
                start_time_eval = datetime.now()
                for eval_data in evalloader:
                    # inputs_eval , labels_eval = eval_data[0], eval_data[1]
                    

                    # if self.config.GPU_COUNT == 1:
                    #     inputs_eval = inputs_eval.to(self.device, non_blocking=True)
                    # labels_val = labels_val.to(self.device, non_blocking=True)

                    # print(img_path_eval)
                    # print(type(img_path_eval))
                    # print(len(img_path_eval))
                    # print(inputs_eval.size())
                    if tta_opt:
                        inputs_eval , labels_eval = eval_data[0], eval_data[1]

                        if self.config.GPU_COUNT == 1:
                            inputs_eval = inputs_eval.to(self.device, non_blocking=True)
                        
                        outputs_eval = tta_model(inputs_eval)
                        labels_eval = labels_eval.to(outputs_eval.device, non_blocking=True)
                        _, eval_preds = torch.max(outputs_eval, 1)
                        y_pred_list.extend(eval_preds.tolist())
                        # ************************************************************************
                        # img_path_eval , labels_eval = eval_data[0], eval_data[1]
                        # batch_preds = self.predict_batch(img_path_eval)
                        # y_pred_list.extend(batch_preds)
                    else:
                        inputs_eval , labels_eval = eval_data[0], eval_data[1]

                        if self.config.GPU_COUNT == 1:
                            inputs_eval = inputs_eval.to(self.device, non_blocking=True)

                        labels_val = labels_val.to(self.device, non_blocking=True)
                        outputs_eval = self.pytorch_model(inputs_eval)
                        _, eval_preds = torch.max(outputs_eval, 1)
                        y_pred_list.extend(eval_preds.tolist())

                    y_gth_list.extend(labels_eval.tolist())
                        
            fail_gth_list  = [np.array(y_gth_list) == class_ for class_ in fail_class_index]
            fail_gth_list = np.sum(fail_gth_list, axis=0)
            total_fail = np.sum(fail_gth_list)
            false_fail_pred_list = [np.array(y_pred_list) == class_ for class_ in fail_class_index]
            false_fail_pred_list = np.invert(np.sum(false_fail_pred_list, axis=0).astype('bool'))
            false_fail_pred_list = false_fail_pred_list * fail_gth_list
            total_underkill = np.sum(false_fail_pred_list)
            UK_rate = (total_underkill / total_fail) * 100

            pass_gth_list = [np.array(y_gth_list) == class_ for class_ in pass_class_index]
            pass_gth_list = np.sum(pass_gth_list, axis=0)
            total_pass = np.sum(pass_gth_list)
            false_pass_pred_list = [np.array(y_pred_list) == class_ for class_ in pass_class_index]
            false_pass_pred_list = np.invert(np.sum(false_pass_pred_list, axis=0).astype('bool'))
            false_pass_pred_list = false_pass_pred_list  * pass_gth_list
            total_overkill = np.sum(false_pass_pred_list)
            OK_rate = (total_overkill / total_pass ) * 100

            end_time_eval = datetime.now()
            print(f"Evaluating time : {end_time_eval-start_time_eval}")

            print(f"Underkill rate: {UK_rate} %")
            print(f"Overkill rate: {OK_rate} %")
            print("====================================")
            print("\n")
            if UK_rate == 0.0 and OK_rate <= min_OK_rate:
                # os.path.join(*(x.split(os.path.sep)[2:]))
                model_info["path"] = os.path.join(*(self.config.WEIGHT_PATH.split("\\")[:-1]))
                model_info["model_name"] = [self.config.WEIGHT_PATH.split("\\")[-1]]
                model_info['Underkill_rate'] = [UK_rate]
                model_info['Overkill_rate'] = [OK_rate]
                # min_OK_rate = OK_rate
                count_model += 1
                model_n = "model_" + str(count_model)
                total_model_info[model_n] = model_info
                # total_model_info.update( model_n : model_info)

        with open("model_info.json", "w") as model_json:
            json.dump(total_model_info, model_json)

class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self, name):
        self.name = name 
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def dict_return(self):
        return {'name': self.name, 'value':self.val, 'count': self.count,'avg': self.avg}
                   
            

    