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

from utils.data_generator import DataGenerator, LoadingData
from utils.custom_dataloader import FastDataLoader
# from .augmentation_setup import custom_augment
from utils.utils import (
    FocalLoss, load_and_crop, preprocess_input, multi_threshold,
    CustomDataParallel, _FP_FN_metric, 
    _check_Fail_Pass_class, GetCenter_MVSD, 
    heatmap2bbox, plot_one_box
    )
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
from tensorboardX import SummaryWriter
import cv2
from datetime import datetime
import ttach as tta
import time


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


class EfficientNetWrapper:
    def __init__(self, config):
        self.config = config
        self.classes = config.CLASS_NAME
        self.input_size = config.INPUT_SIZE
        self.binary_option = config.BINARY
        self.failClasses = config.FAIL_CLASSNAME
        self.passClasses = config.PASS_CLASSNAME
        self.pass_class_index = [self.classes.index(class_) for class_ in self.passClasses]
        self.fail_class_index = [self.classes.index(class_) for class_ in self.failClasses]
        self.pytorch_model = None
        # self.train_generator = None
        # self.val_generator = None
        # self.test_generator = None
        self.class_weights = None
        self.evaluate_generator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tta_rotate_opt_list = [0, 90, 180, 270]
        self.tta_option = tta.Compose([
            tta.Rotate90(self.tta_rotate_opt_list),  # For future developing with CAM
            # tta.HorizontalFlip(),
            # tta.VerticalFlip()
        ])
        # Toggle TTA option
        self.tta_opt = True
        self.global_batch_size = self.config.BATCH_SIZE * self.config.GPU_COUNT if self.config.GPU_COUNT != 0 else self.config.BATCH_SIZE

    def _build_model(self, weight_path=None):
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

        except Exception as e:
            message = f'Invalid Model architecture due to {e}'
            raise Exception(message)

        ## Logic build model
        is_scratch = self.config.WEIGHT_PATH is None
        is_pretrained = not is_scratch and "startingmodel.pth" in self.config.WEIGHT_PATH.split("\\")[-1].lower() \
                        and weight_path is None

        ## Load model from pretrained
        if is_pretrained:
            self.config.LOGGING.write_log(f"Loading pretrained model from {self.config.WEIGHT_PATH}", message_type=0)
            base_model = EfficientNet.from_pretrained(model_class,
                                                      weights_path=self.config.WEIGHT_PATH,
                                                      advprop=False,
                                                      num_classes=len(self.classes),
                                                      image_size=self.config.INPUT_SIZE
                                                      )

        ## Load model from scratch
        else:

            base_model = EfficientNet.from_name(model_class,
                                                num_classes=len(self.classes),
                                                image_size=self.config.INPUT_SIZE
                                                )

            ## Load model from specific path
            if not is_scratch:
                model_path = weight_path if weight_path is not None else self.config.WEIGHT_PATH
                try:
                    self.config.LOGGING.write_log(f"Loading model from {model_path}", message_type=0)
                    base_model.load_state_dict(
                        torch.load(model_path,
                                   map_location=self.device), strict=weight_path is not None)

                except Exception as e:
                    message = f"Cannot load model from {model_path} due to {e}"
                    raise Exception(message)

            else:
                self.config.LOGGING.write_log("Loading model from scratch", message_type=0)

        return base_model

    def check_model_mode(self):

        if self.binary_option:
            init_class = ['Fail', 'Pass']
            self.classes = init_class
            self.num_of_classes = len(init_class)
            self.id_class_mapping = {class_id: class_name for class_id, class_name in enumerate(init_class)}
            self.config.LOGGING.write_log("Using binary classes", message_type=0)

        else:
            self.num_of_classes = len(self.classes)
            self.id_class_mapping = {class_id: class_name for class_id, class_name in enumerate(self.classes)}
            self.config.LOGGING.write_log("Using original classes", message_type=0)

        self.failClasses = ['Fail'] if self.binary_option else self.failClasses
        self.passClasses = ['Pass'] if self.binary_option else self.passClasses
        self.fail_class_index = [self.classes.index(class_) for class_ in self.failClasses]
        self.pass_class_index = [self.classes.index(class_) for class_ in self.passClasses]

        self.config.LOGGING.write_log("Fail class: {}".format(self.failClasses), message_type=0)
        self.config.LOGGING.write_log("Pass class: {}".format(self.passClasses), message_type=0)


    def check_path(self, list_Directory, list_Generator, path_to_check):
        """ Deprecated """
        return [list_Generator[s_value] for s_value in
                [value for value in [list_Directory.index(set_path) for set_path in list_Directory
                                     if path_to_check in set_path.split("\\")[-1].lower()]]]

    def prepare_data(self, test_time=False):

        loadToMemory = False
        if loadToMemory:
            print("Load data into memory...")
        else:
            print("Load data on the fly...")

        list_Directory = [
            os.path.join(self.config.DATASET_PATH, 'Train'),
            os.path.join(self.config.DATASET_PATH, 'Validation'),
            os.path.join(self.config.DATASET_PATH, 'Test'),
            os.path.join(self.config.DATASET_PATH, 'Test_stuff'),
        ]

        self.list_Generator = []
        self.list_data_dict = []

        for diRectory in list_Directory.copy():
            if not os.path.exists(diRectory) or len(os.listdir(diRectory)) == 0:
                list_Directory.remove(diRectory)

        # If point into folder don't have any structure
        if len(list_Directory) == 0:
            list_Directory.append(self.config.DATASET_PATH)

        for diRectory in list_Directory:
            params_dict = {
                "input_dir": diRectory,
                "classes": self.classes,
                "failClasses": self.failClasses,
                "passClasses": self.passClasses,
                "input_size": self.input_size,
                "binary_option": self.binary_option,
                "log": self.config.LOGGING,
                "index_layer": self.config.INDEX_TRAINING_LAYER,
                "loadToMemory": loadToMemory,
                "augmentation": self.config.AU_LIST if "train" in diRectory.split("\\")[
                    -1].lower() and test_time is False else None
            }

            data_dict = LoadingData(**params_dict).load_data()
            self.list_data_dict.append(data_dict)

            if "train" in diRectory.split("\\")[-1].lower():
                self.class_weights = compute_class_weight('balanced', self.classes, data_dict["img_label"])
                self.class_weights = self.class_weights / self.class_weights.sum()

        self.check_model_mode()

        for i, diRectory in enumerate(list_Directory):
            params_dict = {
                "data_dict": self.list_data_dict[i],
                "classes": self.classes,
                "testing": test_time,
                "augmentation": self.config.AU_LIST if "train" in diRectory.split("\\")[
                    -1].lower() and test_time is False else None
            }
            generator = DataGenerator(**params_dict)

            self.list_Generator.append(generator)

        # Create an evaluate_dict
        evaluate_data_dict = {}
        for i in range(len(self.list_data_dict)):
            for key in (self.list_data_dict[i].keys()):
                if key in self.list_data_dict[i]:
                    evaluate_data_dict.setdefault(key, []).extend(self.list_data_dict[i][key])

        evaluate_params_dict = {
            "data_dict": evaluate_data_dict,
            "classes": self.classes,
            "testing": test_time,
        }
        self.evaluate_generator = DataGenerator(**evaluate_params_dict)

        return self.list_Generator

    def optimizer_chosen(self, model_param):
        try:
            optimizer_dict = {
                'sgd': optim.SGD(params=model_param, lr=self.config.LEARNING_RATE,
                                 momentum=self.config.LEARNING_MOMENTUM, nesterov=True),
                'adam': optim.Adam(params=model_param, lr=self.config.LEARNING_RATE),
                'adadelta': optim.Adadelta(params=model_param, lr=self.config.LEARNING_RATE),
                'adagrad': optim.Adagrad(params=model_param, lr=self.config.LEARNING_RATE),
                'adamax': optim.Adamax(params=model_param, lr=self.config.LEARNING_RATE),
                'adamw': optim.AdamW(params=model_param, lr=self.config.LEARNING_RATE),
                'asgd': optim.ASGD(params=model_param, lr=self.config.LEARNING_RATE),
                'rmsprop': optim.RMSprop(params=model_param, lr=self.config.LEARNING_RATE),
                'radam': torch_optimizer.RAdam(params=model_param, lr=self.config.LEARNING_RATE),
                'ranger': torch_optimizer.Ranger(params=model_param, lr=self.config.LEARNING_RATE)
            }[self.config.OPTIMIZER.lower()]

            return optimizer_dict

        except Exception as e:
            message = f"Invalid optimizers {e}"
            raise Exception(message)

    def _train_one_epoch(self, dataloader, optimizer, criterion, current_epoch):

        running_loss = AverageMeter('Loss')
        running_acc = AverageMeter('Acc')

        # enumerate mini batch
        progress_bar = tqdm(dataloader)

        self.pytorch_model.train()

        # Disable batchnorm when batchsize = 1
        denominator = self.config.GPU_COUNT if self.config.GPU_COUNT != 0 else 1
        if self.global_batch_size // denominator == 1:
            if self.config.GPU_COUNT > 1:
                self.pytorch_model.eval()

            else:
                for m in self.pytorch_model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

        else:
            pass

        for iter, data in enumerate(progress_bar):
            inputs, labels = data[0], data[1]

            if self.config.GPU_COUNT == 1:
                inputs = inputs.to(self.device, non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = self.pytorch_model(inputs)

            labels = labels.to(outputs.device, non_blocking=True)

            # forward + backward + optimize
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss.update(loss.item() * inputs.size(0), inputs.size(0))
            running_acc.update(torch.sum(preds == labels.data).double(), inputs.size(0))

            # Update running bar

            progress_bar.set_description(
                'Epoch: {}/{}. {}: {:.5}. {}: {:.5}'.format(
                    current_epoch, self.config.NO_EPOCH,
                    running_loss.name, running_loss.avg,
                    running_acc.name, running_acc.avg))

            progress_bar.update()

        return running_loss, running_acc

    def _valid_one_epoch(self, dataloader, criterion):

        running_val_loss = AverageMeter('Val_Loss')
        running_val_acc = AverageMeter('Val_Acc')

        self.pytorch_model.eval()

        with torch.no_grad():
            for val_data in dataloader:
                inputs_val, labels_val = val_data[0], val_data[1]

                if self.config.GPU_COUNT == 1:
                    inputs_val = inputs_val.to(self.device, non_blocking=True)

                outputs_val = self.pytorch_model(inputs_val)

                labels_val = labels_val.to(outputs_val.device, non_blocking=True)

                val_score, val_preds = torch.max(outputs_val, 1)
                loss = criterion(outputs_val, labels_val)

                running_val_loss.update(loss.item() * inputs_val.size(0), inputs_val.size(0))
                running_val_acc.update(torch.sum(val_preds == labels_val.data).double(), inputs_val.size(0))

        print(f"val Loss: {running_val_loss.avg} val Acc: {running_val_acc.avg}")

        return running_val_loss, running_val_acc

    def _custom_eval_one_epoch(self, dataloader):

        evaluate_dict = {
            "y_gth_list": [],
            "y_pred_list": []
        }

        # Evaluate metric part
        print("====================================")
        print("Calculating Custom Metric.....")
        print("====================================")

        if self.config.GPU_COUNT > 1:
            evaluate_model = self.pytorch_model.module
            tta_model = tta.ClassificationTTAWrapper(evaluate_model, self.tta_option)
            tta_model = nn.DataParallel(tta_model)

        else:
            evaluate_model = self.pytorch_model
            tta_model = tta.ClassificationTTAWrapper(evaluate_model, self.tta_option)

        tta_model.eval()

        with torch.no_grad():
            draft_img = preprocess_input(np.zeros((self.input_size, self.input_size, 3)).astype(np.float32))
            if self.config.GPU_COUNT > 1:
                pass

            else:
                draft_img = draft_img.to(self.device, non_blocking=True)

            if self.tta_opt:
                tta_model(draft_img.unsqueeze(0))

            else:
                self.pytorch_model(draft_img.unsqueeze(0))

            for eval_data in dataloader:
                if self.tta_opt:
                    inputs_eval, labels_eval = eval_data[0], eval_data[1]

                    if self.config.GPU_COUNT == 1:
                        inputs_eval = inputs_eval.to(self.device, non_blocking=True)

                    outputs_eval = tta_model(inputs_eval)
                    labels_eval = labels_eval.to(outputs_eval.device, non_blocking=True)
                    _, eval_preds = torch.max(outputs_eval, 1)
                    evaluate_dict['y_pred_list'].extend(eval_preds.tolist())

                else:
                    inputs_eval, labels_eval = eval_data[0], eval_data[1]

                    if self.config.GPU_COUNT == 1:
                        inputs_eval = inputs_eval.to(self.device, non_blocking=True)

                    labels_val = labels_val.to(self.device, non_blocking=True)
                    outputs_eval = self.pytorch_model(inputs_eval)
                    _, eval_preds = torch.max(outputs_eval, 1)
                    evaluate_dict['y_pred_list'].extend(eval_preds.tolist())

                evaluate_dict['y_gth_list'].extend(labels_eval.tolist())

        UK_ratio = _FP_FN_metric(evaluate_dict['y_pred_list'], evaluate_dict['y_gth_list'], self.fail_class_index)
        OK_ratio = _FP_FN_metric(evaluate_dict['y_pred_list'], evaluate_dict['y_gth_list'], self.pass_class_index)

        # Delete when done evaluate to prevent OOM
        try:
            del evaluate_model

        except NameError:
            pass

        # Tensorboard part
        self.config.LOGGING.write_log(f"False Pass rate: {UK_ratio} %", message_type=0)
        self.config.LOGGING.write_log(f"False Reject rate: {OK_ratio} %", message_type=0)

        print(f"False Pass rate: {UK_ratio} %")
        print(f"False Reject rate: {OK_ratio} %")

        return UK_ratio, OK_ratio

    def train(self):
        train_checkpoint_dir = self.config.LOGS_PATH
        os.makedirs(train_checkpoint_dir, exist_ok=True)

        num_worker = min(8, self.config.GPU_COUNT * 2) if self.config.GPU_COUNT > 1 else 0
        pin_memory = False

        seed_torch()
        trainloader = FastDataLoader(self.list_Generator[0], pin_memory=pin_memory,
                                     worker_init_fn=_init_fn,
                                     batch_size=self.global_batch_size,
                                     shuffle=True,
                                     num_workers=num_worker)

        seed_torch()
        valloader = FastDataLoader(self.list_Generator[1], pin_memory=pin_memory,
                                   worker_init_fn=_init_fn,
                                   batch_size=self.global_batch_size,
                                   shuffle=False,
                                   num_workers=num_worker)

        seed_torch()
        evalloader = FastDataLoader(self.evaluate_generator, pin_memory=pin_memory,
                                    worker_init_fn=_init_fn,
                                    batch_size=self.global_batch_size,
                                    shuffle=False,
                                    num_workers=num_worker)

        self.load_weight()

        print(f"[DEBUG] class_weight : {self.class_weights}")
        if self.class_weights is None:
            pass

        else:
            self.class_weights = torch.tensor(self.class_weights, dtype=torch.float, device=self.device)

        criterion = nn.CrossEntropyLoss(weight=self.class_weights).to(self.device)

        model_parameters = list(self.pytorch_model.parameters())

        optimizer = self.optimizer_chosen(model_parameters)

        # Init tensorboard
        writer = SummaryWriter(log_dir=train_checkpoint_dir)

        start_time = datetime.now()

        value_best = 100

        total_model_info = {}

        count_model = 0

        best_epoch = 1

        best_loss = 1e9

        # enumerate epoch
        for epoch in range(self.config.NO_EPOCH):

            # class_correct = list(0. for i in range(len(self.classes)))
            # class_total = list(0. for i in range(len(self.classes)))

            model_info = {
                "path": [],
                "model_name": [],
                "Underkill_rate": [],
                "Overkill_rate": []
            }
            # Set determinism behavior
            seed_torch(epoch)
            current_epoch = epoch + 1
            self.config.LOGGING.write_log(f"Start training at epoch {current_epoch}", message_type=0)
            print(f"Epoch {current_epoch}/ {self.config.NO_EPOCH}")
            print('-' * 20)

            loss, acc = self._train_one_epoch(trainloader, optimizer, criterion, current_epoch)

            val_loss, val_acc = self._valid_one_epoch(valloader, criterion)

            """
               Save the model first then evaluate later
               To avoid process crashing
            """
            # Save model part
            try:
                message = None
                model_name = None
                if self.config.IS_SAVE_BEST_MODELS:
                    if value_best > val_loss.avg:
                        value_best = val_loss.avg
                        model_name = SaveModelCheckpoint(self.pytorch_model,
                                                         self.config.LOGS_PATH,
                                                         current_epoch,
                                                         value_best,
                                                         True)
                        message = f"Model saved at {os.path.join(self.config.LOGS_PATH, model_name)}"
                    else:
                        pass

                else:
                    model_name = SaveModelCheckpoint(self.pytorch_model, self.config.LOGS_PATH, current_epoch)

                    message = f"Model saved at {os.path.join(self.config.LOGS_PATH, model_name)}"

                if message is not None:
                    self.config.LOGGING.write_log(message, message_type=0)

            except Exception as e:
                message = f"Model cannot be saved at {self.config.LOGS_PATH} due to {e}"
                raise Exception(message)

            UK_ratio, OK_ratio = self._custom_eval_one_epoch(evalloader)

            writer.add_scalars('Loss', {'Train': loss.avg,
                                        'Val': val_loss.avg}, current_epoch)

            writer.add_scalars('Acc', {'Train': acc.avg,
                                       'Val': val_acc.avg}, current_epoch)

            writer.add_scalars('Custom', {"FalsePass_rate": UK_ratio,
                                          "FalseReject_rate": OK_ratio}, current_epoch)

            writer.flush()

            if best_loss >= val_loss.avg:
                best_loss = val_loss.avg
                best_epoch = current_epoch

            if not isinstance(model_name, type(None)):
                model_info["path"] = self.config.LOGS_PATH
                model_info["model_name"] = model_name
                model_info['Underkill_rate'] = UK_ratio
                model_info['Overkill_rate'] = OK_ratio
                model_info['Val_loss'] = val_loss.avg
                count_model += 1
                model_n = "model_" + str(count_model)
                total_model_info[model_n] = model_info

                with open(os.path.join(self.config.LOGS_PATH, f"model_info.json"), "w") as model_json:
                    json.dump(total_model_info, model_json)

        writer.close()
        end_time = datetime.now()
        message = "Training time: {}".format(end_time - start_time)
        self.config.LOGGING.write_log(message, message_type=0)
        print("[INFO] " + message)

        # Evaluating after training.
        if self.config.ENABLE_AUTOMATIC_DETERMINATION:
            self.config.LOGGING.write_log("Testing recommend threshold", message_type=0)
            self._recommend_threshold_evaluate(dataloader=valloader, best_epoch=best_epoch)

    def _recommend_threshold_evaluate(self, dataloader, best_epoch=None):
        try:
            model_path = glob.glob('%s/*[-_]%04d[-_]*.pth' % (self.config.LOGS_PATH, best_epoch))[0]

        except Exception as e:
            message = f'Checkpoint {best_epoch} not found due to {e}'
            raise Exception(message)

        # Re-build model to avoid conflict with multi-gpus model
        self.config.WEIGHT_PATH = model_path
        evaluate_model = self.load_weight(weight_path=model_path, multi_gpus=False)
        evaluate_model.eval()
        tta_model = tta.ClassificationTTAWrapper(evaluate_model, self.tta_option)
        tta_model.eval()
        highest_pass_score = 0.

        with torch.no_grad():
            for val_data in dataloader:
                inputs_val, labels_val = val_data[0], val_data[1]

                # if self.config.GPU_COUNT == 1:
                inputs_val = inputs_val.to(self.device, non_blocking=True)

                if self.tta_opt:
                    outputs_val = tta_model(inputs_val)

                else:
                    outputs_val = evaluate_model(inputs_val)

                # print(torch.nn.Softmax(dim=1)(outputs_val)) # For debug purpose
                labels_val = labels_val.to(outputs_val.device, non_blocking=True)
                val_scores, val_preds = torch.max(outputs_val, 1)

                # # Check if the predict values are in pass class or not
                if_pred_pass = _check_Fail_Pass_class(val_preds, self.pass_class_index)

                # # Check if the label values are in fail class or not
                if_gth_fail = _check_Fail_Pass_class(labels_val, self.fail_class_index)

                # # Final check: if images is actually a fail pass or not
                fail_passes = outputs_val[if_gth_fail & if_pred_pass]
                local_high = torch.softmax(fail_passes, dim=1).max() if fail_passes.size(0) > 0 else 0.0
                if local_high > highest_pass_score:
                    highest_pass_score = local_high

        message = 'Recommended PassThreshold: %.4f' % highest_pass_score
        self.config.LOGGING.write_log(message, message_type=0)
        print('[INFO]' + message)

    def get_recommend_threshold(self, best_epoch=None):

        if best_epoch:
            pass

        else:

            with open(os.path.join(self.config.LOGS_PATH, 'model_info.json'), 'r') as f:
                model_info = json.loads(f.read())

            best_loss, model_path = 1e9, ''
            for info in model_info.values():
                if info['Val_loss'] < best_loss:
                    best_loss = info['Val_loss']
                    best_epoch = int(info["model_name"].split("_")[1])
                    model_path = os.path.join(self.config.LOGS_PATH, info['model_name'])

            print('[INFO] Evaluate checkpoint %s with val_loss of %.4f' % (os.path.basename(model_path), best_loss))

        valloader = FastDataLoader(self.list_Generator[1],
                                   worker_init_fn=_init_fn,
                                   batch_size=self.global_batch_size, shuffle=False,
                                   num_workers=self.config.NUM_WORKERS)

        self._recommend_threshold_evaluate(dataloader=valloader, best_epoch=best_epoch)


    def predict_one(self, img, TTA=True):
        self.pytorch_model.eval()
        with torch.no_grad():
            if TTA:
                img = preprocess_input(img).to(self.device, non_blocking=True)
                tta_model = tta.ClassificationTTAWrapper(self.pytorch_model, self.tta_option)

                outputs = tta_model(img.unsqueeze(0).to(self.device, non_blocking=True))
                propability = torch.nn.Softmax(dim=1)(outputs)

                return self.manage_prediction(propability.tolist())

            else:

                img = preprocess_input(img).to(self.device, non_blocking=True)
                outputs = self.pytorch_model(img.unsqueeze(0))

                propability = torch.nn.Softmax(dim=1)(outputs)
                return self.manage_prediction(propability.tolist())

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

    def load_weight(self, weight_path=None, multi_gpus=True):
        model = self._build_model(weight_path).to(self.device)

        if multi_gpus and self.config.GPU_COUNT > 1:
            model = nn.DataParallel(model)

        else:
            pass

        if weight_path is None:
            self.pytorch_model = model

        else:
            return model

    def initialize_vis(self):
        """Initialize CAM visualizer"""

        from visualisation.core import ClassActivationMapping, GradCam
        # self.vis = ClassActivationMapping(self.pytorch_model, self.device)
        self.vis = GradCam(self.pytorch_model, self.device)

    def cam_testing_v2(self, img_path):
        """Run CAM algorithm on the specific image

        Returns:
            result_image
            prediction
        """
        from visualisation.core.utils import image_net_postprocessing, imshow, cam_tensor_to_numpy

        # It takes the center of image if the image has no json
        img, _ = load_and_crop(img_path, input_size=self.input_size,
                               center_point=(self.input_size / 2, self.input_size / 2))

        img = preprocess_input(img).to(self.device, non_blocking=True)
        # print(img.shape)
        # print(img.unsqueeze(0).shape)
        img_w_hm, heatmap, prediction = self.vis(img.unsqueeze(0), None,
                                                 tta_option=self.tta_rotate_opt_list, tta_transform=self.tta_option,
                                                 postprocessing=image_net_postprocessing)

        result_image = cam_tensor_to_numpy(img_w_hm)

        # return cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR), prediction
        return result_image, heatmap, prediction

    def confusion_matrix_evaluate(self, single_model=True, keep_result=False):

        global gth_name
        csv_path = r"D:\something.csv"

        # csv_path = r"D:\KLA\Jhyn\DLBackend\ODModule\center_points_0010_Mvsd_by_mvsd_MVSD_with_label.csv"

        if_csv_exist = os.path.isfile(csv_path)
        print("Is center points csv file exist?: ", if_csv_exist)
        if if_csv_exist:
            center_point_info = pd.read_csv(csv_path)

        images_result_info = {
            "image_id": [],
            "defect": []
        }
        self.load_weight()
        self.pytorch_model.eval()

        parent_folder = self.config.DATASET_PATH.split("\\")[-1]

        if single_model:
            model_file_name = self.config.WEIGHT_PATH.split("\\")[-1].split(".")[0] + "_" + parent_folder

            result_folder = f"_Result_{model_file_name}"

            workbook = xlsxwriter.Workbook("_model_" + model_file_name + "_result.xlsx")

            cell_format = workbook.add_format()
            cell_format.set_align('center')
            cell_format.set_align('vcenter')
            cell_format.set_text_wrap()

            highlight_format = workbook.add_format()
            highlight_format.set_align('center')
            highlight_format.set_align('vcenter')
            highlight_format.set_bg_color("red")

            Header = ["image_id", "Image", "Label", "Predict"]
            Header.extend(self.classes)
            Header.append("_Underkill_")
            Header.append("_Overkill_")

        image_size_display = min(int(self.input_size * 1.5), 480)  # 192 as Default

        # time_statistic = pd.DataFrame(columns=['Execution_time'])
        self.initialize_vis()

        sub_dirs_path = [
            os.path.join(self.config.DATASET_PATH, "Train"),
            os.path.join(self.config.DATASET_PATH, "Validation"),
            os.path.join(self.config.DATASET_PATH, "Test"),
            os.path.join(self.config.DATASET_PATH, "mvsd_test"),
            os.path.join(self.config.DATASET_PATH, "Underkill_images"),
            os.path.join(self.config.DATASET_PATH, "OD_blind_set"),
            os.path.join(self.config.DATASET_PATH, "OD_productline_set")
        ]
        for diRectory in sub_dirs_path.copy():
            if not os.path.exists(diRectory) or len(os.listdir(diRectory)) == 0:
                sub_dirs_path.remove(diRectory)

        if len(sub_dirs_path) == 0:
            print("There is no sub-folder, will take the main-folder instead")
            sub_dirs_path.append(self.config.DATASET_PATH)

        # Loop over dataset path
        for sub_path in sub_dirs_path:
            ori_sub_path = sub_path
            if "train" in sub_path.lower().split("\\")[-1]:
                sheet_name = sub_path.split("\\")[-1]
                sub_path = os.path.join(sub_path, "OriginImage")

            else:
                sheet_name = sub_path.split("\\")[-1]

            # Skip non-exist folder
            if not os.path.exists(sub_path) or len(os.listdir(sub_path)) == 0:
                continue

            progress_bar = tqdm(glob.glob(os.path.join(sub_path, "*mvsd")) + glob.glob(os.path.join(sub_path, "*bmp")))

            print("sub_dirs_path.index(ori_sub_path): ", sub_dirs_path.index(ori_sub_path))
            print(f"Inspecting PATH: {sub_path}")
            if single_model:
                start_row = 0
                start_column = 1
                # Limit the character in the sheet name to 30
                worksheet = workbook.add_worksheet(sheet_name[:30])
                worksheet.write_row(start_row, start_column, Header, cell_format)
                worksheet.set_column("B:B", 15)
                worksheet.set_column("C:C", int(15 * (image_size_display / 192)))

            y_gth_eval_ls = []
            y_pred_eval_ls = []

            underkill_count = 0
            overkill_count = 0

            for image_path in progress_bar:
                # with torch.no_grad():
                if single_model:
                    Data = [0] * len(Header)
                    start_row += 1
                    worksheet.set_row(start_row, int(90 * (image_size_display / 192)))

                underkill_overkill_flag = 0
                # print(data_eval[1])
                # print(data_eval)

                image_name = image_path.split("\\")[-1]
                images_result_info["image_id"].append(image_name)
                center_points = None

                if if_csv_exist:
                    center_points = center_point_info.loc[
                                        lambda x: x["image_id"] == image_name
                                    ].values.tolist()[0][1:]

                else:
                    # pass
                    if image_path.endswith(".mvsd"):
                        center_points = GetCenter_MVSD(image_path, self.config.INDEX_TRAINING_LAYER)


                img, _ = load_and_crop(image_path, self.input_size, center_points, self.config.INDEX_TRAINING_LAYER)

                # start_inference = time.time()
                pred_id, all_scores, pred_name = self.predict_one(img)
                # end_inference = time.time()
                # img_w_heatmap, heatmap, _ = self.cam_testing_v2(image_path)

                # bboxes = heatmap2bbox(heatmap)
                # for box in bboxes:
                #     plot_one_box(img, box, label=pred_name,
                #                            score=None,
                #                            color=(0, 255, 0),
                #                            line_thickness=1)
                # print(image_name)
                # print(bboxes)
                # img = img_w_heatmap
                # img = img * cv2.cvtColor(bit_image, cv2.COLOR_GRAY2RGB)
                # img = img_w_heatmap

                # inference_time = end_inference - start_inference
                # time_statistic = time_statistic.append({"Execution_time": inference_time}, ignore_index=True)

                jsonpath = image_path + ".json"
                with open(jsonpath, "r") as json_file:
                    img_data = json.load(json_file)

                # Get class groundtruth - this will work even with OD format
                temp_classId = img_data['classId']
                # Check if the field 'classId' in json empty or not
                if len(temp_classId) == 0:
                    continue
                else:
                    pass
                final_classId = []
                if len(temp_classId) > 1:
                    for classId in temp_classId:
                        if "reject" in classId.lower():
                            final_classId.append(classId)
                            gth_name = classId
                            break

                        else:
                            pass

                    if len(final_classId) == 0:
                        gth_name = temp_classId[0]

                elif len(temp_classId) == 1:
                    gth_name = temp_classId[0]

                else:
                    gth_name = "Unknown"

                if (gth_name in self.failClasses or "reject" in gth_name.lower()) and (
                        pred_name in self.passClasses or pred_id == len(self.classes)):  # Underkill
                    underkill_count += 1
                    underkill_overkill_flag = -1

                elif (
                        gth_name in self.passClasses or "overkill" in gth_name.lower()) and pred_name in self.failClasses:  # Overkill
                    overkill_count += 1
                    underkill_overkill_flag = 1

                else:  # Correct result
                    pass

                if single_model:

                    if gth_name == "Empty" and max(all_scores) > 1 / len(self.classes) + 1 / len(self.classes) / 2:
                        mvsd_output_path = os.path.join("mvsd_Result", pred_name)
                        os.makedirs(mvsd_output_path, exist_ok=True)
                        mvsd_path = os.path.join(mvsd_output_path, image_path.split("\\")[-1])
                        shutil.copy(image_path, mvsd_path)

                    os.makedirs(os.path.join(result_folder, gth_name), exist_ok=True)
                    save_path = os.path.join(result_folder, gth_name, image_name.strip(".mvsd") + ".bmp")
                    img = cv2.resize(img, (image_size_display, image_size_display))
                    cv2.imwrite(save_path, img)
                # Don't need the json file for now
                # shutil.copy(image_path + ".json", os.path.join(result_path, image_name + ".json"))

                # Groundtruth
                if "reject" in gth_name.lower():
                    y_gth_eval_ls.append(0)

                elif "overkill" in gth_name.lower():
                    y_gth_eval_ls.append(1)

                else:
                    y_gth_eval_ls.append(2)

                # Result
                if "reject" in pred_name.lower():
                    y_pred_eval_ls.append(0)
                    images_result_info["defect"].append("Reject")

                elif "overkill" in pred_name.lower():
                    y_pred_eval_ls.append(1)
                    images_result_info["defect"].append("Overkill")

                else:
                    y_pred_eval_ls.append(2)
                    images_result_info["defect"].append("Unknown")

                if single_model:
                    Data[0] = image_name.split(".")[0]
                    Data[2] = gth_name
                    Data[3] = pred_name
                    Data[4:4 + len(self.classes)] = all_scores
                    Data[-2] = True if underkill_overkill_flag == -1 else False
                    Data[-1] = True if underkill_overkill_flag == 1 else False
                    # print(f"[DEBUG]:\n{Data}")
                    for index, info in enumerate(Data):
                        excel_format = highlight_format if (
                                Data[index] == True and isinstance(Data[index], bool)) else cell_format

                        worksheet.insert_image(start_row, index + 1, save_path,
                                               {'x_scale': 0.5, 'y_scale': 0.5, 'x_offset': 5, 'y_offset': 5,
                                                'object_position': 1}) if index == 1 else \
                            worksheet.write(start_row, index + 1, Data[index], excel_format)

                progress_bar.update()

                # End the loop over test data when underkill images > 25
                if (underkill_count > 25 or overkill_count > 2000) and not single_model:
                    print("\n")
                    break

            if single_model:
                header = [{'header': head} for head in Header]

                worksheet.add_table(0, 1, start_row, len(Header), {'columns': header})
                worksheet.freeze_panes(1, 0)
                worksheet.hide_gridlines(2)

            confusion_matrix = metrics.confusion_matrix(y_gth_eval_ls, y_pred_eval_ls)
            print(f"Confusion matrix : \n{confusion_matrix}")

        if single_model:
            workbook.close()
            df_result = pd.DataFrame(data=images_result_info)
            df_result.to_csv(f"mvsd_result_{model_file_name}.csv", index=False)

            if not keep_result:
                shutil.rmtree(result_folder)

        else:
            del self.pytorch_model
            model_name = "\\".join(self.config.WEIGHT_PATH.split("\\")[-4:])
            return model_name, confusion_matrix.tolist()[0][1], confusion_matrix.tolist()[1][0]


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
