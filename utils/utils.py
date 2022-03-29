import functools
import random
import sys

from torch.nn import DataParallel

sys.path.append("...")
from Utilities.mvsdUtils.LoadMVSDImage import ReadKLAImage
from torchvision import transforms
import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import skimage
from prettytable import PrettyTable
import cv2
import json
import time


def Read_MVSD(image_path, index_layer=0):
    GDLL = ReadKLAImage()
    global_image = GDLL.Get_Image(image_path, index_training_layer=index_layer)
    return global_image


def GetCenter_MVSD(image_path, index_layer=0):
    GDLL = ReadKLAImage()
    GDLL.read_kla_image_multi_layer(bytes(image_path, 'utf-8'), index_layer)
    x, y = GDLL.GetCenter()
    return x, y


def GetDim_MVSD(image_path, index_layer=0):
    GDLL = ReadKLAImage()
    GDLL.read_kla_image_multi_layer(bytes(image_path, 'utf-8'), index_layer)
    width, height, depth = GDLL.GetImageDim()
    return width, height, depth


def set_GPU(num_of_GPUs, log, memory_restraint=0):
    try:
        from gpuinfo import GPUInfo
        current_memory_gpu = GPUInfo.gpu_usage()[1]
        if not memory_restraint:
            list_available_gpu = np.where(np.array(current_memory_gpu))[0].astype('str').tolist()

        else:
            list_available_gpu = np.where(np.array(current_memory_gpu) < memory_restraint)[0].astype('str').tolist()

    except:
        log.write_log("Cannot find nvidia-smi, please include it into Environment Variables", message_type=0)
        print("[INFO] Cannot find nvidia-smi, please include it into Environment Variables")
        if torch.cuda.is_available():
            list_available_gpu = [str(i) for i in range(num_of_GPUs)]

        else:
            list_available_gpu = []

    list_gpu_using = list_available_gpu[:num_of_GPUs]

    if len(list_available_gpu) < num_of_GPUs and len(list_available_gpu) > 0:
        print("==============Warning==============")
        print("Your process had been terminated")
        print("Please decrease number of gpus you using")
        print(f"number of Devices available:\t{len(list_available_gpu)} gpu(s)")
        print(f"number of Device will use:\t{num_of_GPUs} gpu(s)")
        log.write_log(
            f"number of Devices available:\t{len(list_available_gpu)} gpu(s) < number of Device will use:\t{num_of_GPUs} gpu(s)",
            message_type=2)
        sys.exit()

    elif num_of_GPUs <= len(list_available_gpu) and num_of_GPUs != 0:
        current_available_gpu = ",".join(list_gpu_using)

    elif num_of_GPUs == 0 or len(list_available_gpu) == 0:
        current_available_gpu = "-1"

    print("[INFO] ***********************************************")

    if len(list_gpu_using) > 0:
        tmp_message = f"[INFO] You are using GPU(s): {current_available_gpu}"
    else:
        tmp_message = "[INFO] You are using CPU !"

    print(tmp_message)
    if log is not None:
        log.write_log(tmp_message, message_type=0)

    print("[INFO] ***********************************************")
    os.environ["CUDA_VISIBLE_DEVICES"] = current_available_gpu


def _check_Fail_Pass_class(inspecting_values, inspecting_class_index):
    """
    inspecting_values : torch Tensor
    inspecting_class_index : list class index

    return
    inspecting_result : numpy Array bool -> ex: numpy.array([True True])
    """
    inspecting_values = inspecting_values.cpu()

    if len(inspecting_class_index) == 0:
        inspecting_result = np.array([False] * inspecting_values.shape[0])

    else:
        # inspecting_result = np.sum([inspecting_values == idx for idx in inspecting_class_index], axis=0)
        inspecting_result = np.logical_or.reduce(
            [(inspecting_values == idx).tolist() for idx in inspecting_class_index])

    return inspecting_result


def _FP_FN_metric(y_pred, y_gth, specific_class_index):
    # Get positive groundtruth
    gth_list = [np.array(y_gth) == class_ for class_ in specific_class_index]
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


def preprocess_input(image, advprop=False):
    if advprop:
        normalize = transforms.Lambda(lambda img: img * 2.0 - 1.0)
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                         std=[0.229, 0.224, 0.225])
    preprocess_image = transforms.Compose([transforms.ToTensor(), normalize])(image)
    return preprocess_image


def to_onehot(labels, num_of_classes):
    if type(labels) is list:
        labels = [int(label) for label in labels]
        arr = np.array(labels, dtype=np.int)
        onehot = np.zeros((arr.size, num_of_classes))
        onehot[np.arange(arr.size), arr] = 1
    else:
        onehot = np.zeros((num_of_classes,), dtype=np.int)
        onehot[int(labels)] = 1
    return onehot


def multi_threshold(Y, thresholds):
    if Y.shape[-1] != len(thresholds):
        raise ValueError('Mismatching thresholds and output classes')

    thresholds = np.array(thresholds)
    thresholds = thresholds.reshape((1, thresholds.shape[0]))
    keep = Y > thresholds
    score = keep * Y
    class_id = np.argmax(score, axis=-1)
    class_score = np.max(score, axis=-1)
    if class_score == 0:
        return None
    return class_id, class_score


def load_and_crop(image_path, input_size=128, center_point=None, index_layer=0, keep_origin=False):
    """ Load image and return image with specific crop size

    This function will crop corresponding to json file and will resize respectively input_size

    Input:
        image_path : Ex:Dataset/Train/img01.bmp or data image
        input_size : any specific size
        custom_size : (center_x , center_y)
        keep_origin : keep origin image of data image - this argument just for testing prediction on image have 1 channel
        
    Output:
        image after crop and class gt
    """
    if isinstance(image_path, str):
        if image_path.endswith(".bmp"):
            image = cv2.imread(image_path)

        else:
            image = Read_MVSD(image_path, index_layer)

        json_path = image_path + ".json"

    else:
        image = image_path
        json_path = "abc.json"

    # [H, W, C]
    size_image = image.shape

    if (size_image[-1] == 1 or len(size_image) < 3) and not keep_origin:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    else:
        pass

    if center_point is None:
        if os.path.isfile(json_path):
            with open(json_path, encoding='utf-8') as json_file:
                json_data = json.load(json_file)
                box = json_data['box']
                center_x = box['centerX'][0]
                center_y = box['centerY'][0]
                class_gt = json_data['classId'][0]

        else:
            center_x = size_image[1] // 2
            center_y = size_image[0] // 2
            class_gt = "Empty"

    else:
        center_x = center_point[0]
        center_y = center_point[1]
        class_gt = "Empty"

    new_w = new_h = input_size

    # Prevent center point of mvsd out of the image scope
    if center_x > size_image[1] and center_y > size_image[0]:
        center_x, center_y = size_image[1] // 2, size_image[0] // 2

    else:
        pass

    left, right = center_x - new_w / 2, center_x + new_w / 2
    top, bottom = center_y - new_h / 2, center_y + new_h / 2

    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(size_image[1] - 0, right)), round(min(size_image[0] - 0, bottom))

    if int(bottom) - int(top) != input_size:
        if center_y < new_h / 2:
            bottom = input_size

        else:
            top = size_image[0] - input_size

    if int(right) - int(left) != input_size:
        if center_x < new_w / 2:
            right = input_size

        else:
            left = size_image[1] - input_size

    cropped_image = image[int(top):int(bottom), int(left):int(right)]

    if input_size == size_image[0]:
        cropped_image = image

    elif input_size > size_image[0]:
        cropped_image = cv2.resize(image, (input_size, input_size))

    return cropped_image, class_gt


def metadata_count(input_dir, classes_name_list, label_list, log, show_table):
    Table = PrettyTable()
    print(f"[DEBUG] : {input_dir}")
    # print(classes_name_list)
    # print(label_list)
    Table.field_names = ['Defect', 'Number of images']
    unique_label, count_list = np.unique(label_list, return_counts=True)
    # print(count_list)
    for i in range(len(classes_name_list)):
        for j in range(len(unique_label)):
            if classes_name_list[i] == unique_label[j]:
                log.write_log(f"Number of JSON file of class {classes_name_list[i]} is {count_list[j]} in {input_dir}",
                              message_type=0)

                Table.add_row([classes_name_list[i], count_list[j]])
    if show_table:
        print(f"[DEBUG] : {Table}")
    return classes_name_list, label_list


class FocalLoss(nn.Module):
    # Took from : https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
    # Addition resource : https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    # TODO: clean up FocalLoss class
    def __init__(self, class_weight=1., gamma=2., logits=False, reduction='mean'):
        super(FocalLoss, self).__init__()

        self.class_weight = class_weight
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        # if self.logits:
        #     BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        # else:
        #     BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        # pt = torch.exp(-BCE_loss)
        # F_loss = self.class_weight * (1 - pt)**self.gamma * BCE_loss
        log_prob = F.log_softmax(inputs, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            targets,
            weight=self.class_weight,
            reduction=self.reduction
        )


# TODO: inspect resize_image more carefully
def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    import cv2

    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        # image = cv2.resize(image, (round(h*scale), round(w*scale)), interpolation=cv2.INTER_LINEAR)
        image = skimage.transform.resize(
            image, (round(h * scale), round(w * scale)),
            order=1, mode="constant", preserve_range=True)
    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))

    return image.astype(image_dtype), window, scale, padding, crop


def predict_image(img_width, img_height, img_depth, x_center, y_center, config, model, sock, log):
    # Received image
    try:
        log.write_log("Received image", message_type=1)
        img_byte = sock.recvall(img_width * img_height * img_depth)

    except:
        log.write_log("cannot receive IMAGE", message_type=2)
        print("cannot receive IMAGE")
        sys.exit(0)

    image = np.frombuffer(img_byte, np.uint8).reshape((img_height, img_width, img_depth))

    start = time.time()

    anchor = np.array([
        x_center - config.INPUT_SIZE / 2,
        y_center - config.INPUT_SIZE / 2,
        x_center + config.INPUT_SIZE / 2,
        y_center + config.INPUT_SIZE / 2,
    ], dtype=np.int32)

    log.write_log("Cropping IMAGE...", message_type=1)

    image, _ = load_and_crop(image, config.INPUT_SIZE, (x_center, y_center))

    log.write_log("Predicting IMAGE...", message_type=1)
    pred_id, all_scores, pred_name = model.predict_one(image)
    end = time.time()

    conclusion = "Fail" if pred_name in config.FAIL_CLASSNAME else "Pass" if pred_name in config.PASS_CLASSNAME else "Unknown"

    result_dict = {
        "Classes": model.classes,
        "Scores": all_scores,
        "Box": anchor.tolist(),
        "FinalResult": conclusion,
        "PredResult": pred_name
    }

    sendResult = ""
    sendResult += "Classification result%"
    sendResult += r"Result&&&" + json.dumps(result_dict, sort_keys=True) + "%"
    sendResult += "Done in predict"
    log.write_log(sendResult, message_type=0)
    sock.sendall(sendResult)
    print(sendResult)


class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        self.num_gpus = num_gpus

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')

        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True)) \
                for device_idx in range(len(devices))], [kwargs] * len(devices)


def heatmap2bbox(heatmap):
    """
    heatmap -> (H, W, C)

    bbox -> (min x, min y, max x, max y)

    """
    for channel in range(heatmap.shape[2]):
        """
        Loop over the channel (heatmap return RGB image)

        Follow the rule
        True and True -> True
        True and False -> False
        False and False -> False

        -> We want to take the region with False values (The region that every channel point in the heat map)
        """
        # Try to obtain bit image base on standard deviation threshold
        temp_img = heatmap[:, :, channel] < np.abs(np.mean(heatmap[:, :, channel]) + np.std(heatmap[:, :, channel]))
        # print("Mean: ", np.mean(heatmap[:, :, channel]))
        # print("Std: ", np.std(heatmap[:, :, channel]))
        if channel == 0:
            bit_image = temp_img

        else:
            bit_image *= temp_img

    bboxes = []

    for _ in range(2):
        bit_image = np.invert(bit_image)
        mask_image = (1 * bit_image * 255).astype(np.uint8)
        w, h = bit_image.shape
        region_percentage = w - w * 5 // 100

        _, contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            contour = np.array(contour)
            contour = contour.reshape(contour.shape[0], 2)
            x, y = np.hsplit(contour, 2)
            x = x.flatten().tolist()
            y = y.flatten().tolist()
            bbox = [min(x), min(y), max(x), max(y)]

            if max(x) - min(x) > region_percentage and max(y) - min(y) > region_percentage:
                continue

            bboxes.append(bbox)

    list_area = calculate_bbox_area(bboxes)
    # Return the box with the largest area
    bboxes = [bboxes[list_area.index(max(list_area))]]

    return bboxes


def calculate_bbox_area(list_bbox):
    list_area = []
    for bbox in list_bbox:
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        list_area.append(area)

    return list_area


def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)

    if label:
        tf = max(tl - 2, 1)  # font thickness
        # {:.0%}
        s_size = cv2.getTextSize(str('{:.4f}'.format(score)), 0, fontScale=float(tl) / 3, thickness=tf)[
            0] if score else [0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0] + s_size[0] + 15, c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        # {:.0%}
        if score:
            cv2.putText(img, '{}: {:.4f}'.format(label, score), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                        thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)
        else:
            cv2.putText(img, '{}'.format(label), (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0],
                        thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate
