import os
import torch
import time
import torch.nn as nn
from utils.utils import CustomDataParallel
# from .utils import CustomDataParallel

def SaveModelCheckpoint(model ,PATH, epoch, value=0., save_best_opt=False):
    os.makedirs(PATH,exist_ok=True)
    
    if save_best_opt:
        model_name = 'weights-improvement-epoch-%04d-val_loss-%.04e.pth' % (epoch, value)
    else:
        model_name = '%s_%04d_%s.pth' % (
        time.strftime('%Y%m%d', time.localtime()), epoch, time.strftime('%H%M', time.localtime()))

    if isinstance(model, nn.DataParallel):
        print("Saving multi-gpus model...")
        torch.save(model.module.state_dict(), os.path.join(PATH, model_name))
    else:
        print("Saving model...")
        torch.save(model.state_dict(), os.path.join(PATH, model_name))

    return model_name