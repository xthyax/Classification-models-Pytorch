import json


class Config:
    def __init__(self, param, args):
        self.NO_EPOCH = param.NO_EPOCH
        self.GPU_COUNT = param.NUM_GPU if args.gpu == None else args.gpu
        self.DATASET_PATH = args.dataset
        self.LOGS_PATH = args.logs
        self.BATCH_SIZE = param.BATCH_SIZE
        self.LEARNING_RATE = param.LEANING_RATE
        self.WEIGHT_PATH = args.weight if args.weight else None
        self.LEARNING_MOMENTUM = param.MOMENTUM
        self.WEIGHT_DECAY = param.DECAY
        self.OPTIMIZER = param.OPTIMIZER
        self.NUM_CLASSES = len(param.CLASS_NAME)
        self.CLASS_NAME = param.CLASS_NAME
        self.INPUT_SIZE = param.CHANGE_BOX_SIZE
        self.CLASS_THRESHOLD = param.CLASS_THRESHOLD
        self.AU_LIST = param.AUGMENT_LIST
        if self.AU_LIST == [] or self.AU_LIST == None:
            self.AU_LIST = None
        else:
            self.AU_LIST = param.AUGMENT_LIST
        self.ARCHITECTURE = param.ARCHITECTURE
        self.FAIL_CLASSNAME = param.FAILCLASS_NAME
        self.PASS_CLASSNAME = param.PASSCLASS_NAME
        self.BINARY = bool(param.BINARY)
        self.IS_SAVE_BEST_MODELS = json.loads(param.IS_SAVE_BEST_MODELS.lower())
        self.NUM_WORKERS = param.NUM_WORKERS
        self.ENABLE_AUTOMATIC_DETERMINATION = True
        self.INDEX_TRAINING_LAYER = param.IndexTrainingLayer
        self.LOGGING = None
