# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:configs.py
# software: PyCharm

from easydict import EasyDict

config = EasyDict()

config.MODEL = EasyDict()
config.MODEL.OUT_CHANNELS = 256
config.MODEL.NUM_ANCHORS = 9
config.MODEL.NUM_CLASSES = 6

config.TRAIN = EasyDict()
config.TRAIN.LR1 = 0.0001
config.TRAIN.LR2 = 0.00001
config.TRAIN.BATCH_SIZE1 = 2
config.TRAIN.BATCH_SIZE2 = 2
config.TRAIN.EPOCH1 = 50
config.TRAIN.EPOCH2 = 100  # train longer if you need
config.TRAIN.DECAY_MODE = 'cosine anneal'
config.TRAIN.POSITIVE_THRES = 0.5
config.TRAIN.NEGATIVE_THRES = 0.4
# freeze layers of resnet
# TODO: i will change my architecture to some pretrained models on github and freeze the final 3 layers.
config.TRAIN.FREEZE = 174
config.TRAIN.INPUT_SHAPE = (416, 416)
config.TRAIN.SAVE_INTERVAL = 5

config.TRAIN.TRAIN_TXT = './datas/2077_trainval.txt'
config.TRAIN.TEST_TXT = './datas/2077_test.txt'
config.TRAIN.CLASS_TXT = './datas/danger_source_classes.txt'
config.TRAIN.SAVE_PATH = './logs/first/'
config.TRAIN.DIARY_PATH = './diary/training.log'

config.TRAIN.SIGMA = 2
config.TRAIN.ALPHA = 0.25


config.AUG = EasyDict()
config.AUG.MAX_BOXES = 40

config.DETECT = EasyDict()
config.DETECT.mAP_THRES = 0.5
