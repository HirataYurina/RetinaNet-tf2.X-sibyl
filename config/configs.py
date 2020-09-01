# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:configs.py
# software: PyCharm

from easydict import EasyDict

config = EasyDict()

config.TRAIN = EasyDict()
config.TRAIN.LR1 = 0.001
config.TRAIN.LR2 = 0.0001
config.TRAIN.BATCH_SIZE1 = 32
config.TRAIN.BATCH_SIZE2 = 4
config.TRAIN.DECAY_MODE = 'cosine anneal'
