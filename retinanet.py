# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:retinanet.py
# software: PyCharm

import tensorflow.keras as keras
from net.resnet import ResNet
from net.fpn import FPN
from net.subnet import SubNet


class RetinaNet(keras.Model):
    
    def __init__(self):
        super(RetinaNet, self).__init__()

        self.resnet = ResNet(50)
        self.fpn = FPN()
        self.subnet = SubNet(out_channels=256,
                             num_anchors=9,
                             num_classes=6)

    def call(self, inputs, training=None, mask=None):
        pass
