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
    
    def __init__(self,
                 out_channels,
                 num_anchors,
                 num_classes):
        super(RetinaNet, self).__init__()

        self.resnet = ResNet(50)
        self.fpn = FPN()
        self.subnet = SubNet(out_channels=out_channels,
                             num_anchors=num_anchors,
                             num_classes=num_classes)

    def __call__(self, inputs, training=False, mask=None):
        x = self.resnet(inputs, training=training)
        x = self.fpn(x)
        x = self.subnet(x)

        return x


if __name__ == '__main__':
    inputs = keras.Input(shape=(416, 416, 3))
    retina_model = RetinaNet(256, 9, 6)
    outputs = retina_model(inputs)
    # model = keras.Model(inputs, outputs)
    print(len(retina_model.layers))
