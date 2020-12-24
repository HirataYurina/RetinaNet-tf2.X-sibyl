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
import tensorflow.keras.layers as layers
from net.subnet import class_subnet, box_subnet


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

    def __call__(self, inputs, training=True, mask=None):
        x = self.resnet(inputs, training=training)
        x = self.fpn(x)
        x = self.subnet(x)

        return x


def retinanet(inputs, out_channels, num_classes, num_anchors):
    resnet = ResNet(50)
    C2, C3, C4, C5 = resnet(inputs)

    P5 = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    # 38x38x256
    P5_upsampled = layers.UpSampling2D(name='P5_upsampled')(P5)
    P5 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    # 38x38x256
    P4 = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = keras.layers.Add(name='P4_merged')([P5_upsampled, P4])

    P4_upsampled = layers.UpSampling2D(name='P4_upsampled')(P4)
    P4 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    # 75x75x256
    P3 = keras.layers.Conv2D(256, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = keras.layers.Add(name='P3_merged')([P4_upsampled, P3])
    P3 = keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    P6 = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    P7 = keras.layers.Activation('relu', name='C6_relu')(P6)
    P7 = keras.layers.Conv2D(256, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    features = [P3, P4, P5, P6, P7]

    class_results = []
    box_results = []
    classi_model = class_subnet(out_channels=out_channels, num_anchors=num_anchors, num_classes=num_classes)
    box_model = box_subnet(out_channels=out_channels, num_anchors=num_anchors)

    for feature in features:
        class_results.append(classi_model(feature))
        box_results.append(box_model(feature))
    # concatenate -> (batch, 52*52*9, 4), (batch, 52*52*9, num_classes)
    class_results = layers.Concatenate(axis=1)(class_results)
    box_results = layers.Concatenate(axis=1)(box_results)
    results = [box_results, class_results]

    return keras.Model(inputs, results)


if __name__ == '__main__':
    inputs_ = keras.Input(shape=(416, 416, 3))
    retina_model = RetinaNet(256, 9, 6)
    outputs = retina_model(inputs_)
    # model = keras.Model(inputs, outputs)
    print(len(retina_model.layers))
