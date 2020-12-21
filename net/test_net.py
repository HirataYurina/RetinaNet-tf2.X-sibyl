# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:test_net.py
# software: PyCharm

import tensorflow.keras as keras
from net.resnet import ResNet
from net.fpn import FPN
from net.subnet import class_subnet, box_subnet

inputs = keras.Input(shape=(416, 416, 3))
resnet = ResNet(depth=50)
c2, c3, c4, c5 = resnet(inputs)

fpn = FPN()
p3, p4, p5, p6, p7 = fpn([c2, c3, c4, c5])

box_subnet_model = box_subnet(out_channels=256, num_anchors=9)
class_subnet_model = class_subnet(out_channels=256, num_classes=80, num_anchors=9)

results = [[box_subnet_model(x) for x in [p3, p4, p5, p6, p7]],
           [class_subnet_model(y) for y in [p3, p4, p5, p6, p7]]]

retinanet_model = keras.Model(inputs, results)
# retinanet_model.summary()

# layers that have trainable parameters
length = len([layer for layer in retinanet_model.layers if len(layer.trainable_weights) > 0])
print(length)
print(retinanet_model.layers[176].name)

# load pre-trained model
retinanet_model.load_weights('../datas/resnet50_coco_best_v2.1.0.h5', by_name=True)
print('load weights successfully!')
