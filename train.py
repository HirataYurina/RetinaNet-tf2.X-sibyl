# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:train.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras
from config.configs import config
from retinanet import RetinaNet
from core.loss import smooth_l1, focal_loss


retinanet = RetinaNet()


# 具体的训练策略请参照论文
def train_step(optimizer, x, y,):

    with tf.GradientTape() as tape:
        class_pred, box_pred = retinanet(x)
        labels, delta, label_weights, box_weights = y
        class_loss = focal_loss(labels, class_pred, label_weights)
        box_loss = smooth_l1(box_pred, delta, box_weights)
        losses = class_loss + box_loss

    gradients = tape.gradient(losses, retinanet.trainable_variables)
    optimizer.apply_gradients(zip(gradients, retinanet.trainable_variables))
