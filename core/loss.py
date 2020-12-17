# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:loss.py
# software: PyCharm


import tensorflow as tf
from config.configs import config


def smooth_l1(predictions, delta, box_weights, sigma=3):
    """The smooth l1 loss.
    L1: When box predictions is closed to ground truth, the gradients is not decrease. This will
        make gradients so big.
    L2: At the beginning of optimization, the gradients is so big.
    Smooth L1: Can solve the problems of L1 and L2 loss.
           / 0.5 * x * x  x <= 1
    Loss =
           \ |x| - 0.5    x > 1
    Args:
        predictions: box regression [height, width, num_anchors, 4]
        delta: the delta of ground truth [height, width, num_anchors, 4]
        box_weights: only positive samples can participate in loss computation.
        sigma: sigma can make smooth l1 more linear.

    Returns:
        box_loss: the location loss

    """

    box_weights = tf.expand_dims(box_weights, -1)

    sigma_square = tf.cast(tf.math.square(sigma), tf.float32)
    delta = tf.abs(predictions - delta)
    smooth_l1_loss = tf.where(tf.greater_equal(delta, 1 / sigma_square),
                              delta - 0.5 / sigma_square,
                              0.5 * tf.square(delta) * sigma_square)
    return tf.multiply(smooth_l1_loss, box_weights)


def focal_loss(y_true, y_pred, label_weights, gamma=config.TRAIN.SIGMA, alpha=config.TRAIN.ALPHA):
    """ Use focal loss to solve the problem of negative loss overwhelms positive loss.
    But, focal loss also has some disadvantages that it only considers problem of classification but
    does not consider problem of location regression.
    Location regression is very important in object detection task.
    For example, boxes iou is a key factor of mAp metric.

    focal loss = alpha * (1 - p) ** gamma * log(p)

    Args:
        y_true:         ground truth
        y_pred:         network prediction
        label_weights:  ignored objects do not join in loss computation
        gamma:          2 is best in paper
        alpha:          0.25 is best in paper

    Returns:

    """

    label_weights = tf.expand_dims(label_weights, axis=-1)
    y_pred_sigmoid = tf.sigmoid(y_pred)

    alpha = alpha * y_true + (1 - alpha) * (1 - y_true)
    pt = y_true * y_pred_sigmoid + (1 - y_true) * (1 - y_pred_sigmoid)
    factor = tf.pow(x=(1 - pt), y=gamma)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)

    return label_weights * alpha * factor * cross_entropy


def retina_loss(y_pred, y_true, batch_size):

    batch_float = tf.cast(batch_size, tf.float32)
    # 5 stages and n batch
    num_stages = len(y_pred)

    loss = 0

    for i in range(num_stages):
        y_pred_stage = y_pred[i]
        y_true_stage = y_true[i]

        for j in range(batch_size):
            class_pred = y_pred_stage[0]  # [(batch, 52, 52, 9*num_classes), (batch, 52, 52, 9*4)][0]
            box_pred = y_pred_stage[1]  # (batch, 52, 52, 9*4)
            pred_shape = tf.shape(class_pred)
            h = pred_shape[1]
            w = pred_shape[2]
            class_pred = tf.reshape(class_pred, shape=(batch_size, h, w, 9, -1))
            box_pred = tf.reshape(box_pred, shape=(batch_size, h, w, 9, -1))
            class_pred_batch = class_pred[j]
            box_pred_batch = box_pred[j]

            y_true_batch = y_true_stage[j]
            labels = y_true_batch[0]
            delta = y_true_batch[1]
            labels_weights = y_true_batch[2]
            box_weights = y_true_batch[3]

            box_loss = tf.reduce_sum(smooth_l1(box_pred_batch, delta, box_weights)) / batch_float
            class_loss = tf.reduce_sum(focal_loss(labels, class_pred_batch, labels_weights)) / batch_float
            loss = loss + box_loss + class_loss

    return loss
