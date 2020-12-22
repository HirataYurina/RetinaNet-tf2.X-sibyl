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

    num_pos = tf.where(tf.equal(box_weights, 1))
    num_pos = tf.maximum(tf.constant(1), tf.shape(num_pos)[0])
    num_pos = tf.cast(num_pos, tf.float32)
    return smooth_l1_loss * box_weights, num_pos


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
    # pt = y_true * y_pred_sigmoid + (1 - y_true) * (1 - y_pred_sigmoid)
    # factor = tf.pow(x=(1 - pt), y=gamma)
    pt = y_true * (1 - y_pred_sigmoid) + (1 - y_true) * y_pred_sigmoid
    factor = tf.pow(x=pt, y=gamma)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)

    return label_weights * alpha * factor * cross_entropy


def retina_loss(args):
    """retina loss

    Args:
        y_true: [(batch, num_anchors, 5), (batch, num_anchors, 7)]
        y_pred: [(batch, num_anchors, 4), (batch, num_anchors, 6)]

    Returns:
        loss

    """
    y_true = args[0]
    y_pred = args[1]
    box_pred = y_pred[0]
    class_pred = y_pred[1]
    delta = y_true[0][..., :4]  # (batch, num_anchors, 4)
    labels = y_true[1][..., :6]  # (batch, num_anchors, 6)
    box_weights = y_true[0][..., 4]  # (batch, num_anchors)
    label_weights = y_true[1][..., 6]  # (batch, num_anchors)

    # class loss
    class_loss = tf.reduce_sum(focal_loss(labels, class_pred, label_weights))
    # box loss
    box_loss, num_positive = smooth_l1(box_pred, delta, box_weights)
    box_loss = tf.reduce_sum(box_loss)
    return (class_loss + box_loss) / num_positive


# ###################################
# This function has been deprecated
# ###################################
def retina_loss_other(args, batch_size):

    y_pred = args[0]
    y_true = args[1]

    batch_float = tf.cast(batch_size, tf.float32)
    # 5 stages and n batch
    num_stages = len(y_pred)
    # box_pred = y_pred[0]
    # class_pred = y_pred[1]

    loss = 0
    total_class_losses = 0
    total_box_losses = 0

    for i in range(num_stages):
        y_true_stage = y_true[i]
        y_pred_stage = y_pred[i]

        class_pred_stage = y_pred_stage[0]  # [(batch, 52, 52, 9*num_classes), (batch, 52, 52, 9*4)][0]
        box_pred_stage = y_pred_stage[1]  # (batch, 52, 52, 9*4)
        pred_shape = tf.shape(class_pred_stage)
        h = pred_shape[1]
        w = pred_shape[2]
        class_pred_stage = tf.reshape(class_pred_stage, shape=(batch_size, h, w, 9, -1))
        box_pred_stage = tf.reshape(box_pred_stage, shape=(batch_size, h, w, 9, -1))

        for j in range(batch_size):

            class_pred_batch = class_pred_stage[j]
            box_pred_batch = box_pred_stage[j]

            y_true_batch = y_true_stage[j]
            labels = y_true_batch[0]
            delta = y_true_batch[1]
            labels_weights = y_true_batch[2]
            box_weights = y_true_batch[3]

            box_loss = tf.reduce_sum(smooth_l1(box_pred_batch, delta, box_weights)) / batch_float
            class_loss = tf.reduce_sum(focal_loss(labels, class_pred_batch, labels_weights)) / batch_float
            loss += box_loss + class_loss
            total_box_losses += box_loss
            total_class_losses += class_loss
            # print('class_loss:', class_loss)
            # print('box_loss', box_loss)
    # print(y_true)
    # print(y_pred[0][0][0, :3, :3, :6])
    print('total_class_loss:', total_class_losses)
    print('total_box_loss', total_box_losses)

    return loss
