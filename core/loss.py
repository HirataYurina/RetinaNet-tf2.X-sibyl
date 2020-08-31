# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:loss.py
# software: PyCharm


import tensorflow as tf


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
    sigma_square = tf.cast(tf.math.square(sigma), tf.float32)
    delta = tf.abs(predictions - delta)
    smooth_l1_loss = tf.where(tf.greater_equal(delta, 1 / sigma_square),
                              delta - 0.5 / sigma_square,
                              0.5 * tf.square(delta) * sigma_square)
    return tf.multiply(smooth_l1_loss, box_weights)


def focal_loss(y_true, y_pred, label_weights, gamma=2, alpha=0.25):
    """alpha * (1 - p) ** gamma * log(p)"""

    alpha = alpha * y_true + (1 - alpha) * (1 - y_true)
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    factor = tf.pow(x=(1 - pt), y=gamma)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)

    return label_weights * alpha * factor * cross_entropy
