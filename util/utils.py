# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:utils.py
# software: PyCharm

import tensorflow as tf


def compute_ious(boxes1, boxes2):
    """Compute ious between boxes1 and boxes2

    Args:
        boxes1: shape is (h, w, num_anchors, 4)
        boxes2: shape is (num_gt, 4)

    Returns:
        ious: shape is (h, w, num_anchors, num_gt)

    """
    boxes1 = tf.expand_dims(boxes1, axis=-2)
    boxes1_wh = boxes1[..., 2:] - boxes1[..., :2]
    boxes1_min = boxes1[..., :2]
    boxes1_max = boxes1[..., 2:]
    boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]

    boxes2_wh = boxes2[..., 2:] - boxes2[..., :2]
    boxes2_min = boxes2[..., :2]
    boxes2_max = boxes2[..., 2:]
    boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]

    inter_min = tf.maximum(boxes1_min, boxes2_min)
    inter_max = tf.minimum(boxes1_max, boxes2_max)
    inter_wh = inter_max - inter_min
    inter_wh = tf.maximum(0.0, inter_wh)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    ious = inter_area / (boxes1_area + boxes2_area - inter_area)

    return ious


def trim_zeros(inputs):
    """Usually, the gt boxes are padded with zeros elements,
    so we need to discard invalid boxes in gt boxes.

    Args:
        inputs: ground truth (num_gt, 4)

    Returns:
        valid_gt: (num_valid, 4)
    """
    valid_mask = tf.reduce_sum(tf.abs(inputs), axis=-1)
    valid_mask = tf.cast(valid_mask, tf.bool)

    valid_gt = tf.boolean_mask(inputs, valid_mask)

    return valid_gt


def box2delta(anchors, gt_boxes, means, std):
    # According to the information provided by a keras-retinanet author, they got marginally better results using
    # the following way of bounding box parametrization.
    # See https://github.com/fizyr/keras-retinanet/issues/1273#issuecomment-585828825 for more details
    anchors_w = anchors[..., 2] - anchors[..., 0]
    anchors_h = anchors[..., 3] - anchors[..., 1]

    # gt_x1 = gt_boxes[..., 0]
    # gt_y1 = gt_boxes[..., 1]
    # gt_x2 = gt_boxes[..., 2]
    # gt_y2 = gt_boxes[..., 3]
    # gt_x = (gt_x1 + gt_x2) / 2.0
    # gt_y = (gt_y1 + gt_y2) / 2.0
    # gt_w = gt_x2 - gt_x1
    # gt_h = gt_y2 - gt_y1

    dx1 = (gt_boxes[..., 0] - anchors[..., 0]) / anchors_w  # anchors_w always bigger than 0
    dy1 = (gt_boxes[..., 1] - anchors[..., 1]) / anchors_h
    dx2 = (gt_boxes[..., 2] - anchors[..., 2]) / anchors_w
    dy2 = (gt_boxes[..., 3] - anchors[..., 3]) / anchors_h

    delta = tf.stack([dx1, dy1, dx2, dy2], axis=-1)
    # we need to normalize the delta.
    # I think this operation can make the boxes shift of different images smaller.
    # So, make distribution tighter can make model more stable.
    delta = (delta - means) / std

    return delta


if __name__ == '__main__':

    # test <trim_zeros>
    gt_boxes_ = tf.constant([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
    valid_gt_ = trim_zeros(gt_boxes_)
    print(len(valid_gt_))
