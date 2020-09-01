# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:anchors.py
# software: PyCharm

import tensorflow as tf
from util.utils import compute_ious, trim_zeros, box2delta


class Anchors(object):

    _default = {'sizes': [32, 64, 128, 256, 512],
                'strides': [8, 16, 32, 64, 128],
                'ratios': [0.5, 1.0, 2.0],
                'scales': [2 ** 0, 2 ** (1 / 3), 2 ** (2 / 3)],
                'target_means': [0., 0., 0., 0.],
                'target_stds': [0.1, 0.1, 0.2, 0.2]}

    def __init__(self,
                 positive_threshold=0.5,
                 negative_threshold=0.4):
        self.__dict__.update(self._default)
        self.positive_thres = positive_threshold
        self.negative_thres = negative_threshold

    def anchors_generator(self, img_shape):
        """Generate anchors for predictions at each level.

        Args:
            img_shape: shape of input img [h, w]

        Returns:
            anchors: a list of anchors for different level

        """
        levels = len(self.strides)
        num_anchors = len(self.ratios) * len(self.scales)
        height, width = img_shape

        anchors = []

        for level in range(levels):
            stride = self.strides[level]
            size = self.sizes[level]
            # (3, 3)
            ratios_level, scales_level = tf.meshgrid(self.ratios, self.scales)
            w_level = tf.sqrt(tf.cast(tf.square(size), dtype='float32') / ratios_level)
            w_level_3_3 = w_level * scales_level
            w_level = tf.reshape(w_level_3_3, shape=(num_anchors, 1))
            h_level = tf.reshape(w_level_3_3 * ratios_level, shape=(num_anchors, 1))
            # (9, 2)
            wh_level = tf.concat([w_level, h_level], axis=-1)
            # centers of each anchor
            grid_height = height // stride
            grid_width = width // stride
            grid_xy = tf.meshgrid(tf.range(grid_width), tf.range(grid_height))
            grid_xy = tf.stack(grid_xy, axis=-1)
            # (h, w, 1, 2)
            grid_xy = tf.cast(grid_xy, dtype='float32')
            anchor_centers = tf.expand_dims((grid_xy + 0.5) * tf.cast(stride, dtype='float32'), axis=2)
            # tf.concat does not support broadcast, so we need to change shape of
            # wh_level and anchor_centers.
            # (9, 2) --> (h, w, 9, 2)
            wh_level = tf.tile(tf.reshape(wh_level, shape=(1, 1, 9, 2)), tf.constant([grid_height, grid_width, 1, 1]))
            # (h, w, 1, 2) --> (h, w, 9, 2)
            anchor_centers = tf.tile(anchor_centers, [1, 1, 9, 1])
            anchors_level = tf.concat([anchor_centers, wh_level], axis=-1)
            anchors.append(anchors_level)

        return anchors

    def anchors_target_total(self, anchors, gt_boxes, num_classes):
        """Generate ground truth

        Args:
            anchors:     shape like (13, 13, 3, 4)
            gt_boxes:    (n, 4)
            num_classes:  a scalar

        Returns:
            results

        """

        results = []

        for anchor_level in anchors:
            labels, delta, label_weights, box_weights = \
                self._anchors_target_level(anchor_level, gt_boxes, num_classes)
            results.append([labels, delta, label_weights, box_weights])

        return results

    def _anchors_target_level(self, anchors_level, gt_boxes, num_classes):

        anchors_left = anchors_level[..., :2] - anchors_level[..., 2:] / 2
        anchors_right = anchors_level[..., :2] + anchors_level[..., 2:] / 2
        anchors = tf.concat([anchors_left, anchors_right], axis=-1)

        # 1.discard invalid gt boxes.
        valid_gt = trim_zeros(gt_boxes)
        ious = compute_ious(anchors, valid_gt[..., 0:4])
        gt_class = valid_gt[..., 4]

        # (h, w, 9)
        ious_max = tf.reduce_max(ious, axis=-1)
        ious_argmax = tf.argmax(ious, axis=-1)

        anchors_shape = tf.shape(anchors_level)
        height, width, num_anchors, _ = anchors_shape
        labels = tf.zeros(shape=(height, width, num_anchors, num_classes))

        # 2.if max iou > positive threshold, anchors are assigned to ground truth.
        # (num_pos, 3)
        pos_index = tf.where(ious_max >= 0.5)
        class_id = tf.gather(gt_class, tf.gather_nd(ious_argmax, pos_index))
        class_id = tf.expand_dims(class_id, axis=-1)
        positive_class_index = tf.concat([pos_index, class_id], axis=-1)

        num_positive = tf.shape(positive_class_index)[0]

        labels = tf.tensor_scatter_nd_update(labels, positive_class_index, tf.ones(shape=(num_positive,)))

        # 3.if max iou < negative thredhold, anchors are assigned to background.
        neg_index = tf.where(ious_max < 0.4)

        # 4.transform boxes to delta
        ious_argmax = tf.reshape(ious_argmax, (height * width * num_anchors,))
        valid_gt_box = valid_gt[..., 0:4]
        ious_argmax_box = tf.gather(valid_gt_box, ious_argmax)
        ious_argmax_box = tf.reshape(ious_argmax_box, (height, width, num_anchors, 4))
        delta = box2delta(anchors_level, ious_argmax_box, self.target_means, self.target_stds)

        # 5.create label weights and box weights
        label_weights = tf.zeros(shape=(height, width, num_anchors))
        box_weights = tf.zeros(shape=(height, width, num_anchors))

        label_weights = tf.tensor_scatter_nd_update(label_weights, pos_index, tf.ones(shape=(num_positive,)))

        num_negative = tf.shape(neg_index)[0]
        label_weights = tf.tensor_scatter_nd_update(label_weights, neg_index, tf.ones(shape=(num_negative,)))

        box_weights = tf.tensor_scatter_nd_update(box_weights, pos_index, tf.ones(shape=(num_positive,)))

        return labels, delta, label_weights, box_weights


if __name__ == '__main__':
    anchors = Anchors()

    anchors_generated = anchors.anchors_generator(img_shape=(416, 416))
    print(anchors_generated)
