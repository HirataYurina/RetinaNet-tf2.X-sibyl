# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:get_dataset.py
# software: PyCharm

import tensorflow as tf
from dataset.augment import get_random_data
from core.anchors import Anchors
import numpy as np


class DataGenerator:

    def __init__(self, anno_lines, input_shape, num_classes, batch_size):
        self.anno_lines = anno_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.Anchor = Anchors()

    def data_generate(self):
        """
        data generator

        Returns:
            image_data: tf.tensor
            target_3:   p3 stage's outputs [[labels, delta, label_weights, box_weights]]
            target_4:   p4 stage's outputs [[labels, delta, label_weights, box_weights]]
            target_5:   p5 stage's outputs [[labels, delta, label_weights, box_weights]]
            target_6:   p6 stage's outputs [[labels, delta, label_weights, box_weights]]
            target_7:   p7 stage's outputs [[labels, delta, label_weights, box_weights]]
        """
        anchors = self.Anchor.anchors_generator(self.input_shape)

        n = len(self.anno_lines)
        i = 0

        while True:
            image_data = []
            true_box = []
            true_class = []
            for j in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(self.anno_lines)
                image, boxes = get_random_data(self.anno_lines[i], self.input_shape)
                results = self.Anchor.anchors_target_total(anchors, boxes, self.num_classes, self.input_shape)
                image_data.append(image)
                true_box.append(results[0])
                true_class.append(results[1])

                i = (i + 1) % n

            image_data = tf.stack(image_data, axis=0)
            true_box = tf.stack(true_box, axis=0)
            true_class = tf.stack(true_class, axis=0)

            yield [image_data, [true_box, true_class]], tf.zeros(shape=(self.batch_size,))

    # #########################################
    # this function has been deprecated
    # #########################################
    def data_generate_other(self):
        anchors = self.Anchor.anchors_generator(self.input_shape)

        n = len(self.anno_lines)
        i = 0

        while True:
            targets_3 = []
            targets_4 = []
            targets_5 = []
            targets_6 = []
            targets_7 = []
            image_data = []

            for j in range(self.batch_size):
                if i == 0:
                    np.random.shuffle(self.anno_lines)
                image, boxes = get_random_data(self.anno_lines[i], self.input_shape)
                results = self.Anchor.anchors_target_total(anchors, boxes, self.num_classes, self.input_shape)
                image_data.append(image)
                targets_3.append(results[0])
                targets_4.append(results[1])
                targets_5.append(results[2])
                targets_6.append(results[3])
                targets_7.append(results[4])

                i = (i + 1) % n

            image_data = tf.stack(image_data, axis=0)

            yield [image_data, [targets_3, targets_4, targets_5, targets_6, targets_7]], tf.zeros(shape=(2,))


if __name__ == '__main__':

    with open('../datas/2088_trainval.txt') as f:
        annotations = f.readlines()

    data_gene = DataGenerator(anno_lines=annotations,
                              input_shape=(416, 416),
                              num_classes=6,
                              batch_size=4)
    data = data_gene.data_generate().__next__()
    print(data)
