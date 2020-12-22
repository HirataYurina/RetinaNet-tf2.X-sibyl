# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:subnet.py
# software: PyCharm

from net.resnet import ResNet
from net.fpn import FPN

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from core.initializer import BiasInitializer


class SubNet(keras.Model):

    def __init__(self,
                 out_channels,
                 num_anchors,
                 num_classes,
                 **kwargs):
        super(SubNet, self).__init__(**kwargs)

        self.class_conv1 = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                         bias_initializer=keras.initializers.Constant(value=0),
                                         name='class_conv1')
        self.class_conv2 = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                         bias_initializer=keras.initializers.Constant(value=0),
                                         name='class_conv2')
        self.class_conv3 = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                         bias_initializer=keras.initializers.Constant(value=0),
                                         name='class_conv3')
        self.class_conv4 = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                                         kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                         bias_initializer=keras.initializers.Constant(value=0),
                                         name='class_conv4')
        # ----------------------------------------------------- #
        # The kernel initializer is PriorProbability
        # ----------------------------------------------------- #
        self.class_out = layers.Conv2D(filters=num_classes * num_anchors, kernel_size=3, padding='same',
                                       kernel_initializer=keras.initializers.Constant(value=0),
                                       bias_initializer=BiasInitializer(pi=0.01),
                                       name='class_out')

        self.box_conv1 = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                       bias_initializer=keras.initializers.Constant(value=0),
                                       name='box_conv1')
        self.box_conv2 = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                       bias_initializer=keras.initializers.Constant(value=0),
                                       name='box_conv2')
        self.box_conv3 = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                       bias_initializer=keras.initializers.Constant(value=0),
                                       name='box_conv3')
        self.box_conv4 = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                                       kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                       bias_initializer=keras.initializers.Constant(value=0),
                                       name='box_conv4')
        self.box_out = layers.Conv2D(filters=4 * num_anchors, kernel_size=3, padding='same',
                                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                                     bias_initializer=keras.initializers.Constant(value=0),
                                     name='box_out')

    def __call__(self, inputs):
        results = []

        for input_level in inputs:
            x = self.class_conv1(input_level)
            x = layers.ReLU()(x)
            x = self.class_conv2(x)
            x = layers.ReLU()(x)
            x = self.class_conv3(x)
            x = layers.ReLU()(x)
            x = self.class_conv4(x)
            x = layers.ReLU()(x)
            x = self.class_out(x)
            # x = layers.Activation(tf.nn.sigmoid)(x)

            y = self.box_conv1(input_level)
            y = layers.ReLU()(y)
            y = self.box_conv2(y)
            y = layers.ReLU()(y)
            y = self.box_conv3(y)
            y = layers.ReLU()(y)
            y = self.box_conv4(y)
            y = layers.ReLU()(y)
            y = self.box_out(y)

            results.append([x, y])

        return results


# wrap classification subnet into one layer
def class_subnet(out_channels, num_classes, num_anchors):
    inputs = keras.Input(shape=(None, None, out_channels))
    x = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(value=0),
                      activation='relu',
                      name='pyramid_classification_0')(inputs)
    x = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(value=0),
                      activation='relu',
                      name='pyramid_classification_1')(x)
    x = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(value=0),
                      activation='relu',
                      name='pyramid_classification_2')(x)
    x = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(value=0),
                      activation='relu',
                      name='pyramid_classification_3')(x)
    x = layers.Conv2D(filters=num_classes * num_anchors, kernel_size=3, padding='same',
                      kernel_initializer=keras.initializers.Constant(value=0),
                      bias_initializer=BiasInitializer(pi=0.01),
                      name='pyramid_classification')(x)
    x = layers.Reshape(target_shape=(-1, num_classes))(x)

    return keras.Model(inputs, x, name='classification_submodel')


# wrap box regression subnet into one layer
def box_subnet(out_channels, num_anchors):
    inputs = keras.Input(shape=(None, None, out_channels))
    x = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(value=0),
                      activation='relu',
                      name='pyramid_regression_0')(inputs)
    x = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(value=0),
                      activation='relu',
                      name='pyramid_regression_1')(x)
    x = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(value=0),
                      activation='relu',
                      name='pyramid_regression_2')(x)
    x = layers.Conv2D(filters=out_channels, kernel_size=3, padding='same',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(value=0),
                      activation='relu',
                      name='pyramid_regression_3')(x)
    x = layers.Conv2D(filters=4 * num_anchors, kernel_size=3, padding='same',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.01),
                      bias_initializer=keras.initializers.Constant(value=0),
                      name='pyramid_regression')(x)
    x = layers.Reshape(target_shape=(-1, 4))(x)

    return keras.Model(inputs, x, name='regression_submodel')


if __name__ == '__main__':
    resnet_50 = ResNet(50)
    inputs_ = keras.Input(shape=(416, 416, 3))
    c2, c3, c4, c5 = resnet_50(inputs_)
    p3, p4, p5, p6, p7 = FPN()([c2, c3, c4, c5])

    # 42,402,172 params
    subnet = SubNet(out_channels=256,
                    num_classes=80,
                    num_anchors=9)

    results_ = subnet([p3, p4, p5, p6, p7])

    retinanet = keras.Model(inputs_, results_)
    retinanet.summary()
    print(len(retinanet.layers))
