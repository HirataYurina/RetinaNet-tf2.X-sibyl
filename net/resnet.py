# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:resnet.py
# software: PyCharm

# import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class _ResBlock(keras.Model):

    def __init__(self,
                 kernels,
                 strides,
                 stage,
                 block,
                 downsample=False,
                 **kwargs):
        super(_ResBlock, self).__init__(**kwargs)

        conv_name = 'res' + str(stage) + block + '_branch'
        bn_name = 'bn' + str(stage) + block + '_branch'
        kernels1, kernels2, kernels3 = kernels

        self.downsample = downsample
        self.conv1 = layers.Conv2D(kernels1, 1, strides, padding='same', use_bias=False,
                                   kernel_initializer=keras.initializers.he_normal(), name=conv_name + '2a')
        self.bn1 = layers.BatchNormalization(name=bn_name + '2a')

        self.conv2 = layers.Conv2D(kernels2, 3, padding='same', use_bias=False,
                                   kernel_initializer=keras.initializers.he_normal(), name=conv_name + '2b')
        self.bn2 = layers.BatchNormalization(name=bn_name + '2b')

        self.conv3 = layers.Conv2D(kernels3, 1, padding='same', use_bias=False,
                                   kernel_initializer=keras.initializers.he_normal(), name=conv_name + '2c')
        self.bn3 = layers.BatchNormalization(name=bn_name + '2c')

        if self.downsample:
            self.conv4 = layers.Conv2D(kernels3, 1, strides, padding='same', use_bias=False,
                                       kernel_initializer=keras.initializers.he_normal(), name=conv_name + '1')
            self.bn4 = layers.BatchNormalization(name=bn_name + '1')

    def __call__(self, inputs, training=True, mask=None):
        x = self.conv1(inputs)
        # specify training or not training when you are using bn layer
        x = self.bn1(x, training=training)
        x = layers.ReLU()(x)

        x = self.conv2(x)
        # specify training or not training when you are using bn layer
        x = self.bn2(x, training=training)
        x = layers.ReLU()(x)

        x = self.conv3(x)
        # specify training or not training when you are using bn layer
        x = self.bn3(x)
        # don't compress outputs before add operation.
        # x = layers.ReLU()(x)

        if self.downsample:
            y = self.conv4(inputs)
            y = self.bn4(y)
        else:
            y = inputs

        # short cut
        outputs = layers.Add()([x, y])
        outputs = layers.ReLU()(outputs)

        return outputs


class ResNet(keras.Model):
    """ResNet
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf

                                  x ------------------
                                  |                  |
                                  |                  |
                                  V                  |
                          -----------------          |
                          |  weight layer |          |
                          -----------------          |
                                  |                  |
                                  | Relu             |
                                  V                  |
                          -----------------          |
                          |  weight layer |          |
                          -----------------          |
                  f(x)            |                  |
                                  V                  |
                  f(x) + x        +  <----------------
                                  |
                                  V
                                 Relu   So, we don't compress outputs before add operation.

    """

    def __init__(self, depth, **kwargs):
        super(ResNet, self).__init__(**kwargs)

        if depth not in [50, 101]:
            raise AssertionError('depth must be 50 or 101.')
        self.depth = depth

        self.pad = layers.ZeroPadding2D((3, 3))
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding='valid', use_bias=False,
                                   kernel_initializer=keras.initializers.he_normal(), name='conv1')
        self.bn1 = layers.BatchNormalization(name='bn_conv1')
        
        # block*3
        self.resblock_2a = _ResBlock(kernels=[64, 64, 256], strides=1, stage=2, block='a', downsample=True)
        self.resblock_2b = _ResBlock(kernels=[64, 64, 256], strides=1, stage=2, block='b', downsample=False)
        self.resblock_2c = _ResBlock(kernels=[64, 64, 256], strides=1, stage=2, block='c', downsample=False)
        # block*4
        self.resblock_3a = _ResBlock(kernels=[128, 128, 512], strides=2, stage=3, block='a', downsample=True)
        self.resblock_3b = _ResBlock(kernels=[128, 128, 512], strides=1, stage=3, block='b', downsample=False)
        self.resblock_3c = _ResBlock(kernels=[128, 128, 512], strides=1, stage=3, block='c', downsample=False)
        self.resblock_3d = _ResBlock(kernels=[128, 128, 512], strides=1, stage=3, block='d', downsample=False)
        # block*6
        self.resblock_4a = _ResBlock(kernels=[256, 256, 1024], strides=2, stage=4, block='a', downsample=True)
        self.resblock_4b = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='b', downsample=False)
        self.resblock_4c = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='c', downsample=False)
        self.resblock_4d = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='d', downsample=False)
        self.resblock_4e = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='e', downsample=False)
        self.resblock_4f = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='f', downsample=False)

        if self.depth == 101:
            self.resblock_4g = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='g', downsample=False)
            self.resblock_4h = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='h', downsample=False)
            self.resblock_4i = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='i', downsample=False)
            self.resblock_4j = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='j', downsample=False)
            self.resblock_4k = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='k', downsample=False)
            self.resblock_4l = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='l', downsample=False)
            self.resblock_4m = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='m', downsample=False)
            self.resblock_4n = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='n', downsample=False)
            self.resblock_4o = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='o', downsample=False)
            self.resblock_4p = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='p', downsample=False)
            self.resblock_4q = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='q', downsample=False)
            self.resblock_4r = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='r', downsample=False)
            self.resblock_4s = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='s', downsample=False)
            self.resblock_4t = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='t', downsample=False)
            self.resblock_4u = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='u', downsample=False)
            self.resblock_4v = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='v', downsample=False)
            self.resblock_4w = _ResBlock(kernels=[256, 256, 1024], strides=1, stage=4, block='w', downsample=False)
        # block*3
        self.resblock_5a = _ResBlock(kernels=[512, 512, 2048], strides=2, stage=5, block='a', downsample=True)
        self.resblock_5b = _ResBlock(kernels=[512, 512, 2048], strides=1, stage=5, block='b', downsample=False)
        self.resblock_5c = _ResBlock(kernels=[512, 512, 2048], strides=1, stage=5, block='c', downsample=False)

    def __call__(self, inputs, training=True, mask=None):
        """

        Args:
            inputs: (batch, h, w, c)
            training: True or False
            mask: don't use in this fuction

        Returns:
             c2: h / 4
             c3: h / 8
             c4: h / 16
             c5: h / 32

        """
        x = self.pad(inputs)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        x = self.resblock_2a(x, training=training)
        x = self.resblock_2b(x, training=training)
        x = self.resblock_2c(x, training=training)
        c2 = x

        x = self.resblock_3a(x, training=training)
        x = self.resblock_3b(x, training=training)
        x = self.resblock_3c(x, training=training)
        x = self.resblock_3d(x, training=training)
        c3 = x

        x = self.resblock_4a(x, training=training)
        x = self.resblock_4b(x, training=training)
        x = self.resblock_4c(x, training=training)
        x = self.resblock_4d(x, training=training)
        x = self.resblock_4e(x, training=training)
        x = self.resblock_4f(x, training=training)
        if self.depth == 101:
            x = self.resblock_4g(x, training=training)
            x = self.resblock_4h(x, training=training)
            x = self.resblock_4i(x, training=training)
            x = self.resblock_4g(x, training=training)
            x = self.resblock_4k(x, training=training)
            x = self.resblock_4l(x, training=training)
            x = self.resblock_4m(x, training=training)
            x = self.resblock_4n(x, training=training)
            x = self.resblock_4o(x, training=training)
            x = self.resblock_4p(x, training=training)
            x = self.resblock_4q(x, training=training)
            x = self.resblock_4r(x, training=training)
            x = self.resblock_4s(x, training=training)
            x = self.resblock_4t(x, training=training)
            x = self.resblock_4u(x, training=training)
            x = self.resblock_4v(x, training=training)
            x = self.resblock_4w(x, training=training)
        c4 = x

        x = self.resblock_5a(x, training=training)
        x = self.resblock_5b(x, training=training)
        x = self.resblock_5c(x, training=training)
        c5 = x

        # muti-scale feature maps
        return [c2, c3, c4, c5]


if __name__ == '__main__':

    # -------------------------- #
    # test resnet
    # resnet50: 23,508,032 params
    # resnet101: 41,382,976 params
    # -------------------------- #
    resnet_50 = ResNet(50)
    inputs_ = keras.Input(shape=(608, 608, 3))
    c2_, c3_, c4_, c5_ = resnet_50(inputs_)

    model = keras.Model(inputs_, [c2_, c3_, c4_, c5_])

    print(len(model.layers))

    # weights_path = r'E:\pretrained-model\resnet50_coco_best_v2.1.0.h5'
    # model.load_weights(weights_path, by_name=True)
    # print('load weights successfully!')
