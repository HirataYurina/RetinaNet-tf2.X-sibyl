# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:fpn.py
# software: PyCharm

import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class FPN(keras.Model):

    def __init__(self, out_channels=256, **kwargs):
        super(FPN, self).__init__(**kwargs)

        self.out_channels = out_channels

        self.c3p3 = layers.Conv2D(out_channels, 1, name='c3_reduced')
        self.c4p4 = layers.Conv2D(out_channels, 1, name='c4_reduced')
        self.c5p5 = layers.Conv2D(out_channels, 1, name='c5_reduced')
        self.p5p6 = layers.Conv2D(out_channels, 3, strides=2, padding='same', name='p6')
        self.p6p7 = layers.Conv2D(out_channels, 3, strides=2, padding='same', name='p7')

        self.p5_upsample = layers.UpSampling2D(size=(2, 2), name='p5_upsample')
        self.p4_upsample = layers.UpSampling2D(size=(2, 2), name='p4_upsample')

        self.p3 = layers.Conv2D(out_channels, 3, padding='same', name='p3')
        self.p4 = layers.Conv2D(out_channels, 3, padding='same', name='p4')
        self.p5 = layers.Conv2D(out_channels, 3, padding='same', name='p5')

    def __call__(self, inputs):
        """
        FPN for retina-net.
        This differs slightly from 'Feature pyramid networks for object detection'.
        This minor modifications improve speed while maintaining accuracy.

        Args:
            inputs: the outputs of backbone

        Returns:
            [p3, p4, p5, p6, p7]

        """
        _, c3, c4, c5 = inputs

        p5 = self.c5p5(c5)
        p4 = self.c4p4(c4) + self.p5_upsample(p5)
        p3 = self.c3p3(c3) + self.p4_upsample(p4)
        p5 = self.p5(p5)
        p4 = self.p4(p4)
        p3 = self.p3(p3)

        p6 = self.p5p6(c5)
        p6_relu = layers.ReLU()(p6)
        p7 = self.p6p7(p6_relu)

        return [p3, p4, p5, p6, p7]


if __name__ == '__main__':

    C3 = keras.Input((52, 52, 512))
    C4 = keras.Input((26, 26, 1024))
    C5 = keras.Input((13, 13, 2048))

    fpn = FPN()
    p3_, p4_, p5_, p6_, p7_ = fpn([C3, C4, C5])

    fpn_model = keras.Model([C3, C4, C5],
                            [p3_, p4_, p5_, p6_, p7_])
    fpn_model.summary()
