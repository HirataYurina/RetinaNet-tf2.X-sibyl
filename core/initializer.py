# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:initializer.py
# software: PyCharm

import tensorflow.keras as keras
import numpy as np
import math
import tensorflow as tf


class BiasInitializer(keras.initializers.Initializer):
    """
    4.1 Inference and Training
    For the final conv layer of the classification subnet,
    we set the bias initialization to b = -log((1 - pi) / pi),
    where pi specifies that the start of training every anchor should be labeled
    as foreground with confidence of pi.

    Why we initialize bias to be 0.01?
    If we initialize bias just as "Deep residual learning for image recognition",
    the initial probability will be sigmoid(0) = 0.5.
    The negative object loss loss_negative = -log(0.5) = 0.69. So, the negative loss will overwhelm positive loss.
    If we initializer bias to be 0.01,
    the negative object loss loss_negative = -log(1-0.01) = 0.01.
    """

    def __init__(self, pi=0.01, **kwargs):
        super(BiasInitializer, self).__init__(**kwargs)
        self.pi = pi

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        results = tf.ones(shape=shape, dtype=dtype) * (-tf.math.log((1 - self.pi) / self.pi))
        return results
