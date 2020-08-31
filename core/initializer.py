# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:initializer.py
# software: PyCharm

import tensorflow.keras as keras
import numpy as np
import math


class BiasInitializer(keras.initializers.Initializer):

    def __init__(self, pi=0.01, **kwargs):
        super(BiasInitializer, self).__init__(**kwargs)
        self.pi = pi

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        results = np.ones(shape=shape, dtype=dtype) * -math.log((1 - self.pi) / self.pi)
        return results
