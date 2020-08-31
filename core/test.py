# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:test.py
# software: PyCharm

import tensorflow as tf

a = tf.ones(shape=(10, 4))
b = tf.constant([2.0, 2.0, 2.0, 2.0])
print(a - b)


print(tf.keras.optimizers.Adam().lr.assign(1e-04))
print(tf.keras.optimizers.Adam().lr)
