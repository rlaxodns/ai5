import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(777)

#1. 데이터
x_train = np.array([[[[1], [2], [3]],
                     [[4], [5], [6]],
                     [[7], [8], [9]]]])
print(x_train.shape) # (1, 3, 3, 1)

x = tf.compat.v1.placeholder(tf.float32, [None, 3, 3, 1])
w = tf.compat.v1.constant([[[[1.]], [[0.]]],
                           [[[1.]], [[0.]]]])

print(w) # Tensor("Const:0", shape=(2, 2, 1, 1), dtype=float32)