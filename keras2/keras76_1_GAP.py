import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer, Flatten, GlobalAveragePooling2D
import tensorflow as tf

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)

#1. 데이터
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#2. 모델
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(# weights='imagenet',
              include_top=False, # True가 디폴트 #False적용시 flatten()하단의 Dense layer가 사라짐
              input_shape=(32, 32, 3)
              )

vgg16.trainable = True

model = Sequential()
model.add(vgg16)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

model.summary()

"""
Model: "sequential_Flatten"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, 1000)              138357544

 flatten (Flatten)           (None, 1000)              0

 dense (Dense)               (None, 100)               100100

 dense_1 (Dense)             (None, 100)               10100

 dense_2 (Dense)             (None, 10)                1010

=================================================================
Total params: 138,468,754
Trainable params: 138,468,754
Non-trainable params: 0
_________________________________________________________________

Model: "sequential_GAP"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 vgg16 (Functional)          (None, None, None, 512)   14714688

 global_average_pooling2d (G  (None, 512)              0
 lobalAveragePooling2D)

 dense (Dense)               (None, 100)               51300

 dense_1 (Dense)             (None, 100)               10100

 dense_2 (Dense)             (None, 10)                1010

=================================================================
Total params: 14,777,098
Trainable params: 14,777,098
Non-trainable params: 0
_________________________________________________________________
"""