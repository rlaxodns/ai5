import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, 
              input_shape=(32, 32, 3))
# 13개의 conv layer에 2개의 Dense layer 에 따라서 총 15개의 가중치와
# 15개의 편향을 더해서 총 30개의 print(len(model.weights)) # 30

model : Sequential = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

#1. 전체동결
# model.trainable = False

#2. 전체동결
# for layers in model.layers:
#     layers.trainable = False
# model.summary()

#3. 부분동결
print(model.layers)
# [<keras.engine.functional.Functional object at 0x0000018EAFF7A430>,]
#  <keras.layers.core.flatten.Flatten object at 0x0000018EAFF66160>,
#  <keras.layers.core.dense.Dense object at 0x0000018EAFF53D00>,
#  <keras.layers.core.dense.Dense object at 0x0000018EAFFCE700>

print(model.layers[0])

model.layers[1].trainable = False
print(model.layers)

# exit()
model.summary()


import pandas as pd
pd.set_option('max_colwidth', None)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
print(pd.__version__)

"""
                                                          Layer Type Layer Name  Layer Trainable
0  <keras.engine.functional.Functional object at 0x0000023B8748FE20>      vgg16             True
1   <keras.layers.core.flatten.Flatten object at 0x0000023B87463130>    flatten            False
2       <keras.layers.core.dense.Dense object at 0x0000023B87460220>      dense             True
3       <keras.layers.core.dense.Dense object at 0x0000023B8753E910>    dense_1             True
"""