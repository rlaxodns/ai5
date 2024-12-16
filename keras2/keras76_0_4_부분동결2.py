import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=True, )


model.layers[-3].trainable = False

model.summary()
"""
fc1을 동결
=================================================================
Total params: 138,357,544
Trainable params: 35,593,000
Non-trainable params: 102,764,544
_________________________________________________________________
"""

import pandas as pd
pd.set_option('max_colwidth', None)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
results = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(results)
print(pd.__version__)

"""
                                                            Layer Type    Layer Name  Layer Trainable
0   <keras.engine.input_layer.InputLayer object at 0x000001EF31EB3FA0>       input_1             True
1     <keras.layers.convolutional.Conv2D object at 0x000001EF6C5FF8B0>  block1_conv1             True
2     <keras.layers.convolutional.Conv2D object at 0x000001EF6C5FFEB0>  block1_conv2             True
3     <keras.layers.pooling.MaxPooling2D object at 0x000001EF6C6E12E0>   block1_pool             True
4     <keras.layers.convolutional.Conv2D object at 0x000001EF6C6A6400>  block2_conv1             True
5     <keras.layers.convolutional.Conv2D object at 0x000001EF6C6E9880>  block2_conv2             True
6     <keras.layers.pooling.MaxPooling2D object at 0x000001EF6C6F7F70>   block2_pool             True
7     <keras.layers.convolutional.Conv2D object at 0x000001EF6C6F7580>  block3_conv1             True
8     <keras.layers.convolutional.Conv2D object at 0x000001EF6C941130>  block3_conv2             True
9     <keras.layers.convolutional.Conv2D object at 0x000001EF6C9445E0>  block3_conv3             True
10    <keras.layers.pooling.MaxPooling2D object at 0x000001EF6C941CD0>   block3_pool             True
11    <keras.layers.convolutional.Conv2D object at 0x000001EF6C944EB0>  block4_conv1             True
12    <keras.layers.convolutional.Conv2D object at 0x000001EF6C952280>  block4_conv2             True
13    <keras.layers.convolutional.Conv2D object at 0x000001EF6C94BD90>  block4_conv3             True
14    <keras.layers.pooling.MaxPooling2D object at 0x000001EF6C951340>   block4_pool             True
15    <keras.layers.convolutional.Conv2D object at 0x000001EF6C9560D0>  block5_conv1             True
16    <keras.layers.convolutional.Conv2D object at 0x000001EF6C9628E0>  block5_conv2             True
17    <keras.layers.convolutional.Conv2D object at 0x000001EF6C6FCE80>  block5_conv3             True
18    <keras.layers.pooling.MaxPooling2D object at 0x000001EF6C96B130>   block5_pool             True
19    <keras.layers.core.flatten.Flatten object at 0x000001EF6C96BA00>       flatten             True
20        <keras.layers.core.dense.Dense object at 0x000001EF6C969FD0>           fc1             False
21        <keras.layers.core.dense.Dense object at 0x000001EF6C9700D0>           fc2             True
22        <keras.layers.core.dense.Dense object at 0x000001EF6C97A850>   predictions             True
"""