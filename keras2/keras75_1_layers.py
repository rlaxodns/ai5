import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16

vgg16 = VGG16(weights='imagenet', include_top=False, 
              input_shape=(32, 32, 3))
# 13개의 conv layer에 2개의 Dense layer 에 따라서 총 15개의 가중치와
# 15개의 편향을 더해서 총 30개의 print(len(model.weights)) # 30

vgg16.trainable = False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10, activation='softmax'))

# model.trainable = False
model.summary()

print(len(model.weights)) # 30
print(len(model.trainable_weights)) # 4

"""
                            Trainable: True // model=False // vgg16 = False
len(model.weight)                  30       //     30      //      30
len(model.trainable_weight)        30       //      0      //       4

"""