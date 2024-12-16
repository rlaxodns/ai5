from tensorflow.keras.applications import VGG19, ResNet50, ResNet101, Xception
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, NASNetMobile
from tensorflow.keras.applications import DenseNet121, MobileNetV2, EfficientNetB0

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
import numpy as np
import cv2
from sklearn.metrics import accuracy_score

# 사전 학습된 모델 리스트와 입력 크기 정의
model_list = [
    (Xception(weights='imagenet', include_top=False, input_shape=(71, 71, 3)), (71, 71)),
    (VGG19(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
    (ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
    (ResNet101(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
    (InceptionV3(weights='imagenet', include_top=False, input_shape=(75, 75, 3)), (75, 75)),
    (InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(75, 75, 3)), (75, 75)),
    (DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
    (MobileNetV2(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
    (NASNetMobile(weights='imagenet', include_top=False, input_shape=(224, 224, 3)), (224, 224)),
    (EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 32, 3)), (32, 32)),
]

# CIFAR-10 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 모델 리스트 반복 실행
for base_model, input_size in model_list:
    # 데이터 리사이즈
    x_train_resized = np.array([cv2.resize(x_img, input_size) for x_img in x_train])
    x_test_resized = np.array([cv2.resize(x_img, input_size) for x_img in x_test])

    # 새 Sequential 모델 생성
    model = Sequential()
    model.add(base_model)  # 사전 학습된 모델 추가
    model.add(GlobalAveragePooling2D())  # 글로벌 평균 풀링 레이어 추가
    model.add(Dense(100, activation='softmax'))  # CIFAR-10에 맞는 출력층 추가

    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 모델 학습
    model.fit(x_train_resized, y_train, epochs=1, batch_size=1, validation_split=0.2, verbose=0)

    # 모델 평가 및 예측
    loss, acc = model.evaluate(x_test_resized, y_test, verbose=0)


    # 정확도 출력
    print(f"Model: {base_model.name}, Loss: {loss:.4f}, Accuracy: {acc:.4f}")
