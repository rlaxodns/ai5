# train['종가'] = train['종가'].str.replace(',', '')
# https://www.kaggle.com/c/playground-series-s4e1/data
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from keras.callbacks import EarlyStopping

# 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\train.csv", index_col=[0, 1, 2])
test = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\test.csv", index_col= [0, 1, 2])
submission = pd.read_csv("C:\\ai5\\_data\\kaggle\\bank\\sample_submission.csv", index_col=[0])

print(train.dtypes)
"""Surname             object
CreditScore          int64
Geography           object
Gender              object
Age                float64
Tenure               int64
Balance            float64
NumOfProducts        int64
HasCrCard          float64
IsActiveMember     float64
EstimatedSalary    float64
Exited               int64"""
# France = 1, Spain = 2, Germany = 3 // Male = 1, Female =2

# train['Geography'] = train['Geography'].str.replace('France', '1')
# train['Geography'] = train['Geography'].str.replace('Spain', '2')
# train['Geography'] = train['Geography'].str.replace('Germany', '3')

# train['Gender'] = train['Gender'].str.replace('Male', '1')
# train['Gender'] = train['Gender'].str.replace('Female', '2')

# train = train.astype(float)
# print(train.dtypes)

# test['Geography'] = test['Geography'].str.replace('France', '1')
# test['Geography'] = test['Geography'].str.replace('Spain', '2')
# test['Geography'] = test['Geography'].str.replace('Germany', '3')

# test['Gender'] = test['Gender'].str.replace('Male', '1')
# test['Gender'] = test['Gender'].str.replace('Female', '2')
# test = test.astype(float)
# print(test.dtypes)

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# train['Geography'] = le.fit_transform(train['Geography'])
# train['Gender'] = le.fit_transform(train['Gender'])
# test['Geography'] = le.fit_transform(test['Geography'])
# test['Gender'] = le.fit_transform(test['Gender'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train['Geography'] = le.fit_transform(train['Geography'])
train['Gender'] = le.fit_transform(train['Gender'])

test['Geography'] = le.fit_transform(test['Geography'])
test['Gender'] = le.fit_transform(test['Gender'])

x = train.drop(['Exited'], axis = 1)
y = train['Exited']

print(x.shape, y.shape) #(165034, 10) (165034,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, 
                                                    random_state= 6666)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)

x_train = x_train.reshape(
    x_train.shape[0], 
    x_train.shape[1],
    1
)
x_test = x_test.reshape(
    x_test.shape[0], 
    x_test.shape[1],
    1
)

print(x_train.shape, x_test.shape) #

#2. 모델 구성
from keras.layers import Dropout
from keras.layers import LSTM, Conv1D, Flatten
model = Sequential()

model.add(Conv1D(64,2, input_shape=(x_train.shape[1], 1)))
model.add(Conv1D(32,2))
model.add(Flatten())
model.add(Dense(500, input_dim = 10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dense(2000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1000, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일 및 훈련
from keras.callbacks import ModelCheckpoint
mcp = ModelCheckpoint(
    monitor = 'val_loss', 
    mode = 'auto', 
    patience = 20, 
    verbose=1, 
    save_best_only=True, 
    filepath=".//_save//keras32//keras32_dropout08_save_kaggle_banks.hdf5"
)

es = EarlyStopping(
    monitor='val_loss',
    mode = 'min',
    patience=20,
    restore_best_weights=True
)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
model.fit(x_train, y_train, epochs = 100, batch_size=1024,
          validation_split=0.3, callbacks=[es, mcp])
# validation은 훈련에 있어서 미미하게 영향을 미친다

#4. 평가 및 훈련
loss = model.evaluate(x_test, y_test)
y_pre = np.round(model.predict([x_test]))
result = np.round(model.predict([test]))

acc = accuracy_score(y_test, y_pre)

print(loss, acc)

submission['Exited'] = result
submission.to_csv("C:\\ai5\\_data\\kaggle\\bank\\sample_submission_153311.csv")

# [0.3274446725845337, 0.8632714152336121] 0.8632714272730027
# [0.32849907875061035, 0.8626654744148254] 0.8626654951979883
# [0.3284316956996918, 0.8633320331573486] 0.8633320204805042