import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 데이터 구성
path = "C:\\ai5\\_data\\kaggle\\otto-group-product-classification-challenge\\"
train = pd.read_csv(path + "train.csv", index_col=0)
test = pd.read_csv(path + "test.csv", index_col=0)
sub = pd.read_csv(path + "sampleSubmission.csv", index_col=0, )

le = LabelEncoder()
train["target"] = le.fit_transform(train["target"])

x = train.drop(['target'], axis=1)
y = train['target']


y = pd.get_dummies(y)
print(x.shape, y.shape) #(61878, 93) (61878, 9)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=7777, stratify=y)

####스케일링 적용####
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
mms = MinMaxScaler()
std = StandardScaler()
mas = MaxAbsScaler()
rbs = RobustScaler()

x_train = rbs.fit_transform(x_train)
x_test = rbs.transform(x_test)
test = rbs.transform(test)


#. 모델 구성
from keras.models import load_model
model = load_model("C:\\ai5\\_save\\mcp2\\keras30_13_save_kaggle_otto.hdf5")


# 평가 및 예측
loss = model.evaluate(x_test, y_test)
result = model.predict(test)
y_pre = model.predict(x_test)


print(loss[0], loss[1])

sub[["Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8",
     "Class_9"]] = result
sub.to_csv(path + "sub0011.csv")

# 0.5722733736038208 0.793228805065155
# 0.6055837869644165 0.7788461446762085
# 0.5790331959724426 0.7954912781715393
# 0.5654720664024353 0.7845022678375244