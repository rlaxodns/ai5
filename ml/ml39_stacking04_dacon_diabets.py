# https://dacon.io/competitions/official/236068/overview/description

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier,VotingClassifier,RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
from catboost import CatBoostClassifier, CatBoostRegressor
import time
import warnings
warnings.filterwarnings('ignore')


#1. 데이터 구성
train = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\train.csv", index_col = 0)
test = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\test.csv", index_col=0)
submission = pd.read_csv("C:\\ai5\\_data\\dacon\\diabetes\\sample_submission.csv", index_col = 0)


x = train.drop(['Outcome'], axis=1)
y = train['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=777, 
                                                    stratify=y)

std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

#2. 모델
xgbc = XGBRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

train_list = []
test_list = []
models = [xgbc, rf, cat]

for model in models:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)

    train_list.append(y_predict)
    test_list.append(y_test_predict)

    score = r2_score(y_test, y_test_predict)
    class_name = model.__class__.__name__
    print('{0}acc: {1:.4f}'.format(class_name, score))

x_train_new = np.array(train_list).T
print(x_train_new.shape)   # (455, 3)

x_test_new = np.array(test_list).T
print(x_test_new.shape) # (114, 3)

# 2-1. 모델
model2 = CatBoostClassifier(verbose=0)
model2.fit(x_train_new, y_train)
y_pred = model2.predict(x_test_new)
score2 = r2_score(y_test, y_pred)
print("스태킹 결과", score2)