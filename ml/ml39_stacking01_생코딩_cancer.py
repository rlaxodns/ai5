# Pseudo Labelling 기법: 모델을 돌려서 나온 결과로 결측치를 찾기
# 스태킹: 모델을 몯려 나온거를 컬럼을 구성해서 새로운 데이터 셋을 만들어서 데이터 만듦
        #  한 데이터로 여러 모델을 돌려서 돌리는 족족 컬럼 만들기


import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression  # 로지스틱 리그레션은 분류
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# 배깅은 한 가지 모델 내에서 파라미터를 튜닝하면서 최적을 선택,
# 보팅은 다섯 가지 모델 내에서 파라미터를 튜닝하면서 최적을 선택

#1. 데이터 
x, y = load_breast_cancer(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 3333, 
                                                    stratify = y)

print(x_train.shape, y_train.shape) # (455, 30) (455,)
print(x_test.shape, y_test.shape)   # (114, 30) (114,)

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)

#2. 모델
xgbc = xgb.XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier(verbose=0)

train_list = []
test_list = []
models = [xgbc, rf, cat]

for model in models:
    model.fit(x_train, y_train)
    y_predict = model.predict(x_train)
    y_test_predict = model.predict(x_test)

    train_list.append(y_predict)
    test_list.append(y_test_predict)

    score = accuracy_score(y_test, y_test_predict)
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
score2 = accuracy_score(y_test, y_pred)
print("스태킹 결과", score2)