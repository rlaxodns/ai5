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

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)

# 2. 모델 구성
xgbc = xgb.XGBClassifier()
rf = RandomForestClassifier()
cat = CatBoostClassifier()

# model = xgb.XGBClassifier()
model = VotingClassifier(
    estimators=[('XGB', xgbc), ('RF', rf), ('CAT', cat)],
    voting='hard', # 기본값
    # voting='soft' 
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 및 예측
result = model.score(x_test, y_test)
print('최종점수', result)

y_pre = model.predict(x_test)
acc = accuracy_score(y_test, y_pre)
print('acc', acc)

"""
xgb점수
최종점수 0.9736842105263158
acc 0.9736842105263158

soft
최종점수 0.9824561403508771
acc 0.9824561403508771

hard
최종점수 0.9736842105263158
acc 0.9736842105263158
"""