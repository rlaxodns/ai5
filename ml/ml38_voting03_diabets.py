from sklearn.datasets import load_diabetes
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import BaggingRegressor, VotingRegressor, RandomForestRegressor

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=777,)

#2. 모델
xgb = XGBRegressor()
cat = CatBoostRegressor()
rf = RandomForestRegressor()

model = VotingRegressor(
    estimators=[('XGB',xgb), ('CAT',cat), ('RF', rf)],
    
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 및 예측
result = model.score(x_test, y_test)
print(result)

y_pre = model.predict(x_test)
r2= r2_score(y_test, y_pre)
print(r2)