

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.svm import SVR
import xgboost as xgb
import time

#1. 데이터
path = "C:/Users/kim/Downloads/bike-sharing-demand/"  

train_csv = pd.read_csv(path+"train.csv", index_col=0)  # 첫 번쨰 데이터는 시간 데이터이기 때문에 분할의 필요하여 복잡하니 우선은 인덱스로 잡기
test_csv = pd.read_csv(path+"test.csv", index_col=0)
sampleSubmission_csv = pd.read_csv(path+"sampleSubmission.csv", index_col=0)

## x와 y를 분리
x = train_csv.drop(['casual', 'registered','count'], axis=1) 
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, 
                                                    random_state=777)

kfold = KFold(
    n_splits=5, 
    shuffle=True,
    random_state=777
)
parameter = [
    {"n_jobs":[-1], "n_estimators":[100, 500], 'max_depth':[6,10,12], "min_samples_leaf": [3, 10]}, 
    {"n_jobs":[-1], 'max_depth':[6, 8, 10,12], "min_samples_leaf": [3, 5, 7, 10]},
    {"n_jobs":[-1], "min_samples_leaf": [3, 5, 7, 10], 'min_samples_split':[2, 3, 5, 10]},
    {"n_jobs":[-1], 'min_samples_split':[2, 3, 5, 10]}
    ]

#2. 모델
model = RandomizedSearchCV(xgb.XGBRegressor(),
                    parameter, 
                    cv=kfold, 
                    refit = True,
                    verbose=1,
                    n_jobs=-1,  # cpu의 모든 코어를 활용함
                    n_iter= 10,  # random_search의 횟수 조절
                    random_state=777
                    )

st = time.time()
model.fit(x_train, y_train)
et = time.time()

print('최적의 매개변수', model.best_estimator_) 
# 최적의 매개변수 XGBRegressor(base_score=None, booster=None, callbacks=None,
            #  colsample_bylevel=None, colsample_bynode=None,
            #  colsample_bytree=None, device=None, early_stopping_rounds=None,  
            #  enable_categorical=False, eval_metric=None, feature_types=None,  
            #  gamma=None, grow_policy=None, importance_type=None,
            #  interaction_constraints=None, learning_rate=None, max_bin=None,  
            #  max_cat_threshold=None, max_cat_to_onehot=None,
            #  max_delta_step=None, max_depth=6, max_leaves=None,
            #  min_child_weight=None, min_samples_leaf=3, missing=nan,
            #  monotone_constraints=None, multi_strategy=None, n_estimators=100,      

print('최적의 파라미터', model.best_params_)
# {'max_depth': 6, 'min_samples_leaf': 3, 'n_estimators': 100, 'n_jobs': -1}

print('best_score', model.best_score_) # train을 통한 최고의 점수를 의미하기 때문에 과적합된 점수
# best_score 0.3141384555972085   

print('model.score', model.score(x_test, y_test))
# model.score 0.2969055227670738

y_pre = model.predict(x_test)
print("acc", r2_score(y_test, y_pre))
# acc 0.2969055227670738

y_pre_best = model.best_estimator_.predict(x_test)
print("최적의 튜닝acc", r2_score(y_test, y_pre_best) )
# 최적의 튜닝acc 0.2969055227670738

print("걸린 시간", round(et-st, 2))
# 걸린 시간 10.67

import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)
print(pd.DataFrame(model.cv_results_).sort_values("rank_test_score", ascending=True))
print(pd.DataFrame(model.cv_results_).columns)

##csv파일로 만들기##
path = "C:\\ai5\\study\\_save\\ml15_GS_CV_01\\"
pd.DataFrame(model.cv_results_).sort_values("rank_test_score", ascending=True) \
    .to_csv(path+'ml16_GridSearchCV_03_XGB_kaggle_bike.csv')
