# https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')


# 데이터 구성
path = "C:\\ai5\\_data\\kaggle\\santander-customer-transaction-prediction\\"
train = pd.read_csv(path+"train.csv", index_col=0)
test = pd.read_csv(path+"test.csv", index_col=0)
sub = pd.read_csv(path+"sample_submission.csv", index_col = 0)

x = train.drop(['target'], axis = 1)
y = train['target']

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    stratify=y
                                                    )

#2. 모델
from catboost import CatBoostClassifier
bayesian_params = {
    'learning_rate':(0.001, 0.1),
    'max_depth':(3, 10),
    'num_leaves':(24, 40),
    'min_child_samples':(10, 200),
    'min_child_weight':(1, 50),
    'subsample':(0.5, 1),
    'colsample_bytree':(0.5, 1),
    'max_bin':(9, 500),
    'reg_lambda':(-0.001, 10),
    'reg_alpha':(0.01, 50)
}

def xgb_hamsu(learning_rate, max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample,colsample_bytree, max_bin, reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 100,
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        # 'num_leaves': int(round(num_leaves)),
        'min_child_samples':int(round(min_child_samples)),
        # 'min_child_weight':int(round(min_child_weight)),
        # 'subsample':max(min(subsample, 1), 0),
        # 'colsample_bytree':colsample_bytree,
        'max_bin':max(int(round(max_bin)), 10),
        'reg_lambda':max(reg_lambda, 0),
        # 'reg_alpha':reg_alpha,
    }

    model = CatBoostClassifier(**params,
                               task_type='GPU',         # GPU사용 기본은CPU
                                devices='0',
                                early_stopping_rounds=100, #조기종료 기본값 None
                                verbose=10,  )
    model.fit(x_train, y_train,
              eval_set= (x_test, y_test),
            #   eval_metrics = 'logloss',
              verbose = 0)
    
    y_predict = model.predict(x_test)
    result = accuracy_score(y_test, y_predict)
    
    return result

bay = BayesianOptimization(
    f = xgb_hamsu,
    pbounds=bayesian_params,
    random_state=333,
)

n_iter = 100
st = time.time()
bay.maximize(init_points=5,
             n_iter=n_iter)
et = time.time()

print(bay.max)
print(n_iter, "걸린 시간", round(et-st, 2))

"""
{'target': 0.908225, 'params': {'colsample_bytree': 0.9134852531078208, 'learning_rate': 0.1, 'max_bin': 69.39033541377728, 'max_depth': 10.0, 'min_child_samples': 103.8680293831313, 'min_child_weight': 19.02285940946954, 'num_leaves': 40.0, 'reg_alpha': 42.29386755330169, 'reg_lambda': 10.0, 'subsample': 1.0}}
100 걸린 시간 367.63
"""