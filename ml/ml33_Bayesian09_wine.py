import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, r2_score
from bayes_opt import BayesianOptimization
import time
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_wine(return_X_y=True)
random_state = 777

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=777, stratify=y)

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test  = mms.transform(x_test)


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
{'target': 1.0, 'params': {'colsample_bytree': 0.7716455435126173, 'learning_rate': 0.07316612206200168, 'max_bin': 17.28879000553614, 'max_depth': 5.312371620396405, 'min_child_samples': 80.05714672806884, 'min_child_weight': 3.3668796217250136, 'num_leaves': 25.672483010759535, 'reg_alpha': 4.880901405662105, 'reg_lambda': 2.453278548450057, 'subsample': 0.9367446772294055}}
100 걸린 시간 85.93
"""