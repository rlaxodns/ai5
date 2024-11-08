import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression  # 로지스틱 리그레션은 분류
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

#1. 데이터 
x, y = load_breast_cancer(return_X_y = True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 3333, 
                                                    stratify = y)

mms = MinMaxScaler()
x_train = mms.fit_transform(x_train)
x_test = mms.transform(x_test)

#2. 모델
# model = DecisionTreeClassifier()
# model = BaggingClassifier(LogisticRegression(),
#     n_estimators = 100,
#     n_jobs = 10,
#     random_state= 3333,
#     bootstrap = True, #디폴트 중복허용
#     # bootstrap = False # 중복불가
# )

model = BaggingClassifier(RandomForestClassifier(),
                          n_estimators = 100,
                          random_state = 333,
                          bootstrap = False
                          )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 및 예측
result = model.score(x_test, y_test)
print("최종 점수:", result)

y_pre = model.predict(x_test)
acc = accuracy_score(y_test, y_pre)
print("acc", acc)

#디시젼트리 배깅후
# 최종 점수: 0.9210526315789473
# acc 0.9210526315789473

# 로지스틱리그레션 배깅
# 최종 점수: 0.9912280701754386
# acc 0.9912280701754386