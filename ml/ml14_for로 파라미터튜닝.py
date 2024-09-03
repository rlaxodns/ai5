# 격자탐색_cross_validation

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#1. 데이터
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=777, 
                                                    stratify=y
                                                    )

best_score = 0

for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = SVC(gamma=gamma, C=C)
        model.fit(x_train, y_train)

        score = model.score(x_test, y_test)

        if score > best_score:
            best_score = score
            best_parameter = {'C':C, 'gamma':gamma}

print("최고점수:{:.2f}".format(best_score))
print("최적매개변수: ", best_parameter)