# 최적의 가중치, 최소의 로스
# 로컬미니마, 글로벌미니마

# 순전파를 통해 가중치가 저장되고, 역전파로 가중치가 갱신
# optimizer - learning mate, 
# 동일한 훈련비율로 내려가다보면, 최적의 값을 찾지 못하고 핑퐁되는 문제가 발생할 수 있다.
# 그렇기 때문에 훈련하는 비율을 줄인다면 찾을 수 있겠지만, 훈련비용과 시간이 늘어난다.