import tensorflow as tf

print(tf.__version__)

# pip install protobuf==3.20
# pip install numpy==1.16

print('hello world')

#-----------------------------
hello = tf.constant('hello world')
print(hello) # tensor machine -> hello의 그래프 연산의 상태가 출력된다
#-----------------------------
sess = tf.Session()

print(sess.run(hello)) # 그래프의 실행 결과가 출력된다 : b'hello world'
#-----------------------------