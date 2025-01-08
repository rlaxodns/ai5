import tensorflow as tf
tf.compat.v1.set_random_seed(777)
import numpy as np

#1. 데이터
x_data = [[0,0],
          [0,1],
          [1,0],
          [1,1]]  # (4, 2)

y_data = [[0], [1], [1], [0]]

x = tf.compat.v1.placeholder(tf.float32, shape = [None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1])

w = tf.compat.v1.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.compat.v1.Variable(tf.zeros([1]), name='bias')

# 실습
hypothesis = tf.nn.sigmoid(tf.matmul(x, w) + b)


# 3-1. 컴파일
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(hypothesis), axis = 1))
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=1e-1).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

# 3-3. 훈련
epochs = 2001
for step in range(epochs):
    cost_val, _, w_val, b_val = sess.run([loss, train, w, b], feed_dict={x:x_data, y:y_data})
    if step % 20 == 0:
        print(step, 'loss :', cost_val)

# 4. 평가, 예측
x_test = x_data

y_pred = sess.run(hypothesis, feed_dict={x: x_test})
y_predict = sess.run(tf.cast(y_pred > 0.5, dtype=tf.float32))

print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_data, y_predict)
print('acc : ', acc) 