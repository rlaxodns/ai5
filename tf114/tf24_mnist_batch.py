import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
tf.set_random_seed(333)

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28*28).astype('float32')/255
x_test = x_test.reshape(10000, 28*28).astype('float32')/255
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#############[실습]##############################
x = tf.placeholder(tf.float32, shape=[None, 28*28])
y = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.compat.v1.get_variable('w1', shape=[28*28, 128],
                               initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([128]), name = 'bias1')
layer1 = tf.compat.v1.nn.relu(tf.compat.v1.matmul(x, w1)+b1)


w2 = tf.compat.v1.get_variable('w2', shape = [128, 64],
                            initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([64]), name = 'bias')
layer2 = tf.nn.sigmoid(tf.compat.v1.matmul(layer1, w2)+b2)
layer2 = tf.nn.dropout(layer2, rate = 0.3)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([64, 10]), name = 'weight3')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([10]), name = 'bias')
hypothesis = tf.nn.softmax(tf.compat.v1.matmul(layer2, w3)+b3)

#3-1. 컴파일 
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.nn.log_softmax(hypothesis), axis = 1))
train = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-1).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


############################################################################
#훈련
epochs = 1001
batch_size = 100
total_batch = int(len(x_train) / batch_size)
print(total_batch)

for step in range(epochs):
    avg_cost = 0

    for i in range(total_batch):    
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}

        cost_val, _, w_val, b_val = sess.run([loss, train, w3, b3],
                            feed_dict=feed_dict)
        
        avg_cost += cost_val
    avg_cost /= total_batch

    if step % 10 == 0:
        print(step, 'loss :', avg_cost)
############################################################################


#4. 평가, 예측
print("================================================")
y_predict = sess.run(hypothesis, 
                     feed_dict={x:x_test})
print(y_predict[0], y_predict.shape)  #(10000, 10)

# y_predict = np.argmax(y_predict, 1)
y_predict = sess.run(tf.math.argmax(y_predict, 1))
print(y_predict[0], y_predict.shape)  # 7 (10000,)

y_data = np.argmax(y_test, 1)

# 정확도 계산
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_data)  
print('acc :', acc)     # acc : 1.0

sess.close()