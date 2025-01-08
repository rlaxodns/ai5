import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

sess = tf.Session()

print(sess.run(tf.add(node1, node2)))
print(sess.run(tf.subtract(node1, node2)))
print(sess.run(tf.multiply(node1, node2)))
print(sess.run(tf.divide(node1, node2)))