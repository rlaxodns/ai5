import tensorflow as tf

print(tf.__version__)
print(tf.executing_eagerly())

tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly())

# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0, tf.float32)

# node3 = tf.add(node1, node2)

a = tf.compat.v1.placeholder(tf.float32)
b = tf.compat.v1.placeholder(tf.float32)

add_node = a + b

sess = tf.compat.v1.Session()

print(sess.run(add_node, feed_dict={a:3.0, b:4.0}))
print(sess.run(add_node, feed_dict={a:30.0, b:4.5}))

add_and_triple = add_node * 3

print(add_and_triple)

print(sess.run(add_and_triple, feed_dict={a:3, b:4}))