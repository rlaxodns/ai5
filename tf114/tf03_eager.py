import tensorflow as tf

print('tf version :', tf.__version__)
print('eager execute mode :', tf.executing_eagerly())

# tf.compat.v1.disable_eager_execution() # tensorflow 1.x를 사용할 수 있게 된다

# print('eager execute mode :', tf.executing_eagerly())

tf.compat.v1.enable_eager_execution()
print('eager execute mode :', tf.executing_eagerly())


hello = tf.constant('hello world')
sess = tf.compat.v1.Session()
print(sess.run(hello))

# 1.14.0    disable(default)    b'hello world'
# 1.14.0    enable              RuntimeError
# 2.7.4     disable(default)    b'hello world'
# 2.7.4     enable              RuntimeError

# tensor1 : 그래프 연산 모드
# tensor2 : 즉시실행 모드

# tf.compat.v1.disable_eager_execution() eager 모드 켜 / tensorflow 2.x의 디폴트

# tf.compat.v1.enable_eager_execution() eager 모드 꺼 / 그래프 연산 모드 / tensorflow 1.x 코드를 쓸 수 있다

# tf.executing_eagerly() True이면 즉시실행모드, tensorflow 2.x 코드만 써야한다
#                        False이면 그래프 연산 모드, tensorflow 1.x 코드를 사용 가능