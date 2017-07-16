# These two lines of code disables the warning about
# TensorFlow compilation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


a = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = a * x + b


sess = tf.Session()


# Variables are not initialized when they are created like constants
# To initialize variables, use the following code.
init = tf.global_variables_initializer()
sess.run(init)


# Data set for x
x_values = {x: [1,2,3,4]}

# print(sess.run(linear_model, x_values))

##########################################################

# Loss function calculating how fit the model is (Squared Regression)
goal = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - goal)
loss = tf.reduce_sum(squared_deltas)
# print('loss:', sess.run(loss, {x: [1,2,3,4], goal: [0,-1,-2,-3]}))

##########################################################

# Reasign a and b to reduce the loss
# tf.asign() is used to reasign data to variables
fixA = tf.assign(a, [-1.])
fixB = tf.assign(b, [1.])
sess.run([fixA, fixB])
# print('loss:', sess.run(loss, {x:[1,2,3,4], goal:[0,-1,-2,-3]}))

##########################################################

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Assign a and b with old, imperfect values
sess.run(init)

# Start training the model
for i in range(1000):
	sess.run(train, {x:[1,2,3,4], goal:[0,-1,-2,-3]})

# Evaluate(display) a and b
print(sess.run([a, b]))
print('loss:', sess.run(loss, {x:[1,2,3,4], goal:[0,-1,-2,-3]}))
