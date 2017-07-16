# These two lines of code disables the warning about
# TensorFlow compilation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf


# Defining node a and b
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
# Add two nodes together
adder_node = a + b


# Defining two different set of values for the nodes a and b
# Caution: Do NOT write the dictionary like this:
# dict_A = {'a': 3, 'b': 4.5}
# dict_B = {'a': [1,3], 'b': [2,4]}
# There needs to be no quotation marks for TensorFlow
# to know it is referring to the nodes
dict_A = {a: 3, b: 4.5}
dict_B = {a: [1,3], b: [2,4]}


# Always create a session before evaluating
sess = tf.Session()


# print(sess.run(adder_node, dict_A))
# print(sess.run(adder_node, dict_B))

##########################################################

# Adding another operation to the session
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, dict_A))
print(sess.run(add_and_triple, dict_B))
