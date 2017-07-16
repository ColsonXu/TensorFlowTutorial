# These two lines of code disables the warning about
# TensorFlow compilation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # Also tf.float32 implicitly

# print(node1, node2)

##########################################################

# Creates a session to evaluate the nodes
# A session encapsulates the control and state 
# of the TensorFlow runtime
sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2) # Or node1 + node2
# print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))
