import tensorflow as tf
from .conv1d import Conv1D
def ResBlock(inputs, DIM, kernel_size, name):
    output = inputs
    output = tf.nn.leaky_relu(output)
    output = Conv1D(name+'.1', DIM, DIM, kernel_size, output)
    output = tf.nn.leaky_relu(output)
    output = Conv1D(name+'.2', DIM, DIM, kernel_size, output)
    return inputs + output
