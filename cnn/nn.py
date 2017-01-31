import tensorflow as tf
from utils import *
from functools import reduce

def conv_layer1(name='conv1-layer'):
    def make_layer(input_to_layer):
        # print("Conv1", input_to_layer.get_shape())
        with tf.variable_scope(name, values=[input_to_layer]):
            with tf.name_scope('weights'):
                weights = weight_variable([5, input_to_layer.get_shape()[2], input_to_layer.get_shape()[3], 64], stddev=5e-2)
            with tf.name_scope('biases'):
                bias = bias_variable(64)
            with tf.name_scope('preactivation'):
                conv = tf.nn.conv2d(input_to_layer, weights, strides=[1, 1, 1, 1], padding='VALID')
                preactivation = tf.nn.bias_add(conv, bias)
            with tf.name_scope('output'):
                out = tf.nn.relu(preactivation)
        return out
    return make_layer

def conv_layer2(name='conv2-layer'):
    def make_layer(input_to_layer):
        # print("Conv2", input_to_layer.get_shape())
        with tf.variable_scope(name, values=[input_to_layer]):
            with tf.name_scope('weights'):
                weights = weight_variable([5, input_to_layer.get_shape()[2], input_to_layer.get_shape()[3], 512], stddev=5e-2)
            with tf.name_scope('biases'):
                bias = bias_variable(512)
            with tf.name_scope('preactivation'):
                conv = tf.nn.conv2d(input_to_layer, weights, strides=[1, 1, 1, 1], padding='SAME')
                preactivation = tf.nn.bias_add(conv, bias)
            with tf.name_scope('output'):
                out = tf.nn.relu(preactivation)
        return out
    return make_layer

def max_pool_layer(name='pool1-layer'):
    def make_layer(input_to_layer):
        # print("Max", input_to_layer.get_shape())
        return tf.nn.max_pool(input_to_layer, ksize=[1, 4, 1, 1], strides=[1, 2, 1, 1], padding='VALID', name=name)
    return make_layer

def avg_pool_layer(name='pool1-layer'):
    def make_layer(input_to_layer):
        # print("Mean", input_to_layer.get_shape())
        return tf.nn.avg_pool(input_to_layer, ksize=[1, input_to_layer.get_shape()[1], input_to_layer.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID', name=name)
    return make_layer

def norm_layer(name='norm1-layer'):
    def make_layer(input_to_layer):
        # print("Norm", input_to_layer.get_shape())
        return tf.nn.local_response_normalization(input_to_layer, depth_radius=4, bias=1.0, alpha=0.0001, beta=0.75, name=name)
    return make_layer

def flatten():
    def make_layer(inp):
        # print("Flat", inp.get_shape())
        return tf.reshape(inp, [tf.shape(inp)[0], reduce(lambda x, y: int(x) * int(y), inp.get_shape()[1:], 1)])
    return make_layer

# def fully_connected_layer(size, keep_prob=1.0, name='fc-layer'):
#     def make_layer(input_to_layer):
#         with tf.variable_scope(name, values=[input_to_layer]):
#             with tf.name_scope('weights'):
#                 weights = weight_variable([input_to_layer.get_shape()[1], size], stddev=0.04, wd=0.004)
#             with tf.name_scope('biases'):
#                 bias = bias_variable(size)
#             with tf.name_scope('preactivation'):
#                 preactivation = tf.matmul(input_to_layer, weights) + bias
#                 logits = preactivation
#             with tf.name_scope('output'):
#                 full_output = tf.nn.relu(preactivation)
#                 output = tf.nn.dropout(full_output, keep_prob=keep_prob)
#         return output
#     return make_layer

def softmax_layer(classes, name='softmax-layer'):
    def make_layer(input_to_layer):
        # print("Softmax", input_to_layer.get_shape())
        # with tf.variable_scope(name, values=[input_to_layer]):
        #     with tf.name_scope('weights'):
        #         fanin = input_to_layer.get_shape()[1]
        #         weights = weight_variable([fanin, classes], stddev=1/float(int(fanin)))
        #     with tf.name_scope('biases'):
        #         bias = bias_variable(classes, const=0.0)
        #     with tf.name_scope('preactivation'):
        #         preactivation = tf.matmul(input_to_layer, weights) + bias
        #         logits = preactivation
        #     with tf.name_scope('output'):
        #         with tf.name_scope('probabilities'):
        #             proba = tf.nn.softmax(logits)
        #         with tf.name_scope('predictions'):
        #             prediction = tf.argmax(logits, 1)
        return input_to_layer
    return make_layer

def stack_layers(layers):
    def run_network(inp):
        state = inp
        for layer in layers:
            state = layer(state)
        return state
    return run_network
