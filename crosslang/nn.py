import tensorflow as tf
from utils import *
from functools import reduce

def conv_layer(height, width, channels, name='conv1-layer', reuse=False):
    def make_layer(input_to_layer):
        with tf.variable_scope(name, values=[input_to_layer], reuse=reuse):
            # weights = weight_variable([height, width, input_to_layer.get_shape()[3], channels])
            weights = tf.get_variable("weights", [height, width, input_to_layer.get_shape()[3], channels], initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            # bias = bias_variable(channels,const=0.0)
            bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        conv = tf.nn.conv2d(input_to_layer, weights, [1, 1, 1, 1], padding='SAME')
        preactivation = tf.nn.bias_add(conv, bias)
        out = tf.nn.relu(preactivation)
        return out
    return make_layer

def pool_layer(height, width, vstride, hstride, name='pool1-layer'):
    def make_layer(input_to_layer):
        return tf.nn.max_pool(input_to_layer, ksize=[1, height, width, 1], strides=[1, vstride, hstride, 1], padding='SAME', name=name)
    return make_layer

def norm_layer(name='norm1-layer'):
    def make_layer(input_to_layer):
        return tf.nn.local_response_normalization(input_to_layer, depth_radius=5, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)
    return make_layer

def flatten():
    def make_layer(inp):
        return tf.reshape(inp, [tf.shape(inp)[0], reduce(lambda x, y: int(x) * int(y), inp.get_shape()[1:], 1)])
    return make_layer

def fully_connected_layer(size, keep_prob=1.0, name='fc-layer',reuse=False):
    def make_layer(input_to_layer):
        with tf.variable_scope(name, values=[input_to_layer],reuse=reuse):
            # weights = weight_variable([input_to_layer.get_shape()[1], size])
            weights = tf.get_variable("weights", [input_to_layer.get_shape()[1], size], initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
            # bias = bias_variable(size)
            bias = tf.get_variable("bias", [size], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        preactivation = tf.matmul(input_to_layer, weights) + bias
        full_output = tf.nn.relu(preactivation)
        output = tf.nn.dropout(full_output, keep_prob=keep_prob)
        return output
    return make_layer

def softmax_layer(classes, name='softmax-layer'):
    def make_layer(input_to_layer):
        # with tf.variable_scope(name, values=[input_to_layer]):
        #     with tf.name_scope('weights'):
        #         fanin = input_to_layer.get_shape()[1]
        #         weights = weight_variable([fanin, classes])
        #     with tf.name_scope('biases'):
        #         bias = bias_variable(classes)
        #     with tf.name_scope('preactivation'):
        #         preactivation = tf.matmul(input_to_layer, weights) + bias
        #         logits = preactivation
        #     with tf.name_scope('output'):
        #         with tf.name_scope('probabilities'):
        #             proba = tf.nn.softmax(logits)
        #         with tf.name_scope('predictions'):
        #             prediction = tf.argmax(proba, 1)
        #             prediction = tf.to_int32(prediction, name='ToInt32')
        return input_to_layer
    return make_layer

def stack_layers(layers):
    def run_network(inp):
        state = inp
        for layer in layers:
            state = layer(state)
        return state
    return run_network
