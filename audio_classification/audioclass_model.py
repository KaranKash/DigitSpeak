import os
import tensorflow as tf
from utils import *
from nn import *
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))
SAVED_MODEL_DIR = os.path.join(DIR, "model")
SAVED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model.ckpt")

def forward_propagation(images, labels, dropout=False, train=False):
    network = stack_layers([
        conv_layer(name='conv1-layer'),
        pool_layer(name="max-pool1-layer"),
        # norm_layer(name="norm1-layer"),
        # conv_layer(19, 128, name='conv2-layer', padding='SAME'),
        # # norm_layer(name="norm2-layer"),
        # max_pool_layer(name="max-pool2-layer"),
        # # conv_layer(5, 1024, name='conv3-layer', padding='SAME'),
        # avg_pool_layer(name="avg-pool-layer"),
        flatten(),
        fully_connected_layer(1024, keep_prob=0.5 if train and dropout else 1.0, name="local1-layer"),
        # fully_connected_layer(1024, keep_prob=0.5 if train and dropout else 1.0, name="local2-layer"),
        softmax_layer(11)
    ])
    logits, proba, prediction = network(images)

    with tf.name_scope('accuracy'):
        with tf.name_scope('accuracy'):
            actual = labels
            with tf.name_scope('num_correct'):
                correct = tf.reduce_sum(tf.to_int32(tf.equal(prediction, actual)))

    with tf.name_scope('loss'):
        labels_one_hot = tf.one_hot(labels, 11, on_value=1.0, off_value=0.0)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels_one_hot))

    return correct, loss, proba
