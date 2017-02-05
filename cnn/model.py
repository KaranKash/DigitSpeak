import os
import tensorflow as tf
from utils import *
from nn import *
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))
SAVED_MODEL_DIR = os.path.join(DIR, "model")
SAVED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model.ckpt")

# compute custom loss function using tensors and tensor operations
def compute_loss(embeddings, mnist_batch, mismatch_mnist_batch, mismatch_spec_batch):
    # Linear projection MNIST
    # W_m = tf.Variable(tf.truncated_normal([512,512]), name="W_m")
    # b_m = tf.Variable(tf.truncated_normal([512]), name="b_m")
    # mnist_batch = tf.add(tf.matmul(mnist_batch,W_m),b_m)
    # mismatch_mnist_batch = tf.add(tf.matmul(mismatch_mnist_batch,W_m),b_m)

    # Loss computation
    zero = tf.zeros(embeddings.get_shape()[0].value, tf.float32)
    one = tf.ones(embeddings.get_shape()[0].value, tf.float32)
    mul1 = tf.mul(mismatch_mnist_batch,embeddings)
    dot1 = tf.reduce_sum(mul1,1)
    mul2 = tf.mul(mnist_batch,embeddings)
    dot2 = tf.reduce_sum(mul2,1)
    mul3 = tf.mul(mnist_batch,mismatch_spec_batch)
    dot3 = tf.reduce_sum(mul3,1)
    max1 = tf.maximum(zero, tf.add_n([dot1, tf.negative(dot2), one]))
    max2 = tf.maximum(zero, tf.add_n([dot3, tf.negative(dot2), one]))
    loss = tf.reduce_sum(tf.add(max1,max2))
    return loss

def forward_propagation(images, mnist_batch, mismatch_mnist_batch, indices, train=False):
    network = stack_layers([
        conv_layer(5, 64, name='conv1-layer', padding='VALID'),
        max_pool_layer(name="max-pool1-layer"),
        # norm_layer(name="norm1-layer"),
        conv_layer(5, 512, name='conv2-layer', padding='SAME'),
        # norm_layer(name="norm2-layer"),
        max_pool_layer(name="max-pool2-layer"),
        # conv_layer(5, 1024, name='conv3-layer', padding='SAME'),
        avg_pool_layer(name="avg-pool-layer"),
        flatten(),
        # fully_connected_layer(384, name="local1-layer"),
        # fully_connected_layer(192, keep_prob=0.5 if train and dropout else 1.0, name="local2-layer"),
        softmax_layer(11)
    ])
    embeddings = network(images)
    # embeddings = tf.nn.l2_normalize(embeddings,0)

    with tf.name_scope('loss'):
        total_loss = tf.constant(0)
        if train:
            print("Starting training loss calculation...")
            mismatch_spec_batch = tf.nn.embedding_lookup(embeddings, indices)
            batch_loss = compute_loss(embeddings, mnist_batch, mismatch_mnist_batch, mismatch_spec_batch)
            # tf.summary.scalar('batch loss', batch_loss)
            tf.add_to_collection('losses', batch_loss)
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return embeddings, total_loss
