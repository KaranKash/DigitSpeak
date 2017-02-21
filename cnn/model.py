import os
import tensorflow as tf
from utils import *
from nn import *
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))
SAVED_MODEL_DIR = os.path.join(DIR, "model2")
SAVED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model.ckpt")

# compute custom loss function using tensors and tensor operations
def compute_loss(embeddings, mnist_batch, mismatch_mnist_batch):
    # Linear projection MNIST
    W_m = tf.get_variable("W_m", [mnist_batch.get_shape()[1].value,embeddings.get_shape()[1].value], initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
    b_m = tf.get_variable("b_m", [embeddings.get_shape()[1].value], initializer=tf.constant_initializer(1.0, dtype=tf.float32))
    mnist_batch = tf.add(tf.matmul(mnist_batch,W_m),b_m)
    mismatch_mnist_batch = tf.reshape(mismatch_mnist_batch, [mismatch_mnist_batch.get_shape()[0].value * mismatch_mnist_batch.get_shape()[1].value, -1])
    mismatch_mnist_batch = tf.add(tf.matmul(mismatch_mnist_batch,W_m),b_m)
    mismatch_mnist_batch = tf.reshape(mismatch_mnist_batch, [mnist_batch.get_shape()[0].value, 9, -1])
    mm1,mm2,mm3,mm4,mm5,mm6,mm7,mm8,mm9 = tf.unstack(mismatch_mnist_batch, axis=1)

    # Loss computation
    zero = tf.zeros(embeddings.get_shape()[0].value, tf.float32)
    one = tf.ones(embeddings.get_shape()[0].value, tf.float32)
    base_mul = tf.mul(mnist_batch,embeddings)
    base_dot = tf.reduce_sum(base_mul,1)
    mul1 = tf.mul(mm1,embeddings)
    dot1 = tf.reduce_sum(mul1,1)
    mul2 = tf.mul(mm2,embeddings)
    dot2 = tf.reduce_sum(mul2,1)
    mul3 = tf.mul(mm3,embeddings)
    dot3 = tf.reduce_sum(mul3,1)
    mul4 = tf.mul(mm4,embeddings)
    dot4 = tf.reduce_sum(mul4,1)
    mul5 = tf.mul(mm5,embeddings)
    dot5 = tf.reduce_sum(mul5,1)
    mul6 = tf.mul(mm6,embeddings)
    dot6 = tf.reduce_sum(mul6,1)
    mul7 = tf.mul(mm7,embeddings)
    dot7 = tf.reduce_sum(mul7,1)
    mul8 = tf.mul(mm8,embeddings)
    dot8 = tf.reduce_sum(mul8,1)
    mul9 = tf.mul(mm9,embeddings)
    dot9 = tf.reduce_sum(mul9,1)
    # mul3 = tf.mul(mnist_batch,mismatch_spec_batch)
    # dot3 = tf.reduce_sum(mul3,1)
    max1 = tf.maximum(zero, tf.add_n([dot1, tf.negative(base_dot), one]))
    max2 = tf.maximum(zero, tf.add_n([dot2, tf.negative(base_dot), one]))
    max3 = tf.maximum(zero, tf.add_n([dot3, tf.negative(base_dot), one]))
    max4 = tf.maximum(zero, tf.add_n([dot4, tf.negative(base_dot), one]))
    max5 = tf.maximum(zero, tf.add_n([dot5, tf.negative(base_dot), one]))
    max6 = tf.maximum(zero, tf.add_n([dot6, tf.negative(base_dot), one]))
    max7 = tf.maximum(zero, tf.add_n([dot7, tf.negative(base_dot), one]))
    max8 = tf.maximum(zero, tf.add_n([dot8, tf.negative(base_dot), one]))
    max9 = tf.maximum(zero, tf.add_n([dot9, tf.negative(base_dot), one]))
    # max2 = tf.maximum(zero, tf.add_n([dot3, tf.negative(dot2), one]))
    loss = tf.reduce_sum(tf.add_n([max1,max2,max3,max4,max5,max6,max7,max8,max9]))
    return loss, mnist_batch, mismatch_mnist_batch

def forward_propagation(images, mnist_batch, mismatch_mnist_batch, dropout=False, train=False):
    network = stack_layers([
        conv_layer(name='conv1-layer'),
        pool_layer(name="max-pool1-layer"),
        flatten(),
        fully_connected_layer(1024, keep_prob=0.5 if train and dropout else 1.0, name="local1-layer"),
        fully_connected_layer(512, keep_prob=1.0, name="local2-layer"),
        softmax_layer(11)

        # conv_layer(5, 64, name='conv1-layer', padding='VALID'),
        # max_pool_layer(name="max-pool1-layer"),
        # # norm_layer(name="norm1-layer"),
        # conv_layer(5, 512, name='conv2-layer', padding='SAME'),
        # # norm_layer(name="norm2-layer"),
        # max_pool_layer(name="max-pool2-layer"),
        # # conv_layer(5, 1024, name='conv3-layer', padding='SAME'),
        # avg_pool_layer(name="avg-pool-layer"),
        # flatten(),
        # # fully_connected_layer(384, name="local1-layer"),
        # # fully_connected_layer(192, keep_prob=0.5 if train and dropout else 1.0, name="local2-layer"),
        # softmax_layer(11)
    ])
    embeddings = network(images)
    # embeddings = tf.nn.l2_normalize(embeddings,0)

    with tf.name_scope('loss'):
        print("Starting training loss calculation...")
        # mismatch_spec_batch = tf.nn.embedding_lookup(embeddings, indices)
        batch_loss, mnist, mismatch = compute_loss(embeddings, mnist_batch, mismatch_mnist_batch)

    return embeddings, batch_loss, mnist, mismatch
