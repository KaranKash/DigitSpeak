import os
import tensorflow as tf
from utils import *
from nn import *
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))
SAVED_MODEL_DIR = os.path.join(DIR, "model")
SAVED_MODEL_PATH = os.path.join(SAVED_MODEL_DIR, "model.ckpt")

# shared image network
image_network = stack_layers([
    conv_layer(5, 5, 32, name='image-conv1-layer'),
    pool_layer(2,2,2,2,name="image-max-pool1-layer"),
    conv_layer(5, 5, 64, name='image-conv2-layer'),
    pool_layer(2,2,2,2,name="image-max-pool2-layer"),
    flatten(),
    fully_connected_layer(512, keep_prob=0.5, name="image-local1-layer"),
    fully_connected_layer(512, keep_prob=1.0, name="image-local2-layer"),
    softmax_layer(10)
])

image_network2 = stack_layers([
    conv_layer(5, 5, 32, name='image-conv1-layer',reuse=True),
    pool_layer(2,2,2,2,name="image-max-pool1-layer"),
    conv_layer(5, 5, 64, name='image-conv2-layer',reuse=True),
    pool_layer(2,2,2,2,name="image-max-pool2-layer"),
    flatten(),
    fully_connected_layer(512, keep_prob=0.5, name="image-local1-layer",reuse=True),
    fully_connected_layer(512, keep_prob=1.0, name="image-local2-layer",reuse=True),
    softmax_layer(10)
])

# compute custom loss function using tensors and tensor operations
def compute_loss(embeddings, mnist_batch, mismatch_mnist_batch):
    # Loss computation
    mm1,mm2,mm3,mm4,mm5,mm6,mm7,mm8,mm9 = tf.unstack(mismatch_mnist_batch, axis=1)
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

def english_forward_propagation(images, mnist_batch, mismatch_mnist_batch, dropout=False, train=False):
    english_audio_network = stack_layers([
        conv_layer(5, 23, 64, name='eaudio-conv1-layer'),
        pool_layer(3,4,1,2,name="eaudio-max-pool1-layer"),
        flatten(),
        fully_connected_layer(1024, keep_prob=0.5 if train and dropout else 1.0, name="eaudio-local1-layer"),
        fully_connected_layer(512, keep_prob=1.0, name="eaudio-local2-layer"),
        softmax_layer(11)
    ])

    embeddings = english_audio_network(images)
    mnist_len = mnist_batch.get_shape()[0].value
    mismatch_mnist_batch = tf.reshape(mismatch_mnist_batch, [mismatch_mnist_batch.get_shape()[0].value * mismatch_mnist_batch.get_shape()[1].value, 28, 28, 1])
    mismatch_len = mismatch_mnist_batch.get_shape()[0].value
    joint = tf.unstack(mnist_batch) + tf.unstack(mismatch_mnist_batch)
    feedin = tf.stack(joint)
    output = image_network(feedin)
    mnist_batch = tf.slice(output, [0,0], [mnist_len, output.get_shape()[1].value])
    mismatch_mnist_batch = tf.slice(output, [mnist_len,0], [mismatch_len, output.get_shape()[1].value])
    mismatch_mnist_batch = tf.reshape(mismatch_mnist_batch, [mnist_batch.get_shape()[0].value, 9, -1])

    with tf.name_scope('loss'):
        print("Starting training loss calculation...")
        # mismatch_spec_batch = tf.nn.embedding_lookup(embeddings, indices)
        batch_loss, mnist, mismatch = compute_loss(embeddings, mnist_batch, mismatch_mnist_batch)

    return embeddings, batch_loss, mnist, mismatch

def spanish_forward_propagation(images, mnist_batch, mismatch_mnist_batch, dropout=False, train=False):
    spanish_audio_network = stack_layers([
        conv_layer(5, 40, 64, name='saudio-conv1-layer'),
        pool_layer(3,4,1,2,name="saudio-max-pool1-layer"),
        flatten(),
        fully_connected_layer(1024, keep_prob=0.5 if train and dropout else 1.0, name="saudio-local1-layer"),
        fully_connected_layer(512, keep_prob=1.0, name="saudio-local2-layer"),
        softmax_layer(10)
    ])

    embeddings = spanish_audio_network(images)
    mnist_len = mnist_batch.get_shape()[0].value
    mismatch_mnist_batch = tf.reshape(mismatch_mnist_batch, [mismatch_mnist_batch.get_shape()[0].value * mismatch_mnist_batch.get_shape()[1].value, 28, 28, 1])
    mismatch_len = mismatch_mnist_batch.get_shape()[0].value
    joint = tf.unstack(mnist_batch) + tf.unstack(mismatch_mnist_batch)
    feedin = tf.stack(joint)
    output = image_network2(feedin)
    mnist_batch = tf.slice(output, [0,0], [mnist_len, output.get_shape()[1].value])
    mismatch_mnist_batch = tf.slice(output, [mnist_len,0], [mismatch_len, output.get_shape()[1].value])
    mismatch_mnist_batch = tf.reshape(mismatch_mnist_batch, [mnist_batch.get_shape()[0].value, 9, -1])

    with tf.name_scope('loss'):
        print("Starting training loss calculation...")
        # mismatch_spec_batch = tf.nn.embedding_lookup(embeddings, indices)
        batch_loss, mnist, mismatch = compute_loss(embeddings, mnist_batch, mismatch_mnist_batch)

    return embeddings, batch_loss, mnist, mismatch
