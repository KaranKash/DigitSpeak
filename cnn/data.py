import os
import tensorflow as tf
import numpy as np
from load_data import *

DIR = os.path.dirname(os.path.realpath(__file__))

testdir = '../single_utterances/test'
traindir = '../single_utterances/train'

# Graph ops for loading, parsing, and queuing training images
def input_graph(training=True, partition='test', batch_size=100):
    with tf.name_scope("input"):
        if training or partition == 'train':
            # The training image files:
            # filenames = [os.path.join(DIR, "cifar-10-batches-bin", "data_batch_%d.bin" % i) for i in range(1,6)]
            usedir = traindir
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        elif partition == 'test':
            # filenames = [os.path.join(DIR, "cifar-10-batches-bin", 'test_batch.bin')]
            usedir = testdir
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

        X, Y = ld(usedir)

        # Reshape input vectors for one input channel
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2],NUM_CHANNELS)
        X = tf.Variable(X)

        # Reshape labels to (num_data,11,)
        Y = Y.reshape(Y.shape[0],Y.shape[1],)
        Y = tf.Variable(Y)

        image = X
        record_label = Y

        with tf.name_scope("batching"):
            # Load set of images to start, then continue enqueuing up to capacity
            min_after_dequeue = int(num_examples_per_epoch * 0.8)
            capacity = min_after_dequeue + 20 * batch_size
            kwargs = dict(batch_size=batch_size, capacity=capacity, enqueue_many=True)
            if training:
                batch_fn = tf.train.shuffle_batch
                kwargs["min_after_dequeue"] = min_after_dequeue
            else:
                batch_fn = tf.train.batch
            image_batch, label_batch = batch_fn([image, record_label], **kwargs)

            # The examples and labels for training a single batch
            tf.summary.image("image", image_batch, max_outputs=3)
            # labels = tf.squeeze(label_batch, axis=1)
            return image_batch, label_batch, num_examples_per_epoch
