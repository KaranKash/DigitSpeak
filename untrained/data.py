import os
import tensorflow as tf
import numpy as np
from load_data import *

DIR = os.path.dirname(os.path.realpath(__file__))

# For Audio
testdir = '../single_utterances/test'
traindir = '../single_utterances/train'

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4891
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2325
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 23
NUM_CHANNELS = 1
NUM_CLASSES = 11

def read_from_csv(filename_queue):
    reader = tf.TextLineReader()
    _, csv_row = reader.read(filename_queue)
    record_defaults = [["0"]] + [[0.0]]*2300
    outlist = tf.decode_csv(csv_row, record_defaults=record_defaults)
    image = tf.pack(outlist[1:])
    label = tf.pack([outlist[0]])
    return image, label

# Graph ops for loading, parsing, and queuing training images
def input_graph(training=True, partition='test', batch_size=100):
    with tf.name_scope("input"):
        if training or partition == 'train':
            usedir = traindir
            target = "train.csv"
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN

            # # Organizing MNIST data into mnist images and labels
            # train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
            # train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
            #
            # # Extract it into numpy arrays.
            # mnist_data = extract_data(train_data_filename, 60000)
            # mnist_labels = extract_labels(train_labels_filename, 60000)

        elif partition == 'test':
            usedir = testdir
            target = "test.csv"
            num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

            # # Organizing MNIST data into mnist images and labels
            # test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
            # test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')
            #
            # # Extract it into numpy arrays.
            # mnist_data = extract_data(test_data_filename, 10000)
            # mnist_labels = extract_labels(test_labels_filename, 10000)

        if not os.path.isfile(target):
            ld(usedir,target)

        # Organizing audio data into spectrogram images and labels
        filename_queue = tf.train.string_input_producer([target])
        image, record_label = read_from_csv(filename_queue)
        image = tf.cast(tf.reshape(image, [IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS]),tf.float32)

        with tf.name_scope("batching"):
            # Load set of images to start, then continue enqueuing up to capacity
            min_after_dequeue = 1000
            capacity = min_after_dequeue + 3 * batch_size
            kwargs = dict(batch_size=batch_size, capacity=capacity)
            if training:
                batch_fn = tf.train.shuffle_batch
                kwargs["min_after_dequeue"] = min_after_dequeue
            else:
                batch_fn = tf.train.batch
            image_batch, label_batch = batch_fn([image, record_label], **kwargs)

            # The examples and labels for training a single batch
            tf.summary.image("image", image_batch, max_outputs=3)
            return image_batch, label_batch, num_examples_per_epoch
