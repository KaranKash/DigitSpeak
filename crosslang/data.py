import os
import tensorflow as tf
import numpy as np
from load_data import *

DIR = os.path.dirname(os.path.realpath(__file__))

IMAGE_HEIGHT = 100
NUM_CHANNELS = 1

E_NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4891
E_NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2325
E_IMAGE_WIDTH = 23
E_NUM_CLASSES = 11

S_NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 11830
S_NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 5070
S_IMAGE_WIDTH = 40
S_NUM_CLASSES = 11

def english_read_from_csv(filename_queue):
    reader = tf.TextLineReader()
    _, csv_row = reader.read(filename_queue)
    record_defaults = [["0"]] + [[0.0]]*2300
    outlist = tf.decode_csv(csv_row, record_defaults=record_defaults)
    image = tf.pack(outlist[1:])
    label = tf.pack([outlist[0]])
    return image, label

# Graph ops for loading, parsing, and queuing training images
def english_input_graph(training=True, partition='test', batch_size=100):
    with tf.name_scope("input"):
        if training or partition == 'train':
            target = "english_train.csv"
            num_examples_per_epoch = E_NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            target = "english_test.csv"
            num_examples_per_epoch = E_NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

        # Organizing audio data into spectrogram images and labels
        filename_queue = tf.train.string_input_producer([target])
        image, record_label = english_read_from_csv(filename_queue)
        image = tf.cast(tf.reshape(image, [IMAGE_HEIGHT, E_IMAGE_WIDTH, NUM_CHANNELS]),tf.float32)

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

def spanish_read_from_csv(filename_queue):
    reader = tf.TextLineReader()
    _, csv_row = reader.read(filename_queue)
    record_defaults = [["0"]] + [[0.0]]*4000
    outlist = tf.decode_csv(csv_row, record_defaults=record_defaults)
    image = tf.pack(outlist[1:])
    label = tf.pack([outlist[0]])
    return image, label

# Graph ops for loading, parsing, and queuing training images
def spanish_input_graph(training=True, partition='test', batch_size=100):
    with tf.name_scope("input"):
        if training or partition == 'train':
            target = "spanish_train.csv"
            num_examples_per_epoch = S_NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
        else:
            target = "spanish_test.csv"
            num_examples_per_epoch = S_NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

        # Organizing audio data into spectrogram images and labels
        filename_queue = tf.train.string_input_producer([target])
        image, record_label = spanish_read_from_csv(filename_queue)
        image = tf.cast(tf.reshape(image, [IMAGE_HEIGHT, S_IMAGE_WIDTH, NUM_CHANNELS]),tf.float32)

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
