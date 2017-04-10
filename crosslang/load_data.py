import os
import argparse
import gzip
import sys
import time
import numpy as np
from multiprocessing import Pool
from contextlib import closing
import csv
import tensorflow as tf
from six.moves import urllib

e_bin_freq = 23
e_spect_width = e_bin_freq  # Don't add one pixel of zeros on either side of the image
window_size = 100
e_dim_Y = 11

s_bin_freq = 40
s_spect_width = s_bin_freq
s_dim_Y = 10

# For MNIST
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

# MNIST embeddings directory
dataMnistdir = '../mnistAct/'
MNIST_DIM = 512

WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255

# mnist_data = {digit: np.loadtxt(dataMnistdir + str(digit) + ".txt") for digit in range(10)}

label_map = {'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'o':0,'z':0,'0':0}

def maybe_download(filename):
    """Download the data from Yann's website, unless it's already here."""
    if not tf.gfile.Exists(WORK_DIRECTORY):
        tf.gfile.MakeDirs(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not tf.gfile.Exists(filepath):
        filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
        with tf.gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath

def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
        return data

def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


# Organizing MNIST data into mnist images and labels
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

# Extract it into numpy arrays.
mnist_values = extract_data(test_data_filename, 10000)
mnist_labels = extract_labels(test_labels_filename, 10000)

mnist_data = {i:[] for i in range(10)}
for i in range(len(mnist_labels)):
    lab = mnist_labels[i]
    val = mnist_values[i]
    mnist_data[lab].append(val)

def get_mnist_embedding(label):
    digit = label[1]
    data = mnist_data[digit]
    i = np.random.randint(0,len(data))
    return label[0],data[i]

def get_label_map(s):
    return label_map[s]

def get_mismatch_mnist_embedding(label):
    i = label[1]
    out = None
    for j in range(10):
        if i != j:
            _, data = get_mnist_embedding((label[0],j))
            if out is None:
                out = data
            else:
                out = np.vstack([out, data])
    return label[0],out.flatten()

def generate_mnist_set(labels):
    matches = []
    mismatches = []
    labels = [(i, label_map[labels[i].decode('utf-8')]) for i in range(len(labels))]
    with closing(Pool()) as pool:
        matches = pool.map(get_mnist_embedding, labels)
        mismatches = pool.map(get_mismatch_mnist_embedding, labels)
    matches = np.array([match[1] for match in sorted(matches)])
    mismatches = np.array([mismatch[1] for mismatch in sorted(mismatches)])
    return matches, mismatches.reshape((len(labels),9,-1))
