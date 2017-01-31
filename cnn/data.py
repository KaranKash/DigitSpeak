import os
import tensorflow as tf
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 23
NUM_CHANNELS = 1
NUM_CLASSES = 11
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 4891
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 2325

bin_freq = 23
spect_width = bin_freq  # Don't add one pixel of zeros on either side of the image
window_size = 100
dim_Y = 11

testdir = '/Users/karan/research/single_utterances/test'
traindir = '/Users/karan/research/single_utterances/train'

def load_from_file(f):
    '''Given a file, returns a list of the string values in that value'''
    data = []
    for line in f:
        vector = []
        line = line.replace("[", "")
        line = line.replace("]", "")
        line_chars = line.split()
        for char in line_chars:
            vector.append(float(char))
        try:
            assert len(vector) == bin_freq
            data.append(vector)
        except AssertionError:
            if len(vector) == 0:
                pass
            else:
                print len(vector)
                raise AssertionError

    # Now we have a list of length-23 vectors which we need to trim/pad to
    # window_size
    if len(data)>window_size:
        #cut excess rows
        cut = 1.*(len(data) - window_size)
        data = data[int(np.floor(cut/2)):-int(np.ceil(cut/2))]
    else:
        # pad data with excess rows of zeros about center
        cut = 1.*(window_size - len(data))
        data = [[0]*bin_freq]*int(np.floor(cut/2)) + data + [[0]*bin_freq]*int(np.ceil(cut/2))
    #Convert data to a numpy array and invert it
    data = np.flipud(np.array(data,dtype=np.float32))
    #Pad one pixel of zeros on top and bottom of array
    zero = np.zeros((bin_freq,))
    data[0] = zero
    data[-1] = zero
    return data

def ld(rootdir):
    '''Given a directory, load all the files within it as described on top'''
    X = []
    Y = []
    for subdir, dirs, files in os.walk(rootdir):
        for filename in files:
            try:
                y = int(filename[3])
                y_val = np.zeros((dim_Y,1), dtype=np.float32)
                y_val[y] = 1
                f = open(os.path.join(subdir, filename))
                row = load_from_file(f)
                f.close()
                #check to ensure data has the right dimension
                assert (window_size, spect_width) == row.shape
                X.append(row)
                Y.append(y_val)
            except ValueError:
                if filename[3]=='o':
                    y_val = np.zeros((dim_Y,1))
                    y_val[dim_Y-1] = 1 #the last entry (index 10) set as 1 for 'o'
                    f = open(os.path.join(subdir, filename))
                    row = load_from_file(f)
                    f.close()
                    #check to ensure data has the right dimension
                    assert (window_size, spect_width) == row.shape
                    X.append(row)
                    Y.append(y_val)
                if filename[3]=='z':
                    y_val = np.zeros((dim_Y,1))
                    y_val[0] = 1 #the first entry (index 0) set as 1 for 'z' or zero
                    f = open(os.path.join(subdir, filename))
                    row = load_from_file(f)
                    f.close()
                    #check to ensure data has the right dimension
                    assert (window_size, spect_width) == row.shape
                    X.append(row)
                    Y.append(y_val)
    return np.array(X,dtype=np.float32), np.array(Y, dtype=np.float32)

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
