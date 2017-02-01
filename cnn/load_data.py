import os
import numpy as np
from multiprocessing import Pool
from contextlib import closing

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

# MNIST embeddings directory
dataMnistdir = '../mnistAct/'
MNIST_DIM = 512

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
                # print len(vector)
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

def get_mnist_embedding(label):
    digit = label[1]
    data = np.loadtxt(dataMnistdir + str(digit) + ".txt")
    i = np.random.randint(0,len(data))
    return label[0],data[i]

def get_mismatch_mnist_embedding(label):
    i = label[1]
    while (i == label[1]):
        i = np.random.randint(0,10)
    return get_mnist_embedding((label[0],i))

def generate_mnist_set(labels):
    matches = []
    mismatches = []
    labels = [(i, np.nonzero(labels[i])[0][0] % 10) for i in range(len(labels))]
    # define the number of CPU cores to be used concurrently
    with closing(Pool()) as pool:
        matches = pool.map(get_mnist_embedding, labels)
        mismatches = pool.map(get_mismatch_mnist_embedding, labels)
    # now transfer the list of return statements to the logfile
    # for label in labels:
    #     y_index = np.nonzero(label)[0][0] % 10
    #     match = get_mnist_embedding(y_index)
    #     mismatch = get_mismatch_mnist_embedding(y_index)
    #     matches.append(match)
    #     mismatches.append(mismatch)
    matches = np.matrix([match[1] for match in sorted(matches)])
    mismatches = np.matrix([mismatch[1] for mismatch in sorted(mismatches)])
    return matches, mismatches

def get_new_index(inp):
    index,labels = inp
    i = np.random.randint(0,len(labels))
    while (labels[index] == labels[i]):
        i = np.random.randint(0,len(labels))
    return index,i

def permute_batch(labels):
    labels = [np.nonzero(label)[0][0] % 10 for label in labels]
    inp = [(i,labels) for i in range(len(labels))]
    # out = [get_new_index(i,labels) for i in range(len(labels))]
    with closing(Pool()) as pool:
        indices = pool.map(get_new_index,inp)
    indices = [index[1] for index in sorted(indices)]
    return indices
