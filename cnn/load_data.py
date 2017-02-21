import os
import numpy as np
from multiprocessing import Pool
from contextlib import closing
import csv

bin_freq = 23
spect_width = bin_freq  # Don't add one pixel of zeros on either side of the image
window_size = 100
dim_Y = 11
MEAN_SPEC = 10.786225977

# MNIST embeddings directory
dataMnistdir = '../mnistAct/'
MNIST_DIM = 512

mnist_data = {digit: np.loadtxt(dataMnistdir + str(digit) + ".txt") for digit in range(10)}

label_map = {'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'o':0,'z':0}

def load_from_file(f):
    '''Given a file, returns a list of the string values in that value'''
    data = []
    for line in f:
        vector = []
        line = line.replace("[", "")
        line = line.replace("]", "")
        line_chars = line.split()
        for char in line_chars:
            # vector.append(float(char)-MEAN_SPEC)
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
    # #Pad one pixel of zeros on top and bottom of array
    # zero = np.zeros((bin_freq,))
    # data[0] = zero
    # data[-1] = zero
    return data.flatten().tolist()

def ld(rootdir,target):
    with open(target, 'wb') as datafile:
        writer = csv.writer(datafile)
        for subdir, dirs, files in os.walk(rootdir):
            for filename in files:
                y = filename[3]
                f = open(os.path.join(subdir, filename))
                row = load_from_file(f)
                f.close()
                writer.writerow([y] + row)

def get_mnist_embedding(label):
    digit = label[1]
    data = mnist_data[digit] # np.loadtxt(dataMnistdir + str(digit) + ".txt")
    i = np.random.randint(0,len(data))
    return label[0],data[i]

def get_label_map(s):
    return label_map[s]

def get_mismatch_mnist_embedding(label):
    i = label[1]
    out = None
    for j in xrange(10):
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
    labels = [(i, label_map[labels[i]]) for i in range(len(labels))]
    with closing(Pool()) as pool:
        matches = pool.map(get_mnist_embedding, labels)
        mismatches = pool.map(get_mismatch_mnist_embedding, labels)
    matches = np.array([match[1] for match in sorted(matches)])
    mismatches = np.array([mismatch[1] for mismatch in sorted(mismatches)])
    return matches, mismatches.reshape((len(labels),9,-1))

# def get_new_index(inp):
#     index,labels = inp
#     i = np.random.randint(0,len(labels))
#     while (labels[index] == labels[i]):
#         i = np.random.randint(0,len(labels))
#     return index,i
#
# def permute_batch(labels):
#     labels = [label_map[label] for label in labels]
#     inp = [(i,labels) for i in range(len(labels))]
#     with closing(Pool()) as pool:
#         indices = pool.map(get_new_index,inp)
#     indices = [index[1] for index in sorted(indices)]
#     return indices
