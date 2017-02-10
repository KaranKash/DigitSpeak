import os
import numpy as np
import csv

bin_freq = 23
spect_width = bin_freq  # Don't add one pixel of zeros on either side of the image
window_size = 100
dim_Y = 11
MEAN_SPEC = 10.786225977

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
    with open(target, 'w', newline='') as datafile:
        writer = csv.writer(datafile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for subdir, dirs, files in os.walk(rootdir):
            for filename in files:
                y = filename[3]
                if filename[3]=='z':
                    y = 0
                if filename[3]=='o':
                    y = 10
                y = int(y)
                f = open(os.path.join(subdir, filename))
                row = load_from_file(f)
                f.close()
                writer.writerow([y] + row)
