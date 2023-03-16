#import numpy as np
#import scipy.io
#import pickle as pkl
#import scipy.io.wavfile as wav
#import matplotlib.pyplot as plt
#
#def split_signal(sig):
#    res = []
#    for i in range(60):
#        chunk = sig[i * 1000 : i * 1000 + 1000]
#        cut_chunk = chunk[488:]
#        res.append(cut_chunk)
#
#    return np.array(res)
#
#
#def load_PCG():
#
#    X_train = []
#    y_train = []
#
#    X_test = []
#    y_test = []
#
#    for i in range(1, 2):
#        data = scipy.io.loadmat("data/S" + str(i) + ".mat")
#        clean_data = scipy.io.loadmat("data/S" + str(i) + "_Clean.mat")
#
#        sig = data["x"]
#        clean_sig = clean_data["PCG"]
#
#        split_sig = split_signal(sig)
#        clean_split_sig = split_signal(clean_sig)
#
#        print(split_sig.shape)
#        X_train = split_sig[:50]
#        print(X_train.shape)
#        X_test = split_sig[50:]
#
#        y_train = clean_split_sig[:50]
#        y_test = clean_split_sig[50:]
#
#
#    Dataset = [X_train, y_train, X_test, y_test]
#
#    return Dataset

import numpy as np
import scipy.io
import pickle as pkl
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

#def split_signal(sig):
#    res = []
#    for i in range(60):
#        chunk = sig[i * 1000 : i * 1000 + 1000]
#        cut_chunk = chunk[488:]
#        res.append(cut_chunk)
#
#    return np.array(res)

def cut_chunk(chunk, chunk_clean):
    abs_chunk = np.abs(chunk_clean)
    chunk_len = len(chunk)
    
    max_idx = np.argmax(abs_chunk)

    lower_bound = -256
    if lower_bound + max_idx < 0:
        lower_bound = 0

    upper_bound = lower_bound + 512
    if max_idx + upper_bound > chunk_len:
        upper_bound = chunk_len
        lower_bound = upper_bound - 512

    return (np.array(chunk[max_idx + lower_bound : max_idx + upper_bound]),
            np.array(chunk_clean[max_idx + lower_bound : max_idx + upper_bound]))



def split_signal(sig, sig_clean, nb_beats):
    chunk_len = int(len(sig) / nb_beats)

    chunks = np.zeros((nb_beats, 512, 1))
    chunks_clean = np.zeros((nb_beats, 512, 1))

    for i in range(nb_beats):
        chunk = sig[i * chunk_len : i * chunk_len + chunk_len]
        chunk_clean = sig_clean[i * chunk_len : i * chunk_len + chunk_len]

        (chunk, chunk_clean) = cut_chunk(chunk, chunk_clean)

        if(chunk.shape[0] < 512):
            break
        chunks[i] = chunk
        chunks_clean[i] = chunk_clean

    return (np.array(chunks), np.array(chunks_clean))

def load_PCG():

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    # Data loading
    data1 = scipy.io.loadmat("data/S1.mat")
    data1_clean = scipy.io.loadmat("data/S1_Clean.mat")

    data2 = scipy.io.loadmat("data/S2.mat")
    data2_clean = scipy.io.loadmat("data/S2_Clean.mat")

    data3 = scipy.io.loadmat("data/S3.mat")
    data3_clean = scipy.io.loadmat("data/S3_Clean.mat")

    # Signal extraction
    sig1 = data1["x"]
    sig1_clean = data1_clean["PCG"]

    sig2 = data2["x"]
    sig2_clean = data2_clean["PCG"]

    sig3 = data3["x"]
    sig3_clean = data3_clean["PCG"]
    
    # Signal splitting
    split_sig1, split_sig1_clean = split_signal(sig1, sig1_clean, 60)
    split_sig2, split_sig2_clean = split_signal(sig2, sig2_clean, 48)
    split_sig3, split_sig3_clean = split_signal(sig3, sig3_clean, 13)

    X_train = np.concatenate((split_sig1[0:50], split_sig2[0:40], split_sig3[0:9]))
    y_train = np.concatenate((split_sig1_clean[0:50], split_sig2_clean[0:40], split_sig3_clean[0:9]))

    X_test = np.concatenate((split_sig1[50:], split_sig2[40:], split_sig3[9:]))
    y_test = np.concatenate((split_sig1_clean[50:], split_sig2_clean[40:], split_sig3_clean[9:]))


    Dataset = [X_train, y_train, X_test, y_test]

    return Dataset
