import numpy as np
import scipy.io
import pickle as pkl
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

def split_signal(sig):
    res = []
    for i in range(60):
        chunk = sig[i * 1000 : i * 1000 + 1000]
        cut_chunk = chunk[488:]
        res.append(cut_chunk)

    return np.array(res)


def load_PCG():

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for i in range(1, 2):
        data = scipy.io.loadmat("data/S" + str(i) + ".mat")
        clean_data = scipy.io.loadmat("data/S" + str(i) + "_Clean.mat")

        sig = data["x"]
        clean_sig = clean_data["PCG"]

        split_sig = split_signal(sig)
        clean_split_sig = split_signal(clean_sig)

        print(split_sig.shape)
        X_train = split_sig[:50]
        print(X_train.shape)
        X_test = split_sig[50:]

        y_train = clean_split_sig[:50]
        y_test = clean_split_sig[50:]


    Dataset = [X_train, y_train, X_test, y_test]

    return Dataset
