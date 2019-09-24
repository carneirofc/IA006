import numpy as np
import pandas
import math
import time

def read_data(fname):
    return pandas.read_csv(fname)


def config_plt():
    import tkinter
    import matplotlib
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn-whitegrid')
    matplotlib.use('TkAgg')


def shuffle(train_inp, train_out):
    indices = np.arange(train_inp.shape[0])
    np.random.shuffle(indices)
    return train_inp[indices], train_out[indices]


def norm_lim(data, min_, max_):
    return (data - min_)/(max_ - min_)*(2.) - 1.

def dnorm_lim(data, min_, max_):
    return ((data)*(max_ - min_) + min_ + max_ )/2.

def dnorm_zscore(data, mean, std):
    return data*std + mean

def norm_zscore(data, mean, std):
    '''
    Z-score norm
    :param data: np.array
    :return :Tuple (norm np.array, mean, std)
    '''
    return (data - mean)/std

def Kfold(folds, train_inp, train_out):
    """
    K-Fold
    :return _train_inp, _train_out, validation_inp, validation_out:
    """
    _train_inp = []
    _train_out = []
    validation_inp = []
    validation_out = []

    lenght = len(train_inp)
    folds_size = int(lenght/folds) if (lenght%folds==0) else int(lenght/folds + 1)
    for k in range(folds):
        val_start = folds_size*k
        _aux = folds_size*(k+1)
        val_end = _aux if _aux < lenght else lenght

        validation_inp.append(train_inp[val_start:val_end])
        validation_out.append(train_out[val_start:val_end])

        _train_inp.append( np.append(train_inp[0:val_start], train_inp[val_end:], axis = 0))
        _train_out.append( np.append(train_out[0:val_start],train_out[val_end:], axis = 0))
    return _train_inp, _train_out, validation_inp, validation_out


def get_inp_no_bias():
    pass

def generate_attributes(inp_original, T, k, winp):
    '''
    :param inp_original: original input matrix
    :param T: qty of attributes to be generated
    :param k: qty of original attributes to be used
    :param winp: weight matrix for the transformation
    :return: inp
    '''
    inp = np.zeros((len(inp_original), T+1)) # +1 for the bias

    # input dataset
    for i in range(len(inp_original)): # For every input ...
        inp[i][0] = 1 # Bias
        for j in range(T):
            inp[i][j+1] = math.tanh(inp_original[i][1:].dot(winp[j].T))
    return inp


def get_inp_res(data, k):
    inp = np.zeros((len(data)-k, k+1))
    res = np.zeros((len(data)-k, 1))

    for i in range(len(data)-1, k-1, -1):
        rw = i - k
        inp[rw][0] = 1
        res[rw][0] = data[i]
        for j in range(1, k+1):
            inp[rw][j] = data[i -j]
    return inp, res

def get_weight(fi, y, lamb=None, ident=None):
    # p1 = (fi*fiT)-1
    # p2 = fiT*y
    fiT = fi.transpose()
    if lamb != None:
        p1 = np.linalg.inv(fiT.dot(fi)+ (lamb*ident)) 
    else:
        p1 = np.linalg.inv(fiT.dot(fi))
    p2 = fiT.dot(y)
    w = p1.dot(p2)
    return w