import numpy as np
import pandas
import math
def read_data(fname):
    return pandas.read_csv(fname)


def config_plt():
    plt.style.use('seaborn-whitegrid')
    matplotlib.use('TkAgg')


def shuffle(train_inp, train_out):
    indices = np.arange(train_inp.shape[0])
    np.random.shuffle(indices)
    return train_inp[indices], train_out[indices]


def norm_data(data):
    '''
    Z-score norm
    :param data: np.array
    :return :Tuple (norm np.array, mean, std)
    '''
    mean = data.mean()
    std = data.std()
    return (data - mean)/std, mean, std

def get_tanh_inp_res(data, k):
    """
    Apply tanh(w*inp), w random generated weights following a linear dist
    :param data:
    :param k:
    :return inp, res, w:
    """
    inp = np.zeros((len(data)-k, k+1))
    res = np.zeros((len(data)-k, 1))

    for i in range(len(data)-1, k-1, -1):
        rw = i - k
        inp[rw][0] = 1
        res[rw][0] = data[i]
        for j in range(1, k+1):
            inp[rw][j] = data[i -j]
    w = np.ones(len(res[0]))
    #@todo: Randon gen w
    inp = math.tanh(w*inp)
    return inp, res, w


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
