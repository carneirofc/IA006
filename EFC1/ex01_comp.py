#!../venv/bin/python
import utils
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import pandas
import datetime
import numpy as np
from statistics import mean

def config_plt():
    plt.style.use('seaborn-whitegrid')
    matplotlib.use('TkAgg')

def getTrainData(data, k, folds):
    pass

def getInpRes(data, k):
    inp = np.zeros((len(data)-k, k+1))
    res = np.zeros((len(data)-k, 1))

    for i in range(len(data)-1, k-1, -1):
        rw = i - k
        inp[rw][0] = 1
        res[rw][0] = data[i]
        for j in range(1, k+1):
            inp[rw][j] = data[i -j]
    return inp, res

def getWeight(fi, y):
    # p1 = (fi*fiT)-1
    # p2 = fiT*y
    fiT = fi.transpose()
    p1 = np.linalg.inv(fiT.dot(fi))
    p2 = fiT.dot(y)
    w = p1.dot(p2)
    return w


def evalModel(w, test_inp, test_resp):
    y_est = test_inp.dot(w)
    rm = (test_resp - y_est)**2
    print('{}\t{}\t{}\t{}'.format(len(w)-1, np.sum(rm)/len(rm), np.amin(rm), np.amax(rm)))
    return rm

if __name__ == '__main__':
    df_data = utils.read_data(fname='daily-minimum-temperatures.csv')
    df_data['Date'] = pandas.to_datetime(df_data['Date'])

    # Split dataset ...
    # >= 1990-01-01 Test
    date_max = datetime.datetime(1990,1,1)
    df_test = df_data.loc[df_data.Date >= date_max]
    df_train = df_data.loc[df_data.Date < date_max]

    for k in range(1,31):
        train_inp, train_resp = getInpRes(df_train['Temp'].values, k)
        test_inp, test_resp = getInpRes(df_test['Temp'].values, k)

        w = getWeight(train_inp, train_resp)
        evalModel(w, test_inp, test_resp)
