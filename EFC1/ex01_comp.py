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


def shuffle(train_inp, train_resp):
    indices = np.arange(train_inp.shape[0])
    np.random.shuffle(indices)
    return train_inp[indices], train_resp[indices]


def getBestFold(folds, percent, train_inp, train_resp):
    '''
    return w, fold_detail
    '''
    # print('Folding ...')
    folds_detail = []
    folds_rms = []
    folds_w = []
    split = int(len(train_inp)*percent)
    for fold in range(folds):
        s_train_inp, s_train_resp = shuffle(train_inp, train_resp)     # shuffle the bois
        w = getWeight(s_train_inp[0:split], s_train_resp[0:split])     # train
        rms = evalModel(w, s_train_inp[split:], s_train_resp[split:])  # test
        avg = np.sum(rms)/len(rms)
        folds_rms.append(avg)
        folds_detail.append({'var':np.var(rms),'avg':avg, 'min':np.amin(rms), 'max':np.amax(rms)})
        folds_w.append(w)
    idx = folds_rms.index(min(folds_rms))
    #print('Finish folding ({} folds)... Best fold:idx {}\t{}'.format(folds, idx, folds_detail[idx]))
    return folds_w[idx], folds_detail[idx]

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
    return rm

if __name__ == '__main__':
    df_data = utils.read_data(fname='daily-minimum-temperatures.csv')
    df_data['Date'] = pandas.to_datetime(df_data['Date'])
    temp = df_data['Temp'].values
    print('Dataset {} {}'.format(mean(temp), np.var(temp)))

    # Split dataset ...
    # >= 1990-01-01 Test
    # 1981 -> 1988
    date_max = datetime.datetime(1990,1,1)
    df_test = df_data.loc[df_data.Date >= date_max]
    #df_train = df_data.loc[df_data.Date < date_max]
    #for year in range(1,10):
    for year in range(1,2):
        df_train = df_data.loc[(df_data.Date < date_max)&(df_data.Date >= datetime.datetime(1980+year,1,1))]

        estimators = []
        #print('---------------------------')
        for k in range(1,31):
            train_inp, train_resp = getInpRes(df_train['Temp'].values, k)
            test_inp, test_resp = getInpRes(df_test['Temp'].values, k)

            w_optimum, folds_detail = getBestFold(50, 0.8, train_inp, train_resp)
            rm = evalModel(w_optimum, test_inp, test_resp)
            det = {'k':len(w_optimum)-1,'avg':np.mean(rm), 'var':np.var(rm), 'min':np.amin(rm), 'max':np.max(rm), 'w':w_optimum, 'fold_det':folds_detail}
            #print(det['k'],det['avg'],det['var'])
            estimators.append(det)

        da_best = min(estimators, key=lambda x:x['avg'])
        #print('---------------------------')
        print('{} to 1989'.format(1980+year), da_best['k'], da_best['avg'],da_best['var'])
