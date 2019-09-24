#!../venv/bin/python
import pandas
import datetime
import numpy as np
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import pickle
from utils import read_data, config_plt, shuffle, get_inp_res, Kfold

#def eval_fold(folds, percent, train_inp, train_out):
def eval_fold(_train_inp, _train_out, _test_inp, _test_out):
    '''
    return weights, details, folds_rms.index(min(folds_rms))
    '''
    folds_detail = []
    folds_rms = []
    folds_w = []
    for s_train_inp, s_train_out, s_validation_inp, s_validation_out in zip(_train_inp, _train_out, _test_inp, _test_out):
        w = get_weight(s_train_inp, s_train_out)                # train
        rms = eval_model(w, s_validation_inp, s_validation_out) # test
        avg = np.mean(rms)
        folds_rms.append(avg)
        folds_detail.append({'var':np.var(rms),'avg':avg, 'min':np.amin(rms), 'max':np.amax(rms)})
        folds_w.append(w)
    return folds_rms, folds_detail, folds_w

def get_weight(fi, y):
    # p1 = (fi*fiT)-1
    # p2 = fiT*y
    fiT = fi.transpose()
    p1 = np.linalg.inv(fiT.dot(fi))
    p2 = fiT.dot(y)
    w = p1.dot(p2)
    return w

def eval_model(w, test_inp, test_out):
    y_est = test_inp.dot(w)
    rm = (test_out - y_est)**2
    return rm

if __name__ == '__main__':
    config_plt()
    df_data = read_data(fname='daily-minimum-temperatures.csv')
    df_data['Date'] = pandas.to_datetime(df_data['Date'])
    temp = df_data['Temp'].values
    #print('Dataset {} {}'.format(np.mean(temp), np.var(temp)))

    # Split dataset ...
    # >= 1990-01-01 Test
    # 1981 -> 1988
    date_max = datetime.datetime(1990,1,1)
    df_test = df_data.loc[df_data.Date >= date_max]
    df_train = df_data.loc[df_data.Date < date_max]
    folds = 10

    # Plot dataset
    fig, ax = plt.subplots()
    ax.set(xlabel='date', ylabel='ºC', title='Dataset')
    ax.plot(df_train['Date'].values, df_train['Temp'].values, '-.', markerfacecolor='blue', label='Train and Validation')
    ax.plot(df_test['Date'].values, df_test['Temp'].values, '-.', markerfacecolor='red', label='Test')
    ax.legend()
    fig.savefig("ex01/dataset.png", dpi=300)

    with open('ex01/info.txt', 'w+') as f:
        f.write('Folds = {}\n)')

    # Locate best K value
    kfolds_info= []
    for k in range(1,101):
        train_inp, train_out = get_inp_res(df_train['Temp'].values, k)

        # Fold and return the best weight (W)
        _train_inp, _train_out, _validation_inp, _validation_out = Kfold(folds, train_inp, train_out)
        ms, folds_detail, folds_w = eval_fold(_train_inp, _train_out, _validation_inp, _validation_out)#, idx
        rms = np.sqrt(ms)
        #print('K{} ... mean {}'.format(k,np.mean(rms)))
        det = {'k':k,'avg':np.mean(rms), 'var':np.var(rms), 'min':np.amin(rms), 'max':np.max(rms)}
        kfolds_info.append(det)
    data = {'kfolds_info':kfolds_info, 'df_test':df_test, 'df_train':df_train}
    with open('ex01/data', 'wb+') as f:
        pickle.dump(data, f)
        print('Data saved at ex01/data')
