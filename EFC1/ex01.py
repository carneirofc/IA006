#!../venv/bin/python
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import pandas
import datetime
import numpy as np
from statistics import mean
from utils import read_data, config_plt, shuffle, get_inp_res, norm_data


def get_best_fold(folds, percent, train_inp, train_out):
    '''
    return weights, details, folds_rms.index(min(folds_rms))
    '''
    # print('Folding ...')
    folds_detail = []
    folds_rms = []
    folds_w = []
    split = int(len(train_inp)*percent)
    for fold in range(folds):
        s_train_inp, s_train_out = shuffle(train_inp, train_out)     # shuffle the bois
        w = get_weight(s_train_inp[0:split], s_train_out[0:split])     # train
        rms = eval_model(w, s_train_inp[split:], s_train_out[split:])  # test
        avg = np.mean(rms)
        folds_rms.append(avg)
        folds_detail.append({'var':np.var(rms),'avg':avg, 'min':np.amin(rms), 'max':np.amax(rms)})
        folds_w.append(w)
    idx = folds_rms.index(min(folds_rms))
    #print('Finish folding ({} folds)... Best fold:idx {}\t{}'.format(folds, idx, folds_detail[idx]))
    return folds_w, folds_detail, idx

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
    df_data = read_data(fname='daily-minimum-temperatures.csv')
    df_data['Date'] = pandas.to_datetime(df_data['Date'])
    temp = df_data['Temp'].values
    print('Dataset {} {}'.format(mean(temp), np.var(temp)))
    #temp, m, std = norm_data(temp)
    #df_data['Temp'] = temp

    # Split dataset ...
    # >= 1990-01-01 Test
    # 1981 -> 1988
    date_max = datetime.datetime(1990,1,1)
    df_test = df_data.loc[df_data.Date >= date_max]
    df_train = df_data.loc[df_data.Date < date_max]

    folds = 10
    percent = 0.8

    estimators = []
    with open('01info.txt', 'w+') as f:
        f.write('Folds = {}\n)')
        f.write('Test+Validation = {}%\n'.format(folds, percent*100))

    with open('01delays.csv', 'w+') as f:
        f.write('"k","avg","var","min","max"\n')
        for k in range(1,31):
            print('K {} ...'.format(k))
            train_inp, train_out = get_inp_res(df_train['Temp'].values, k)
            test_inp, test_out = get_inp_res(df_test['Temp'].values, k)

            # Fold and return the best weight (W)
            w_optimum, folds_detail, idx = get_best_fold(folds, percent, train_inp, train_out)
            w_optimum, folds_detail = w_optimum[idx], folds_detail[idx]

            # Test the model using W
            rm = eval_model(w_optimum, test_inp, test_out)
            det = {'k':len(w_optimum)-1,'avg':np.mean(rm), 'var':np.var(rm), 'min':np.amin(rm), 'max':np.max(rm), 'w':w_optimum, 'fold_det':folds_detail}
            estimators.append(det)
            f.write('{},{},{},{},{}\n'.format(det['k'], det['avg'], det['var'], det['min'], det['max']))

    # Locate the min avg(RMS) from K delays
    da_best = min(estimators, key=lambda x:x['avg'])
    print(da_best['k'], da_best['avg'], da_best['var'])
    with open('01estimator.txt', 'w+') as f:
        f.write('W =\n {}\n'.format(da_best['w']))
    with open('01estimator.csv', 'w+') as f:
        f.write('"Real","Est","Error"\n')
        test_inp, test_out = get_inp_res(df_test['Temp'].values, da_best['k'])
        y_est = test_inp.dot(da_best['w'])
        e = (test_out - y_est)
        for i in range(len(e)):
            f.write('{},{},{}\n'.format(test_out[i][0], y_est[i][0], e[i][0]))


