#!../venv/bin/python
import pandas
import datetime
import numpy as np
import math
from utils import read_data, config_plt, shuffle, get_inp_res, norm_data, Kfold

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
    return folds_rms, folds_detail, folds_w#, idx

def get_weight(fi, y, lamb, ident):
    # p1 = (fi*fiT)-1
    # p2 = fiT*y
    fiT = fi.transpose()
    p1 = np.linalg.inv(fiT.dot(fi)) + (lamb*ident)
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
    print('Dataset {} {}'.format(np.mean(temp), np.var(temp)))

    # Split dataset ...
    # >= 1990-01-01 Test
    # 1981 -> 1988
    date_max = datetime.datetime(1990,1,1)
    df_test = df_data.loc[df_data.Date >= date_max]
    df_train = df_data.loc[df_data.Date < date_max]

    # Normalize dataset using Z-score
    mean, std = df_train['Temp'].values.mean(), df_train['Temp'].values.std()
    min_, max_= df_train['Temp'].values.min(), df_train['Temp'].values.max()
    #norm_df_train = df_train['Temp'].values
    #norm_df_train = norm_data(df_train['Temp'].values, mean, std)
    norm_df_train = (df_train['Temp'].values - min_)/(max_ - min_)*(2) - 1
    #exit(0)
    # Folds
    folds = 10

    with open('ex02/info.txt', 'w+') as f:
        f.write('Folds = {}\n)')
    
    # 5 delays
    k = 5
    lamb_max  =  10e-6
    lamb_step =  10e-9
    train_inp_original, train_out = get_inp_res(norm_df_train, k)

    results = []
    # Total attributes
    for T in range(1, 101):
        res = {'T':T}
        train_inp = np.zeros((len(train_inp_original), T+1)) # +1 for the bias
        winp = np.random.rand(T+1,k+1)
        res['winp'] = winp
        # input dataset
        for i in range(len(train_inp_original)): # For every input ...
            #train_inp[i][0] = 1
            for j in range(T+1):
                train_inp[i][j] = math.tanh(train_inp_original[i].dot(winp[j].T))
                #train_inp[i][j+1] = math.tanh(train_inp_original[i][1:].dot(winp[j].T))
        ident = np.identity(T+1)
        ident[0][0] = 0
        
        # Lambdas ....
        lamb = 0.
        while lamb < lamb_max:
            lamb += lamb_step
            _train_inp, _train_out, _validation_inp, _validation_out = Kfold(folds, train_inp, train_out)

            # Testing lambda  k-fold ...
            rmss = []
            for s_train_inp, s_train_out, s_validation_inp, s_validation_out in\
                 zip(_train_inp, _train_out, _validation_inp, _validation_out):
                w = get_weight(s_train_inp, s_train_out, lamb, ident)
                y_est = s_validation_inp.dot(w)
                y_est_real = ((y_est)*(max_ - min_) + min_ + max_ )/(2)
                s_real = ((s_validation_out)*(max_ - min_) + min_ + max_ )/(2)
                rms = (s_real - y_est_real)**2
                #for i in range(len(s_real)):
                #    print(s_real[i], y_est_real[i])
                #rms = ((s_validation_out - y_est)*std + mean)**2
                rmss.append(rms)
            print(T, lamb, np.mean(rmss)) 
    # Locate best K value
