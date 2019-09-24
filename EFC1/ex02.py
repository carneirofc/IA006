#!../venv/bin/python
import pandas
import datetime
import pickle
import math

import numpy as np

from utils import read_data, shuffle, get_inp_res, get_weight,\
    norm_zscore, dnorm_zscore, norm_lim, dnorm_lim, Kfold, generate_attributes

def eval_model(w, test_inp, test_out):
    y_est = test_inp.dot(w)
    rm = (test_out - y_est)**2
    return rm

if __name__ == '__main__':
    T_min, T_max = 1, 101 # Generated attribute
    folds = 10  # Folds
    k = 5       # Delays
    
    # Ridge penality
    lamb_max  =  10
    lamb_ini  =  0
    lamb_step =  0.01

    df_data = read_data(fname='daily-minimum-temperatures.csv')
    df_data['Date'] = pandas.to_datetime(df_data['Date'])
    temp = df_data['Temp'].values

    # Split dataset ...
    # >= 1990-01-01 Test
    # 1981 -> 1988
    date_max = datetime.datetime(1990,1,1)
    df_test  = df_data.loc[df_data.Date >= date_max]
    df_train = df_data.loc[df_data.Date < date_max]

    # Normalize dataset using Z-score
    mean, std = df_train['Temp'].values.mean(), df_train['Temp'].values.std()
    min_, max_= df_train['Temp'].values.min(), df_train['Temp'].values.max()
 
    #norm_df_train = norm_zscore(df_train['Temp'].values, mean, std)
    norm_df_train = norm_lim(df_train['Temp'].values, min_, max_)

    with open('ex02/info.txt', 'w+') as f:
        f.write('T_min, T_max  = {}, {}\n'.format(T_min, T_max))
        f.write('folds = {}\n'.format(folds)) 
        f.write('k = {}\n'.format(k))
        # Ridge penality
        f.write('lamb_max = {}\n'.format(lamb_max))
        f.write('lamb_ini = {}\n'.format(lamb_ini))
        f.write('lamb_step = {}\n'.format(lamb_step))
    
    train_inp_original, train_out = get_inp_res(norm_df_train, k)

    results = []
    # Total attributes
    for T in range(T_min, T_max):
        print('T', T)
        res = {'T':T}
        results.append(res)
        winp = np.random.uniform(-1, 1,(T,k))
        res['winp'] = winp
        train_inp = generate_attributes(train_inp_original, T, k, winp)
        
        # Pseudo identity
        ident = np.identity(T+1)
        ident[0][0] = 0

        res['lamb'] = []
        # Lambdas ....
        #lamb = 0.
        lamb = lamb_ini
        while lamb < lamb_max:
            _train_inp, _train_out, _validation_inp, _validation_out = Kfold(folds, train_inp, train_out)
            lamb += lamb_step

            # Testing lambda k-fold ...
            krms = []
            for s_train_inp, s_train_out, s_validation_inp, s_validation_out in\
                 zip(_train_inp, _train_out, _validation_inp, _validation_out):
                w = get_weight(s_train_inp, s_train_out, lamb, ident)
                y_est = s_validation_inp.dot(w)

                y_est_real = dnorm_lim(y_est, min_, max_)
                s_real     = dnorm_lim(s_validation_out, min_, max_)
                #y_est_real = dnorm_zscore(y_est, mean, std)
                #s_real     = dnorm_zscore(s_validation_out, mean, std)
                ms = (s_real - y_est_real)**2
                rms = np.sqrt(ms)
                krms.append(rms)
            np.mean(krms)
            res['lamb'].append({'mean':np.mean(krms), 'val':lamb})
    
    # Locate best K value
    best_lambs = []
    for r in results:
        res = min(r['lamb'], key=lambda x:x['mean'])
        best_lambs.append({'lamb_val':res['val'], 'mean':res['mean'], 'T':r['T'], 'winp':r['winp']})
    
    data = {
        'lamb_max':lamb_max, 'lamb_min':lamb_ini, 'lamb_step':lamb_step,
        'best_lambs':best_lambs, 'norm_min':min_, 'norm_max':max_, 'folds':folds,
        'df_train':df_train, 'df_test':df_test}

    with open('ex02/data', 'wb+') as f:
        pickle.dump(data, f)
        print('Data saved at ex02/data')
