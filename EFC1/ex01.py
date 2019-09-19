#!../venv/bin/python
import pandas
import datetime
import numpy as np
from utils import read_data, config_plt, shuffle, get_inp_res, norm_data, Kfold

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
    return folds_rms, folds_detail, folds_w#, idx

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

import tkinter
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
    config_plt()
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
    folds = 10

    with open('ex01/info.txt', 'w+') as f:
        f.write('Folds = {}\n)')

    # Locate best K value
    kfolds_info= []
    with open('ex01/folds.csv', 'w+') as f:
        f.write('"k","avg","var","min","max"\n')
        for k in range(1,31):
            train_inp, train_out = get_inp_res(df_train['Temp'].values, k)

            # Fold and return the best weight (W)
            _train_inp, _train_out, _validation_inp, _validation_out = Kfold(folds, train_inp, train_out)
            rms, folds_detail, folds_w = eval_fold(_train_inp, _train_out, _validation_inp, _validation_out)#, idx
            print('K{} ... mean {}'.format(k,np.mean(rms)))
            det = {'k':k,'avg':np.mean(rms), 'var':np.var(rms), 'min':np.amin(rms), 'max':np.max(rms)}
            kfolds_info.append(det)
            f.write('{},{},{},{},{}\n'.format(det['k'], det['avg'], det['var'], det['min'], det['max']))

    # Plot RMSE per k delays
    ks = [ e['k'] for e in kfolds_info]
    avgs = [ e['avg'] for e in kfolds_info]
    fig, ax = plt.subplots()
    ax.set(xlabel='k delays', ylabel='RMSE', title='RMSE per k delays')
    ax.plot(ks, avgs, label='Validation data')
    ax.legend()
    fig.savefig("ex01/folds.png")
    plt.show()
   
    # Using the best K, train and test a model using the entire training dataset
    best_fold = min(kfolds_info, key=lambda x:x['avg'])
    k = best_fold['k']
    train_inp, train_out = get_inp_res(df_train['Temp'].values, k)
    test_inp, test_out = get_inp_res(df_test['Temp'].values, k)
    
    w = get_weight(train_inp, train_out)
    y_est = test_inp.dot(w)
    e = (test_out - y_est)
    erms = e * e

    # Plot model
    fig, ax = plt.subplots()
    ax.set(xlabel='nth sample', ylabel='RMSE', title='Model test result')
    ax.plot(erms, label='Test data') 
    ax.legend()
    fig.savefig("ex01/model.png")
    plt.show()

    fig, ax = plt.subplots()
    ax.set(xlabel='nth sample', ylabel='ºC', title='Test result')
    ax.plot(test_out, label='Expected') 
    ax.plot(y_est, label='Estimated') 
    ax.legend()
    fig.savefig("ex01/model_comp.png")
    plt.show()
     
    print('Best fold: {}\nW {}'.format(best_fold, w))
    with open('ex01/model.csv', 'w+') as f:
        f.write('"y","ŷ","e","erms"\n')
        for i in range(len(e)):
            f.write('{},{},{},{}\n'.format(test_out[i][0], y_est[i][0], e[i][0], erms[i][0]))

    with open('ex01/model.txt', 'w+') as f:
        f.write('Model={}\n'.format(best_fold))        
        f.write('w={}\n'.format(w))        
