#!/usr/bin/python3
import math
import sys
import pickle
from math import log10 as log

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def load():
    return pd.read_csv('dados_voz_genero.csv')

def dataset(data, split=0.8):
    man = data.loc[data['label'] == 1]
    woman = data.loc[data['label'] == 0]

    man.iloc[:int(len(man)*split),:]
    man.iloc[int(len(man)*split):,:]

    woman.iloc[:int(len(woman)*split),:]
    woman.iloc[int(len(woman)*split):,:]

    df_train = pd.concat([
            man.iloc[:int(len(man)*split),:],
            woman.iloc[:int(len(woman)*split),:]],
        ignore_index=True)
    df_test = pd.concat([
        man.iloc[int(len(man)*split):,:],
        woman.iloc[int(len(woman)*split):,:]],
    ignore_index=True)
    
    df_train = shuffle(df_train)
    df_test = shuffle(df_test)
    
    return df_train, df_test

def plot_data(df):
    fig, ax = plt.subplots()
    df.hist(figsize=(30,15), ax=ax)
    fig.savefig('img1/data_hist.png', dpi=600)

    fig, ax = plt.subplots()
    plt.matshow(df.corr())
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    fig.savefig('img1/data_corr.png', dpi=600)

    fig, ax = plt.subplots()
    scatter_matrix(data, figsize=(35, 35), ax=ax)
    fig.savefig('img1/data_corr_scatter.png', dpi=600)

    plt.show()

def sigmoid(x):
    return 1. / (1. + math.exp(-1*x)) 

def cost_func(y, yest):
    return -1*y*log(yest) + -1*(1-y)*log(1 + -1*yest)

def estimate(w, fi):
    return np.apply_along_axis(sigmoid, 1, fi.T.dot(w)).reshape(-1, 1)

def train(w, fi, eta, tol, y):
    err = tol
    while err >= tol:
        yest = estimate(w, fi)
        e = y - yest
        delta = -1*(e.T.dot(fi.T))/len(fi)
        w = w - delta.T*eta
        
        err = np.sqrt(e**2).mean()

        sys.stdout.write("\r{}".format(err))
        sys.stdout.flush()
        
    return w 

if __name__ == '__main__':
    data = load()
    df_train, df_test = dataset(data, split=0.8)
    #plot_data(data)
    
    # Data normalization
    scaler = StandardScaler() 
    df_train_out = df_train.loc[:,'label']
    df_train_inp = df_train.drop('label', axis=1)
    scaler.fit(df_train_inp) 
    df_train_inp_scaled = scaler.transform(df_train_inp)

    # w and fi
    ones = np.ones([len(df_train_inp_scaled),1])
    fi = np.concatenate((ones, df_train_inp_scaled),axis=1).T
    w = np.random.uniform(0,1,(len(fi[:,0]), 1))

    # Train
    eta = 1e-3
    tol = 1e-3
    w = train(w, fi, eta, tol, df_train_out.to_numpy().reshape(-1, 1))
 
    with open('data1', 'wb+') as f:
        pickle.dump(w, f)

