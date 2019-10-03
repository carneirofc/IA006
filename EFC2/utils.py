#!/usr/bin/python3
import pandas

def load_data(fname):
    return pandas.read_csv(fname)

def shuffle_dataframe(df):
    return df.sample(frac=1).reset_index(drop=True)