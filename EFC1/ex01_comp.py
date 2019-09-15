#!../venv/bin/python
import utils
import tkinter
import matplotlib
import matplotlib.pyplot as plt
import pandas
import datetime

def config_plt():
    plt.style.use('seaborn-whitegrid')
    matplotlib.use('TkAgg')


def getModel(y, k):
    '''
    :param y: Temperature array
    :param k: delay quantity
    :returns I and Y matrix
    '''
    for i in range(len(y)-1, 0, -1):
        if i - k < 0:
            break
        res = '{}) y={}'.format(i, y[i])
        res += ' x0=1'
        for x in range(1, k+1):
            res += ' x{}={}'.format(x, y[i-x])
        print(res)

if __name__ == '__main__':
    config_plt()
    fig = plt.figure()
    ax = plt.axes()

    df_data = utils.read_data(fname='daily-minimum-temperatures.csv')
    df_data['Date'] = pandas.to_datetime(df_data['Date'])
    temp = df_data['Temp'].values
    date = df_data['Date'].values

    #ax.plot(date, temp)
    #plt.draw()
    #plt.show()

    # Split dataset ...
    # >= 1990-01-01 Test
    date_max= datetime.datetime(1990,1,1)
    df_test = df_data.loc[df_data.Date >= date_max]
    df_train = df_data.loc[df_data.Date < date_max]
    print('Test DF', df_test.head())
    print('Train DF', df_train.head())

    getModel(df_train['Temp'].values, 2)
