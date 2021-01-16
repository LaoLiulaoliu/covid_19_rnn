#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuande Liu <miraclecome (at) gmail.com>

import time
import numpy as np
import pandas as pd
import mxnet as mx
from sklearn.metrics import r2_score


def massive_missing(df, threshod=0.6):
    nan_sum = df.isnull().sum()
    return nan_sum[nan_sum > df.shape[0] * threshod]

def fill_missing_values(df):
    nan_sum = df.isnull().sum()
    nan_sum = nan_sum[nan_sum > 0]
    for column in list(nan_sum.index):
        if df[column].dtype == 'object':
            df[column].fillna(df[column].value_counts().index[0], inplace=True)
        elif df[column].dtype == 'int64' or 'float64':
            df[column].fillna(df[column].median(), inplace=True)


class LSTM(mx.gluon.nn.Block):
    """
        input: (30438, 1, 229)
        LSTM:        (229, 64) -> (30438, 1, 64)
        dense drop:            -> (30438, 1)
        decoder:               -> (30438, 1)
    """
    def __init__(self, num_hiddens, input_size):
        super(LSTM, self).__init__()
        self.encoder = mx.gluon.rnn.LSTM(hidden_size=num_hiddens, input_size=input_size)
        self.middle = mx.gluon.nn.Dense(1)
        self.drop = mx.gluon.nn.Dropout(0.05)
        self.decoder = mx.gluon.nn.Dense(1)

    def forward(self, inputs):
        outputs = self.encoder(inputs)
        outs = self.decoder(self.drop(self.middle(outputs)))
        return outs


def train_lstm(features, labels, num_epochs, net, trainer, loss, validation_x=None, validation_y=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()

        with mx.autograd.record():
            y_hat = net(features)
            l = loss(y_hat, labels)

        l.backward()
        trainer.step(labels.shape[0])

        train_l_sum += l.sum().asscalar()
        n += l.size
        if epoch & 63 == 0:
            if validation_x is not None:
                validation_loss = loss(net(validation_x), validation_y).mean().asscalar()
                msg = 'epoch %d, train_loss %.4f, validation_loss %.4f, time %.4fs' % (
                epoch + 1, train_l_sum / n, validation_loss, time.time() - start)
            else:
                msg = 'epoch %d, train_loss %.4f, time %.4fs' % (epoch + 1, train_l_sum / n, time.time() - start)
            print(msg)
        
def run_mxnet(train_x, train_y, net, trainer, num_epochs, learning_rate, validation_x=None, validation_y=None):
    if validation_x is not None:
        train_lstm(mx.nd.array(train_x), mx.nd.array(train_y), num_epochs, net, trainer, loss, mx.nd.array(validation_x), mx.nd.array(validation_y))
    else:
        train_lstm(mx.nd.array(train_x), mx.nd.array(train_y), num_epochs, net, trainer, loss)


def gen_lstm_data(data, step=10):
    x, y =[], []
    for i in range(len(data) - step):
        x_temp = data[i:i+step, 1:].ravel()[:-1].reshape(1, -1)
        y_temp = data[i+step:i+step+1, -1]
        x.append(x_temp)
        y.append(y_temp.tolist())
    return np.array(x), np.array(y)


def preprocessing(df, country_code):
    df.drop(['iso_code', 'continent'], axis=1, inplace=True)
    # 删除国家代码和日期
    cols = df.columns.to_list()
    cols.remove('location')
    cols.remove('date')

    # 归一化
    numeric_feature = df[cols].dtypes[df[cols].dtypes != 'object'].index
    mean_df = df[numeric_feature].mean()
    std_df = df[numeric_feature].std()
    df[numeric_feature] = (df[numeric_feature] - mean_df) / std_df

    # 填充缺失值
    fill_missing_values(df)

    # 把国家，字符串转换数字
    df['location'] = df['location'].apply(lambda x: country_code[x])

    # 去除极强相关的冗余变量，否则train不出
    train = df.drop(['total_cases', 'new_cases_smoothed', 'new_cases',
                     'total_deaths', 'new_deaths_smoothed',
                     'total_cases_per_million', 'new_cases_smoothed_per_million',
                     'new_cases_per_million',
                     'total_deaths_per_million', 'new_deaths_smoothed_per_million',
                     'aged_65_older'], axis=1)
    return train, df

def evaluate_mxnet(net, X, Y):
    loaded = time.time()

    Y_hat = net(mx.nd.array(X)).asnumpy()
    print(f'mxnet r2 score: {r2_score(Y_hat, Y)}')
    print('mxnet percentage error: {:.4f}%'.format(((Y_hat - Y) / Y).mean() * 100))

data = pd.read_csv('owid-covid-data.csv')

# 缺失率大于25% 的数据column 丢弃
missing_series = massive_missing(data, 0.25)
df_rnn = data.drop(missing_series.index, axis=1)
countries = list(set(df_rnn['location'].to_list()))
country_code = {c: i for i, c in enumerate(countries)}

train, df_rnn = preprocessing(df_rnn, country_code)


# 设置X， Y
y_name = 'new_deaths'
df_rnn.index = df_rnn['date']
Y = df_rnn.pop(y_name)
df_rnn.insert(len(df_rnn.columns), 'Y', Y)
df_rnn.drop(['date', 'location'], axis=1, inplace=True)


loss = mx.gluon.loss.L2Loss()

step = 10
train_ratio = 0.7
num_epochs = 200
learning_rate = 0.0005

data_x, data_y = gen_lstm_data(df_rnn.values)
print(df_rnn.shape, data_x.shape, data_y.shape) # (43494, 24) (43484, 1, 229) (43484, 1)

train_length = int(data_x.shape[0] * train_ratio)
train_x = data_x[:train_length , :]
train_y = data_y[:train_length , :]
validation_x = data_x[train_length: , :]
validation_y = data_y[train_length: , :]

# (30438, 1, 229) (30438, 1) (13046, 1, 229) (13046, 1)
print(train_x.shape, train_y.shape, validation_x.shape, validation_y.shape)


net = LSTM(64, train_x.shape[2])
net.initialize(mx.init.Xavier())
trainer = mx.gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})
run_mxnet(train_x, train_y, net, trainer, num_epochs, learning_rate, validation_x, validation_y)

evaluate_mxnet(net, validation_x, validation_y)

