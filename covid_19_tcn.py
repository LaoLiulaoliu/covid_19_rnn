import os
import sys

import paddle
import paddle.nn as nn
import numpy as np

import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from paddlenlp.seq2vec import TCNEncoder

%matplotlib inline
%config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 14, 10
register_matplotlib_converters()

#https://aistudio.baidu.com/aistudio/projectdetail/1471138
#!pip install paddlenlp>=2.0.0b -i https://pypi.org/simple
# !wget https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv

df_all = pd.read_csv('time_series_covid19_confirmed_global.csv')

df = df_all.iloc[:, 4:]
daily_cases = df.sum(axis=0)
daily_cases.index = pd.to_datetime(daily_cases.index)

daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)

TEST_DATA_SIZE = 30

train_data = daily_cases[:-TEST_DATA_SIZE]
test_data = daily_cases[-TEST_DATA_SIZE:]

print("The number of the samples in train set is : %i"%train_data.shape[0])


scaler = MinMaxScaler()
train_data = scaler.fit_transform(np.expand_dims(train_data, axis=1)).astype('float32')
test_data = scaler.transform(np.expand_dims(test_data, axis=1)).astype('float32')


SEQ_LEN = 10

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length+1):
        x = data[i:i+seq_length-1]
        y = data[i+seq_length-1]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

x_train, y_train = create_sequences(train_data, SEQ_LEN)
test_data = np.concatenate((train_data[-SEQ_LEN+1:],test_data),axis=0)
x_test, y_test = create_sequences(test_data, SEQ_LEN)

print("The shape of x_train is: %s"%str(x_train.shape))
print("The shape of y_train is: %s"%str(y_train.shape))
print("The shape of x_test is: %s"%str(x_test.shape))
print("The shape of y_test is: %s"%str(y_test.shape))


class CovidDataset(paddle.io.Dataset):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        super(CovidDataset, self).__init__()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return [self.feature[index], self.label[index]]

train_dataset = CovidDataset(x_train, y_train)
test_dataset = CovidDataset(x_test, y_test)

class TimeSeriesNetwork(nn.Layer):

  def __init__(self, input_size, next_k=1, num_channels=[64,128,256]):
    super(TimeSeriesNetwork, self).__init__()

    self.last_num_channel = num_channels[-1]

    self.tcn = TCNEncoder(
      input_size=input_size,
      num_channels=num_channels,
      kernel_size=2, 
      dropout=0.2
    )

    self.linear = nn.Linear(in_features= self.last_num_channel, out_features=next_k)

  def forward(self, x):
    tcn_out = self.tcn(x)
    y_pred = self.linear(tcn_out)
    return y_pred

network = TimeSeriesNetwork(input_size=1)
LR = 1e-3
model = paddle.Model(network)
optimizer = paddle.optimizer.Adam(
        learning_rate=LR, parameters=model.parameters())
loss = paddle.nn.MSELoss(reduction='sum')
model.prepare(optimizer, loss)


USE_GPU = False
TRAIN_EPOCH = 100
LOG_FREQ = 10
SAVE_DIR = os.path.join(os.getcwd(),"save_dir")
SAVE_FREQ = 10

if USE_GPU:
    paddle.set_device("gpu")
else:
    paddle.set_device("cpu")

model.fit(train_dataset, 
    batch_size=32,
    drop_last=True,
    epochs=TRAIN_EPOCH,
    log_freq=LOG_FREQ,
    save_dir=SAVE_DIR,
    save_freq=SAVE_FREQ,
    verbose=1
)

preds = model.predict(test_data=test_dataset)


true_cases = scaler.inverse_transform(
    np.expand_dims(y_test.flatten(), axis=0)
).flatten()

predicted_cases = scaler.inverse_transform(
  np.expand_dims(np.array(preds).flatten(), axis=0)
).flatten()


plt.plot(
  daily_cases.index[:len(train_data)], 
  scaler.inverse_transform(train_data).flatten(),
  label='Historical Daily Cases'
)

plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)], 
  true_cases,
  label='Real Daily Cases'
)

plt.plot(
  daily_cases.index[len(train_data):len(train_data) + len(true_cases)], 
  predicted_cases, 
  label='Predicted Daily Cases'
)

plt.legend();
