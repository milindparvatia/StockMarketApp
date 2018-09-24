import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
import requests
import pandas as pd
import datetime
import numpy as np
import random
import collections
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint

data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=NSE:TCS&outputsize=full&apikey=6G6EDTRGV2N1F9SP')
        
data=data.json()
data=data['Time Series (Daily)']
df=pd.DataFrame(columns=['date','open','high','low','close','volume'])
for d,p in data.items():
    if float(p['3. low'])!=0:
        date=datetime.datetime.strptime(d,'%Y-%m-%d')
        data_row=[date,float(p['1. open']),float(p['2. high']),float(p['3. low']),float(p['4. close']),int(p['6. volume'])]
        df.loc[-1,:]=data_row
        df.index=df.index+1
data=df.sort_values('date')
data.set_index("date", inplace=True)
# print(data)
# data['date']=data['date'].astype(datetime.datetime)
# data['1min']=np.round(data['close'].rolling(window=1).mean(),2)
# data[['1000min','close']].plot()
SEQ_LEN = 100  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 1  # how far into the future are we trying to predict?


def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):

    for col in df.columns:  # go through all of the columns
        if col != "target":  # normalize all ... except for the target itself!
            df[col] = df[col].pct_change()  # pct change "normalizes" the different currencies (each crypto coin has vastly diff values, we're really more interested in the other coin's movements)
            df.dropna(inplace=True)  # remove the nas created by pct_change
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1.

    df.dropna(inplace=True)  # cleanup again... jic.

    # print(df)
    sequential_data = []  # this is a list that will CONTAIN the sequences
    prev_days = collections.deque(maxlen=SEQ_LEN)  # These will be our actual sequences. They are made with deque, which keeps the maximum length by popping out older values as new ones come in
    
    for i in df.values:  # iterate over the values
        prev_days.append([n for n in i[:-1]])  # store all but the target
        # if len(prev_days) == SEQ_LEN:# print(len(prev_days))
        sequential_data.append([np.array(prev_days), i[-1]])  # append those bad boys!
            

    random.shuffle(sequential_data)  # shuffle for good measure.
    # print(sequential_data)
    # print(len(sequential_data))
    buys = []  # list that will store our buy sequences and targets
    sells = []  # list that will store our sell sequences and targets

    for seq, target in sequential_data:  # iterate over the sequential data
        if target == 0:  # if it's a "not buy"
            sells.append([seq, target])  # append to sells list
        elif target == 1:  # otherwise if the target is a 1...
            buys.append([seq, target])  # it's a buy!

    random.shuffle(buys)  # shuffle the buys
    random.shuffle(sells)  # shuffle the sells!

    lower = min(len(buys), len(sells))  # what's the shorter length?

    buys = buys[:lower]  # make sure both lists are only up to the shortest length.
    sells = sells[:lower]  # make sure both lists are only up to the shortest length.

    # print(len(sells))
    # print(len(buys))

    sequential_data = buys+sells  # add them together
    random.shuffle(sequential_data)  # another shuffle, so the model doesn't get confused with all 1 class then the other.
    # print(len(sequential_data))
    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)  # X is the sequences
        y.append(target)  # y is the targets/labels (buys vs sell/notbuy)

    return np.array(X), y  # return X and y...and make X a numpy array! ..import numpy as np

data.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
data.dropna(inplace=True)

arr = data['close'].shift(-FUTURE_PERIOD_PREDICT)
data['target'] = list(map(classify, data['close'], arr))
# print(len(data))
# print(data.head())


times = sorted(data.index.values)
last_5pct = times[-int(0.05*len(times))]

# print(last_5pct)

validation_data = data[(data.index >= last_5pct)]  # make the validation data where the index is in the last 5%
data = data[(data.index < last_5pct)]  # now the data is all the data up to the last 5%
#print(validation_data)
# print(len(validation_data))
  
# print(data.head())

# preprocess_df(validation_data)
train_x, train_y = preprocess_df(data)
validation_x, validation_y = preprocess_df(validation_data)

print(f"train data: {len(train_x)} validation: {len(validation_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")


print(train_x.shape[1:])

# print(data.head())
# print(data.tail())

# print(len(data))

# print(train_x.ndim)

# def arraycahnge(x):
#     for i in range (len(x)):
#         y=len(x[i])
#         x[i]=x[i].reshape(y,1)
#         x[i]=[x[i]]
#     # x=x.reshape(len(x),200,1)
#     print(x)
#     return x

# arraycahnge(train_x)
# train_x1=train_x[3073].reshape(200,1)
# print(train_x[3073])
# print(train_x1.ndim)

# EPOCHS = 10  # how many passes through our data
# BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
# NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model

# model = Sequential()
# model.add(LSTM(128, input_shape=(train_x.shape[:1]), return_sequences=True))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())  
# #normalizes activation outputs, same reason you want to normalize your input data.

# model.add(LSTM(128, return_sequences=True))
# model.add(Dropout(0.1))
# model.add(BatchNormalization())

# model.add(LSTM(128))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())

# model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))

# model.add(Dense(2, activation='softmax'))

# opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# # Compile model
# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=opt,
#     metrics=['accuracy']
# )

# tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

# filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
# checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones

# # Train model
# history = model.fit(
#     train_x, train_y,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCHS,
#     validation_data=(validation_x, validation_y),
#     callbacks=[tensorboard, checkpoint],
# )

# # Score model
# score = model.evaluate(validation_x, validation_y, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# # Save model
# model.save("models/{}".format(NAME))
