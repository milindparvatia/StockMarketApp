import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn.preprocessing import Imputer,StandardScaler
import requests
import pandas as pd
import datetime
import numpy as np
import random
from collections import deque
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=AAPL&outputsize=compact&apikey=6G6EDTRGV2N1F9SP')
 
data=data.json()
data=data['Time Series (Daily)']
df=pd.DataFrame(columns=['date','open','high','low','close','volume'])
for d,p in data.items():
    if float(p['3. low'])!=0:
        date=datetime.datetime.strptime(d,'%Y-%m-%d')
        data_row=[date,float(p['1. open']),float(p['2. high']),float(p['3. low']),float(p['4. close']),int(p['6. volume'])]
        df.loc[-1,:]=data_row
        df.index=df.index+1
main_df=df.sort_values('date')
main_df['date'] = main_df['date'].dt.strftime('%Y%m%d')
main_df.set_index("date", inplace=True)
# print(df.head())

x,y=[],[]
columns=['open','high','low','close','volume']
# print(df)
# for i in range(10):
#     X=[]
#     for j in columns:
#         X.append(main_df[j][i:i+10])
#     Y=main_df['close'][i+11]
#    # X=np.asarray(X)
#     #Y=np.asarray(Y)
x=main_df.iloc[:,]
y=main_df.iloc[:,3]    
    # x.append(X)
    # y.append(Y)
for i in range(len(y)-2):
    y[i]=y[i+1]
# print(l)
# print(main_df['open'][15])

# def baseline_model():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(13, input_dim=5, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(1, kernel_initializer='normal'))
# 	# Compile model
# 	model.compile(loss='mean_squared_error', optimizer='adam')
# 	return model

# # fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)
# # evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=10, batch_size=5, verbose=0)

# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, x, y, cv=kfold, n_jobs=1)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression

cl = LogisticRegression(random_state=0)     
cl.fit(x_train,y_train)

y_pred = cl.predict(x_test)

for i in range(len(y_pred)):
    print(y_pred, y_test)