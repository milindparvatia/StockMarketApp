import matplotlib.pyplot as plt
from matplotlib import style

import requests
import pandas as pd
import datetime
import numpy as np
style.use('fivethirtyeight')
data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&interval=1min&symbol=AAPL&apikey=6G6EDTRGV2N1F9SP')

data=data.json()
data=data['Monthly Time Series']
df=pd.DataFrame(columns=['date','open','high','low','close','volume'])
for d,p in data.items():
    date=datetime.datetime.strptime(d,'%Y-%m-%d')
    data_row=[date,float(p['1. open']),float(p['2. high']),float(p['3. low']),float(p['4. close']),int(p['5. volume'])]
    df.loc[-1,:]=data_row
    df.index=df.index+1
data=df.sort_values('date')
# data['close']=data['close'].astype(float)
# data['date']=data['date'].astype(datetime.datetime)
# data['1min']=np.round(data['close'].rolling(window=1).mean(),2)
print(data)
# data[['1000min','close']].plot()
# plt.show()
