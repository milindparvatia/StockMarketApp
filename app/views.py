from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from rest_framework import generics
from django.views.generic import View   
from rest_framework.views import APIView
from rest_framework.response import Response
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import requests
import pandas as pd
import urllib3, json
import urllib3.request as urllib3
import datetime
import numpy as np
from .forms import NameForm
from twitterscraper import query_tweets
import os
from urllib.parse import unquote
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import defaultdict
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
from rest_framework import viewsets
from app.serializers import CompanyListSerializer
import django_filters.rest_framework
from rest_framework import generics,filters
from django_filters.rest_framework import DjangoFilterBackend
from app.models import CompanyList


class CompanyListView(APIView):
    def get(self, request):
        queryset = CompanyList.objects.all()
        serializer_class = CompanyListSerializer(queryset, many=True)
        #return the serialize JSON data
        return Response(serializer_class.data)


def get_name(request):
    # if this is a POST request we need to process the form data
    if request.GET.get('tvwidgetsymbol'):
        # create a form instance and populate it with data from the request:
        dataval = request.GET['tvwidgetsymbol']
        # process the data in form.cleaned_data as required
        valArray = unquote(dataval)
        Array = valArray.split(":")
        value = Array[1]
        val = valArray

        with open('searchVal.txt','w+') as f:
            #convert to string:
            f.seek(0)
            f.write(val)
            f.truncate()
            f.close()
        # redirect to a new URL:
        response = requests.get('https://newsapi.org/v2/everything?q="'+value+'"&apiKey=4df8d4c46e5f41bca7e6e1331b63ad7d')
        geodata = response.json()
            
        return render(request, 'app/search.html', {'allnews': geodata['articles'],'val': val,'value':value})
    else:
        form = NameForm()
        return render(request, 'app/name.html', {'form': form})

class TimeSeriesDailyAdjusted(APIView):
    authentication_classes = []
    permission_classes = []
        
    def get(self, request, format=None):
        # create a form instance and populate it with data from the request: 
        # dataval = self.request.query_params.get('tvwidgetsymbol')
        # dataval = unquote(dataval)
        search_val = open('searchVal.txt','r').read()
        
        data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='+search_val+'&outputsize=full&apikey=6G6EDTRGV2N1F9SP')
        data=data.json()
        data=data['Time Series (Daily)']
        df=pd.DataFrame(columns=['date','open','high','low','close','volume'])
        for d,p in data.items():
            if float(p['3. low'])!=0:
                date=datetime.datetime.strptime(d,'%Y-%m-%d')
                data_row=[date,float(p['1. open']),float(p['2. high']),float(p['3. low']),float(p['4. close']),int(p['6. volume'])]
                df.loc[-1,:]=data_row
                df.index=df.index+1
        dataDaily=df.sort_values('date')

        data=requests.get('https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol='+search_val+'&outputsize=compact&apikey=6G6EDTRGV2N1F9SP')
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
        # main_df['date'] = main_df['date'].dt.strftime('%Y%m%d')
        main_df.set_index("date", inplace=True)

        x,y=[],[]
        columns=['open','high','low','close','volume']

        def prediction(x1,y):
            scaler=MinMaxScaler()
            scaler.fit(x1)
            tt=scaler.transform(x1)
            x=pd.DataFrame(data=tt)
            
            for i in range(len(y)-1):
                y[i]=y[i+1]
            
            y=y.shift(1, freq='D')

            x_train=x[0:90]  
            x_test=x[90:100]
            y_train=y[0:90]
            y_test=y[90:100]

            sc = StandardScaler()
            
            x_train = sc.fit_transform(x_train)
            x_test = sc.fit_transform(x_test)
            
            model = make_pipeline(PolynomialFeatures(3), Ridge())
            model.fit(x_train, y_train)
            y_plot = model.predict(x_test)
                
            return y_plot,y_test
            
        x1=main_df.iloc[:,]

        yplotDF = pd.DataFrame()
        ytestDF = pd.DataFrame()
            
        for i in range(len(columns)):
            y=main_df.iloc[:,i]   
            y=y.astype('int')
            y_pred,y_orig = prediction(x1,y)
            yplotDF[i] = pd.Series(y_pred)
            ytestDF[i] = pd.Series(y_orig)
        
        ytestDF['date'] = pd.to_datetime(ytestDF.index)
        yplotDF['date'] = pd.to_datetime(ytestDF.index)

        ytestDF=ytestDF.drop(ytestDF.index[[len(ytestDF)-1]])
        
        yplotDF.columns = ['open','high','low','close','volume','date']
        ytestDF.columns = ['open','high','low','close','volume','date']
        
        data=requests.get('https://www.alphavantage.co/query?function=EMA&symbol='+search_val+'&interval=weekly&time_period=9&series_type=open&apikey=6G6EDTRGV2N1F9SP')
        data=data.json()
        data=data['Technical Analysis: EMA']
        df=pd.DataFrame(columns=['date','EMA'])
        for d,p in data.items():
            date=datetime.datetime.strptime(d,'%Y-%m-%d')
            data_row=[date,float(p['EMA'])]
            df.loc[-1,:]=data_row
            df.index=df.index+1
        dataEMA1=df.sort_values('date')


        data=requests.get('https://www.alphavantage.co/query?function=EMA&symbol='+search_val+'&interval=weekly&time_period=26&series_type=open&apikey=6G6EDTRGV2N1F9SP')
        data=data.json()
        data=data['Technical Analysis: EMA']
        df=pd.DataFrame(columns=['date','EMA'])
        for d,p in data.items():
            date=datetime.datetime.strptime(d,'%Y-%m-%d')
            data_row=[date,float(p['EMA'])]
            df.loc[-1,:]=data_row
            df.index=df.index+1
        dataEMA2=df.sort_values('date')


        # os.remove("tweets1234.json")
        # Array = search_val.split(":")
        # value = Array[1]
        # os.system('twitterscraper #'+value+' --limit 50 -bd 2018-09-17 -ed 2018-09-20 --output=tweets1234.json')
        # punctuation = list(string.punctuation)
        # stop = stopwords.words('english') + punctuation + ['rt', 'via']
        
        # with open('tweets1234.json', 'r') as f:
        #     line = f.read() # read only the first tweet/line
        #     total = list()
        #     sentiment = 0.0
        #     pos = 0.0
        #     neg = 0.0
        #     tweet = json.loads(line) # load it as Python dict
        #     type(tweet)
        #     for key in tweet:
        #         snt = analyser.polarity_scores(key['text'])
        #         sentiment = sentiment + snt['compound']
        #         pos = pos + snt['pos']
        #         neg = neg + snt['neg']
        #         terms_stop = [term for term in word_tokenize(key['text']) if term not in stop] #Using Nltk to tokenize
        #         total.extend(terms_stop)
        
        # for key in total:
        #     if(len(key) < 3):
        #         total.remove(key)

        # for i in range(len(total)):
        #     total[i] = total[i].lower()
        # f.close()

        
        # sentiment=str(sentiment)
        # str(neg)
        # str(pos)

        def leaders(xs, top=20):
            counts = defaultdict(int)
            for x in xs:
                counts[x] += 1
            return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:top]

        response = requests.get("https://api.stocktwits.com/api/2/streams/symbol/GOOG.json")

        dat = response.json()

        with open("data5.json","w") as outfile:
            json.dump(dat,outfile)
            
        with open('data5.json', 'r') as f:
            line = f.read()  # read only the first tweet/line
            
            tweet = json.loads(line) # load it as Python dict
            print(type(tweet))
            msg = tweet['messages']
            print(type(msg))
            #loop in through msg by indexes to get all messages
            #message[body] will give you tweet text
            message = msg[0]
            print(type(message))
            ent = message['entities']
            print(type(ent))
            sentiment = ent['sentiment']
            print(type(sentiment))
            bulltotal = list()
            beartotal = list()
            neutraltotal = list()
            sentiment = 0.0
            pos = 0.0
            neg = 0.0
            count = 0

            punctuation = list(string.punctuation)
            stop = stopwords.words('english') + punctuation + ['rt', 'via']
            
            for tweets in msg:
                text = tweets['body']  #Actual message
                count = count + 1
                #Analyser
                snt = analyser.polarity_scores(text)
                sentiment = sentiment + snt['compound']
                pos = pos + snt['pos']
                neg = neg + snt['neg']
                entity = tweets['entities']
                sentiments = entity['sentiment']
                #print(text)

                if(bool(sentiments)):
                    if(sentiments['basic'] == 'Bullish'):
                        terms_stop = [term for term in word_tokenize(text) if term not in stop]  # Using Nltk to tokenize
                        bulltotal.extend(terms_stop)
                    else:
                        terms_stop = [term for term in word_tokenize(text) if term not in stop]  # Using Nltk to tokenize
                        beartotal.extend(terms_stop)
                else:
                    terms_stop = [term for term in word_tokenize(text) if term not in stop]  # Using Nltk to tokenize
                    neutraltotal.extend(terms_stop)        
        f.close()
        
        # freq1 = leaders(bulltotal)
        # freq2 = leaders(beartotal)
        # freq3 = leaders(neutraltotal)

        sentimentData = sentiment/count
        predict = yplotDF
        original = ytestDF
        defaultEMA1 = dataEMA1
        defaultEMA2 = dataEMA2
        defaultDaily = dataDaily
        alldata = {
            "defaultDaily": defaultDaily,
            "defaultEMA2":defaultEMA2,
            "defaultEMA1":defaultEMA1,
            "predict":predict,
            "original":original,
            "sentiment": sentimentData,
        }
        return Response(alldata)


def search(request):
    return render(request,'app/search.html')

def about(request):
    return render(request,'app/about.html')

def contact(request):
    return render(request,'app/contact.html')

def index(request):
    return render(request,'app/home.html')
    
    
def register(request):
    if request.method == 'POST':    
        form = UserCreationForm(request.POST)
    
        if form.is_valid():
            form.save()
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']
            user = authenticate(username = username, password = password)
            login(request, user)
            return redirect('index')
    else:
        form=UserCreationForm()

    context={'form' : form}
    return render(request,'registration/register.html',context)