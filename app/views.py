from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from django.views.generic import View
from rest_framework.views import APIView
from rest_framework.response import Response
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
        url = ('https://newsapi.org/v2/everything?q="'+value+'"&apiKey=4df8d4c46e5f41bca7e6e1331b63ad7d')
        response = requests.get(url)
        geodata = response.json()
            
        return render(request, 'app/search.html', {'allnews': geodata['articles'],'val': val})
    else:
        form = NameForm()
        return render(request, 'app/name.html', {'form': form})

# def get_name(request):
#     # if this is a POST request we need to process the form data
#     if request.method == 'POST':
#         # create a form instance and populate it with data from the request:
#         form = NameForm(request.POST)
#         # check whether it's valid:
#         if form.is_valid():
#             # process the data in form.cleaned_data as required
#             valArray = form.cleaned_data['search']
#             Array = valArray.split(":")
#             value = Array[1]
#             val = form.cleaned_data['search']
#             with open('searchVal.txt','w+') as f:
#                 #convert to string:
#                 f.seek(0)
#                 f.write(val)
#                 f.truncate()
#                 f.close()
#             # redirect to a new URL:
#             url = ('https://newsapi.org/v2/everything?q="'+value+'"&apiKey=4df8d4c46e5f41bca7e6e1331b63ad7d')
#             response = requests.get(url)
#             geodata = response.json()
            
#             return render(request, 'app/search.html', {'allnews': geodata['articles'],'val': val})
            
#     # if a GET (or any other method) we'll create a blank form
#     else:
#         form = NameForm()
#         return render(request, 'app/name.html', {'form': form})

class TimeSeriesDailyAdjusted(APIView):
    authentication_classes = []
    permission_classes = []
        
    def get(self, request, format=None):
        # create a form instance and populate it with data from the request: 
        dataval = self.request.query_params.get('tvwidgetsymbol')
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

        data=requests.get('https://www.alphavantage.co/query?function=MACD&symbol='+search_val+'&interval=daily&series_type=open&apikey=6G6EDTRGV2N1F9SP')
        data=data.json()
        data=data['Technical Analysis: MACD']
        df=pd.DataFrame(columns=['date','MACD_Signal','MACD','MACD_Hist'])
        for d,p in data.items():
            date=datetime.datetime.strptime(d,'%Y-%m-%d')
            data_row=[date,float(p['MACD_Signal']),float(p['MACD']),float(p['MACD_Hist'])]
            df.loc[-1,:]=data_row
            df.index=df.index+1
        dataMACD=df.sort_values('date')

        defaultDaily = dataDaily
        defaultMACD = dataMACD
        alldata = {
            "defaultDaily": defaultDaily,
            "defaultMACD": defaultMACD,
        }
        return Response(alldata)


# class MACD(APIView):
#     authentication_classes = []
#     permission_classes = []
        
#     def get(self, request, format=None):
#         search_val = open('searchVal.txt','r').read()
#         data=requests.get('https://www.alphavantage.co/query?function=MACD&symbol='+search_val+'&interval=daily&series_type=open&apikey=6G6EDTRGV2N1F9SP')
#         data=data.json()
#         data=data['Technical Analysis: MACD']
#         df=pd.DataFrame(columns=['date','MACD_Signal','MACD','MACD_Hist'])
#         for d,p in data.items():
#             date=datetime.datetime.strptime(d,'%Y-%m-%d')
#             data_row=[date,float(p['MACD_Signal']),float(p['MACD']),float(p['MACD_Hist'])]
#             df.loc[-1,:]=data_row
#             df.index=df.index+1
#         data=df.sort_values('date')

#         default_items = data
#         alldata = {
#                 "default": default_items,
#         }
#         return Response(alldata)

class Tweeter(APIView):
    authentication_classes = []
    permission_classes = []
        
#     def get(self, request, format=None):
#         search_val = open('searchVal.txt','r').read()
#         result_array = np.array([])
#         for tweet in query_tweets("Trump OR Clinton", 10)[:10]:
#             result=tweet.user.encode('utf-8')
#             return result
#         tdata  = json.loads(result)
#         default_items = tdata
#         alldata = {
#                 "default": default_items,
#         }
#         return Response(alldata)


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