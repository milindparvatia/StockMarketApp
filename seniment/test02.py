import os
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import defaultdict
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def leaders(xs, top=500):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:top]

if __name__ == '__main__':
    os.remove("tweets1234.json")
    os.system('twitterscraper #GOOGL --limit 100 -bd 2018-01-10 -ed 2018-09-20 --output=tweets1234.json')
    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['rt', 'via']
    
 
    with open('tweets1234.json', 'r') as f:
        line = f.read() # read only the first tweet/line
        total = list()
        sentiment = 0.0
        pos = 0.0
        neg = 0.0
        tweet = json.loads(line) # load it as Python dict
        type(tweet)
        for key in tweet:
            #print("\n")
            #print("\n Tweet : ")
            terms_stop = [term for term in word_tokenize(key['text']) if term not in stop] #Using Nltk to tokenize
            total.extend(terms_stop)
    
    for key in total:
        if(len(key) < 3):
            total.remove(key)

    for i in range(len(total)):
        total[i] = total[i].lower()

    
    with open('bulltest.json','r') as temp:
        bull = json.load(temp)
        print(bull)

    with open('beartest.json', 'r') as temp:
        bear = json.load(temp)
        print(bear)

    f.close()
    sentpos = 0.0
    sentneg = 0.0

    freq = leaders(total)

    for key1 in freq:
        #t1 = list(key) #convert tuple to list for comparing
        for key2 in bull:
            if(key1[0].lower() == key2[0].lower()):
                sentpos = sentpos + (key2[1] * key1[1])
        for key3 in bear:
            if(key1[0].lower() == key3[0].lower()):
                sentneg = sentneg - (key3[1] * key1[1]) 


    print("\n\n")
    
    # print(freq)
    print(sentpos)
    print(sentneg)

    print(sentpos+sentneg)
