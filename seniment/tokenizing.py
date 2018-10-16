import os
import json
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import defaultdict
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


analyser = SentimentIntensityAnalyzer()

def leaders(xs, top=20):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:top]


if __name__ == '__main__':
    with open('data1.json', 'r') as f:
        line = f.read()  # read only the first tweet/line
        total = list()
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

       # print(sentiment['basic'])

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


        '''    
        for key in total:
            if(len(key) < 3):
                total.remove(key)

        for i in range(len(total)):
            total[i] = total[i].lower()
        '''
    
    f.close()
    print("\n\n")
    freq1 = leaders(bulltotal)
    freq2 = leaders(beartotal)
    freq3 = leaders(neutraltotal)

    print(freq1)
    print("\n\n")
    print(freq2)
    print("\n\n")
    print(freq3)
    print("\n\n")
    print("\n Sentiment Index Compound : " + str(sentiment/count))
    print("\n Sentiment Index Negative : " + str(neg/count))
    print("\n Sentiment Index Positive: " + str(pos/count))
