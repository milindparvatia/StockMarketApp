import os
import json
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
    os.remove("tweets1234.json")
    search_val = open('searchVal.txt','r').read()
    os.system('twitterscraper #'+search_val+' --limit 50 -bd 2018-09-17 -ed 2018-09-20 --output=tweets1234.json')
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
            snt = analyser.polarity_scores(key['text'])
            sentiment = sentiment + snt['compound']
            pos = pos + snt['pos']
            neg = neg + snt['neg']
            terms_stop = [term for term in word_tokenize(key['text']) if term not in stop] #Using Nltk to tokenize
            total.extend(terms_stop)
    
    for key in total:
        if(len(key) < 3):
            total.remove(key)

    for i in range(len(total)):
        total[i] = total[i].lower()

    
    f.close()

    print("\n Sentiment Index Compound : " + str(sentiment))
    print("\n Sentiment Index Negative : " + str(neg))
    print("\n Sentiment Index Positive: " + str(pos))