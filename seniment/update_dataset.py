import os
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import defaultdict
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()


def leaders(xs, top=10000):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:top]


if __name__ == '__main__':
    #os.remove("tweets12.json")
    os.system('twitterscraper #AAPL --limit 10000 -bd 2018-01-10 -ed 2018-09-20 --output=tweettest.json')
    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['rt', 'via']

    with open('tweettest.json', 'r') as f:
        line = f.read()  # read only the first tweet/line
        total = list()
        sentiment = 0.0
        pos = 0.0
        neg = 0.0
        count = 0

        bullfinal = list()  # contains normalizes bullish words with intensity
        bearfinal = list()

        bullwords = list()  # contains bullish words
        bearwords = list()
        neutralwords = list()

        tweet = json.loads(line)  # load it as Python dict
        type(tweet)
        for key in tweet:
            count = count + 1
            #print("\n")
            #print("\n Tweet : ")
            snt = analyser.polarity_scores(key['text'])
            sentiment = sentiment + snt['compound']
            pos = pos + snt['pos']
            neg = neg + snt['neg']

            terms_stop = [term for term in word_tokenize(key['text']) if term not in stop]  # Using Nltk to tokenize
            total.extend(terms_stop)

    '''
    for key in total:
        if(len(key) < 3):
            total.remove(key)
    '''

    
    
    for key in total:
        snt = analyser.polarity_scores(key)
        #print(key)
        #print(str(snt))
        if(snt['pos'] > 0):
            bullwords.append(key)
        if(snt['neg'] > 0):
            bearwords.append(key)
        if(snt['neu'] > 0):
            neutralwords.append(key)

    f.close()


    freqbull = leaders(bullwords)
    freqbear = leaders(bearwords)

    print(type(freqbull))

    for key in freqbull:
        #print(type(key))
        temp = list(key)
        temp[1] = float(temp[1])/count
        #print(temp)
        bullfinal.append(temp)

    for key in freqbear:
        #print(type(key))
        temp = list(key)
        temp[1] = float(temp[1])/count
        #print(temp)
        bearfinal.append(temp)

    print(bullfinal)
    print(bearfinal)
    print("\n\n")

    with open('bulltest.json', 'w') as outfile:
        json.dump(bullfinal, outfile)

    with open('beartest.json', 'w') as outfile:
        json.dump(bearfinal, outfile)


    '''
    print("\n\n")

    for i in range(len(total)):
        total[i] = total[i].lower()
    '''
    
    freq = leaders(total)

    print(freq)
