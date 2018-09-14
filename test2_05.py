import os
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from collections import defaultdict
import re

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)'  # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')',re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def leaders(xs, top=20):
    counts = defaultdict(int)
    for x in xs:
        counts[x] += 1
    return sorted(counts.items(), reverse=True, key=lambda tup: tup[1])[:top]




if __name__ == '__main__':
    os.system('twitterscraper AAPL --limit 20 -bd 2018-09-12 -ed 2018-09-13 --output=tweets1234.json')
    #os.system('twitterscraper Trump -l 10 -bd 2017-01-01 -ed 2017-06-01 -o tweets.json')
    punctuation = list(string.punctuation)
    stop = stopwords.words('english') + punctuation + ['rt', 'via']
    
 
    with open('tweets1234.json', 'r') as f:
        line = f.read() # read only the first tweet/line
        #count = 1;
        #for line in f:
        total = list()
        total2 = list() #optional
        tweet = json.loads(line) # load it as Python dict
        type(tweet)
        for key in tweet:
            #result = key['text']
            #print(result)
            terms_stop = [term for term in word_tokenize(key['text']) if term not in stop]
            total.extend(terms_stop)
            print(key['text'])
            print(word_tokenize(key['text']))
            print("\n\n")

            #temp = preprocess(key['text'])
            temp = [term for term in preprocess(key['text']) if term not in stop and not term.startswith(('#', '@', 'http' , 'https'))]
            total2.extend(temp)
            print("\n\n")
            print(temp)

    f.close();
    print("\n\n")
    freq = leaders(total)
    print(freq)


    freq2 = leaders(total2)
    print("\n\n")
    print(freq2)
            #print (result)
            #print(json.dumps(tweet[7*count], indent=4)) # pretty-print
            #count = count + 1;
    

