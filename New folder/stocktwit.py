import os
import json
import requests

response = requests.get("https://api.stocktwits.com/api/2/streams/symbol/AAPL.json")

dat = response.json();

with open("data5.json","w") as outfile:
    json.dump(dat,outfile)
