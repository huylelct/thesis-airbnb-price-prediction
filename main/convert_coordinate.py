import requests
import pandas as pd

data = pd.read_csv('../data/data_cleaned.csv')

data = data.drop(columns=[
    'Unnamed: 0.1',
    'Unnamed: 0.1.1',
])
data.to_csv("../data/airbnb-hcm.csv")

for i in range(len(data)):
    if i < 8770:
        continue
    print("Crawl {} ...".format(i))
    lat = data.loc[i, 'latitude']
    long = data.loc[i, 'longitude']

    URL = "https://api.apple-mapkit.com/v1/reverseGeocode?loc={}%2C{}1&lang=en-GB".format(lat, long)

    headers = {
        'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiJ9.eyJpc3MiOiJtYXBzYXBpIiwidGlkIjoiWUg2V005S1o0VSIsImFwcGlkIjoiWUg2V005S1o0VS5tYXBzLm9yZy5ncHMtY29vcmRpbmF0ZXMiLCJpdGkiOmZhbHNlLCJpcnQiOmZhbHNlLCJpYXQiOjE1OTE0MzQ4MTMsImV4cCI6MTU5MTQzNjYxM30.Pj48dsxtMw2aCH8h2OTDQqc-5GIMiXaw6HrLr-gUm2zhYfujkzsU-wBQvStnkRZKRLvTvDPPo4eQhlyb4T6eqA'
    }

    r = requests.get(url=URL, headers=headers)

    data_res = r.json()

    data.loc[i, 'address'] = data_res["results"][0]["dependentLocalities"][0]
    data.to_csv("../data/data_cleaned.csv")
