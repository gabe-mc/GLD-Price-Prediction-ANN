"""FMP API to download real time stock prices"""

import requests
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datetime import timedelta
import json
import csv

# 512 Data points
# 13 previous trading days (17 days), data from each 10 minutes = 507


# Data
api_key = "7PUXRnPbbIlTyRlU5PblgwJrvcODH8Qe"
ticker = "VIX"
timeframe = "4hour"
timeseries = "8"
today = datetime.today()
# from_date = today.date() - timedelta(days=800)
from_date = "2015-1-1"
to_date = today.date().isoformat()
# params = {
#     "from": from_date,
#     "to":
#     "apikey": api_key,
# }

# Amex ticker
url = "https://financialmodelingprep.com/api/v3/historical-price-full/SPY?from=1990-01-01&apikey=7PUXRnPbbIlTyRlU5PblgwJrvcODH8Qe"
res = requests.get(url)
if res.status_code != 200:
    raise Exception(f"Failed to fetch data: {res.status_code}")
res_json = res.json()

# DATA VARIABLES
gld_data = [res_json['historical'][x]['open'] for x in range(len(res_json['historical']))]
gld_data.reverse()

times_data = [res_json['historical'][x]['date'] for x in range(len(res_json['historical']))]
times_data.reverse()

# with open('spy_prices.csv', 'w') as file:
#     writer = csv.writer(file)
#     for i in range(len(gld_data)):
#         writer.writerow([times_data[i][5:7]+'/'+times_data[i][8:]+'/'+times_data[i][:4],gld_data[i]])

all_dates = [times_data[i][5:7]+'/'+times_data[i][8:]+'/'+times_data[i][:4] for i in range(len(times_data))]
# print(all_dates)


# print(len(gld_data))

# with open("gold_historical_price.json", "w") as f:
#     f.write(json.dumps(res_json, indent=2))
#     f.flush()
# print(f"Successfully wrote chart data (timeframe {timeframe}, from {from_date} to {to_date}).")

accum = []
m= -1
with open("VIX_History - VIX_History.csv", 'r') as file:
    reader = csv.reader(file)
    for i in reader:
        if i[0] not in all_dates:
            m += 1
            accum+= [i[0]]
            

print(accum)
# print(all_dates)
# print('06/11/2004' in all_dates)
print(m)



# def get_price(ticker: str):
#     """Get the post-market price data for the current day, for the given ticker"""
#     date = datetime.today().strftime("%m/%d/%y")
#     url = f"https://financialmodelingprep.com/api/v3/quote-short/{
#         ticker}?apikey=7PUXRnPbbIlTyRlU5PblgwJrvcODH8Qe"
#     price_data = requests
#     return price_data[price]


# a = "https://financialmodelingprep.com/api/v3/quote-short/AAPL?apikey=7PUXRnPbbIlTyRlU5PblgwJrvcODH8Qe"
# print(requests.get(a).json().price())


# Historical Intraday Price Fetch
# AAPL_ticker = "https://financialmodelingprep.com/api/v3/historical-chart/5min/SDG.V?from=2023-08-10&to=2023-09-10&apikey=7PUXRnPbbIlTyRlU5PblgwJrvcODH8Qe"
# response = requests.get(AAPL_ticker)
# item = response.json()
# print(item)

# fig, ax = plt.subplots()
# ax.set_title(item['ticker'])
# ax.plot([x for x in range(0,item['queryCount'])], data)
# plt.show()
