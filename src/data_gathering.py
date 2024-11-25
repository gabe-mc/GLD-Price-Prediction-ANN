"""FMP API to download real time stock prices"""

import requests
import os

from dotenv import load_dotenv
from datetime import datetime
from datetime import timedelta

load_dotenv()
api_key = os.getenv('API_Key')


def get_price(ticker: str, prev=False, apikey=api_key) -> float:
    """
    Gets the closing price of the given stock or commodity, on today's date.

    Args:
        ticker: The three digit ticker for the stock, or the three digit symbol for the commodity.

    Returns:
        A float representing the today's price in USD.

    Throws:
        Exception "Failed to fetch data"
    """
    time_offset = 0
    if prev:
        time_offset = 1
    if datetime.now().strftime("%A") == "Sunday":
        date = (datetime.now() - timedelta(days=(2 + time_offset))).strftime("%Y-%m-%d")
    else:
        date = (datetime.now() - timedelta(days=(1 + time_offset)).strftime("%Y-%m-%d"))
    url = "https://financialmodelingprep.com/api/v3/historical-price-full/" + \
        ticker + "?from=" + date + "&apikey=" + apikey
    req = requests.get(url)

    if req.status_code != 200:
        raise Exception(f"Failed to fetch data: {req.status_code}")
    req_json = req.json()

    return req_json['historical'][0]['close']
