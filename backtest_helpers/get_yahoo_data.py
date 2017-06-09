# NOTE: see AAAA GET YAHOO DATA.ipynb for yahoo data

import pandas as pd
from pandas.compat import StringIO

import re
from urllib.request import urlopen, Request, URLError
import calendar
import datetime
import getopt
import sys
import time

crumble_link = 'https://finance.yahoo.com/quote/{0}/history?p={0}'
crumble_regex = r'CrumbStore":{"crumb":"(.*?)"}'
cookie_regex = r'Set-Cookie: (.*?); '
quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events={}&crumb={}'


def get_crumble_and_cookie(symbol):
    link = crumble_link.format(symbol)
    response = urlopen(link)
    match = re.search(cookie_regex, str(response.info()))
    cookie_str = match.group(1)
    text = response.read().decode("utf-8")
    match = re.search(crumble_regex, text)
    crumble_str = match.group(1)
    return crumble_str , cookie_str


def download_quote(symbol, date_from, date_to,events):
    time_stamp_from = calendar.timegm(datetime.datetime.strptime(date_from, "%Y-%m-%d").timetuple())
    next_day = datetime.datetime.strptime(date_to, "%Y-%m-%d") + datetime.timedelta(days=1)
    time_stamp_to = calendar.timegm(next_day.timetuple())

    attempts = 0
    while attempts < 5:
        crumble_str, cookie_str = get_crumble_and_cookie(symbol)
        link = quote_link.format(symbol, time_stamp_from, time_stamp_to, events,crumble_str)
        r = Request(link, headers={'Cookie': cookie_str})

        try:
            response = urlopen(r)
            text = response.read()
            return text
        except URLError:
            attempts += 1
            time.sleep(2*attempts)
    return b

def fetch_yahoo_data(symbol, from_date='1980-01-01', to_date=datetime.datetime.today(), event='history'):
    symbol_val = symbol
    from_val = from_date
    to_val = to_date.isoformat()[:10]
    event_val = event
    text = download_quote(symbol_val, from_val, to_val, event_val)
    return pd.read_csv(StringIO(text.decode("utf-8")), index_col=['Date'], parse_dates=True).sort_index(ascending=True)['Adj Close']

def get_yahoo_data(tickers, from_date='1990-01-01', to_date=datetime.datetime.today(), event='history' ):

    data = pd.DataFrame(columns=tickers)
    for symbol in tickers:
        print(symbol, )
        data[symbol] = fetch_yahoo_data(symbol)
    return data

# data = get_yahoo_data(['SPY','EEM'])
# data = data