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
        #print link
        r = Request(link, headers={'Cookie': cookie_str})

        try:
            response = urlopen(r)
            text = response.read()
            print ("{} downloaded".format(symbol))
            return text
        except URLError:
            print ("{} failed at attempt # {}".format(symbol, attempts))
            attempts += 1
            time.sleep(2*attempts)
    return b

def fetch_yahoo_data(symbol):
    symbol_val = symbol
    # assume thios is earliest from data
    from_val = '1990-01-01'
    now = datetime.datetime.today()
    to_val = now.isoformat()[:10]
    event_val = 'history'
    print("downloading {}".format(symbol_val))
    text = download_quote(symbol_val, from_val, to_val, event_val)
    return pd.read_csv(StringIO(text.decode("utf-8")), index_col=['Date'], parse_dates=True).sort_index(ascending=True)['Adj Close']

def get_yahoo_data(tickers):

    data = pd.DataFrame(columns=tickers)
    for symbol in tickers:
        print(symbol, )
#        url = 'http://chart.finance.yahoo.com/table.csv?s=' + symbol + '&ignore=.csv'
#        data[symbol] = pd.read_csv(url, parse_dates=True, index_col='Date').sort_index(ascending=True)['Adj Close']
        data[symbol] = fetch_yahoo_data(symbol)
    inception_dates = pd.DataFrame([data[ticker].first_valid_index() for ticker in data.columns],
                                   index=data.keys(), columns=['inception'])

    #     print (inception_dates)
    return data
