## Downloaded from
## https://stackoverflow.com/questions/44044263/yahoo-finance-historical-data-downloader-url-is-not-working
## Modified for Python 3
## Added --event=history|div|split   default = history
## changed so "to:date" is included in the returned results
## usage: download_quote(symbol, date_from, date_to, events).decode('utf-8')

# Here an example how to invoke it:
# python new_get_yahoo_data.py --symbol=IBM --from=2017-01-01 --to=2017-05-25 -o IBM.csv
# This will download IBM historical prices from 2017-01-01 to 2017-05-25 and save them in IBM.csv file.

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
    return b''

if __name__ == '__main__':
    print (get_crumble_and_cookie('KO'))
    from_arg = "from"
    to_arg = "to"
    symbol_arg = "symbol"
    event_arg = "event"
    output_arg = "o"
    opt_list = (from_arg+"=", to_arg+"=", symbol_arg+"=", event_arg+"=")
    try:
        options, args = getopt.getopt(sys.argv[1:],output_arg+":",opt_list)
    except getopt.GetoptError as err:
        print (err)

    symbol_val = ""
    from_val = ""
    to_val = ""
    output_val = ""
    event_val = "history"
    for opt, value in options:
        if opt[2:] == from_arg:
            from_val = value
        elif opt[2:] == to_arg:
            to_val = value
        elif opt[2:] == symbol_arg:
            symbol_val = value
        elif opt[2:] == event_arg:
            event_val = value
        elif opt[1:] == output_arg:
            output_val = value

    print ("downloading {}".format(symbol_val))
    text = download_quote(symbol_val, from_val, to_val,event_val)
    if text:
        with open(output_val, 'wb') as f:
            f.write(text)
        print ("{} written to {}".format(symbol_val, output_val))
