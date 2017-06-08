import pandas as pd

def get_yahoo_data(tickers):

    data = pd.DataFrame(columns=tickers)
    for symbol in tickers:
        print(symbol, )
        url = 'http://chart.finance.yahoo.com/table.csv?s=' + symbol + '&ignore=.csv'
        data[symbol] = pd.read_csv(url, parse_dates=True, index_col='Date').sort_index(ascending=True)['Adj Close']

    inception_dates = pd.DataFrame([data[ticker].first_valid_index() for ticker in data.columns],
                                   index=data.keys(), columns=['inception'])

    #     print (inception_dates)
    return data