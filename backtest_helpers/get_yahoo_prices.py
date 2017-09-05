def get_history_prices(p):

    from pandas_datareader import data
    from datetime import datetime

    if isinstance(p.prices, str):
        if p.prices == 'yahoo':
            tickers = p.symbols.copy()
            if p.cash_proxy != 'CASHX':
                tickers = list(set(tickers + [p.cash_proxy]))
            try:
                if isinstance(p.risk_free, str):
                    tickers = list(set(tickers + [p.risk_free]))

            except:
                pass

            start = '2000-01-01'
            end = datetime.today().strftime('%Y-%m-%d')
            data_panel = data.DataReader(tickers, "yahoo", start, end)

            close = data_panel['Adj Close'].sort_index(ascending=True)

            return close.copy().dropna()
    else:
         return p.prices