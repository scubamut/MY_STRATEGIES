from .set_start_end import set_start_end

def get_yahoo_prices(p):

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

            if p.start >= p.end:
                raise ('start must be < end')

            start, end = set_start_end()

            data_panel = data.DataReader(tickers, "yahoo", start, end)

            close = data_panel['Adj Close'].sort_index(ascending=True)

            return close.copy().dropna()
    else:
         return p.prices