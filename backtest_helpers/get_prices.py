from .get_yahoo_data import get_yahoo_data

def get_prices(p):

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

            data = get_yahoo_data(tickers)

            return data.copy().dropna()
    else:
         return p.prices