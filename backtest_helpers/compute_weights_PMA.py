import pandas as pd
from .endpoints import endpoints
from .backtest import backtest

def compute_weights_PMA(name, parameters):

    print(name)

    symbols = parameters['symbols']
    cash_proxy = parameters['cash_proxy']

    risk_lookback = parameters['risk_lookback']
    allocations = parameters['allocations']
    frequency = parameters['frequency']

    tickers = list(set(symbols + [cash_proxy]))

    data = pd.DataFrame(columns=symbols)
    for symbol in tickers:
        print(symbol)
        url = 'http://chart.finance.yahoo.com/table.csv?s=' + symbol + '&ignore=.csv'
        data[symbol] = pd.read_csv(url, parse_dates=True, index_col='Date').sort_index(ascending=True)['Adj Close']

    inception_dates = pd.DataFrame([data[ticker].first_valid_index().date() for ticker in data.columns],
                                   index=data.keys(), columns=['inception'])

    #     print ('INCEPTION DATES:\n\n{}'.format(inception_dates))

    prices = data.copy().dropna()

    end_points = endpoints(period=frequency, trading_days=prices.index)
    prices_m = prices.loc[end_points]

    # elligibility rule
    SMA = pd.rolling_mean(prices_m, risk_lookback).dropna()
    rebalance_dates = SMA.index
    rule = prices_m.loc[rebalance_dates][symbols] > SMA[symbols]

    # fixed weight allocation
    weights = allocations * rule

    # downside protection
    weights[cash_proxy] = 1 - weights[symbols].sum(axis=1)

    # backtest
    p_value, p_holdings, p_weights = backtest(prices, weights, 10000., offset=0, commission=10.)

    p_value.plot(figsize=(15, 10), grid=True, legend=True, label=name)

    return p_value, p_holdings, p_weights