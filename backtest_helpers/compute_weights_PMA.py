import pandas as pd
from .endpoints import endpoints
from .backtest import backtest
from .get_yahoo_data import get_yahoo_data

def compute_weights_PMA(name, parameters):

    print(name)

    symbols = parameters['symbols']
    cash_proxy = parameters['cash_proxy']
    prices = parameters['prices']

    risk_lookback = parameters['risk_lookback']
    allocations = parameters['allocations']
    frequency = parameters['frequency']

    tickers = list(set(symbols + [cash_proxy]))
    if isinstance(prices, str):
        if prices == 'yahoo':
            data = get_yahoo_data(tickers)
            prices = data.copy().dropna()

    end_points = endpoints(period=frequency, trading_days=prices.index)
    prices_m = prices.loc[end_points]

    # elligibility rule
    SMA = prices_m.rolling(risk_lookback).mean().dropna()
    rebalance_dates = SMA.index
    rule = prices_m.loc[rebalance_dates][symbols] > SMA[symbols]

    # fixed weight allocation
    weights = allocations * rule

    # downside protection
    weights[cash_proxy] = 1 - weights[symbols].sum(axis=1)

    # backtest
    p_value, p_holdings, p_weights = backtest(prices, weights, 10000., offset=0, commission=10.)

    # p_value.plot(figsize=(15, 10), grid=True, legend=True, label=name)

    return p_value, p_holdings, p_weights, prices