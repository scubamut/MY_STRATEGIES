import pandas as pd
from .Parameters import Parameters
from .get_yahoo_prices import get_yahoo_prices
from .endpoints import endpoints
from .backtest import backtest

def compute_weights_PMA(name, parameters):

    print(name)

    p = Parameters(parameters)

    prices = get_yahoo_prices(p)

    end_points = endpoints(period=p.frequency, trading_days=prices.index)
    prices_m = prices.loc[end_points]

    # elligibility rule
    SMA = prices_m.rolling(p.risk_lookback).mean().dropna()
    rebalance_dates = SMA.index
    rule = prices_m.loc[rebalance_dates][p.symbols] > SMA[p.symbols]

    # fixed weight allocation
    weights = p.allocations * rule

    # downside protection
    weights[p.cash_proxy] = 1 - weights[p.symbols].sum(axis=1)

    # backtest
    p_value, p_holdings, p_weights = backtest(prices, weights, 10000., offset=0, commission=10.)

    # p_value.plot(figsize=(15, 10), grid=True, legend=True, label=name)

    return p_value, p_holdings, p_weights, prices