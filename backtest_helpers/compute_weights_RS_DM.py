import pandas as pd
from .endpoints import endpoints
from .backtest import backtest
from .get_yahoo_data import get_yahoo_data

def compute_weights_RS_DM(name, parameters):

    print('Strategy : {}'.format(name))

    symbols = parameters['symbols']
    cash_proxy = parameters['cash_proxy']
    risk_free = parameters['risk_free']
    prices = parameters['prices']

    rs_lookback = parameters['rs_lookback']
    risk_lookback = parameters['risk_lookback']
    n_top = parameters['n_top']

    frequency = parameters['frequency']

    if isinstance(prices, str):
        if prices == 'yahoo':
            tickers = symbols.copy()
            if cash_proxy != 'CASHX':
                tickers = list(set(tickers + [cash_proxy]))
            if isinstance(risk_free, str):
                tickers = list(set(tickers + [risk_free]))

            data = get_yahoo_data(tickers)

            prices = data.copy().dropna()

    end_points = endpoints(period=frequency, trading_days=prices.index)
    prices_m = prices.loc[end_points]

    returns = prices_m[symbols].pct_change(rs_lookback)[rs_lookback:]

    if isinstance(risk_free, int):
        excess_returns = returns
    else:
        risk_free_returns = prices_m[risk_free].pct_change(rs_lookback)[rs_lookback:]
        excess_returns = returns.subtract(risk_free_returns, axis=0).dropna()

    absolute_momentum = prices_m[symbols].pct_change(risk_lookback)[risk_lookback:]
    absolute_momentum_rule = absolute_momentum > 0
    rebalance_dates = excess_returns.index.join(absolute_momentum_rule.index, how='inner')

    # relative strength ranking
    ranked = excess_returns.loc[rebalance_dates][symbols].rank(ascending=False, axis=1, method='dense')
    # elligibility rule - top n_top ranked securities
    elligible = ranked[ranked <= n_top] > 0

    # equal weight allocations
    elligible = elligible.multiply(1. / elligible.sum(1), axis=0)

    # downside protection
    weights = pd.DataFrame(0., index=elligible.index, columns=prices.columns)
    if cash_proxy == 'CASHX':
        weights[cash_proxy] = 0
        prices[cash_proxy] = 1.
    weights[symbols] = (elligible * absolute_momentum_rule).dropna()
    weights[cash_proxy] += 1 - weights[symbols].sum(axis=1)

    # backtest

    p_value, p_holdings, p_weights = backtest(prices, weights, 10000., offset=0, commission=10.)

    # p_value.plot(figsize=(15, 10), grid=True, legend=True, label=name)

    return p_value, p_holdings, p_weights, prices

