import pandas as pd
from .Parameters import Parameters
from .get_yahoo_prices import get_yahoo_prices
from .endpoints import endpoints
from .backtest import backtest

def compute_weights_RS_DM(name, parameters):

    print('Strategy : {}'.format(name))

    p = Parameters(parameters)

    prices = get_yahoo_prices(p)

    end_points = endpoints(period=p.frequency, trading_days=prices.index)
    prices_m = prices.loc[end_points]

    returns = prices_m[p.symbols].pct_change(p.rs_lookback)[p.rs_lookback:]

    if isinstance(p.risk_free, int):
        excess_returns = returns
    else:
        risk_free_returns = prices_m[p.risk_free].pct_change(p.rs_lookback)[p.rs_lookback:]
        excess_returns = returns.subtract(risk_free_returns, axis=0).dropna()

    absolute_momentum = prices_m[p.symbols].pct_change(p.risk_lookback)[p.risk_lookback:]
    absolute_momentum_rule = absolute_momentum > 0
    rebalance_dates = excess_returns.index.join(absolute_momentum_rule.index, how='inner')

    # relative strength ranking
    ranked = excess_returns.loc[rebalance_dates][p.symbols].rank(ascending=False, axis=1, method='dense')
    # elligibility rule - top n_top ranked securities
    elligible = ranked[ranked <= p.n_top] > 0

    # equal weight allocations
    elligible = elligible.multiply(1. / elligible.sum(1), axis=0)

    # downside protection
    weights = pd.DataFrame(0., index=elligible.index, columns=prices.columns)
    if p.cash_proxy == 'CASHX':
        weights[p.cash_proxy] = 0
        prices[p.cash_proxy] = 1.
    weights[p.symbols] = (elligible * absolute_momentum_rule).dropna()
    weights[p.cash_proxy] += 1 - weights[p.symbols].sum(axis=1)

    # backtest

    p_value, p_holdings, p_weights = backtest(prices, weights, 10000., offset=0, commission=10.)

    # p_value.plot(figsize=(15, 10), grid=True, legend=True, label=name)

    return p_value, p_holdings, p_weights, prices

