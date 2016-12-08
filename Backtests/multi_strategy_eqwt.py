import pandas as pd
import ffn
import pandas as pd
import ffn
import os

from backtest_helpers.compute_weights_RS_DM import compute_weights_RS_DM
from backtest_helpers.compute_weights_PMA import compute_weights_PMA
from backtest_helpers.endpoints import endpoints
from backtest_helpers.backtest import backtest
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

os.chdir('C:\\users\\scuba\\pycharmprojects\\simplebacktester')
os.getcwd()


strategies = {
    'DM0003': {'symbols': ['VCVSX', 'VWEHX', 'VFIIX', 'FGOVX', 'VWAHX'], 'prices': 'yahoo',
               'rs_lookback': 1, 'risk_lookback': 1, 'n_top': 2, 'frequency': 'M',
               'cash_proxy': 'CASHX', 'risk_free': 0},
    'RS0002': {'symbols': ['MMHYX', 'FAGIX', 'VFIIX'], 'prices': 'yahoo',
               'rs_lookback': 3, 'risk_lookback': 2, 'n_top': 1, 'frequency': 'M',
               'cash_proxy': 'CASHX', 'risk_free': 0},
    'RS0003': {'symbols': ['MMHYX', 'FAGIX', 'VFIIX'], 'prices': 'yahoo',
               'rs_lookback': 1, 'risk_lookback': 1, 'n_top': 1, 'frequency': 'Q',
               'cash_proxy': 'CASHX', 'risk_free': 0},
    'DM0001': {'symbols': ['VCVSX', 'VWINX', 'VWEHX', 'VGHCX', 'VUSTX', 'VFIIX', 'VWAHX', 'FGOVX', 'FFXSX'],
               'prices': 'yahoo',
               'rs_lookback': 1, 'risk_lookback': 1, 'n_top': 3, 'frequency': 'M',
               'cash_proxy': 'CASHX', 'risk_free': 'FFXSX'},
    'DM0002': {'symbols': ['VCVSX', 'VUSTX', 'VWEHX', 'VFIIX', 'VGHCX', 'FRESX'], 'prices': 'yahoo',
               'rs_lookback': 1, 'risk_lookback': 1, 'n_top': 5, 'frequency': 'M',
               'cash_proxy': 'VFIIX', 'risk_free': 'FFXSX'},
    'PMA001': {'symbols': ['VCVSX', 'VFIIX'], 'prices': 'yahoo',
               'risk_lookback': 3, 'frequency': 'M', 'allocations': [0.6, 0.4],
               'cash_proxy': 'VUSTX'},
    'PMA002': {'symbols': ['VCVSX', 'VWINX', 'VWEHX'], 'prices': 'yahoo',
               'risk_lookback': 3, 'frequency': 'M', 'allocations': [0.6, 0.2, 0.2],
               'cash_proxy': 'VUSTX'},
    'PMA003': {'symbols': ['VCVSX', 'FAGIX', 'VGHCX'], 'prices': 'yahoo',
               'risk_lookback': 2, 'frequency': 'M', 'allocations': [1. / 3., 1. / 3., 1. / 3.],
               'cash_proxy': 'VUSTX'},
}

strategy_values = pd.DataFrame(columns=strategies.keys())
security_weights = {}
security_holdings = {}
security_prices = {}

for name in strategies:
    if 'PMA' in name:
        s_value, s_holdings, s_weights, s_prices = compute_weights_PMA(name, strategies[name])
    else:
        s_value, s_holdings, s_weights, s_prices = compute_weights_RS_DM(name, strategies[name])

    strategy_values[name] = s_value
    security_weights[name] = s_weights
    security_holdings[name] = s_holdings
    security_prices[name] = s_prices

for filename in [strategy_values, security_weights, security_holdings, security_prices] :
    filename.to_pickle(str(filename) + 'pkl')

index = strategy_values.dropna().index
rebalance_dates = endpoints(period='M', trading_days=index)

# find the set of all portfolio symbols
n = len(strategies)
l = [list(security_weights[name].columns) for name in strategies]
s = []
for i in range(n):
    s = s + l[i]

aggregated_weights = pd.DataFrame(0, index=rebalance_dates, columns=list(set(s)))
all_prices = pd.DataFrame(0, index=index, columns=list(set(s)))

# for equally weighted strategies
strategy_weights = pd.Series([1. / n for i in range(n)], index=list(strategies.keys()))

prices = security_prices.copy()
for name in strategies:
    aggregated_weights[security_weights[name].columns] += security_weights[name].loc[rebalance_dates] * \
                                                          strategy_weights[name]
    all_prices = prices[name].loc[index].combine_first(all_prices)

# equally weighted
p_value, p_holdings, p_weights = backtest(all_prices, aggregated_weights, 10000., offset=0, commission=10.)

p_value.plot(figsize=(15, 10), grid=True)

ffn.calc_perf_stats(p_value).display()