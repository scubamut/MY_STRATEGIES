import pandas as pd
import itable
import ffn
import talib
import pandas as pd
import ffn
import os

from backtest_helpers.compute_weights_RS_DM import compute_weights_RS_DM
from backtest_helpers.compute_weights_PMA import compute_weights_PMA
from backtest_helpers.monthly_return_table import monthly_return_table
from backtest_helpers.endpoints import endpoints
from backtest_helpers.backtest import backtest

def side_by_side(*objs, **kwds):
    from pandas.formats.printing import adjoin
    space = kwds.get('space', 4)
    reprs = [repr(obj).split('\n') for obj in objs]
    print(adjoin(space, *reprs))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

os.chdir('C:\\Users\\scuba\\Google Drive\\PycharmProjects\\SimpleBacktester')
os.getcwd()


strategies = {
    'DM0003': {'symbols': ['VCVSX', 'VWEHX', 'VFIIX', 'FGOVX', 'VWAHX'], 'prices': 'yahoo',
               'start': '1986-01-01', 'end': 'today',
               'rs_lookback': 1, 'risk_lookback': 1, 'n_top': 2, 'frequency': 'M',
               'cash_proxy': 'CASHX', 'risk_free': 0},
    'RS0002': {'symbols': ['MMHYX', 'FAGIX', 'VFIIX'], 'prices': 'yahoo',
               'start': '1986-01-01', 'end': 'today',
               'rs_lookback': 3, 'risk_lookback': 2, 'n_top': 1, 'frequency': 'M',
               'cash_proxy': 'CASHX', 'risk_free': 0},
    'RS0003': {'symbols': ['MMHYX', 'FAGIX', 'VFIIX'], 'prices': 'yahoo',
               'start': '1986-01-01', 'end': 'today',
               'rs_lookback': 1, 'risk_lookback': 1, 'n_top': 1, 'frequency': 'Q',
               'cash_proxy': 'CASHX', 'risk_free': 0},
    'DM0001': {'symbols': ['VCVSX', 'VWINX', 'VWEHX', 'VGHCX', 'VUSTX', 'VFIIX', 'VWAHX', 'FGOVX', 'FFXSX'],
               'prices': 'yahoo', 'start': '1986-01-01', 'end': 'today',
               'rs_lookback': 1, 'risk_lookback': 1, 'n_top': 3, 'frequency': 'M',
               'cash_proxy': 'CASHX', 'risk_free': 'FFXSX'},
    'DM0002': {'symbols': ['VCVSX', 'VUSTX', 'VWEHX', 'VFIIX', 'VGHCX', 'FRESX'], 'prices': 'yahoo',
               'start': '1986-01-01', 'end': 'today',
               'rs_lookback': 1, 'risk_lookback': 1, 'n_top': 5, 'frequency': 'M',
               'cash_proxy': 'VFIIX', 'risk_free': 'FFXSX'},
    'PMA001': {'symbols': ['VCVSX', 'VFIIX'], 'prices': 'yahoo',
               'start': '1986-01-01', 'end': 'today',
               'risk_lookback': 3, 'frequency': 'M', 'allocations': [0.6, 0.4],
               'cash_proxy': 'VUSTX'},
    'PMA002': {'symbols': ['VCVSX', 'VWINX', 'VWEHX'], 'prices': 'yahoo',
               'start': '1986-01-01', 'end': 'today',
               'risk_lookback': 3, 'frequency': 'M', 'allocations': [0.6, 0.2, 0.2],
               'cash_proxy': 'VUSTX'},
    'PMA003': {'symbols': ['VCVSX', 'FAGIX', 'VGHCX'], 'prices': 'yahoo',
               'start': '1986-01-01', 'end': 'today',
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

##################################################
# DUAL MOMENTUM
###############

strategy_prices = strategy_values.dropna().copy()
# need to add prices for cash_proxy and, if necessary, risk_free

from backtest_helpers.get_history_prices import get_history_prices
strategy_prices['FFXSX'] = get_history_prices(['FFXSX']).loc[index]

strategies1 = {
    'MUTLTI-RS': { 'symbols': list(strategies.keys()), 'prices': strategy_prices,
               'rs_lookback': 1, 'risk_lookback': 1, 'n_top': 8, 'frequency': 'm',
              'cash_proxy': 'FFXSX', 'risk_free': 0}}

for name in strategies1 :
    s_value, s_holdings, s_weights, s_prices =  compute_weights_RS_DM (name, strategies1[name])

# poorer return but lower drawdown and better SR
ffn.calc_perf_stats(s_value).display()
########################################################################################################
# This to calculate the orders at each rebalance date
#####################################################◙

# get weights from backtest
strategy_weights = s_weights.loc[rebalance_dates].copy()
aggregated_weights = pd.DataFrame(0, index=rebalance_dates, columns=list(set(s + ['FFXSX'])))
all_prices = pd.DataFrame(0, index=index, columns=list(set(s + ['FFXSX'])))

prices = security_prices.copy()
for name in strategies:
    aggregated_weights[security_weights[name].columns] += security_weights[name].loc[rebalance_dates].mul(
        strategy_weights[name], axis=0)
    all_prices = prices[name].loc[index].combine_first(all_prices)

# need to add in the cash_proxy weights
aggregated_weights['FFXSX'] = aggregated_weights['FFXSX'].add(strategy_weights['FFXSX'], axis=0)

from backtest_helpers.backtest import backtest
aggregated_weights = aggregated_weights[aggregated_weights.sum(1) > 0]
values, holdings, weights = backtest(all_prices, aggregated_weights, 10000., offset=0, commission=10.)

transactions = (holdings - holdings.shift(1).fillna(0))
transactions = transactions[transactions.sum(1) != 0]

def generate_orders(transactions, prices) :
    orders = pd.DataFrame()
    for i in range(len(transactions)):
        for j in range(len(transactions.columns)):
            t = transactions.ix[i]
            qty = abs(t[j])
            if qty >= 1.:
                if transactions.ix[i][j] < 0 :
                    orders = orders.append([[t.name.date().year, t.name.date().month, t.name.date().day, t.index[j],\
                                             'Sell', -abs(t[j]), prices.ix[t.name][t.index[j]]]])
                if transactions.ix[i][j] > 0 :
                    orders = orders.append([[t.name.date().year, t.name.date().month, t.name.date().day, t.index[j],\
                                             'Buy', abs(t[j]), prices.ix[t.name][t.index[j]]]])
    orders.columns = ['Year', 'Month', 'Day', 'Symbol', 'Action', 'Qty', 'Price']
    orders
    return orders

orders = generate_orders(transactions, all_prices)
