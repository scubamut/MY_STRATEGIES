import ffn
import os

from backtest_helpers.compute_weights_PMA import compute_weights_PMA

os.chdir('C:\\users\\scuba\\pycharmprojects\\simplebacktester')
os.getcwd()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

strategies = {'PMA002': {'symbols': ['VCVSX', 'VWINX', 'VWEHX'], 'prices': 'yahoo',
               'risk_lookback': 3, 'frequency': 'M', 'allocations': [0.6, 0.2, 0.2],
               'cash_proxy': 'VUSTX'}}

for name in strategies:
    s_value, s_holdings, s_weights, s_prices = compute_weights_PMA(name, strategies[name])

s_value.plot(figsize=(15, 10), grid=True)

ffn.calc_perf_stats(s_value).display()

pass