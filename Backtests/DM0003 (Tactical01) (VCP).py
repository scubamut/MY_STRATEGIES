import ffn
import os

from backtest_helpers.compute_weights_RS_DM import compute_weights_RS_DM

os.chdir('C:\\users\\scuba\\pycharmprojects\\simplebacktester')
os.getcwd()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

strategies = {
    'DM0003': {'symbols': ['VCVSX','VGHCX', 'VWEHX','VFIIX','DRGIX','VWAHX'], 'prices': 'yahoo',
               'rs_lookback': 1, 'risk_lookback': 1, 'n_top': 2, 'frequency': 'M',
               'cash_proxy': 'VFIIX', 'risk_free': 0}}

for name in strategies:
    s_value, s_holdings, s_weights, s_prices = compute_weights_RS_DM(name, strategies[name])

s_value.plot(figsize=(15, 10), grid=True)

ffn.calc_perf_stats(s_value).display()

pass