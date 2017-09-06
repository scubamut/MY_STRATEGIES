# http://seekingalpha.com/article/4026626-defensive-etf-bond-strategy-application?source=all_articles_title

import ffn
import os

from backtest_helpers.compute_weights_RS_DM import compute_weights_RS_DM

os.chdir('C:\\Users\\scuba\\Google Drive\\PycharmProjects\\SimpleBacktester')
os.getcwd()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

strategies = {
    'DM0004': {'symbols': ['HYS', 'MBB', 'HYMB'], 'prices': 'yahoo',
               'start': '1986-01-01', 'end': 'today',
               'rs_lookback': 3, 'risk_lookback': 3, 'n_top': 1, 'frequency': 'M',
               'cash_proxy': 'CASHX', 'risk_free': 0}}

for name in strategies:
    s_value, s_holdings, s_weights, s_prices = compute_weights_RS_DM(name, strategies[name])

s_value.plot(figsize=(15, 10), grid=True)

ffn.calc_perf_stats(s_value).display()

pass