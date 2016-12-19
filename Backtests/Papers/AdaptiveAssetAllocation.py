# http://systematicinvestor.wordpress.com/2012/08/14/adaptive-asset-allocation/

import pandas as pd
from collections import OrderedDict
import time, datetime as dt
import pytz
import matplotlib.pyplot as plt
# import mpld3
# mpld3.enable_notebook()
import numpy as np
from pandas.tseries.offsets import BDay
import talib as t

from backtest_helpers.finhelpers3 import *
from backtest_helpers.mlhelpers3 import *
from backtest_helpers.cla import *

#*****************************************************************
# Load historical data
#******************************************************************

# these are the ETFs used by Systematic Investor (2005 - today)
# tickers = ['SPY','EFA','EWJ','EEM','IYR','RWX','IEF','TLT','DBC','GLD']
# these are the funds used as proxies for the Philbrick results
tickers = ['^GSPC', 'VEURX', 'FJPNX', 'FEMKX', 'FRESX','EGLRX', 'VFITX', 'VUSTX', 'VGPMX', 'GOLDX' ]
data_path = 'C:\\NOTEBOOKS\\Data\\'
start_date = dt.datetime(1990,1,1)
end_date = dt.datetime(2015,1,19)

# NOTE: IF ANY INPUTS CHANGE, DELETE make sure you create new pkl !!!

try :
	data = pd.read_pickle(data_path + 'TAA-long-term.pkl')
except :
    # this uses finlib to load data
    data = get_history(tickers, start_date, end_date, data_path)
    data.major_axis = data.major_axis.tz_localize(pytz.utc)
    data.minor_axis = np.array(['open', 'high', 'low', 'close', 'volume', 'price'], dtype=object)
    data.to_pickle(data_path + 'TAA-long-term.pkl')

inception_dates = pd.DataFrame([data[ticker].first_valid_index().date() for ticker in tickers], index=tickers, columns=['inception'])
print(inception_dates.T[tickers])

# reset start date to latest inception date if it is > start_date

s = max(start_date.date(), inception_dates.values.max())
start_date = dt.datetime(s.year, s.month, s.day)
print(start_date)

data_prices = data.ix[:,:,'price'][start_date:end_date]

#*****************************************************************
# Code Strategies
#******************************************************************
capital = 100000.
prices = data_prices.copy()
n = len(prices.columns)

# can have several portfolios
portfolios = {}

# find period ends
period_ends = endpoints(start_date, end_date, 'M', data_prices.index)

# Adaptive Asset Allocation parameters
n_top = 5       # number of momentum positions
n_mom = 6 * 22    # length of momentum look back
n_vol = 1 * 22    # length of volatility look back

# fix the data by removing NAs
prices = data_prices.dropna()

ret_log = np.log(1. + prices.pct_change())

# *****************************************************************
# Adaptive Asset Allocation (AAA)
# weight positions in the Momentum Portfolio according to
# the minimum variance algorithm
# *****************************************************************

p_name = 'AAA'
portfolios = add_portfolio(p_name, portfolios)

momentum = prices.pct_change(n_mom)
p = momentum.loc[period_ends]

rankings = p.rank(axis=1, ascending=False)
weights = ntop(rankings, n_top)
weights = weights[weights.sum(1) > 0]

index = [d for d in weights.index if d in period_ends]
w_mv = pd.DataFrame(0., index=index, columns=weights.columns)
for i in range(len(index)):
    if weights.iloc[i].sum() != 0:
        symbols = [column for column in weights.columns if weights[column].ix[i] > 0]
        idx = ret_log.index.searchsorted(period_ends[i])
        hist = ret_log.ix[idx - n_vol + 1:idx + 1][symbols]

        ia = create_historical_ia(symbols, hist)

        s0 = ia['std_deviation']
        mu_vec = pd.DataFrame(ia['expected_return'])
        sigma_mat = ia['correlation'] * pd.DataFrame(s0).dot(pd.DataFrame(s0).T)

        mean = mu_vec.values
        lB = np.array([[0. for j in range(len(mu_vec))]]).T
        uB = np.array([[1. for j in range(len(mu_vec))]]).T
        covar = sigma_mat.values
        # print (mean, covar, lB, uB)

        cla = CLA(mean, covar, lB, uB)
        cla.solve()
        w_mv.ix[i] = (w_mv.ix[i] + pd.Series(cla.getMinVar()[1].T[0], index=symbols).T).fillna(0)
        print(w_mv.index[i], '\n', w_mv.ix[i])

p_value, p_holdings, p_weights = backtest(prices, w_mv, capital, offset=1, commission=0.)
print_stats(p_value)