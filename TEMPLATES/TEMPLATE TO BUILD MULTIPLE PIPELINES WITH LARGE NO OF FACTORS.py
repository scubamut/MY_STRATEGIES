# https://www.quantopian.com/posts/alpha-combination-via-clustering
# adapted for Zipline

"""
This is a template algorithm on Zipline for you to adapt and fill in.
"""

from zipline.api import attach_pipeline, pipeline_output
from zipline import run_algorithm
from zipline.api import symbols, get_datetime, schedule_function
from zipline.utils.events import date_rules, time_rules
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.factors import CustomFactor, Returns, DailyReturns
from zipline.pipeline.filters import StaticAssets
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn import preprocessing
from datetime import datetime
import pytz

WINDOW_LENGTH = 5
WIN_LIMIT = 0


# flag used for first WINDOW_LENGTH days, where the algo is "only" innitialising buffers. One can avoid that using a second pipeline, which is call only at initialization and compute the alphas for the entire window... But I have not yet found a good solution for this!

def preprocess(a):
    a = a.astype(np.float64)
    a[np.isinf(a)] = np.nan
    a = np.nan_to_num(a - np.nanmean(a))
    a = winsorize(a, limits=[WIN_LIMIT, WIN_LIMIT])

    return preprocessing.scale(a)


def make_factor():
    class Direction(CustomFactor):
        inputs = [USEquityPricing.open, USEquityPricing.close]
        window_length = 21
        window_safe = True

        def compute(self, today, assets, out, open, close):
            p = (close - open) / close
            out[:] = preprocess(np.nansum(-p, axis=0))

    class mean_rev(CustomFactor):
        inputs = [USEquityPricing.high, USEquityPricing.low, USEquityPricing.close]
        window_length = 30
        window_safe = True

        def compute(self, today, assets, out, high, low, close):
            p = (high + low + close) / 3

            m = len(close[0, :])
            n = len(close[:, 0])

            b = np.zeros(m)
            a = np.zeros(m)

            for k in range(10, n + 1):
                price_rel = np.nanmean(p[-k:, :], axis=0) / p[-1, :]
                wt = np.nansum(price_rel)
                b += wt * price_rel
                price_rel = 1.0 / price_rel
                wt = np.nansum(price_rel)
                a += wt * price_rel

            out[:] = preprocess(b - a)

    factors = {
        'Direction': Direction,
        'mean_rev': mean_rev
    }

    return factors


class Factor_N_Days_Ago(CustomFactor):

    def compute(self, today, assets, out, input_factor):
        out[:] = input_factor[0]


def initialize(context):
    """
    Called once at the start of the algorithm.
    """

    c = context

    c.etf_universe = StaticAssets(symbols('XLY', 'XLP', 'XLE', 'XLF', 'XLV',
                                          'XLI', 'XLB', 'XLK', 'XLU'))
    c.alphas = pd.DataFrame()

    # Rebalance every day, 1 hour after market open.
    schedule_function(
        rebalance,
        date_rules.every_day(),
        time_rules.market_open(hours=1),
    )

    # Record tracking variables at the end of each day.
    schedule_function(
        record_vars,
        date_rules.every_day(),
        time_rules.market_close(),
    )

    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(context), 'pipeline')
    attach_pipeline(make_pipeinit(context), 'pipeinit')

    c.first_trading_day = True
    c.factor_name_list = make_factor().keys()


def make_pipeinit(context):
    universe = context.etf_universe
    factors = make_factor()

    pipeline_columns = {}
    for f in factors.keys():
        for days_ago in reversed(range(WINDOW_LENGTH)):
            pipeline_columns[f + '-' + str(days_ago)] = Factor_N_Days_Ago([factors[f](mask=universe)],
                                                                          window_length=days_ago + 1,
                                                                          mask=universe)

    pipe = Pipeline(columns=pipeline_columns,
                    screen=universe)

    return pipe


def make_pipeline(context):
    universe = context.etf_universe
    all_factors = make_factor()
    factors = {a: all_factors[a]() for a in all_factors}
    pipe = Pipeline(
        columns=factors,
        screen=universe
    )
    return pipe


def before_trading_start(context, data):
    if context.first_trading_day == True:
        df = (pipeline_output("pipeinit")).astype('float32')
        df = df.stack()
        df.index.names = ['stock', 'alphas']
        df = df.reset_index(level=['alphas', 'stock'])
        alphaname = np.empty(df['alphas'].values.size, dtype='object')
        dayaname = np.empty(df['alphas'].values.size, dtype='int')
        for i, a in enumerate(df['alphas'].values):
            pos = a.find('-')
            alphaname[i] = a[:pos]
            dayaname[i] = a[pos + 1:]

        df['factor'] = pd.Series(alphaname, index=df.index)
        df['day'] = pd.Series(dayaname, index=df.index)
        df = df.drop('alphas', axis=1)
        df = df.set_index(['stock', 'factor', 'day'])
        df = df[0]
        df = df.unstack(level=2)
        context.alphas = df

        context.first_trading_day = False
    else:
        df = (pipeline_output("pipeline")).astype('float32')
        df = df.stack().to_frame()
        df.index.names = ['stock', 'factor']
        context.alphas = context.alphas.drop([4], axis=1)
        context.alphas.columns = [1, 2, 3, 4]
        context.alphas = pd.concat([df, context.alphas], axis=1)


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    pass


def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    pass


def handle_data(context, data):
    """
    Called every minute.
    """
    pass


if __name__ == "__main__":
    start = datetime(2013, 1, 1, 0, 0, 0, 0, pytz.utc)
    end = datetime(2013, 1, 10, 0, 0, 0, 0, pytz.utc)
    #     end = datetime.today().replace(tzinfo=timezone.utc)
    capital_base = 100000

    result = run_algorithm(start=start, end=end, initialize=initialize, \
                           capital_base=capital_base, \
                           before_trading_start=before_trading_start,
                           bundle='etfs_bundle')

    print(result[-3:])