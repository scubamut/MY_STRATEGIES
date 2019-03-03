# https://www.quantopian.com/posts/alpha-combination-via-clustering

"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
import quantopian.algorithm as algo
from quantopian.algorithm import attach_pipeline, pipeline_output
import quantopian.optimize as opt
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.factors import CustomFactor, Returns, DailyReturns
import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn import preprocessing

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

    context.alphas = pd.DataFrame()

    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(hours=1),
    )

    # Record tracking variables at the end of each day.
    algo.schedule_function(
        record_vars,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
    )

    # Create our dynamic stock selector.
    attach_pipeline(make_pipeline(), 'pipeline')
    attach_pipeline(make_pipeinit(), 'pipeinit')

    context.first_trading_day = True
    context.factor_name_list = make_factor().keys()


def make_pipeinit():
    factors = make_factor()

    pipeline_columns = {}
    for f in factors.keys():
        for days_ago in reversed(range(WINDOW_LENGTH)):
            pipeline_columns[f + '-' + str(days_ago)] = Factor_N_Days_Ago([factors[f](mask=QTradableStocksUS())],
                                                                          window_length=days_ago + 1,
                                                                          mask=QTradableStocksUS())

    pipe = Pipeline(columns=pipeline_columns,
                    screen=QTradableStocksUS())

    return pipe


def make_pipeline():
    all_factors = make_factor()
    factors = {a: all_factors[a]() for a in all_factors}
    pipe = Pipeline(
        columns=factors,
        screen=QTradableStocksUS()
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