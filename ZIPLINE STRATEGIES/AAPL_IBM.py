# Save this as spy_tlt.py

"""
A simple Pipeline algorithm that longs SPY and TLT each day.
"""
from six import viewkeys
from my_zipline.api import (
    attach_pipeline,
    order_target_percent,
    pipeline_output,
    record,
    schedule_function,
)
from my_zipline.utils.run_algo import run_algorithm
from my_zipline.utils.events import date_rules, time_rules
from my_zipline.utils.events import date_rules, time_rules
from my_zipline.api import get_environment, symbol
from my_zipline.pipeline import Pipeline
from my_zipline.algorithm import TradingAlgorithm
from my_zipline.api import order, record, symbol, order_target_percent, get_datetime, symbols
from my_zipline.api import attach_pipeline, schedule_function
from my_zipline.pipeline.data import USEquityPricing
from my_zipline.pipeline.filters import StaticAssets

import os
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, timezone
import pytz
import numpy as np
import xarray as xr

from fintools.get_DataArray import get_DataArray


# def make_pipeline():
#     """
#     Create our pipeline.
#     """
#     # Set up a filter for two securities AAPL and IBM
#     spy_tlt = StaticAssets([symbol('AAPL'), symbol('IBM')])
#
#     print(symbol('AAPL'), symbols('AAPL'))
#
#     # Create an arbitrary factor
#     price = USEquityPricing.close.latest
#
#     pipe = Pipeline(
#         screen=spy_tlt,
#         columns={
#             'price': price,
#         }
#     )
#
#     return pipe


def rebalance(context, data):
    pipeline_data = context.pipeline_data
    all_assets = pipeline_data.index

    record(universe_size=len(all_assets))

    for asset in all_assets:
        order_target_percent(asset, 0.5)


def initialize(context):

    spy_tlt = StaticAssets([symbol('AAPL'), symbol('IBM')])

    print(symbol('AAPL'), symbols('AAPL'))

    # Create an arbitrary factor
    price = USEquityPricing.close.latest

    pipe = Pipeline(
        screen=spy_tlt,
        columns={
            'price': price,
        }
    )

    attach_pipeline(pipe, 'my_pipeline')

    schedule_function(rebalance, date_rules.every_day())

    pass

def handle_data(context, data):
    pass


def before_trading_start(context, data):
    context.pipeline_data = pipeline_output('my_pipeline')


# RUN THE ALGO
assets = ['AAPL', 'IBM']

start = datetime(2011, 1, 1, 0, 0, 0, 0, pytz.utc)
# end = datetime(2013, 1, 1, 0, 0, 0, 0, pytz.utc)
end = datetime.today().replace(tzinfo=timezone.utc)
da = get_DataArray(assets,start,end)
data = da.to_pandas().transpose(1,2,0)

algo = TradingAlgorithm(initialize=initialize,
                        handle_data=handle_data,
                        before_trading_start=before_trading_start)

algo.run(data)

# HOW TO EXECUTE run_pipeline()????