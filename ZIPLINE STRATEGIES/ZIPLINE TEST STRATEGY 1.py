from six import viewkeys
from zipline.api import (
    symbol, symbols,
    get_datetime,
    attach_pipeline,
    order, order_target_percent,
    pipeline_output,
    record,
    schedule_function,
)

from zipline.utils.events import date_rules, time_rules
from zipline.algorithm import TradingAlgorithm
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.data.bundles.core import load
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.factors import RSI, Returns, AnnualizedVolatility
import zipline
import os
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, timezone
import pytz
import numpy as np
import xarray as xr

from fintools.get_DataArray import get_DataArray

import matplotlib as plt
# %matplotlib inline

WEIGHT1 = WEIGHT2 = WEIGHT3 = WEIGHT4 = WEIGHT5 = 1.0


def initialize(context, data):
    context.universe = StaticAssets(symbols(
        'XLY',  # Select SPDR U.S. Consumer Discretionary
        'XLP',  # Select SPDR U.S. Consumer Staples
        'XLE',  # Select SPDR U.S. Energy
        'XLF',  # Select SPDR U.S. Financials
        'XLV',  # Select SPDR U.S. Healthcare
        'XLI',  # Select SPDR U.S. Industrials
        'XLB',  # Select SPDR U.S. Materials
        'XLK',  # Select SPDR U.S. Technology
        'XLU',  # Select SPDR U.S. Utilities
    ))

    #     my_pipe = Pipeline()
    my_pipe = make_pipeline(context, data)

    # context.universe = StaticAssets(etfs)
    attach_pipeline(my_pipe, 'my_pipeline')

    schedule_function(func=rebalance,
                      date_rule=date_rules.month_end(days_offset=2),
                      # date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=30))


def handle_data(context, data):
    pass


def make_pipeline(context, data):
    '''
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    '''
    universe = context.universe

    # Factor of yesterday's close price.
    day1mo_ret = Returns(inputs=[USEquityPricing.close], window_length=21, mask=universe)
    day3mo_ret = Returns(inputs=[USEquityPricing.close], window_length=63, mask=universe)
    day6mo_ret = Returns(inputs=[USEquityPricing.close], window_length=126, mask=universe)
    day9mo_ret = Returns(inputs=[USEquityPricing.close], window_length=189, mask=universe)
    day1yr_ret = Returns(inputs=[USEquityPricing.close], window_length=252, mask=universe)

    volatility = AnnualizedVolatility(mask=universe)
    score = ((WEIGHT1 * day1mo_ret) + (WEIGHT2 * day3mo_ret) + (WEIGHT3 * day6mo_ret)
             + (WEIGHT3 * day9mo_ret) + (WEIGHT5 * day1yr_ret)) / (volatility)

    high = USEquityPricing.high.latest
    low = USEquityPricing.low.latest
    open_price = USEquityPricing.open.latest
    close = USEquityPricing.close.latest
    volume = USEquityPricing.volume.latest

    pipe_columns = {
        'Score': score,
        'Day1mo': day1mo_ret,
        'high': high,
        'low': low,
        'close': close,
        'open_price': open_price,
        'volume': volume,
    }

    pipe = Pipeline()
    pipe = Pipeline(columns=pipe_columns)
    pipe.set_screen = universe

    return pipe


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('my_pipeline')
    print(context.output)


def rebalance(context, data):
    pass
