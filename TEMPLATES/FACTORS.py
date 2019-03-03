from six import viewkeys
from my_zipline.api import (
    symbol, symbols,
    get_datetime,
    attach_pipeline,
    order, order_target_percent,
    pipeline_output,
    record,
    schedule_function,
)

from my_zipline.utils.calendars import get_calendar
from my_zipline.utils.run_algo import run_algorithm
from my_zipline.utils.events import date_rules, time_rules
from my_zipline.algorithm import TradingAlgorithm
from my_zipline.data.data_portal import DataPortal
from my_zipline.pipeline.loaders import USEquityPricingLoader
from my_zipline.data.bundles.core import register, load
from my_zipline.pipeline.engine import SimplePipelineEngine
from my_zipline.pipeline import Pipeline
from my_zipline.pipeline.data import USEquityPricing
from my_zipline.pipeline.filters import StaticAssets
from my_zipline.pipeline.factors import RSI, CustomFactor
import os
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, timezone
import pytz
import numpy as np
import xarray as xr
import statsmodels.api as sm

from fintools.get_DataArray import get_DataArray

import matplotlib as plt


'''
    Fundamentals, Factors ...
'''


def nanfill(
        _in):  # From https://stackoverflow.com/questions/41190852/most-efficient-way-to-forward-fill-nan-values-in-numpy-array
    # Includes a way to count nans on webpage at
    #   https://www.quantopian.com/posts/forward-filling-nans-in-pipeline

    # return _in            # uncomment to not run the code below
    mask = np.isnan(_in)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    _in[mask] = _in[np.nonzero(mask)[0], idx[mask]]
    return _in


def beta(ts, benchmark, benchmark_var):
    return np.cov(ts, benchmark)[0, 1] / benchmark_var


def slope(in_):  # Slope of regression line. Make sure input has no nans or screen its output later
    # https://www.quantopian.com/posts/slope-calculation
    return sm.OLS(in_, sm.add_constant(range(-len(in_) + 1, 1))).fit().params[-1]  # slope


def curve(_in):  # ndarray   see https://www.quantopian.com/posts/curve-calculation
    return sm.OLS(_in[-len(_in) / 2:], sm.add_constant(range(-len(_in[-len(_in) / 2:]) + 1, 1))).fit().params[-1] - \
           sm.OLS(_in[0:len(_in) / 2], sm.add_constant(range(-len(_in[0:len(_in) / 2]) + 1, 1))).fit().params[-1]


class AroonUp(CustomFactor):
    window_length = 100
    inputs = [USEquityPricing.high]

    def compute(self, today, assets, out, highs):
        out[:] = (np.argmax(highs, 0).astype(float) / float(self.window_length)) * 100.0


class AroonDown(CustomFactor):
    window_length = 100
    inputs = [USEquityPricing.low]

    def compute(self, today, assets, out, lows):
        out[:] = (np.argmin(lows, 0).astype(float) / float(self.window_length)) * 100.0


class AvgDailyDollarVolumeTraded(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.volume];
    window_length = 42

    def compute(self, today, assets, out, close, volume):
        volume = nanfill(volume)
        close = nanfill(close)
        out[:] = np.mean(close * volume, axis=0)


class ATR(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.high, USEquityPricing.low]
    window_length = 21

    def compute(self, today, assets, out, close, high, low):
        close = nanfill(close)
        high = nanfill(high)
        low = nanfill(low)
        hml = high - low
        hmpc = np.abs(high - np.roll(close, 1, axis=0))
        lmpc = np.abs(low - np.roll(close, 1, axis=0))
        tr = np.maximum(hml, np.maximum(hmpc, lmpc))
        atr = np.mean(tr[1:], axis=0)
        out[:] = atr


class Beta(CustomFactor):
    inputs = [USEquityPricing.close];
    window_length = 60

    def compute(self, today, assets, out, close):
        close = nanfill(close)
        returns = pd.DataFrame(close, columns=assets).pct_change()[1:]
        spy_returns = returns[symbol('SPY')]
        spy_returns_var = np.var(spy_returns)
        out[:] = returns.apply(beta, args=(spy_returns, spy_returns_var,))


class CrossSectionalMomentum(CustomFactor):
    inputs = [USEquityPricing.close];
    window_length = 252

    def compute(self, today, assets, out, closes):
        closes = nanfill(closes)
        closes = pd.DataFrame(closes)
        R = (closes / closes.shift(100))
        out[:] = (R.T - R.T.mean()).T.mean()


class Curve(CustomFactor):
    inputs = [USEquityPricing.close];
    window_length = 6

    def compute(self, today, assets, out, closes):
        closes = nanfill(closes)
        out[:] = curve(closes)


class Downward(CustomFactor):
    inputs = [USEquityPricing.close];
    window_length = 5

    def compute(self, today, assets, out, close):
        close = nanfill(close)
        ratio_avg = (close[-1] / np.mean(close, axis=0))
        out[:] = ((close[-1] / close[0]) + ratio_avg)


class EfficiencyRatio(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.high, USEquityPricing.low]
    window_length = 30

    def compute(self, today, assets, out, close, high, low):
        lb = self.window_length
        e_r = np.zeros(len(assets), dtype=np.float64)
        a = np.array(([high[1:(lb):1] - low[1:(lb):1], abs(high[1:(lb):1] - close[0:(lb - 1):1]),
                       abs(low[1:(lb):1] - close[0:(lb - 1):1])]))
        b = a.T.max(axis=1)
        c = b.sum(axis=1)
        e_r = abs(close[-1] - close[0]) / c
        out[:] = e_r


class MACD(CustomFactor):
    inputs = [USEquityPricing.close]
    window_length = 60

    def ema(self, data, window):  # Initial value for EMA is taken as trialing SMA
        import numpy as np
        c = 2.0 / (window + 1)
        ema = np.mean(data[-(2 * window) + 1:-window + 1], axis=0)
        for value in data[-window + 1:]:
            ema = (c * value) + ((1 - c) * ema)
        return ema

    def compute(self, today, assets, out, close):
        close = nanfill(close)
        fema = self.ema(close, 12)
        sema = self.ema(close, 26)
        macd_line = fema - sema
        macd = []
        macd.insert(0, self.ema(close, 12) - self.ema(close, 26))
        for i in range(1, 15, 1):
            macd.insert(0, self.ema(close[:-i], 12) - self.ema(close[:-i], 26))
        signal = self.ema(macd, 9)
        out[:] = macd_line - signal


class MaxGap(CustomFactor):  # the biggest absolute overnight gap in the previous 90 sessions
    inputs = [USEquityPricing.close];
    window_length = 90

    def compute(self, today, assets, out, close):
        close = nanfill(close)
        abs_log_rets = np.abs(np.diff(np.log(close), axis=0))
        max_gap = np.max(abs_log_rets, axis=0)
        out[:] = max_gap


class MedianValue(CustomFactor):
    inputs = [USEquityPricing.close];
    window_length = 42

    def compute(self, today, assets, out, close):
        close = nanfill(close)
        out[:] = np.nanmedian(close, axis=0)


class Momentum(CustomFactor):
    inputs = [USEquityPricing.close];
    window_length = 20

    def compute(self, today, assets, out, close):
        close = nanfill(close)
        out[:] = close[-1] / close[0]


class MultipleOutputs(CustomFactor):
    # Define inputs and outputs.
    inputs = [USEquityPricing.close]
    # Specify and name the different outputs.
    outputs = ['highs', 'lows']
    window_length = 10

    def compute(self, today, assets, out, close):
        highs = np.nanmax(close, axis=0)
        lows = np.nanmin(close, axis=0)
        # Write the desired return values into `out.<output_name>` for each output name in `self.outputs`.
        out.highs[:] = highs
        out.lows[:] = lows


class NormalizedRelativeStrengthOscillator(CustomFactor):
    """
    Compute the following:
    Normalized Relative Strength Oscillator = ((% change in stock + 1) / (% change in benchmark + 1) - 1) * 100
    """
    params = ('market_sid',)
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, close, market_sid):
        nrsoRankTable = pd.DataFrame(index=assets)

        returns = (close[-1] - close[0]) / close[0]
        market_idx = assets.get_loc(market_sid)
        nrsoRankTable["NRSO"] = (((returns + 1) / (returns[market_idx] + 1)) - 1) * 100

        out[:] = nrsoRankTable["NRSO"].rank(ascending=False)


class RelativeStrength(CustomFactor):
    """
    Compute the following:
    relative strength = ( (% change in price in stock / % change in benchmark) - 1) * 100
    """
    params = ('market_sid',)
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, close, market_sid):
        rsRankTable = pd.DataFrame(index=assets)

        returns = (close[-1] - close[0]) / close[0]
        market_idx = assets.get_loc(market_sid)
        rsRankTable["RS"] = (((returns + 1) / (returns[market_idx] + 1)) - 1) * 100

        out[:] = rsRankTable["RS"].rank(ascending=False)


class Returns(CustomFactor):
    """
    this factor outputs the returns over the period defined by
    business_days, ending on the previous trading day, for every security.
    """
    window_length = 20
    inputs = [USEquityPricing.close]

    def compute(self, today, assets, out, price):
        out[:] = (price[-1] - price[0]) / price[0] * 100


class Slope(CustomFactor):
    inputs = [USEquityPricing.close];
    window_length = 10

    def compute(self, today, assets, out, closes):
        closes = nanfill(closes)
        out[:] = slope(closes)


class TenDayRange(CustomFactor):
    """
    Computes the difference between the highest high in the last 10
    days and the lowest low.

    Pre-declares high and low as default inputs and `window_length` as
    10.
    """

    inputs = [USEquityPricing.high, USEquityPricing.low]
    window_length = 10

    def compute(self, today, assets, out, highs, lows):
        from numpy import nanmin, nanmax

        highest_highs = nanmax(axis=0)
        lowest_lows = nanmin(axis=0)
        out[:] = highest_highs - lowest_lows


class Volatility1(CustomFactor):
    inputs = [USEquityPricing.close];
    window_length = 252

    def compute(self, today, assets, out, close):
        close = nanfill(close)
        close = pd.DataFrame(data=close, columns=assets)
        # Rank largest is best, need to invert the sdev.
        out[:] = 1 / np.log(close).diff().std()


class Volatility2(CustomFactor):
    inputs = [USEquityPricing.close];
    window_length = 252

    def compute(self, today, assets, out, close):
        close = nanfill(close)
        close = pd.DataFrame(data=close, columns=assets)
        # Rank largest is best, need to invert the sdev.
        out[:] = np.log(close).diff().std()


class Volatility3(CustomFactor):
    inputs = [USEquityPricing.close];
    window_length = 122

    def compute(self, today, assets, out, close):
        close = nanfill(close)
        # 6-month volatility, starting before the five-day mean reversion period
        daily_returns = np.log(close[1:-6]) - np.log(close[0:-7])
        out[:] = daily_returns.std(axis=0)


class VolumeMinimum(CustomFactor):
    inputs = [USEquityPricing.volume];
    window_length = 42

    def compute(self, today, assets, out, volume):
        volume = nanfill(volume)
        out[:] = np.min(np.array(volume), axis=0)  # .astype(int)


class VolumeMin(CustomFactor):
    inputs = [USEquityPricing.volume];
    window_length = 42

    def compute(self, today, assets, out, volume):
        volume = nanfill(volume)
        out[:] = np.min(volume, axis=0)  # & VolumeMin().top(200)


class VolumeMax(CustomFactor):
    inputs = [USEquityPricing.volume];
    window_length = 42

    def compute(self, today, assets, out, volume):
        volume = nanfill(volume)
        out[:] = np.max(volume, axis=0)


class VolumeMean(CustomFactor):
    inputs = [USEquityPricing.volume];
    window_length = 42

    def compute(self, today, assets, out, volume):
        volume = nanfill(volume)
        out[:] = np.mean(volume, axis=0)
