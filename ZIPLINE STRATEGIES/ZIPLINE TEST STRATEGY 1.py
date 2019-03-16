
from zipline import run_algorithm
from zipline.api import (symbols,attach_pipeline,pipeline_output,schedule_function)
from zipline.utils.events import date_rules, time_rules
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.factors import Returns, AnnualizedVolatility
from datetime import datetime
import pytz

WEIGHT1 = WEIGHT2 = WEIGHT3 = WEIGHT4 = WEIGHT5 = 1.0

def initialize(context):
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
    my_pipe = make_pipeline(context)

    # context.universe = StaticAssets(etfs)
    attach_pipeline(my_pipe, 'my_pipeline')

    schedule_function(func=rebalance,
                      date_rule=date_rules.month_end(days_offset=2),
                      # date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_close(minutes=30))


def handle_data(context, data):
    pass


def make_pipeline(context):
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

    pipe = Pipeline(columns=pipe_columns,screen=universe)

    return pipe


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = pipeline_output('my_pipeline')
    print(context.output)


def rebalance(context, data):
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

    print(result[:3])