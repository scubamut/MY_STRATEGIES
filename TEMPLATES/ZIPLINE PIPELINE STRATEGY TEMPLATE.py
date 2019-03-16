"""
This is a template algorithm on Zipline for you to adapt and fill in.
"""

from zipline.api import attach_pipeline, pipeline_output
from zipline import run_algorithm
from zipline.api import symbols, get_datetime, schedule_function
from zipline.utils.events import date_rules, time_rules
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.filters import StaticAssets
from datetime import datetime
import pytz


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    # Rebalance every day, 1 hour after market open.
    schedule_function(
        rebalance,
        date_rules.month_end(),
        time_rules.market_open(hours=1)
    )

    # Record tracking variables at the end of each day.
    schedule_function(
        record_vars,
        date_rules.every_day(),
        time_rules.market_close(),
    )

    # Create our dynamic stock selector.
    print('ATTACH PIPELINE')
    attach_pipeline(make_pipeline(), 'pipeline')
    print('PIPELINE ATTACHED')


def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    base_universe = StaticAssets(symbols('XLY', 'XLP', 'XLE', 'XLF', 'XLV',
                                         'XLI', 'XLB', 'XLK', 'XLU'))

    # Factor of yesterday's close price.
    yesterday_close = USEquityPricing.close.latest

    pipeline = Pipeline(
        columns={
            'close': yesterday_close,
        },
        screen=base_universe
    )
    return pipeline


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    print('GET PIPELINE')
    context.output = pipeline_output('pipeline')
    print('PIPELINE OUTPUT')
    print(context.output)

    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index
    print('SECURITY LIST : ', context.security_list)


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    print('REBALANCE - DATE', get_datetime())
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

    print(result[:3])