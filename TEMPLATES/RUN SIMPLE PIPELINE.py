from datetime import datetime, timezone, timedelta
import pytz

from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.utils.calendars import get_calendar
from zipline.data.bundles import register,load
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.engine import SimplePipelineEngine
from fintools.make_pipeline_engine import make_pipeline_engine

etfs = [
        # ----------------------------------------- #
        # SPDRS/State Street Global Advisors (SSGA)
         'XLY' , # Select SPDR U.S. Consumer Discretionary
         'XLP' , # Select SPDR U.S. Consumer Staples
         'XLE' , # Select SPDR U.S. Energy
         'XLF' , # Select SPDR U.S. Financials
         'XLV' , # Select SPDR U.S. Healthcare
         'XLI' , # Select SPDR U.S. Industrials
         'XLB' , # Select SPDR U.S. Materials
         'XLK' , # Select SPDR U.S. Technology
         'XLU' , # Select SPDR U.S. Utilities
         ]

def make_pipeline(assets):
    pipe = Pipeline(
    columns={
        'price': USEquityPricing.close.latest,
    },
    screen=StaticAssets(assets)
                )
    return pipe

start = datetime(2016, 1, 5, 0, 0, 0, 0, pytz.utc)
end = datetime(2016, 1, 7, 0, 0, 0, 0, pytz.utc)
# pipeline engine, Equity() assets
assets, engine = make_pipeline_engine(symbols=etfs, bundle='etfs_bundle')
#run pipeline
pipeline_output = engine.run_pipeline(make_pipeline(assets),start,end)
print(pipeline_output[:5])