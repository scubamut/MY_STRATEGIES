import numpy as np
import pandas as pd
from datetime import datetime
import pytz

from my_zipline.pipeline.data import Column
from my_zipline.pipeline.data import DataSet
from my_zipline.api import symbols
from my_zipline.pipeline.loaders import USEquityPricingLoader
from my_zipline.utils.calendars import get_calendar
from my_zipline.data.bundles import register, load
from my_zipline.pipeline import Pipeline
from my_zipline.pipeline.data import USEquityPricing
from my_zipline.pipeline.factors import Returns, AnnualizedVolatility
from my_zipline.pipeline.filters import StaticAssets
from my_zipline.pipeline.engine import SimplePipelineEngine
from my_zipline.pipeline.loaders.frame import DataFrameLoader
from fintools.make_pipeline_engine import make_pipeline_engine


def make_pipeline(assets):

    WEIGHT1 = 1.0
    WEIGHT2 = 1.0
    WEIGHT3 = 1.0
    WEIGHT4 = 1.0

    etf_universe = StaticAssets(assets)

    day20_ret = Returns(inputs=[USEquityPricing.close], window_length=21, mask=etf_universe)
    day3mo_ret = Returns(inputs=[USEquityPricing.close], window_length=63, mask=etf_universe)
    day6mo_ret = Returns(inputs=[USEquityPricing.close], window_length=126, mask=etf_universe)
    day1yr_ret = Returns(inputs=[USEquityPricing.close], window_length=252, mask=etf_universe)

    volatility = AnnualizedVolatility(mask=etf_universe)
    score = ((WEIGHT1 * day20_ret) + (WEIGHT1 * day3mo_ret) + (WEIGHT3 * day6mo_ret) + (WEIGHT3 * day1yr_ret)) / (
        volatility)

    high = USEquityPricing.high.latest
    low = USEquityPricing.low.latest
    open_price = USEquityPricing.open.latest
    close = USEquityPricing.close.latest
    volume = USEquityPricing.volume.latest

    pipe = Pipeline(
        columns={
            'Score': score,
            'Day20': day20_ret,
            'high': high,
            'low': low,
            'close': close,
            'open_price': open_price,
            'volume': volume,
        },
        screen=etf_universe
    )
    return pipe


etfs = [
    # ----------------------------------------- #
    # SPDRS/State Street Global Advisors (SSGA)
    'XLY',  # Select SPDR U.S. Consumer Discretionary
    'XLP',  # Select SPDR U.S. Consumer Staples
    'XLE',  # Select SPDR U.S. Energy
    'XLF',  # Select SPDR U.S. Financials
    'XLV',  # Select SPDR U.S. Healthcare
    'XLI',  # Select SPDR U.S. Industrials
    'XLB',  # Select SPDR U.S. Materials
    'XLK',  # Select SPDR U.S. Technology
    'XLU',  # Select SPDR U.S. Utilities
    'KRE',  # SPDR S&P Regional Banking ETF
    'KBE',  # SPDR S&P Bank ETF
    'XOP',  # SPDR S&P Oil & Gas Explor & Product
    'GLD',  # SPDR Gold Trust
    'SLV',  # SPDR Silver Trust
    'SPY',  # SPDR S&P 500
    'JNK',  # SPDR Barclays Capital High Yield Bond ETF
    'DIA',  # SPDR Dow Jones Industrial Avg. ETF
    'XHB',  # SPDR Homebuilders ETF
    'MDY',  # SPDR S&P MidCap 400 ETF
    'FEZ',  # SPDR Euro Stoxx 50 ETF
    # ----------------------------------------- #
    # iShares
    'AGG',  # iShares Core U.S. Aggregate Bond ETF
    'IAU',  # iShares Gold Trust
    'IXC',  # iShares Global Energy ETF
    'IWR',  # iShares Russell Mid-Cap ETF
    'IWB',  # iShares Russell 1000 ETF
    'IJR',  # iShares Core S&P Small-Cap ETF
    'IJH',  # iShares Core S&P Mid-Cap ETF
    'EWT',  # iShares MSCI Taiwan ETF
    'EEM',  # iShares MSCI Emerging Markets ETF
    'IWM',  # iShares Russell 2000 ETF
    'EWG',  # iShares MSCI Germany ETF
    'EWJ',  # iShares MSCI Japan ETF
    'EFA',  # iShares MSCI EAFE ETF
    'EWZ',  # iShares MSCI Brazil Capped ETF
    'TLT',  # iShares 20+ Year Treasury Bond ETF
    'INDA',  # iShares MSCI India ETF
    'ECH',  # iShares MSCI Chile Capped ETF
    'EWU',  # iShares MSCI United Kingdom ETF
    'EWI',  # iShares MSCI Italy Capped ETF
    'EWP',  # iShares MSCI Spain Capped ETF
    'EWQ',  # iShares MSCI France ETF
    'EWL',  # iShares MSCI Switzerland Capped ETF
    'EWD',  # iShares MSCI Sweden ETF
    'EWT',  # iShares MSCI Taiwan ETF
    'EWY',  # iShares MSCI South Korea Capped ETF
    'EWA',  # iShares MSCI Australia ETF
    'EWS',  # iShares MSCI Singapore ETF
    'IYM',  # iShares Dow Jones U.S. Basic Materials Index
    'IYK',  # iShares Dow Jones U.S. Consumer Goods Index
    'IYC',  # iShares Dow Jones U.S. Consumer Services Index
    'IYE',  # iShares Dow Jones U.S. Energy Index
    'IYF',  # iShares Dow Jones U.S. Financial Sector Index
    'IYG',  # iShares Dow Jones U.S. Financial Services Index
    'IYH',  # iShares Dow Jones U.S. Healthcare Index
    'IYJ',  # iShares Dow Jones U.S. Industrial Index
    'IYR',  # iShares Dow Jones U.S. Real Estate Index
    'IYW',  # iShares Dow Jones U.S. Technology Index
    'IYZ',  # iShares Dow Jones U.S. Telecommunications Index
    'IYT',  # iShares Dow Jones Transportation Average Index
    'IDU',  # iShares Dow Jones U.S. Utilities Index
    'ICF',  # iShares Cohen & Steers Realty Majors Index
    'AAXJ',  # iShares MSCI All Country Asia ex Japan ETF
    'FXI',  # iShares China Large-Cap ETF
    'ACWI',  # iShares MSCI ACWI ETF
    'EZU',  # iShares MSCI Eurozone ETF
    'EWH',  # iShares MSCI Hong Kong ETF
    'EWM',  # iShares MSCI Malaysia ETF
    # ----------------------------------------- #
    # Vanguard
    'VGK',  # Vanguard FTSE Europe ETF
    'VEA',  # Vanguard Developed Market FTSE
    'VPU',  # Vangaurd Utilities ETF
    'VDE',  # Vanguard Energy ETF
    'VEU',  # Vanguard FTSE All-World ex-US ETF
    'VXUS',  # VanguardTotal Int'l Stock ETF
    'VOO',  # Vanguard S&P 500
    'VO',  # Vanguard Mid-Cap ETF
    'VB',  # Vanguard Small-Cap ETF
    'VOX',  # Vanguard Telecom Services ETF
    # ----------------------------------------- #
    # Market Vectors
    'SMH',  # Market Vectors Semiconductor ETF
    'GDX',  # Market Vectors TR Gold Miners
    'OIH',  # Market Vectors Oil Services ETF
    'RSX',  # Market Vectors Russia ETF
    'GDXJ',  # Market Vectors Junior Gold Miners ETF
    # ----------------------------------------- #
    # Powershares (Invesco)
    'QQQ',  # Powershares (Invesco) NASDAQ 100
    # ----------------------------------------- #
    # Uncategorized
    'AMLP',  # Alerian MLP ETF
    #           'HACK' , # Purefunds ISE Cyber Security ETF
    'FDN',  # First Trust Dow Jones Internet Index ETF
    'HEDJ',  # WisdomTree Europe Hedged Equity ETF
    'EPI'  # WisdomTree India Earnings ETF
]
start = datetime(2016, 1, 5, 0, 0, 0, 0, pytz.utc)
end = datetime(2016, 1, 7, 0, 0, 0, 0, pytz.utc)
# pipeline engine, Equity() assets
assets, engine = make_pipeline_engine(symbols=etfs, bundle='etfs_bundle')
# run pipeline
pipeline_output = engine.run_pipeline(make_pipeline(assets), start, end)
print(pipeline_output[:5])