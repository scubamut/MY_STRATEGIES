# ZIPLINE IMPORTS

import pandas as pd
import numpy as np
import re
import scipy
from collections import OrderedDict
from cvxopt import solvers, matrix, spdiag
import talib
from zipline import TradingAlgorithm
from zipline.api import attach_pipeline, pipeline_output, get_datetime
from zipline import run_algorithm
from zipline.api import set_symbol_lookup_date, order_target_percent, get_open_orders
from zipline.api import order, record, set_commission
from zipline.api import symbol, symbols, get_datetime, schedule_function, get_environment
from zipline.finance import commission
from zipline.utils.events import date_rules, time_rules
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.filters import StaticAssets
from datetime import datetime, timezone
import pytz

# CONSTANTS

GTC_LIMIT = 10
VALID_PORTFOLIO_ALLOCATION_MODES = ['EW','FIXED','PROPORTIONAL','MIN_VARIANCE','MAX_SHARPE',
                                    'BY_FORMULA', 'RISK_PARITY','VOLATILITY_WEIGHTED','RISK_TARGET', 'MIN_CORRELATION']
VALID_STRATEGY_ALLOCATION_MODES = ['EW','FIXED','MIN_VARIANCE','MAX_SHARPE', 'BRUTE_FORCE_SHARPE',
                                    'BY_FORMULA', 'RISK_PARITY','VOLATILITY_WEIGHTED','RISK_TARGET', 'MIN_CORRELATION']
VALID_PORTFOLIO_ALLOCATION_FORMULAS = [None]
VALID_SECURITY_SCORING_METHODS = [None, 'RS', 'EAA']
VALID_PORTFOLIO_SCORING_METHODS = [None, 'RS']
VALID_PROTECTION_MODES = [None, 'BY_RULE', 'RAA', 'BY_FORMULA']
VALID_PROTECTION_FORMULAS = [None, 'DPF']
VALID_ALGO_ALLOCATION_MODES = ['EW','FIXED','PROPORTIONAL','MIN_VARIANCE','MAX_SHARPE',
                                    'BY_FORMULA', 'RISK_PARITY','VOLATILITY_WEIGHTED','RISK_TARGET', 'MIN_CORRELATION']
VALID_STRATEGY_ALLOCATION_FORMULAS = [None, 'PAA']
VALID_STRATEGY_ALLOCATION_RULES = [None]
NONE_NOT_ALLOWED = ['portfolios', 'portfolio_allocation_modes', 'cash_proxies', 'strategy_allocation_mode']

from talib._ta_lib import BBANDS, DEMA, EMA, HT_TRENDLINE, KAMA, MA, MAMA, MAVP, MIDPOINT, MIDPRICE, SAR, \
    SAREXT, SMA, T3, TEMA, TRIMA, WMA, ADD, DIV, MAX, MAXINDEX, MIN, MININDEX, MINMAX, \
    MINMAXINDEX, MULT, SUB, SUM, BETA, CORREL, LINEARREG, LINEARREG_ANGLE, \
    LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR, ADX, ADXR, APO, AROON, \
    AROONOSC, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM, \
    PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, STOCH, STOCHF, STOCHRSI, \
    TRIX, ULTOSC, WILLR, ATR, NATR, TRANGE, ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, \
    FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH, AD, ADOSC, OBV, AVGPRICE, MEDPRICE, \
    TYPPRICE, WCLPRICE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE

TALIB_FUNCTIONS = [BBANDS, DEMA, EMA, HT_TRENDLINE, KAMA, MA, MAMA, MAVP, MIDPOINT, MIDPRICE, SAR, \
                   SAREXT, SMA, T3, TEMA, TRIMA, WMA, ADD, DIV, MAX, MAXINDEX, MIN, MININDEX, MINMAX, \
                   MINMAXINDEX, MULT, SUB, SUM, BETA, CORREL, LINEARREG, LINEARREG_ANGLE, \
                   LINEARREG_INTERCEPT, LINEARREG_SLOPE, STDDEV, TSF, VAR, ADX, ADXR, APO, AROON, \
                   AROONOSC, BOP, CCI, CMO, DX, MACD, MACDEXT, MACDFIX, MFI, MINUS_DI, MINUS_DM, MOM, \
                   PLUS_DI, PLUS_DM, PPO, ROC, ROCP, ROCR, ROCR100, RSI, STOCH, STOCHF, STOCHRSI, TRIX, \
                   ULTOSC, WILLR, ATR, NATR, TRANGE, ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, \
                   LOG10, SIN, SINH, SQRT, TAN, TANH, AD, ADOSC, OBV, AVGPRICE, MEDPRICE, TYPPRICE, \
                   WCLPRICE, HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Algo():

    def __init__(self, context, strategies=[], allocation_model=None,
                 scoring_model=None, regime=None):

        if get_environment('platform') == 'zipline':
            context.day_no = 0

        self.ID = 'algo'
        self.type = 'Algorithm'

        self.strategies = strategies
        self.allocation_model = allocation_model
        self.regime = regime

        context.strategies = self.strategies

        context.max_lookback = self._compute_max_lookback(context)
        log.info('MAX_LOOKBACK = {}'.format(context.max_lookback))

        self.weights = [0. for s in self.strategies]
        context.strategy_weights = self.weights
        self.strategy_IDs = [s.ID for s in self.strategies]
        self.active = [s.ID for s in self.strategies] + [p.ID for s in self.strategies for p in s.portfolios]

        if self.allocation_model == None:
            raise ValueError('\n *** FATAL ERROR : ALGO ALLOCATION MODEL CANNOT BE NONE ***\n')

        context.prices = pd.Series()
        context.returns = pd.Series()
        context.log_returns = pd.Series()
        context.covariances = dict()
        context.sharpe_ratio = pd.Series()

        self.all_assets = self._set_all_assets()
        context.all_assets = self.all_assets[:]
        self.allocations = pd.Series(0, index=context.all_assets)
        self.previous_allocations = pd.Series(0, index=context.all_assets)
        context.scoring_model = scoring_model
        self.score = 0.

        context.data = Data(self.all_assets)
        context.algo_data = context.data

        set_symbol_lookup_date('2016-01-01')

        self._instantiate_rules(context)

        context.securities = []  # placeholder securities in portfolio

        if get_environment('platform') == 'zipline':
            context.count = context.max_lookback
        else:
            context.count = 0

        self.rebalance_count = 1  # default rebalance interval = 1
        self.first_time = True

        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # looks for any 'lookback' kwargs
    def _compute_max_lookback(self, context):

        kwargs_list = self._get_all_kwargs(context)
        for kwargs in kwargs_list:
            if 'lookback' in kwargs:
                lookback = kwargs['lookback']
                try:
                    period = kwargs['period']
                except:
                    period = 'D'
                # add additional days to cater for 'sip_period'
                if period == 'D':
                    lookback_days = 5 + lookback
                elif period == 'W':
                    lookback_days = 6 + lookback * 5
                elif period == 'M':
                    lookback_days = 25 + lookback * 25
                else:
                    raise RuntimeError('UNKNOWN LOOKBACK PERIOD TYPE {} for strategy {}'.format(period, self.ID))

                context.max_lookback = max(context.max_lookback, lookback_days)

        return context.max_lookback

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_all_kwargs(self, context):
        # creates a list of all kwargs containing 'lookback' labels
        kwargs_list = self._get_portfolio_and_strategy_kwargs(context)
        kwargs_list = kwargs_list + self._get_transform_kwargs(context)
        return kwargs_list

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_portfolio_and_strategy_kwargs(self, context):
        kwargs_list = []
        for strategy in context.strategies:
            kwargs_list = kwargs_list + [strategy.allocation_model.kwargs]
            for pfolio in strategy.portfolios:
                kwargs_list = kwargs_list + [pfolio.allocation_model.kwargs]
        non_trivial_kwargs_list = [kwargs for kwargs in kwargs_list if kwargs not in [None, [], {}, [{}]]]
        return non_trivial_kwargs_list

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_transform_kwargs(self, context):
        kwargs_list = []
        for transform in context.transforms:
            if transform.kwargs not in [None, [], {}, [{}]]:
                kwargs_list = kwargs_list + [transform.kwargs]

        return kwargs_list

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _instantiate_rules(self, context):
        context.rules = {}
        for r in context.algo_rules:
            context.rules[r.name] = r
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _set_all_assets(self):
        all_assets = [s.all_assets for s in self.strategies]
        self.all_assets = list(set([i for sublist in all_assets for i in sublist]))
        return self.all_assets

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _allocate_assets(self, context):
        log.debug('STRATEGY WEIGHTS = {}\n'.format(self.weights))
        for i, s in enumerate(self.strategies):
            self.allocations = self.allocations.add(self.weights[i] * s.allocations,
                                                    fill_value=0)
        if self.allocations.sum() == 0:
            # not enough price data yet
            return self.allocations

        # if 1. - sum(self.allocations) > 1.e-15 :
        #     raise RuntimeError ('SUM OF ALLOCATIONS = {} - SHOULD ALWAYS BE 1'.format(sum(self.allocations)))

        return self.allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def check_signal_trigger(self, context, data):

        holdings = context.portfolio.positions
        if self.first_time or context.rules['rebalance_rule'].apply_rule(context)[holdings].any():
            # force rebalance
            self.rebalance(context, data)
            self.first_time = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def rebalance(self, context, data):

        # log.info('REBALANCE >> REBALANCE INTERVAL = ' + str(context.rebalance_interval))

        # make sure there's algo data
        # if not isinstance(context.algo_data, dict):
        if not context.data:
            return
        elif not self.first_time:
            if self.rebalance_count != context.rebalance_interval:
                self.rebalance_count += 1
                return

        self.first_time = False

        self.rebalance_count = 1

        log.info('----------------------------------------------------------------------------')

        self.allocations = pd.Series(0., index=context.all_assets)
        self.elligible = pd.Index(self.strategy_IDs)

        # if self.scoring_model != None:
        #     self.scoring_model.caller = self
        #     context.symbols = self.strategy_IDs[:]
        #     self.score = self.scoring_model.compute_score (context)
        #     self.elligible =  self.scoring_model.apply_ntop ()

        self.allocation_model.caller = self
        if self.regime == None:
            self._get_strategy_and_portfolio_allocations(context)
        else:
            self._check_for_regime_change_and_set_active(context)

        self.weights = self.allocation_model.get_weights(context)
        self.allocations = self._allocate_assets(context)

        self._execute_orders(context, data)

        return self.allocations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_strategy_and_portfolio_allocations(self, context):
        for s_no, s in enumerate(self.strategies):
            s.allocations = pd.Series(0., index=s.all_assets)
            for p_no, p in enumerate(s.portfolios):
                p.allocations = pd.Series(0., index=p.all_assets)
                p.allocations = p.reallocate(context)
            s.allocations = s.reallocate(context)
        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_for_regime_change_and_set_active(self, context):
        self.current_regime = self.regime.get_current(context)
        log.debug('REGIME : {} \n'.format(self.current_regime))
        if self.regime.detect_change(context):
            self.regime.set_new_regime()
            self.active = self.regime.get_active()
        else:
            log.info('REGIME UNCHANGED. JUST REBALANCE\n')
        return
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _execute_orders(self, context, data):

        for security in self.allocations.index:
            if context.portfolio.positions[security].amount > 0 and self.allocations[security] == 0:
                order_target_percent(security, 0)
            elif self.allocations[security] != 0:
                if get_open_orders(security):
                    continue

                current_value = context.portfolio.positions[security].amount * data.current(security, 'price')
                portfolio_value = context.portfolio.portfolio_value
                if portfolio_value == 0:  # before first purchases
                    portfolio_value = context.account.available_funds
                target_value = portfolio_value * self.allocations[security]

                if np.abs(target_value / current_value - 1) < context.threshold:
                    continue

                order_target_percent(security, self.allocations[security] * context.leverage)
                qty = int(
                    context.account.net_liquidation * self.allocations[security] / data.current(security, 'price'))
                log.debug('ORDERING {} : {}%  QTY = {}'.format(security.symbol,
                                                               self.allocations[security] * 100, qty))

        context.gtc_count = GTC_LIMIT

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def check_for_unfilled_orders(self, context, data):
        unfilled = {o.sid: o.amount - o.filled for oo in get_open_orders() for o in get_open_orders(oo)}
        context.outstanding = {u: unfilled[u] for u in unfilled if unfilled[u] != 0}
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def fill_outstanding_orders(self, context, data):
        if context.outstanding == {}:
            context.show_positions = False
            return
        elif context.gtc_count > 0:
            for s in context.outstanding:
                order(s, context.outstanding[s])
                log.debug('ORDER {} OUTSTANDING {} SHARES'.format(context.outstanding[s], s.symbol))

            context.gtc_count -= 1
        else:
            log.info('GTC_COUNT EXPIRED')
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def show_records(self, context, data):
        record('LEVERAGE', context.account.leverage)
        # record('CONTEXT_LEVERAGE', context.leverage)
        # record('PV', context.account.total_positions_value)
        # record('PV1',context.portfolio.positions_value)
        # record('TOTAL', context.portfolio.portfolio_value)
        # record('CASH', context.portfolio.cash)
        # for s in context.strategies:
        #     # record(s.ID + '_prices', s.prices.iloc[-1])
        #     for p in s.portfolios:
        #         record(p.ID + '_prices', p..ilocprices[-1])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def show_positions(self, context, data):

        if context.portfolio.positions == {}:
            return

        log.info('\nPOSITIONS\n')
        for asset in self.all_assets:
            if context.portfolio.positions[asset].amount > 0:
                log.info(
                    '{0} : QTY = {1}, COST BASIS {2:3.2f}, CASH = {3:7.2f}, POSITIONS VALUE = {4:7.2f}, TOTAL = {5:7.2f}'
                    .format(asset.symbol, context.portfolio.positions[asset].amount,
                            context.portfolio.positions[asset].cost_basis,
                            context.portfolio.cash,
                            context.portfolio.positions[asset].amount * data.current(asset, 'price'),
                            context.portfolio.portfolio_value))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Strategy():

    def __init__(self, context, ID='', portfolios=[], allocation_model=None,
                 scoring_model=None):

        self.ID = ID
        self.type = 'Strategy'
        self.portfolios = portfolios
        self.portfolio_IDs = [p.ID for p in self.portfolios]
        self.weights = [0. for p in portfolios]

        self.prices = pd.Series()
        self.returns = pd.Series()
        self.covariances = dict()
        self.sharpe_ratio = pd.Series()

        if allocation_model == None:
            self.allocation_model = AllocationModel(context, mode='EW')
        else:
            self.allocation_model = allocation_model
        self.scoring_model = scoring_model
        self.score = 0.

        self._set_all_assets()
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _set_all_assets(self):
        all_assets = [p.all_assets for p in self.portfolios]
        self.all_assets = set([i for sublist in all_assets for i in sublist])
        return self.all_assets
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def allocate_assets(self, context):
        self.allocations = pd.Series(0., index=self.all_assets)
        log.debug('STRATEGY {} PORTFOLIO WEIGHTS = {}\n'.format(self.ID, [round(w, 2) for w in self.weights]))
        for i, p in enumerate(self.portfolios):
            self.allocations = self.allocations.add(self.weights[i] * p.allocations,
                                                    fill_value=0)
        log.debug('SECURITY ALLOCATIONS for {} \n{}\n'.format(self.ID, self.allocations.round(2)))
        return self.allocations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def reallocate(self, context):
        self.elligible = pd.Index(self.portfolio_IDs)

        if self.scoring_model != None:
            self.scoring_model.caller = self
            context.symbols = self.portfolio_IDs[:]
            self.score = self.scoring_model.compute_score(context)
            self.elligible = self.scoring_model.apply_ntop()

        self.allocation_model.caller = self
        self.weights = self.allocation_model.get_weights(context)
        self.allocations = self.allocate_assets(context)
        self.holdings = (self.allocations * context.portfolio.portfolio_value).divide(
            context.algo_data['price'][self.all_assets]).round(0)
        return self.allocations
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Portfolio():

    def __init__(self, context, ID='',
                 securities=[], allocation_model=None,
                 scoring_model=None,
                 downside_protection_model=None,
                 cash_proxy=None, allow_shorts=False):

        self.ID = ID
        self.type = 'Portfolio'
        self.securities = securities
        self.weights = [0. for s in securities]
        self.allocation_model = allocation_model
        self.scoring_model = scoring_model
        self.score = 0.
        self.downside_protection_model = downside_protection_model
        if cash_proxy == None:
            log.info('NO CASH_PROXY SPECIFIED FOR PORTFOLIO {}'.format(self.ID))
            raise ValueError('INITIALIZATION ERROR')
        self.cash_proxy = cash_proxy

        self.prices = pd.Series()
        self.returns = pd.Series()
        self.covariances = dict()
        self.sharpe_ratios = pd.Series()

        for s in [context.market_proxy, self.cash_proxy, context.risk_free]:
            if s in self.securities:
                log.warn('{} is included in the portfolio'.format(s.symbol))

        self.all_assets = list(set(self.securities + [context.market_proxy, self.cash_proxy, context.risk_free]))

        self.allocations = pd.Series([0.0] * len(self.all_assets), index=self.all_assets)

        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def reallocate(self, context):

        self.allocations = pd.Series(0., index=self.all_assets)
        self.elligible = pd.Index(self.securities)

        if self.scoring_model != None:
            self.scoring_model.caller = self
            context.symbols = self.securities[:]
            self.score = self.scoring_model.compute_score(context)
            self.elligible = self.scoring_model.apply_ntop()

        self.allocation_model.caller = self
        self.weights = self.allocation_model.get_weights(context)
        self.allocations[self.elligible] = self.weights

        log.debug('ALLOCATIONS FOR {} : {}\n'.format(self.ID,
                                                     [(self.allocations.index[i].symbol, round(v, 2))
                                                      for i, v in enumerate(self.allocations)
                                                      if v > 0]))

        if self.downside_protection_model != None:
            self.downside_protection_model.caller = self
            self.allocations = self.downside_protection_model.apply_protection(context,
                                                                               self.allocations,
                                                                               self.cash_proxy,
                                                                               [self.securities, self.score])
            log.debug('AFTER DOWNSIDE PROTECTION {} : {}\n'.format(self.ID,
                                                                   [(self.allocations.index[i].symbol, round(v, 2))
                                                                    for i, v in enumerate(self.allocations)
                                                                    if v > 0]))

        self.holdings = (self.allocations * context.portfolio.portfolio_value).divide(
            context.algo_data['price'][self.all_assets]).round(0)

        return self.allocations


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Regime():

    def __init__(self, transitions):
        """Initialize Regime object. Set init state and transition table."""
        self.transitions = transitions
        # set current != new to always detect change on first reallocation
        self.current_regime = 0
        self.new_regime = 1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def detect_change(self, context):
        self.new_regime = self.get_current(context)
        return [False if self.current_regime == self.new_regime else True][0]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_current(self, context):
        for k in self.transitions.keys():
            rule_name = self.transitions[k][0]
            rule = context.rules[rule_name]
            if rule.apply_rule(context):
                return k
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def set_new_regime(self):
        self.current_regime = self.new_regime
        record('REGIME', self.current_regime)
        return self.current_regime
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_active(self):
        return self.transitions[self.current_regime][1]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Data():

    def __init__(self, assets):
        self.all_assets = assets
        # self.fallbacks = {'EDV' : symbol('TLT')}
        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def update(self, context, data):

        ''' generates context.raw_data (dictionary of context.max_lookback values)  and context.algo_data (dictioanary current values) for  'high', 'open', 'low', 'close', 'volume', 'price' and all transforms '''

        # log.info('\n{} GENERATING ALGO_DATA...'.format(get_datetime().date()))

        # dataframe for each of 'high', 'open', 'low', 'close', 'volume', 'price'
        context.raw_data = self.get_raw_data(context, data)

        # add a dataframe for each transform
        context.raw_data = self.generate_frame_for_each_transform(context, data)

        # only need the current value for each security (Series)
        context.algo_data = self.current_algo_data(context, data)

        return context.algo_data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_tradeable_assets(self, data):
        tradeable_assets = [asset for asset in self.all_assets if data.can_trade(asset)]
        if len(self.all_assets) > len(tradeable_assets):
            non_tradeable = [s.symbol for s in self.all_assets if data.can_trade(s) == False]
            log.error('*** FATAL ERROR : MISSING DATA for securities {}'.format(non_tradeable))
            raise ValueError('FATAL ERROR: SEE LOG FOR MISSING DATA')
        return tradeable_assets

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_raw_data(self, context, data):

        context.raw_data = dict()

        tradeable_assets = self.get_tradeable_assets(data)

        for item in ['high', 'open', 'low', 'close', 'volume', 'price']:
            try:
                context.raw_data[item] = data.history(tradeable_assets, item, context.max_lookback, '1d')
            except:
                log.warn('FATAL ERROR: UNABLE TO LOAD HISTORY DATA FOR {}'.format(item))
                # force exit
                raise ValueError(' *** FATAL ERROR : INSUFFICIENT DATA - SEE LOG *** ')

            if np.isnan(context.raw_data[item].values).any():
                # log.warn ('\n WARNING : THERE ARE NaNs IN THE DATA FOR {} \n FILL BACKWARDS.......'
                #           .format([k.symbol for k in context.raw_data[item].keys() if
                #                    np.isnan(context.raw_data[item][k][0])]))
                context.raw_data[item] = context.raw_data[item].bfill()

        return context.raw_data

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def generate_frame_for_each_transform(self, context, data):

        for transform in context.transforms:
            # result = apply_transform(context, transform)
            result = transform.apply_transform(context)
            outputs = transform.outputs
            if type(result) == pd.Panel:
                context.raw_data.update(dict([(o, result[o]) for o in outputs]))
            elif type(result) == pd.DataFrame:
                context.raw_data[outputs[0]] = result
            else:
                log.error('\n INVALID TRANSFORM RESULT\n')

        return context.raw_data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def current_algo_data(self, context, data):

        context.algo_data = dict()
        for k in [key for key in context.raw_data.keys()
                  if type(context.raw_data[key]) == pd.DataFrame]:
            context.algo_data[k] = context.raw_data[k].ix[-1]
            if np.isnan(context.algo_data[k].values).any():
                security = [s.symbol for s in context.raw_data[k].ix[-1].index
                            if np.isnan(context.raw_data[k][s].ix[-1])][0]
                log.warn('*** WARNING: FOR ITEM {} THERE IS A NAN IN THE DATA FOR {}'.format(k, security))
        return context.algo_data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # prices are NOMINAL prices used for individual portfolio/strategy variance/cov calculations
    def update_portfolio_and_strategy_metrics(self, context, data):
        for s_no, s in enumerate(context.strategies):
            self._update_strategy_metrics(context, data, s, s_no)
            self._update_portfolio_metrics(context, data, s)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _update_strategy_metrics(self, context, data, s, s_no):
        ''' calculate and store current price of strategies used by algo '''
        strategy_price = s.holdings.multiply(context.algo_data['price'][s.all_assets]).sum()
        s.prices[get_datetime()] = strategy_price
        s.sharpe_ratio[get_datetime()] = self._calculate_sharpe_ratio(context, data, s)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _update_portfolio_metrics(self, context, data, s):
        for p_no, p in enumerate(s.portfolios):
            portfolio_price = p.holdings.multiply(context.algo_data['price'][p.all_assets]).sum()
            p.prices[get_datetime()] = portfolio_price
            p.sharpe_ratios[get_datetime()] = self._calculate_sharpe_ratio(context, data, p)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _calculate_sharpe_ratio(self, context, data, s_or_p):
        if len(s_or_p.prices) <= context.SR_lookback:
            # not enought data yet
            return 0
        rets = s_or_p.prices.pct_change()[-context.SR_lookback:]
        # s_or_p_rets = (rets * s_or_p.allocation_model.weights).sum(axis=1)[-context.SR_lookback:]
        risk_free_rets = data.history(context.risk_free, 'price', context.SR_lookback, '1d').pct_change()
        excess_returns = rets[1:].values - risk_free_rets[1:].values
        return excess_returns.mean() / rets.std()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ScoringModel():

    def __init__(self, context, factors=None, method=None, n_top=1):
        self.factors = factors
        self.method = method
        if self.factors == None:
            raise ValueError('Unable to score model with no factors')
        # if self.method == None :
        #     raise ValueError ('Unable to score model with no method')
        self.n_top = n_top
        self.score = 0
        self.methods = {'RS': self._relative_strength,
                        'EAA': self._eaa
                        }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def compute_score(self, context):
        self.symbols = context.symbols
        self.score = self.methods[self.method](context)
        # log.debug ('\nSCORE\n\n{}\n'.format(self.score))
        return self.score

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _relative_strength(self, context):
        self.score = 0.
        for name in self.factors.keys():

            if np.isnan(context.algo_data[name[1:]][self.symbols]).any():
                if isinstance(self.symbols[0], str):
                    sym = [(self.symbols[s], v)
                           for s, v in enumerate(context.algo_data[name[1:]][self.symbols]) if np.isnan(v)][0][0]
                else:
                    sym = [(self.symbols[s].symbol, v)
                           for s, v in enumerate(context.algo_data[name[1:]][self.symbols]) if np.isnan(v)][0][0]
                print('SCORING ERROR : FACTOR {} VALUE FOR {} IS nan'.format(name, sym))
                raise RuntimeError()

            if name[0] == '+':
                # log.debug('Values for factor {} :\n\{}\nRANKS : \n{}'.format(name[1:],
                #                                                              [(s.symbol, context.algo_data[name[1:]][s]) for s in self.securities],
                #                                                              [(s.symbol, context.algo_data[name[1:]][self.securities].rank(ascending=False)[s])
                #                                                               for s in self.securities]))

                try:
                    # highest value gets highest rank / score
                    self.score = self.score + context.algo_data[name[1:]][self.symbols].rank(ascending=True) \
                                 * self.factors[name]
                except:
                    raise RuntimeError(
                        '\n *** FATAL ERROR : UNABLE TO SCORE FACTOR {}. CHECK TRANSFORM & FACTOR DEFINITIONS\n'
                        .format(name[1:]))

            elif name[0] == '-':
                # log.debug('Values for factor {} :\n\{}\nRANKS : \n{}'.format(name[1:],
                #                                                              [(s.symbol, context.algo_data[name[1:]][s]) for s in self.securities],
                #                                                              [(s.symbol, context.algo_data[name[1:]][self.securities].rank(ascending=True)[s])
                #                                                               for s in self.securities]))

                try:
                    # lowest value gets highest rank /score
                    self.score = self.score + context.algo_data[name[1:]][self.symbols].rank(ascending=False) \
                                 * self.factors[name]
                except:
                    raise RuntimeError('\n UNABLE TO SCORE FACTOR {}. CHECK TRANSFORM & FACTOR DEFINITIONS\n'
                                       .format(name[1:]))

        # log.debug('Scores for factor {} :\n\n{}'.format(name[1:],
        #                                                 [(s.symbol, self.score[s]) for s in self.securities]))

        return self.score
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _eaa(self, context):

        # only valid for securities, not portfolios or strategies (?)

        if self.caller.type != 'Portfolio':
            raise RuntimeError('SCORING MODEL EAA ONLY VALID FOR PORTFOLIO, NOT {}'.format(self.method))

        # prices = data.history(self.securities, 'price', 280, '1d')
        prices = context.raw_data['price'][self.symbols]

        monthly_prices = prices.resample('M').last()[self.symbols]
        monthly_returns = monthly_prices.pct_change().ix[-12:]

        # nominal return correlation to equi-weight portfolio
        N = len(self.symbols)
        equal_weighted_index = monthly_returns.mean(axis=1)
        C = pd.Series([0.0] * N, index=self.symbols)
        for s in C.index:
            C[s] = monthly_returns[s].corr(equal_weighted_index)

        R = context.algo_data['R'][self.symbols]
        V = monthly_returns.std()

        # Apply factor weights
        # wi ~ zi = ( ri^wR * (1-ci)^wC / vi^wV )^wS
        wR = self.factors['R']
        wC = self.factors['C']
        wV = self.factors['V']
        wS = self.factors['S']
        eps = self.factors['eps']

        # Generalized Momentum Score
        self.score = ((R ** wR) * ((1 - C) ** wC) / (V ** wV)) ** (wS + eps)

        return self.score

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def apply_ntop(self):

        N = len(self.symbols)
        if self.method == 'EAA':
            self.n_top = min(np.ceil(N ** 0.5) + 1, N / 2)
            elligible = self.score.sort_values().index[-self.n_top:]
        else:
            # best score gets lowest rank
            ranks = self.score.rank(ascending=False, method='dense')
            elligible = ranks[ranks <= self.n_top].index

        return elligible


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class AllocationModel():

    def __init__(self, context, mode='EW', weights=None, rule=None, formula=None, kwargs={}):
        self.mode = mode
        self.formula = formula
        self.weights = weights
        self.rule = rule
        self.kwargs = kwargs

        self.modes = {'EW': self._equal_weight_allocation,
                      'FIXED': self._fixed_allocation,
                      'PROPORTIONAL': self._proportional_allocation,
                      'MIN_VARIANCE': self._min_variance_allocation,
                      'BRUTE_FORCE_SHARPE': self._brute_force_sharpe_allocation,
                      'MAX_SHARPE': self._max_sharpe_allocation,
                      'BY_FORMULA': self._allocation_by_formula,
                      'REGIME_EW': self.allocate_by_regime_EW,
                      'RISK_PARITY': self._risk_parity_allocation,
                      'VOLATILITY_WEIGHTED': self._volatility_weighted_allocation,
                      'RISK_TARGET': self._risk_targeted_allocation,
                      'MIN_CORRELATION': self._get_reduced_correlation_weights,
                      }

        if mode not in self.modes.keys():
            raise ValueError('UNKNOWN MODE "{}"'.format(mode))

        self.caller = None  # portfolio or strategy object calling the model

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def get_weights(self, context):
        self.prices = self._get_caller_prices(context)
        if self.mode not in ['EW', 'FIXED', 'PROPORTIONAL']:
            # all other modes need prices for at least 'lookback' period
            if self.kwargs is not None and 'lookback' in self.kwargs:
                # unable to allocate weights until more than 'lookback' prices
                if len(self.prices) <= self.kwargs['lookback']:
                    # default to 'EW' to be able to generate prices
                    self.caller_weights = [1. / len(self.caller.elligible) for i in self.caller.elligible]
                    return self.caller_weights
        if self.mode.startswith('REGIME') and self.caller.ID != 'algo':
            raise ValueError('ILLEGAL REGIME ALLOCATION : REGIME ALLOCATION MODEL ONLY ALLOWED AT ALGO LEVEL')
        return self.modes[self.mode](context)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_caller_prices(self, context):
        if self.caller.type == 'Portfolio':
            prices = context.raw_data['price'][self.caller.elligible]
        elif self.caller.type == 'Strategy':
            # portfolio prices for portfolios in strategy
            prices = self._get_strategy_prices(context)

        elif self.caller.type == 'Algorithm':
            # strategy prices for strategies in algorithm
            prices = self._get_algo_prices(context)

        return prices

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_strategy_prices(self, context):
        prices_dict = OrderedDict({p.ID: p.prices for s in context.strategies for p in s.portfolios})
        index = context.strategies[0].portfolios[0].prices.index
        columns = [p.ID for s in context.strategies for p in s.portfolios]
        return pd.DataFrame(prices_dict, index=index, columns=columns)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_algo_prices(self, context):
        prices_dict = OrderedDict({s.ID: s.prices for s in context.strategies})
        index = context.strategies[0].prices.index
        columns = [s.ID for s in context.strategies]
        return pd.DataFrame(prices_dict, index=index, columns=columns)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _equal_weight_allocation(self, context):
        elligible = self.caller.elligible
        if len(elligible) > 0:
            self.caller.weights = [1. / len(elligible) for i in elligible]
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _fixed_allocation(self, context):
        # we are going to change these weights, so be careful to keep a copy!
        self.caller.weights = self.caller.allocation_model.weights[:]
        return self.caller.weights

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _proportional_allocation(self, context):
        elligible = self.caller.elligible
        score = self.caller.score
        self.caller.weights = score[elligible] / score[elligible].sum()
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _risk_parity_allocation(self, context):
        lookback = self.kwargs['lookback']
        prices = self.prices[-lookback:]
        ret_log = np.log(1. + prices.pct_change())[1:]
        hist_vol = ret_log.std(ddof=0)

        adj_vol = 1. / hist_vol

        self.caller.weights = adj_vol.div(adj_vol.sum(), axis=0)
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _volatility_weighted_allocation(self, context):

        elligible = self.caller.elligible
        lookback = self.kwargs['lookback']
        ret_log = np.log(1. + self.prices.pct_change())
        hist_vol = ret_log.rolling(window=lookback, center=False).std()[elligible]

        adj_vol = 1. / hist_vol

        self.caller.weights = adj_vol.div(adj_vol.sum())
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _risk_targeted_allocation(self, context):
        lookback = self.kwargs['lookback']
        target_risk = self.kwargs['target_risk']
        shorts = self.kwargs['shorts']
        prices = self.prices[self.caller.elligible][-lookback:]
        sigma_mat = self._compute_covariance_matrix(prices)
        mu_vec = self._compute_expected_returns(prices)
        risk_free = context.raw_data['price'][context.risk_free].pct_change()[-lookback:].mean()
        self.caller.weights = self._compute_target_risk_portfolio(mu_vec, sigma_mat,
                                                                  target_risk=target_risk,
                                                                  risk_free=risk_free,
                                                                  shorts=shorts)[0]
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _min_variance_allocation(self, context):
        lookback = self.kwargs['lookback']
        shorts = self.kwargs['shorts']
        prices = self.prices[self.caller.elligible][-lookback:]
        sigma_mat = self._compute_covariance_matrix(prices)
        mu_vec = self._compute_expected_returns(prices)
        self.caller.weights = self._compute_global_min_portfolio(mu_vec=mu_vec,
                                                                 sigma_mat=sigma_mat,
                                                                 shorts=shorts)[0]
        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _max_sharpe_allocation(self, context):
        # calculate security weights for max sharpe portfolio
        elligible = self.caller.elligible
        lookback = self.kwargs['lookback']
        shorts = self.kwargs['shorts']
        prices = self.prices[elligible][-lookback:]
        sigma_mat = self._compute_covariance_matrix(prices)
        mu_vec = self._compute_expected_returns(prices)
        risk_free = context.raw_data['price'][context.risk_free].pct_change()[-lookback:].mean()
        self.caller.weights = self._compute_tangency_portfolio(mu_vec=mu_vec,
                                                               sigma_mat=sigma_mat,
                                                               risk_free=risk_free,
                                                               shorts=shorts)[0]
        return self.caller.weights

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # this only works at strategy level
    # it could feasibly work at algo level too
    def _brute_force_sharpe_allocation(self, context):
        if isinstance(self.caller, Strategy):
            portfolio_SRs = [p.sharpe_ratios[-1] for p in self.caller.portfolios]
            # select the portfolio(s) with the highest SR - could be more than 1
            self.caller.weights = [1. if s == np.max(portfolio_SRs) else 0 for s in portfolio_SRs]
            # in case there are more than 1, normalize
            return self.caller.weights / np.sum(self.caller.weights)
        else:
            raise RuntimeError('BRUTE_FORCE_SHARPE_ALLOCATION ONLY WORKS AT STRATEGY LEVEL')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_reduced_correlation_weights(self, context):
        """
        Implementation of minimum correlation algorithm.
        ref: http://cssanalytics.com/doc/MCA%20Paper.pdf

        :Params:
            :returns <Pandas DataFrame>:Timeseries of asset returns
            :risk_adjusted <boolean>: If True, asset weights are scaled
                                      by their standard deviations
        """
        elligible = self.caller.elligible
        lookback = self.kwargs['lookback']
        risk_adjusted = self.kwargs['risk_adjusted']

        prices = self.prices[elligible][-lookback:]
        returns = prices.pct_change()[1:]

        correlations = returns.corr()
        adj_correlations = self._get_adjusted_cor_matrix(correlations)
        initial_weights = adj_correlations.T.mean()

        ranks = initial_weights.rank()
        ranks /= ranks.sum()

        weights = adj_correlations.dot(ranks)
        weights /= weights.sum()

        if risk_adjusted:
            weights = weights / returns.std()
            weights /= weights.sum()
        return weights

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_adjusted_cor_matrix(self, cor):
        values = cor.values.flatten()
        mu = np.mean(values)
        sigma = np.std(values)
        distribution = scipy.stats.norm(mu, sigma)
        return 1 - cor.apply(lambda x: distribution.cdf(x))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _allocation_by_formula(self, context):
        # for Protective Asset Allocation (PAA), strategy assumed to have 2 portfolios
        if self.formula == 'PAA':
            if len(self.caller.elligible) != 2:
                raise ValueError('Protective Asset Allocation (PAA) Srategy has {} Portfolio; must have 2')
            else:
                self.caller.allocations = self._allocate_by_PAA_formula(context)
        return self.caller.allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _allocate_by_PAA_formula(self, context):
        try:
            protection_factor = self.kwargs['protection_factor']
        except:
            raise RuntimeError(
                'MISSING STRATEGY ALLOCATION KWARG "protection_factor" FOR STRATEGY {}'.format(self.caller.ID))
        securities = self.caller.portfolios[0].securities
        N = len(securities)
        n = context.rules[self.rule].apply_rule(context)[securities].sum()
        dpf = (N - n) / (N - protection_factor * n / 4.)
        # log.debug ('For portfolio {}, n = {}, N = {}, dpf = {}'.format(self.caller.ID, n, N, dpf))
        record('DPF', dpf)
        self.caller.weights = [1. - dpf, dpf]
        return self.caller.weights

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def allocate_by_regime_EW(self, context):

        # log.debug('\nACTIVE : {} \n'.format(self.caller.active))

        if self.caller.type != 'Algorithm':
            raise RuntimeError('REGIME SWITCHING ONLY ALLOWED AT ALGO LEVEL')

        self._reset_strategy_and_portfolio_weights(context)

        for s in self.caller.strategies:
            s.allocations = pd.Series(0, index=s.all_assets)

            for p in s.portfolios:
                if s.ID in self.caller.active:
                    p_weight = 1. / len(s.portfolios)
                elif p.ID in self.caller.active:
                    p_weight = 1. / np.sum([1 if pfolio.ID in self.caller.active else 0 for pfolio in s.portfolios])
                elif s.ID not in self.caller.active and p.ID not in self.caller.active:
                    continue

                p.allocations = p.reallocate(context)
                s.allocations = s.allocations.add(p_weight * p.allocations, fill_value=0)

        active_strategies = set([s.ID for s in context.strategies
                                 for p in s.portfolios if s.ID in self.caller.active
                                 or p.ID in self.caller.active])
        self.caller.weights = [1. / len(active_strategies) if s.ID in active_strategies else 0 for s in
                               context.strategies]
        context.strategy_weights = self.caller.weights

        return self.caller.weights
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _reset_strategy_and_portfolio_weights(self, context):

        for s_no, s in enumerate(self.caller.strategies):
            self.caller.weights[s_no] = 0
            context.strategy_weights[s_no] = 0
            for p_no, p in enumerate(s.portfolios):
                s.weights[p_no] = 0
        return
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _get_no_of_active_portfolios(self):
        # Note : if strategy in active, all its portfolios are active
        number = 0
        for s in self.caller.strategies:
            if s.ID in self.caller.active:
                # all portfolios are active
                for p in s.portfolios:
                    number += 1
            for p in s.portfolios:
                if p.ID in self.caller.active:
                    number += 1

        return number

    # Portfolio Helper Functions

    # Functions:
    #    1. compute_efficient_portfolio        compute minimum variance portfolio
    #                                            subject to target return
    #    2. compute_global_min_portfolio       compute global minimum variance portfolio
    #    3. compute_tangency_portfolio         compute tangency portfolio
    #    4. compute_efficient_frontier         compute Markowitz bullet
    #    5. compute_portfolio_mu               compute portfolio expected return
    #    6. compute_portfolio_sigma            compute portfolio standard deviation
    #    7. compute_covariance_matrix          compute covariance matrix
    #    8. compute_expected_returns           compute expected returns vector

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_covariance_matrix(self, prices):
        # calculates the cov matrix for the period defined by prices
        returns = np.log(1 + prices.pct_change())[1:]
        excess_returns_matrix = returns - returns.mean()
        return 1. / len(returns) * (excess_returns_matrix.T).dot(excess_returns_matrix)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_expected_returns(self, prices):
        mu_vec = np.log(1 + prices.pct_change(1))[1:].mean()
        return mu_vec

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_portfolio_mu(self, mu_vec, weights_vec):
        if len(mu_vec) != len(weights_vec):
            raise RuntimeError('mu_vec and weights_vec must have same length')
        return mu_vec.T.dot(weights_vec)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_portfolio_sigma(self, sigma_mat, weights_vec):

        if len(sigma_mat) != len(sigma_mat.columns):
            raise RuntimeError('sigma_mat must be square\nlen(sigma_mat) = {}\nlen(sigma_mat.columns) ={}'.
                               format(len(sigma_mat), len(sigma_mat.columns)))
        return np.sqrt(weights_vec.T.dot(sigma_mat).dot(weights_vec))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_efficient_portfolio(self, mu_vec, sigma_mat, target_return, shorts=True):

        # compute minimum variance portfolio subject to target return
        #
        # inputs:
        # mu_vec                  N x 1 DataFrame expected returns
        #                         with index = asset names
        # sigma_mat               N x N DataFrame covariance matrix of returns
        #                         with index = columns = asset names
        # target_return           scalar, target expected return
        # shorts                  logical, allow shorts is TRUE
        #
        # output is portfolio object with the following elements
        #
        # mu_p                   portfolio expected return
        # sig_p                  portfolio standard deviation
        # weights                N x 1 DataFrame vector of portfolio weights
        #                        with index = asset names

        # check for valid inputs
        #

        if len(mu_vec) != len(sigma_mat):
            print("dimensions of mu_vec and sigma_mat do not match")
            raise ValueError
        if np.matrix([sigma_mat.ix[i][i] for i in range(len(sigma_mat))]).any() <= 0:
            print('Covariance matrix not positive definite')
            raise TypeError

        #
        # compute efficient portfolio
        #

        solvers.options['show_progress'] = False
        P = 2 * matrix(sigma_mat.values)
        q = matrix(0., (len(sigma_mat), 1))
        G = spdiag([-1. for i in range(len(sigma_mat))])
        A = matrix(1., (1, len(sigma_mat)))
        A = matrix([A, matrix(mu_vec.T.values).T], (2, len(sigma_mat)))
        b = matrix([1.0, target_return], (2, 1))

        if shorts == True:
            h = matrix(1., (len(sigma_mat), 1))

        else:
            h = matrix(0., (len(sigma_mat), 1))

        # weights_vec = pd.DataFrame(np.array(solvers.qp(P, q, G, h, A, b)['x']),\
        #                                     sigma_mat.columns)
        try:
            weights_vec = pd.Series(list(solvers.qp(P, q, G, h, A, b)['x']), index=sigma_mat.columns)
        except:
            log.info('W A R N I N G : unable to compute optimal weights; setting to equal weights')
            weights_vec = pd.Series(1. / len(sigma_mat), index=sigma_mat.columns)

            #
        # compute portfolio expected returns and variance
        #
        # print ('*** Debug ***\n_compute_efficient_portfolio:\nmu_vec:\n', self.mu_vec, '\nsigma_mat:\n',
        #        self.sigma_mat, '\nweights:\n', self.weights_vec )
        weights_vec.index = mu_vec.index
        mu_p = self._compute_portfolio_mu(mu_vec, weights_vec)
        sigma_p = self._compute_portfolio_sigma(sigma_mat, weights_vec)

        return weights_vec, mu_p, sigma_p
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _compute_global_min_portfolio(self, mu_vec, sigma_mat, shorts=True):

        solvers.options['show_progress'] = False
        P = 2 * matrix(sigma_mat.values)
        q = matrix(0., (len(sigma_mat), 1))
        G = spdiag([-1. for i in range(len(sigma_mat))])
        A = matrix(1., (1, len(sigma_mat)))
        b = matrix(1.0)

        if shorts == True:
            h = matrix(1., (len(sigma_mat), 1))
        else:
            h = matrix(0., (len(sigma_mat), 1))

        # print ('\nP\n\n{}\n\nq\n\n{}\n\nG\n\n{}\n\nh\n\n{}\n\nA\n\n{}\n\nb\n\n{}\n\n'.format(P,q,G,h,A,b))
        # weights_vec = pd.DataFrame(np.array(solvers.qp(P, q, G, h, A, b)['x']),\
        #                                     index=sigma_mat.columns)
        weights_vec = pd.Series(list(solvers.qp(P, q, G, h, A, b)['x']), index=sigma_mat.columns)

        #
        # compute portfolio expected returns and variance
        #
        # print ('*** Debug ***\n_Global Min Portfolio:\nmu_vec:\n', mu_vec, '\nsigma_mat:\n',
        #        sigma_mat, '\nweights:\n', weights_vec)

        mu_p = self._compute_portfolio_mu(mu_vec, weights_vec)
        sigma_p = self._compute_portfolio_sigma(sigma_mat, weights_vec)

        return weights_vec, mu_p, sigma_p
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _compute_efficient_frontier(self, mu_vec, sigma_mat, risk_free=0, points=100, shorts=True):

        efficient_frontier = pd.DataFrame(index=range(points), dtype=object, columns=['mu_p', 'sig_p', 'sr_p', 'wts_p'])

        gmin_wts, gmin_mu, gmin_sigma = self._compute_global_min_portfolio(mu_vec, sigma_mat, shorts=shorts)

        xmax = mu_vec.max()
        if shorts == True:
            xmax = 2 * mu_vec.max()
        for i, mu in enumerate(np.linspace(gmin_mu, xmax, points)):
            w_vec, portfolio_mu, portfolio_sigma = self._compute_efficient_portfolio(mu_vec, sigma_mat, mu,
                                                                                     shorts=shorts)
            efficient_frontier.ix[i]['mu_p'] = w_vec.dot(mu_vec)
            efficient_frontier.ix[i]['sig_p'] = np.sqrt(w_vec.T.dot(sigma_mat.dot(w_vec)))
            efficient_frontier.ix[i]['sr_p'] = (efficient_frontier.ix[i]['mu_p'] - risk_free) / \
                                               efficient_frontier.ix[i]['sig_p']
            efficient_frontier.ix[i]['wts_p'] = w_vec

        return efficient_frontier

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_tangency_portfolio(self, mu_vec, sigma_mat, risk_free=0, shorts=True):

        efficient_frontier = self._compute_efficient_frontier(mu_vec, sigma_mat, risk_free, shorts=shorts)
        index = efficient_frontier.index[efficient_frontier['sr_p'] == efficient_frontier['sr_p'].max()]

        wts = efficient_frontier['wts_p'][index].values[0]
        mu_p = efficient_frontier['mu_p'][index].values[0]
        sigma_p = efficient_frontier['sig_p'][index].values[0]
        sharpe_p = efficient_frontier['sr_p'][index].values[0]

        return wts, mu_p, sigma_p, sharpe_p

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _compute_target_risk_portfolio(self, mu_vec, sigma_mat, target_risk, risk_free=0, shorts=True):

        efficient_frontier = self._compute_efficient_frontier(mu_vec, sigma_mat, risk_free, shorts=shorts)
        if efficient_frontier['sig_p'].max() <= target_risk:
            log.warn('TARGET_RISK {} > EFFICIENT FRONTIER MAXIMUM {}; SETTING IT TO MAXIMUM'.
                     format(target_risk, efficient_frontier['sig_p'].max()))
            index = len(efficient_frontier) - 1
        elif efficient_frontier['sig_p'].min() >= target_risk:
            log.warn('TARGET RISK {} < GLOBAL MINIMUM {}; SETTING IT TO GLOBAL MINIMUM'.
                     format(target_risk, efficient_frontier['sig_p'].max()))
            index = 1
        else:
            index = efficient_frontier.index[efficient_frontier['sig_p'] >= target_risk][0]

        wts = efficient_frontier['wts_p'][index]
        mu_p = efficient_frontier['mu_p'][index]
        sigma_p = efficient_frontier['sig_p'][index]
        sharpe_p = efficient_frontier['sr_p'][index]

        return wts, mu_p, sigma_p, sharpe_p


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DownsideProtectionModel():

    def __init__(self, context, mode=None, rule=None, formula=None, *args):

        self.mode = mode
        self.rule = rule
        self.formula = formula
        self.args = args

        self.modes = {'BY_RULE': self._by_rule,
                      'RAA': self._apply_RAA,
                      'BY_FORMULA': self._by_formula
                      }

        self.caller = None  # portfolio or strategy object calling the model

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def apply_protection(self, context, allocations, cash_proxy=None, *args):

        # apply downside protection model to cash_proxy, if it fails, set cash_proxy to risk_free

        if context.allow_cash_proxy_replacement:
            if context.raw_data['price'][cash_proxy][-1] < context.algo_data['price'][-43:].mean():
                cash_proxy = context.risk_free

        new_allocations = self.modes[self.mode](context, allocations, cash_proxy, *args)

        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _by_rule(self, context, allocations, cash_proxy, *args):
        try:
            triggers = context.rules[self.rule].apply_rule(context)[allocations.index]
        except:
            raise RuntimeError('UNABLE TO APPLY RULE {} FOR {}'.format(self.rule, self.caller.ID))

        new_allocations = pd.Series([0 if triggers[a] else allocations[a] for a in allocations.index],
                                    index=allocations.index)
        new_allocations[cash_proxy] = new_allocations[cash_proxy] + (1 - new_allocations.sum())

        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _apply_RAA(self, context, allocations, cash_proxy, *args):
        excess_returns = context.algo_data['EMOM']

        tmp1 = [0.5 if excess_returns[asset] > 0 else 0. for asset in allocations.index]

        prices = context.algo_data['price']
        MA = context.algo_data['smma']

        tmp2 = [0.5 if prices[asset] > MA[asset] else 0. for asset in allocations.index]

        dpf = pd.Series([x + y for x, y in zip(tmp1, tmp2)], index=allocations.index)

        new_allocations = allocations * dpf
        new_allocations[cash_proxy] = new_allocations[cash_proxy] + (1 - np.sum(new_allocations))

        record('BOND EXPOSURE', new_allocations[cash_proxy])

        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _by_formula(self, context, allocations, cash_proxy, *args):
        if self.formula == 'DPF':
            try:
                new_allocations = self._apply_DPF(context, allocations, cash_proxy, *args)
            except:
                raise ValueError('FORMULA "{}" DOES NOT EXIST OR ERROR CALCULATING FORMULA'.formmat(self.formula))
        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _apply_DPF(self, context, allocations, cash_proxy, *args):
        securities = args[0][0]
        N = len(securities)
        try:
            triggers = context.rules[self.rule].apply_rule(context)[securities]
        except:
            raise ValueError('UNABLE TO APPLY RULE {}'.format(self.rule))

        num_neg = triggers[triggers == True].count()
        dpf = float(num_neg) / N
        log.info("DOWNSIDE PROTECTION FACTOR = {}".format(dpf))

        new_allocations = (1. - dpf) * allocations
        new_allocations[cash_proxy] += dpf

        return new_allocations


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DownsideProtectionModel():

    def __init__(self, context, mode=None, rule=None, formula=None, *args):

        self.mode = mode
        self.rule = rule
        self.formula = formula
        self.args = args

        self.modes = {'BY_RULE': self._by_rule,
                      'RAA': self._apply_RAA,
                      'BY_FORMULA': self._by_formula
                      }

        self.caller = None  # portfolio or strategy object calling the model

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def apply_protection(self, context, allocations, cash_proxy=None, *args):

        # apply downside protection model to cash_proxy, if it fails, set cash_proxy to risk_free

        if context.allow_cash_proxy_replacement:
            if context.raw_data['price'][cash_proxy][-1] < context.algo_data['price'][-43:].mean():
                cash_proxy = context.risk_free

        new_allocations = self.modes[self.mode](context, allocations, cash_proxy, *args)

        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _by_rule(self, context, allocations, cash_proxy, *args):
        try:
            triggers = context.rules[self.rule].apply_rule(context)[allocations.index]
        except:
            raise RuntimeError('UNABLE TO APPLY RULE {} FOR {}'.format(self.rule, self.caller.ID))

        new_allocations = pd.Series([0 if triggers[a] else allocations[a] for a in allocations.index],
                                    index=allocations.index)
        new_allocations[cash_proxy] = new_allocations[cash_proxy] + (1 - new_allocations.sum())

        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _apply_RAA(self, context, allocations, cash_proxy, *args):
        excess_returns = context.algo_data['EMOM']

        tmp1 = [0.5 if excess_returns[asset] > 0 else 0. for asset in allocations.index]

        prices = context.algo_data['price']
        MA = context.algo_data['smma']

        tmp2 = [0.5 if prices[asset] > MA[asset] else 0. for asset in allocations.index]

        dpf = pd.Series([x + y for x, y in zip(tmp1, tmp2)], index=allocations.index)

        new_allocations = allocations * dpf
        new_allocations[cash_proxy] = new_allocations[cash_proxy] + (1 - np.sum(new_allocations))

        record('BOND EXPOSURE', new_allocations[cash_proxy])

        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _by_formula(self, context, allocations, cash_proxy, *args):
        if self.formula == 'DPF':
            try:
                new_allocations = self._apply_DPF(context, allocations, cash_proxy, *args)
            except:
                raise ValueError('FORMULA "{}" DOES NOT EXIST OR ERROR CALCULATING FORMULA'.formmat(self.formula))
        return new_allocations

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _apply_DPF(self, context, allocations, cash_proxy, *args):
        securities = args[0][0]
        N = len(securities)
        try:
            triggers = context.rules[self.rule].apply_rule(context)[securities]
        except:
            raise ValueError('UNABLE TO APPLY RULE {}'.format(self.rule))

        num_neg = triggers[triggers == True].count()
        dpf = float(num_neg) / N
        log.info("DOWNSIDE PROTECTION FACTOR = {}".format(dpf))

        new_allocations = (1. - dpf) * allocations
        new_allocations[cash_proxy] += dpf

        return new_allocations


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Rule():
    functions = {'EQ': lambda x, y: x == y,
                 'LT': lambda x, y: x < y,
                 'GT': lambda x, y: x > y,
                 'LE': lambda x, y: x <= y,
                 'GE': lambda x, y: x >= y,
                 'NE': lambda x, y: x != y,
                 'AND': lambda x, y: x & y,
                 'OR': lambda x, y: x | y,
                 }

    def __init__(self, context, name='', rule='', apply_to='all'):

        self.name = name
        # remove spaces
        self.rule = rule.replace(' ', '')
        self.temp = ''
        self.apply_to = apply_to

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def apply_rule(self, context):

        ''' routine to evaluate a rule consisting of a string formatted as 'conditional [AND|OR conditional]'
            where conditionals are logical expressions, pandas series of logical expressions
            or pandas dataframes of logical expressions. Returns True or False,
            pandas series of True/False or pandas dataframe of True/False respectively.
        '''

        if self.rule == 'always_true':
            return True

        self.temp = self._replace_operators(self.rule)
        # get the first condition of the rule and evaluate it
        condition, result, cjoin = self._get_next_conditional(context)

        # log.debug ('result = {}'.format(result))

        while cjoin != None:
            # get the rest of the rule
            self.temp = self.temp[len(condition) + len(cjoin):]
            # get the next conditional of the rule and evaluate it
            func = Rule.functions[cjoin]
            condition, tmp_result, cjoin = self._get_next_conditional(context)

            result = func(result, tmp_result)

            # log.debug ('intermediate result = {}'.format(result))

        # log.debug ('final result = {}'.format(result))
        return result

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_next_conditional(self, context):
        condition, cjoin = self._get_conditional(self.temp)
        result = self._evaluate_condition(context, condition)
        if self.apply_to != 'all':
            result = result[self.apply_to]
        return condition, result, cjoin
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _replace_operators(self, s):

        ''' to make it easy to find operators in the rule s, replace ['=', '>', '<', '>=', '<=', '!=', 'and', 'or']
            with ['EQ', 'GT', 'LT', 'GE', 'LE', 'NE', 'AND', 'OR'] respectively
        '''

        s1 = s.replace('and', 'AND').replace('or', 'OR').replace('!=', 'NE').replace('<=', 'LE').replace('>=', 'GE')
        s1 = s1.replace('=', 'EQ').replace('<', 'LT').replace('>', 'GT')
        return s1

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_conditional(self, s):

        ''' routine to find first ocurrence of "AND" or "OR" in rule s. Returns
        conditional to the left of AND/OR and either 'AND', 'OR' or None '''

        pos_AND = [s.find('AND') if s.find('AND') != -1 else len(s)][0]
        pos_OR = [s.find('OR') if s.find('OR') != -1 else len(s)][0]
        condition, cjoin = [(s.split('AND')[0], 'AND') if pos_AND < pos_OR else (s.split('OR')[0], 'OR')][0]
        if pos_AND == len(s) and pos_OR == len(s):
            cjoin = None
        return condition, cjoin

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_operator(self, condition):

        '''routine to extract the operator and its position from the conditional expression
        '''
        for o in ['EQ', 'GT', 'LT', 'GE', 'LE', 'NE', 'AND', 'OR']:
            if condition.find(o) > 0:
                return o, condition.find(o)
        raise ('UNKNOWN OPERATOR')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _get_operand_value(self, context, operand):
        if operand.startswith('('):
            tuple_0 = operand[1:operand.find(',')].strip("'").strip('"')
            tuple_1 = operand[operand.find(',') + 1:-1]
            return context.algo_data[tuple_0][tuple_1]
        if operand[0].isdigit() or operand.startswith('.') or operand.startswith('-'):
            return float(operand)
        elif isinstance(operand, str):
            return context.algo_data[operand.strip("'").strip('"')]
        else:
            op = context.algo_data[operand[0]]
            if operand[1] != None:
                op = context.algo_data[operand[0].strip("'").strip('"')][operand[1]]
            return op

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _evaluate_condition(self, context, condition):
        operator, position = self._get_operator(condition)
        x = self._get_operand_value(context, condition[:position])
        y = self._get_operand_value(context, condition[position + 2:])
        # log.debug ('x = {}, y = {}, operator = {}'.format(x, y, operator))
        func = Rule.functions[operator]

        return func(x, y)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Transform():

    def __init__(self, context, name='', function='', inputs=[], kwargs={}, outputs=[]):

        self.name = name
        self.function = function
        self.inputs = inputs
        self.kwargs = kwargs
        self.outputs = outputs

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def apply_transform(self, context):

        # transform format [([<data_items>], function, <data_item>, args)]

        context.dp = pd.Panel(context.raw_data)

        if self.function in TALIB_FUNCTIONS:
            return self._apply_talib_function(context)

        elif self.function.__name__.startswith('roll') or self.function.__name__.startswith(
                'expand') or self.function.__name__ == '<lambda>':
            return self._apply_pandas_function(context)

        else:
            return self.function(self, context)

        raise ValueError('UNKNOWN TRANSFORM {}'.format(self.function))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _apply_talib_function(self, context):

        '''
        Routine to apply transform to data provided as a pandas Panel.
        Inputs:
        dp: pandas dataPanel consisting of a DataFrame for each item in ['open', 'high', 'low', 'close', 'volume',
            'price']; each DataFrame has column names = asset names
        inputs : list of dp items to be used as inputs. If empty (=[]), routine will use default input
                        names from the talib function DOC string
        function : talib function name (e.g. RSI, MACD, ADX etc.) - see list of imported functions above
        output_names : list of names for the tranforms DataFrames
        NOTE: names must be unique and there must be a name for each output (some transforms produce more than
                one output e.g MACD produces 3 outputs)
        args : empty list (=[]), in which case default values are obtained from talib function DOC string.
                otherwise, custom parameters may be provided as a list of integers, the parameters matching
                the FULL parameter list, as per the function DOC string

        Outputs:
            pandas DataPanel with new items (DataFrames) appended for each transform output.

        '''

        # parameters = [a for a in self.args]
        parameters = [self.kwargs[key] for key in iter(self.kwargs)]
        if parameters == []:
            parameters = [int(s) for s in re.findall('\d+', self.function.__doc__)]
        data_items = re.findall("(?<=\')\w+", self.function.__doc__)
        if data_items == []:
            inputs = self.inputs
        else:
            inputs = data_items

        for output in self.outputs:
            context.dp[output] = pd.DataFrame(0, index=context.dp.major_axis, columns=context.dp.minor_axis)

        for asset in context.dp.minor_axis:
            data = [context.dp.transpose(2, 1, 0)[asset][i].values for i in inputs]
            args = data + parameters
            transform = self.function(*args)
            if len(transform) == len(self.outputs) or len(transform) > 3:
                pass
            else:
                raise ValueError('** ERROR : must be output_names for each output')

            if len(self.outputs) == 1:
                context.dp[self.outputs[0]][asset] = transform
            else:
                for i, output in enumerate(self.outputs):
                    context.dp[output][asset] = transform[i]

        # for some reason, if you don't do this, then dp.transpose(2,1,0) gives dp[output][asset] as 0 !!
        for name in self.outputs:
            context.dp[name] = context.dp[name]

        return context.dp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _apply_pandas_function(self, context):

        '''
        Routine to apply pandas function to column(s) of data provided as Pandas DataFrame.
        Allowed functions include all the pandas.rolling_ and pandas.expanding_ functions.
        NOTE: corr and cov are NOT allowed here, but must be implemented as CUSTOM FUNCTIONS
        Inputs:
            dp = Pandas DataPanel with data to be transformed in one (or more) panel items
            NOTE: in the case of CORR or COV, columns contain price data for each stock.
            inputs = name(s) of item(s) containing data to be transformed (DataFrames with columns = asset names)
            function = name of pandas function provided by user (pd.rolling_  or pd.expanding_ )
            args = list of arguments required by function
        Output:
            Pandas DataPanel with appended items containing the transformed data as DataFrames, or,
            as in the case of CORR and COV functions, the item is a DataPanel of correlations/covariances

        '''
        if 'corr' in self.function.__name__ or 'cov' in self.function.__name__:
            raise ValueError('** ERROR: Correlation and Covariance must be implemented as CUSTOM FUNCTIONS')

        for asset in context.dp.minor_axis:
            context.dp[self.outputs[0]] = self.function(context.dp[self.inputs[0]], *self.args)

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Custom Transforms

    def n_period_return(self, context):

        '''
        percentage return (optionally, excess return) over n periods
        most recent period can optionally be skipped

        kwargs[0] = 'no of periods'
        kwargs[1] = 'period' : 'D'|'W'|'M' (day|week||month)
        kwargs[2] = 'skip_period' (optional = False)

        '''
        try:
            skip_period = self.kwargs['skip_period']
        except:
            skip_period = False

        # TODO : need to return excess_return, depending on risk_free

        prices = context.dp[self.inputs[0]]

        no_of_periods = self.kwargs['lookback']
        # if no 'period' kwarg, assume 'D'
        try:
            period = self.kwargs['period']
        except:
            period = 'D'

        if period in ['W', 'M']:
            returns = prices.resample(period).last().pct_change(no_of_periods)
        elif period == 'D':
            returns = prices.pct_change(no_of_periods)

        idx = -1
        if skip_period:
            idx = - 2

        df = pd.DataFrame(0, index=context.dp.major_axis,
                          columns=context.dp.minor_axis)
        if not isinstance(context.risk_free, int):
            returns = returns.sub(returns[context.risk_free], axis=0)

        ds = returns.iloc[idx]
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def simple_mean_monthly_average(self, context):

        h = context.dp[self.inputs[0]]
        lookback = self.kwargs['lookback']
        ds = h.resample('M').last()[-lookback - 1:-1].mean()

        df = pd.DataFrame(0, index=h.index, columns=h.columns)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def momentum(self, context):

        lookback = self.kwargs['lookback']
        ds = context.dp[self.inputs[0]].iloc[-1] / context.dp[self.inputs[0]].iloc[-lookback] - 1

        df = pd.DataFrame(0, index=context.dp.major_axis, columns=context.dp.minor_axis)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def daily_returns(self, context):

        context.dp[self.outputs[0]] = context.dp['price'].pct_change(1)

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def excess_momentum(self, context):

        lookback = self.kwargs['lookback']
        ds = context.dp['price'].pct_change(lookback).iloc[-1] - \
             context.dp['price'][context.risk_free].pct_change(lookback).iloc[-1]

        df = pd.DataFrame(0, index=context.dp.major_axis, columns=context.dp.minor_axis)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def log_returns(self, context):

        try:
            context.dp[self.outputs[0]] = np.log(1. + context.dp['price'].pct_change(1))
        except:
            raise RuntimeError("Inputs must be ['price']")

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def historic_volatility(self, context):

        lookback = self.kwargs['lookback']
        try:
            ret_log = np.log(1. + context.dp['price'].pct_change())
        except:
            raise RuntimeError("Inputs must be ['price']")

        # this is for pandas < 0.18
        # hist_vol = pd.rolling_std(ret_log, lookback)

        # this is for pandas ver > 0.18
        hist_vol = ret_log.rolling(window=lookback, center=False).std()

        context.dp[self.outputs[0]] = hist_vol * np.sqrt(252 / lookback)

        return context.dp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def average_excess_return_momentum(self, context):

        '''
        Average Excess Return Momentum

        average_excess_return_momentum is the average of monthly returns in excess of the risk_free rate for multiple
        periods (1,3,6,12 months). In addtion, average momenta < 0 are set to 0.

        '''
        h = context.dp[self.inputs[0]].copy()
        hm = h.resample('M').last()
        hb = h.resample('M').last()[context.risk_free]

        ds = (hm.ix[-1] / hm.ix[-2] - hb.ix[-1] / hb.ix[-2] + hm.ix[-1] / hm.ix[-4]
              - hb.ix[-1] / hb.ix[-4] + hm.ix[-1] / hm.ix[-7] - hb.ix[-1] / hb.ix[-7]
              + hm.ix[-1] / hm.ix[-13] - hb.ix[-1] / hb.ix[-13]) / 22
        ds[ds < 0] = 0
        df = pd.DataFrame(0, index=h.index, columns=h.columns)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def paa_momentum(self, context):

        ds = context.dp[self.inputs[0]].iloc[-1] / context.dp[self.inputs[1]].iloc[-1] - 1

        df = pd.DataFrame(0, index=context.dp.major_axis, columns=context.dp.minor_axis)
        df.iloc[-1] = ds

        context.dp[self.outputs[0]] = df

        return context.dp

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def crossovers(self, context):
        df1 = context.dp[self.inputs[0]]
        df2 = context.dp[self.inputs[1]]
        down = (df1 > df2) & (df1.shift(1) < df2.shift(1)).astype(int)
        up = (df1 < df2) & (df1.shift(1) > df2.shift(1)).astype(int)
        # returns +1 for crosses above and -1 for crosses below
        return down * (-1) + up
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def slope(self, context):

        lookback = self.kwargs['lookback']
        ds = pd.Series(index=context.dp.minor_axis)
        for asset in context.dp.minor_axis:
            ds[asset] = talib.LINEARREG_SLOPE(context.dp[self.inputs[0]][asset].values, lookback)[-1]
        df = pd.DataFrame(0, index=context.dp.major_axis, columns=context.dp.minor_axis)
        df.iloc[-1] = ds
        return df


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Configurator():
    '''
    The Configurator uses the Strategy Parameters set up by the StrategyParameters Class and dictionary of global
    parameters to create all the objects required for the algorithm.

    '''

    # def __init__ (self, context, strategies, global_parameters=None) :
    def __init__(self, context, strategies):
        self.strategies = strategies
        # self.global_parameters = global_parameters
        # self._set_global_parameters (context)
        context.tranforms = define_transforms(context)

        context.algo_rules = define_rules(context)
        self._configure_algo_strategies(context)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _configure_algo_strategies(self, context):
        for s in self.strategies:
            self._check_valid_parameters(context, s)
            self._configure_strategy(context, s)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # TODO: would be better to make this table-driven
    # TODO : check strategy names are unique
    # TODO : compute context.max_lookback

    def _check_valid_parameters(self, context, strategy):
        N = len(strategy.portfolios)
        s = strategy
        self._check_valid_parameter(context, s, strategy.portfolios, 'portfolios', list, N, list, ''),
        self._check_valid_parameter(context, s, strategy.portfolio_allocation_modes, 'portfolio_allocation_modes',
                                    list, N, str, VALID_PORTFOLIO_ALLOCATION_MODES),
        self._check_valid_parameter(context, s, strategy.security_weights, 'security_weights', list, N, list, ''),
        self._check_valid_parameter(context, s, strategy.portfolio_allocation_formulas, 'portfolio_allocation_formulas',
                                    list,
                                    N, str, VALID_PORTFOLIO_ALLOCATION_FORMULAS),
        self._check_valid_parameter(context, s, strategy.security_scoring_methods, 'security_scoring_methods', list, N,
                                    str, VALID_SECURITY_SCORING_METHODS),
        self._check_valid_parameter(context, s, strategy.security_scoring_factors, 'security_scoring_factors', list, N,
                                    dict, ''),
        self._check_valid_parameter(context, s, strategy.security_n_tops, 'security_n_tops', list, N, int, '')
        self._check_valid_parameter(context, s, strategy.portfolio_scoring_method, 'portfolio_scoring_method', list, 1,
                                    str, VALID_PORTFOLIO_SCORING_METHODS),
        self._check_valid_parameter(context, s, strategy.portfolio_scoring_factors, 'portfolio_scoring_factors', list,
                                    1, dict, ''),
        self._check_valid_parameter(context, s, strategy.portfolio_n_top, 'portfolio_n_top', list, 1, int, '')
        self._check_valid_parameter(context, s, strategy.protection_modes, 'protection_modes', list, N,
                                    str, VALID_PROTECTION_MODES),
        self._check_valid_parameter(context, s, strategy.protection_rules, 'protection_rules', list, N, str, ''),
        self._check_valid_parameter(context, s, strategy.protection_formulas, 'protection_formulas', list, N,
                                    str, VALID_PROTECTION_FORMULAS),
        self._check_valid_parameter(context, s, strategy.cash_proxies, 'cash_proxies', list, N, type(symbols('SPY')[0]),
                                    ''),
        self._check_valid_parameter(context, s, strategy.strategy_allocation_mode, 'strategy_allocation_mode', str,
                                    1, str, VALID_STRATEGY_ALLOCATION_MODES)
        self._check_valid_parameter(context, s, strategy.portfolio_weights, 'portfolio_weights', list, N, float, ''),
        self._check_valid_parameter(context, s, strategy.strategy_allocation_formula, 'strategy_allocation_formula',
                                    str,
                                    1, str, VALID_STRATEGY_ALLOCATION_FORMULAS)
        self._check_valid_parameter(context, s, strategy.strategy_allocation_rule, 'strategy_allocation_rule', str,
                                    1, str, '')
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _check_valid_parameter(self, context, s, param, name, param_type, param_length, item_type, valid_params):

        if name in ['strategy_allocation_mode', 'portfolio_weights', 'strategy_allocation_formula',
                    'strategy_parameters', 'strategy_allocation_rule', 'portfolio_scoring_method',
                    'portfolio_scoring_factors', 'portfolio_n_top']:
            self._check_strategy_parameters(context, s, param, name, param_type, param_length, item_type, valid_params)
        else:
            # if param is None and name in NONE_NOT_ALLOWED :
            #     raise RuntimeError ('"None" not allowed for parameter {}'.format(name))
            # if param is None and 'FIXED' in s.portfolio_allocation_modes:
            #     raise RuntimeError ('Parameter {} cannot be None for portfolio_allocation_mode "FIXED"'.format(name))
            # else:
            #     # valid None parameter
            #     return

            self._check_param_type(name, param, param_type)

            if len(param) != param_length:
                raise RuntimeError('Parameter {} must be of length {} not {}'.format(name, param_length, len(param)))
            for n in range(param_length):
                if param[n] == None and name in NONE_NOT_ALLOWED:
                    raise RuntimeError('"None" not allowed for parameter {}'.format(name))
                elif param[n] == None:
                    if name == 'scoring_factors' and s.protection_modes == 'RS':
                        self._check_valid_scoring_factors(name, param[n])
                    # if name == 'security_n_tops' and s.portfolio_allocation_modes[n] == 'FIXED' :
                    #     if param[n] != len(s.security_weights[n]) :
                    #         raise RuntimeError ('For portfolio_allocation_mode = "FIXED", n_tops must equal no of security weights')
                    continue
                if valid_params != "":
                    if param[n] not in valid_params:
                        raise RuntimeError('Invalid {} {}'.format(name, param[n]))
                if not isinstance(param[n], item_type):
                    raise RuntimeError('Items of {} must be of type {} not {}'.format(name, item_type, type(param[n])))
                if name == 'portfolios':
                    self._check_valid_portfolio(param[n])

                if name.endswith('_weights') and np.sum(param[n]) != 1.:
                    raise RuntimeError('Sum of {} must equal 1, not {}'.format(name, np.sum(param)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_strategy_parameters(self, context, s, param, name, param_type, param_length, item_type, valid_params):
        if name == 'strategy_allocation_mode':
            if param not in valid_params:
                raise RuntimeError('Invalid strategy_allocation_mode {}'.format(param))
        elif name == 'portfolio_weights' and s.strategy_allocation_mode == 'FIXED':
            if np.sum(param) != 1.:
                raise RuntimeError('portfolio_weights must be a list of floating point numbers with sum = 1')
        elif name == 'strategy_allocation_formula':
            if param not in valid_params:
                raise RuntimeError('Invalid strategy_allocation_formula {}'.format(param))
        elif name == 'strategy_allocation_rule' and s.strategy_allocation_rule != None:
            valid_rules = [rule.name for rule in context.algo_rules]
            if s.strategy_allocation_rule not in valid_rules:
                raise RuntimeError(
                    'Strategy rule {} not found. Check rule definitions'.format(s.strategy_allocation_rule))
        elif name == 'portfolio_scoring_method':
            if param not in valid_params:
                raise RuntimeError('Invalid strategy_allocation_formula {}'.format(param))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _check_param_type(self, name, param, param_type):
        if not isinstance(param, param_type):
            raise RuntimeError('Parameter {} must be of type {} not {}'.format(name, param_type, type(param)))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_valid_scoring_factors(self, name, factors):
        sum_of_weights = 0.

        for key in factors.keys():
            if not key[0] in ['+', '-']:
                raise RuntimeError('First character of scoring factor {}, must be "+" or "-"'.format(key))
            sum_of_weights += factors[key]
        if sum_of_weights != 1.:
            raise RuntimeError('Sum of {} weights must equal 1, not {}'.format(name, sum_of_weights))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def _check_valid_portfolio(self, pfolio):
        if len(pfolio) < 1:
            raise RuntimeError('Portfolio must have at least one item')
        for n in range(len(pfolio)):
            if not isinstance(pfolio[n], type(symbols('SPY')[0])):
                raise RuntimeError('portfolio item {} must be of type '.format(type(symbols('SPY')[0])))
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _configure_strategy(self, context, s):

        portfolios = []

        for n in range(len(s.portfolios)):
            if s.security_scoring_methods[n] is None:
                scoring_model = None
            else:
                scoring_model = ScoringModel(context,
                                             method=s.security_scoring_methods[n],
                                             factors=s.security_scoring_factors[n],
                                             n_top=s.security_n_tops[n])

            if s.protection_modes[n] == None:
                downside_protection_model = None
            else:
                downside_protection_model = DownsideProtectionModel(context,
                                                                    mode=s.protection_modes[n],
                                                                    rule=s.protection_rules[n],
                                                                    formula=s.protection_formulas[n])

            portfolios = portfolios + \
                         [Portfolio(context,
                                    ID=s.ID + '_p' + str(n + 1),
                                    securities=s.portfolios[n],
                                    allocation_model=AllocationModel(context,
                                                                     mode=s.portfolio_allocation_modes[n],
                                                                     weights=s.security_weights[n],
                                                                     formula=s.portfolio_allocation_formulas[n],
                                                                     kwargs=s.portfolio_allocation_kwargs[n]
                                                                     ),
                                    scoring_model=scoring_model,
                                    downside_protection_model=downside_protection_model,
                                    cash_proxy=s.cash_proxies[n]
                                    )]

        if s.portfolio_scoring_method is None:
            scoring_model = None
        else:
            scoring_model = ScoringModel(context,
                                         method=s.portfolio_scoring_method,
                                         factors=s.portfolio_scoring_factors,
                                         n_top=s.portfolio_n_top)
        s.strategy = Strategy(context,
                              ID=s.ID,
                              allocation_model=AllocationModel(context,
                                                               mode=s.strategy_allocation_mode,
                                                               weights=s.portfolio_weights,
                                                               formula=s.strategy_allocation_formula,
                                                               kwargs=s.strategy_allocation_kwargs,
                                                               rule=s.strategy_allocation_rule),
                              scoring_model=scoring_model,
                              portfolios=portfolios
                              )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class StrategyParameters():
    '''
    StrategyParameters hold the parameters for each strategy for a single or multistrategy algoritm

    calling:

    strategy = StrategyParameters(context, portfolios, portfolio_allocation_modes, security_weights,
                         portfolio_allocation_formulas,
                         scoring_methods, scoring_factors, n_tops,
                         protection_modes, protection_rules, protection_formulas,
                         cash_proxies, strategy_allocation_mode, portfolio_weights=None,
                         strategy_allocation_formula, strategy_allocation_rule)

    see below for definition of each parameter

    '''

    # NOTE: kwarg label 'lookback' should be ALWAYS be used for timeseries lookback periods!
    def __init__(self, context, ID, portfolios=[],
                 portfolio_allocation_modes=[], security_weights=None,
                 portfolio_allocation_formulas=None,
                 portfolio_allocation_kwargs=None,
                 security_scoring_methods=None, security_scoring_factors=None,
                 security_n_tops=None,
                 protection_modes=None, protection_rules=None, protection_formulas=None,
                 cash_proxies=[],
                 strategy_allocation_mode='EW', portfolio_weights=None,
                 portfolio_scoring_method=None, portfolio_scoring_factors=None,
                 portfolio_n_top=None,
                 strategy_allocation_formula=None,
                 strategy_allocation_kwargs=None,
                 strategy_allocation_rule=None):

        # strategy ID, must be unique str
        # eg 'strat1'
        self.ID = ID
        # list of n valid security lists (must be at least one security list)
        # eg [symbols('SPY','EEM')] or [symbols('SPY','EEM'), symbols('TLT','JNK','SHY'),....]
        self.portfolios = portfolios
        n = len(portfolios)
        # list of n VALID_PORTFOLIO_ALLOCATION_MODES, one for each portfolio
        # eg ['EW'] or ['EW', 'PROPORTIONAL',.....]
        self.portfolio_allocation_modes = portfolio_allocation_modes
        # either None or list of n kwargs each containing kwargs matching porfolio_allocation_modes
        # eg None or [kwargs1] or [kwargs1, kwargs2, ....] where kargsn = dict of kwargs relevant to modes
        self.portfolio_allocation_kwargs = portfolio_allocation_kwargs
        if portfolio_allocation_kwargs is None:
            self.portfolio_allocation_kwargs = [None for i in range(n)]
        # None or list of n lists of security weights for 'FIXED' portfolio_allocation_modes, else None
        # eg None or [[0.2,0.8]] or [[0.5,0.5],[0.1,0.7,0.2]...] where sum of each list = 1
        self.security_weights = security_weights
        if security_weights is None:
            self.security_weights = [None for i in range(n)]
            # None or list of n VALID_PORTFOLIO_ALLOCATION_FORMULAS for 'BY_FORMULA'
        # portfolio_allocation_modes, else None
        # eg None or [valid formula] or [None, valid formula, ...] for each portfolio where allocation 'BY_FORMULA'
        self.portfolio_allocation_formulas = portfolio_allocation_formulas
        if portfolio_allocation_formulas is None:
            self.portfolio_allocation_formulas = [None for i in range(n)]
            # None or list of n VALID_SECURITY_SCORING_METHODS, one for each portfolio
        # eg None or ['RS'] or [None, 'EAA', ....]
        self.security_scoring_methods = security_scoring_methods
        if security_scoring_methods is None:
            self.security_scoring_methods = [None for i in range(n)]
            # None or list of n dicts of scoring factors, relevant to each scoring method
        # eg None or [factors1] or [None, factors2, ...] where factorsn = dict of factors relevant to scoring methods
        self.security_scoring_factors = security_scoring_factors
        if security_scoring_factors is None:
            self.security_scoring_factors = [None for i in range(n)]
            # None or list of n_tops, one for each ranked portfolio, else None; n_top <= len(portfolio) - 1
        # eg None or [1], [1,2,...]
        self.security_n_tops = security_n_tops
        if security_n_tops is None:
            self.security_n_tops = [None for i in range(n)]
            # None or list of n VALID_PROTECTION_MODES, one for each portfolio
        # eg None or ['RAA'] or [None, 'BY_RULE','BY_FORMULA', ....]
        self.protection_modes = protection_modes
        if protection_modes is None:
            self.protection_modes = [None for i in range(n)]
            # None or list of n valid rules for portfolios with protection mode 'BY_RULE', else None
        # eg None or [valid rule] or [None, valid rule, ...] for each portfolio where allocation 'BY_RULE'
        self.protection_rules = protection_rules
        if protection_rules is None:
            self.protection_rules = [None for i in range(n)]
            # None or list of n VALID_PROTECTION_FORMULAS for portfolios with protection mode 'BY_FORMULA', else None
        # eg None or [valid formula] or [None, valid formula, ...] for each portfolio where allocation 'BY_FORMULA'
        self.protection_formulas = protection_formulas
        if protection_formulas is None:
            self.protection_formulas = [None for i in range(n)]
            # list of n valid securities to be used as cash proxies, one for each portfolio
        # eg [symbol('SHY')] or [symbol('SHY'), symbol('TLT'),...]  NOTE: symbol NOT symbols!!
        self.cash_proxies = cash_proxies
        # any one of VALID_STRATEGY_ALLOCATION_MODES
        # eg 'RISK_TARGET'
        self.strategy_allocation_mode = strategy_allocation_mode
        # None or any kwargs relevant to the strategy_allocation_mode
        # eg {'lookback': 100, 'target_risk': 0.01}
        self.strategy_allocation_kwargs = strategy_allocation_kwargs
        # None or list of n portfolio weights (sum = 1) if strategy_allocation_mode is 'FIXED'
        self.portfolio_weights = portfolio_weights
        if portfolio_weights is None:
            self.portfolio_weights = [None for i in range(n)]
            # None or one of VALID_STRATEGY_ALLOCATION_FORMULAS, if strategy_allocation_mode is 'BY_FORMULA'
        # eg 'PAA'
        self.strategy_allocation_formula = strategy_allocation_formula
        # None or one of VALID_PORTFOLIO_SCORING_METHODS
        # eg 'RS'
        self.portfolio_scoring_method = portfolio_scoring_method
        # None or dict of factors to be used for scoring (ranking) portfolios
        # eg {'+factor1': 10, '-factor2':20} - NOTE that factor names must be prefixed by '+' or '-'
        # to indicate whether to rank factor ascending (+) or descending (-)
        self.portfolio_scoring_factors = portfolio_scoring_factors
        # None or integer <= no of portfolios - 1
        # eg 2
        self.portfolio_n_top = portfolio_n_top
        # None or one of VALID_STRATEGY_ALLOCATION_RULES
        # eg None
        self.strategy_allocation_rule = strategy_allocation_rule
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def handle_data(context, data):

#   TRAILING STOPS
# if trailing stops not required, this can be commented out

#     if not context.use_trailing_stops:
#         return

#     # see https://www.quantopian.com/posts/trailing-stop-loss-with-multiple-securities
#     for security in context.portfolio.positions:
#         current_position = context.portfolio.positions[security].amount
#         context.stop_price[security] = max(context.stop_price[security] if security in context.stop_price
#                                         else 0, context.stop_pct * data.current(security, 'price'))
#         if (data.current(security, 'price') < context.stop_price[security]) and (current_position > 0):
#             order_target_percent(security, 0)
#             del context.stop_price[security]
#             log.info("Trail Selling {} at {}".format(security.symbol, data.current(security, 'price')))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def before_trading_start(context, data):
    """
    Called every day before market open.
    """

    # log.info('PLATFORM = ' + get_environment('platform') + str(context.day_no))

    # ONLY IF WE REQUIRE TO FILL THE PIPELINE WITH DATA (IE NOT REQUIRED FOR QUANTOPIUAN)
    # if get_environment('platform') == 'zipline':
    #     # allow data buffer to fill in the ZIPLINE ENVIRONMENT
    #     if context.day_no <= context.max_lookback:
    #         context.day_no += 1
    #         return

    # generate updated algo data
    # log.info('GENERATE DATA')
    context.algo_data = context.data.update(context, data)

    if np.sum(context.strategies[0].weights) > 1.e-07:
        # wait until first allocation to generate portfolio and strategy metrics
        context.data.update_portfolio_and_strategy_metrics(context, data)

    return context.algo_data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# define transforms
#########################################################################################################
#########################################################################################################
# the following routines contain all the configuration details
# any transform which relies on lookback data MUST have a 'lookback' kwarg
# and, optionally, 'period' = <no. of days> |'W'| 'M'
# NOTE: kwarg label 'lookback' should be ALWAYS be used for timeseries lookback periods!

def define_transforms (context) :
# Define transforms
# select transforms required and make sure correct parameters are used
# no need to comment out unused transforms, but they will slow algo down

    log.info('DEFINE TRANSFORMS')

    context.transforms = [
        Transform(context, name='momentum', function=Transform.n_period_return, inputs=['price'],
                  kwargs={'lookback':45, 'risk_free': 0, 'skip_period': False}, outputs=['momentum']),
        Transform(context, name='mom_A', function=talib.ROCP, inputs=['price'],
                  kwargs={'lookback':43}, outputs=['mom_A']),
        Transform(context, name='mom_B', function=talib.ROCP, inputs=['price'],
                  kwargs={'lookback':21}, outputs=['mom_B']),
        Transform(context, name='daily_returns', function=Transform.daily_returns, inputs=['price'],
                  kwargs={}, outputs=['daily_returns']),
        Transform(context, name='vol_C', function=talib.STDDEV, inputs=['daily_returns'],
                  kwargs={'lookback':20}, outputs=['vol_C']),
        Transform(context, name='hist_vol', function=Transform.historic_volatility, inputs=['price'],
                  kwargs={'lookback':45}, outputs=['hist_vol']),
        Transform(context, name='slope', function=Transform.slope, inputs=['price'],
                  kwargs={'lookback':100}, outputs=['slope']),
        Transform(context, name='TMOM', function=Transform.momentum, inputs=['price'],
                  kwargs={'lookback':43}, outputs=['TMOM']),
        Transform(context, name='MA', function=talib.SMA, inputs=['price'],
                  kwargs={'lookback': 100}, outputs=['MA']),
        Transform(context, name='R', function=Transform.average_excess_return_momentum, inputs=['price'],
                  kwargs={'lookback':13, 'period':'M'}, outputs=['R']),
        Transform(context, name='RMOM', function=Transform.momentum, inputs=['price'],
                  kwargs={'lookback':43}, outputs=['RMOM']),
        Transform(context, name='TMOM', function=Transform.excess_momentum, inputs=['price'],
                  kwargs={'lookback':43}, outputs=['TMOM']),
        Transform(context, name='EMOM', function=Transform.momentum, inputs=['price'],
                  kwargs={'lookback':43}, outputs=['EMOM']),
        Transform(context, name='volatility', function=talib.STDDEV, inputs=['daily_returns'],
                  kwargs={'lookback':43}, outputs=['volatility']),
        Transform(context, name='smma', function=Transform.simple_mean_monthly_average, inputs=['price'],
                  kwargs={'lookback':1, 'period':'M'}, outputs=['smma']),
        Transform(context, name='mom', function=Transform.paa_momentum, inputs=['price', 'smma'],
                  kwargs={'lookback':2, 'period':'M'}, outputs=['mom']),
        Transform(context, name='smma_12', function=Transform.simple_mean_monthly_average, inputs=['price'],
                  kwargs={'lookback':12, 'period':'M'}, outputs=['smma_12']),
        Transform(context, name='rebalance_signal', function=Transform.crossovers, inputs=['price','MA'],
                  kwargs={'timeperiods':100}, outputs=['rebalance_signal']),
    ]

    return context.transforms


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def define_rules(context):  # Define rules
    # select rules required and make sure correct transform names are used
    # no need to comment out unused rules

    log.info('DEFINE RULES')

    context.algo_rules = [
        # Rule(context, name='absolute_momentum_rule', rule="'price' < 'smma' "),
        # Rule(context, name='dual_momentum_rule', rule="'TMOM' < 0"),
        Rule(context, name='smma_rule', rule="'price' < 'smma'"),
        # Rule(context, name='complex_rule', rule="'price' < smma or 'TMOM' < 0"),
        Rule(context, name='momentum_rule', rule="'price' < 'MA'"),
        Rule(context, name='EAA_rule', rule="'R' <= 0"),
        Rule(context, name='paa_rule', rule="'mom' <= 0"),
        Rule(context, name='paa_filter', rule="'mom' > 0"),
        Rule(context, name='momentum_rule1', rule="'price' < 'smma_12'"),
        Rule(context, name='riskon', rule="'price' > 'smma_12'", apply_to=context.market_proxy),
        Rule(context, name='riskoff', rule="'price' <= 'smma_12'", apply_to=context.market_proxy),
        Rule(context, name='neutral', rule="'slope' <= 0.1 and 'slope' >= -0.1",
             apply_to=context.market_proxy),
        Rule(context, name='bull', rule="'slope' > 0.1", apply_to=context.market_proxy),
        Rule(context, name='bear', rule="'slope' < -0.1", apply_to=context.market_proxy)
        # Rule(context, name='rebalance_rule', rule="'rebalance_signal' != 0"),
    ]

    return context.algo_rules


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def set_global_parameters(context):

    # set the following parameters as required

    context.show_positions = True
    # select records to show in algo.show_records()
    context.show_records = True

    # replace cash_proxy with risk_free if context.allow_cash_proxY_replacement is True
    # and cash_proxy price is <= average cash_proxy price over last context.cash_proxy_lookback days
    context.allow_cash_proxy_replacement = False
    context.cash_proxy_lookback = 43  # must be <= context.max_lookback

    context.use_trailing_stops = False
    context.stop_pct = 0.92
    context.stop_price = {}

    # to calculate portfolio and strategy Sharpe ratios
    context.SR_lookback = 63
    context.SD_factor = 0

    # position only changed if percentage change > threshold
    context.threshold = 0.01

    # the following can be changed
    context.market_proxy = symbol('SPY')
    context.risk_free = symbol('SHY')

    set_commission(commission.PerTrade(cost=10.0))
    context.leverage = 1.0

    # parameters for rebalance period
    context.rebalance_period = 'ME'  # 'D'|'WS'|'WE'|'MS'|'ME'|'A'
    context.days_offset = 2
    context.on_open = True  # if false, then market_close
    context.hours = 0
    context.minutes = 1

    context.rebalance_interval = 1  # rebalancing will occur every balance_interval * balance_period


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def set_strategy_parameters(context):
    # If not required, parameters may be omitted
    # no need to comment out unused strategies
    # strategies used by the algo selected in set_algo_parameters

    # configure strategies below
    # ####################################################################################################
    #     # single RS portfolio with downside protection
    #     s1 = StrategyParameters(context, ID='s1',
    #                     portfolios=[symbols( 'IVV', 'IJH', 'IJR', 'VEA',
    #                                         'VWO', 'VNQ', 'AGG')],
    #                     portfolio_allocation_modes=['EW'],
    #                     security_scoring_methods=['RS'],
    #                     security_scoring_factors=[{'+momentum': 1.0}],
    #                     security_n_tops=[2],
    #                     protection_modes=['BY_RULE'],
    #                     protection_rules=['smma_rule'],
    #                     cash_proxies=[symbol('TLT')],
    #                     strategy_allocation_mode='EW',
    #                    )
    # ####################################################################################################
    #     # RAA - Robust Asset Allocation (4 portfolios)
    #     #
    s2 = StrategyParameters(context, ID='s2',
                            portfolios=[symbols('MDY', 'EFA'), symbols('VNQ', 'RWX'),
                                        symbols('GLD', 'AGG'),
                                        symbols('EDV', 'EMB')],
                            portfolio_allocation_modes=['EW', 'EW', 'EW', 'EW'],
                            security_scoring_methods=['RS', 'RS', 'RS', 'RS'],
                            security_scoring_factors=[{'+EMOM': 1.}, {'+EMOM': 1.},
                                                      {'+EMOM': 1.}, {'+EMOM': 1.}],
                            security_n_tops=[1, 1, 1, 1],
                            protection_modes=['RAA', 'RAA', 'RAA', 'RAA'],
                            cash_proxies=[symbol('TLT'), symbol('TLT'), symbol('TLT'), symbol('TLT')],
                            strategy_allocation_mode='MAX_SHARPE',
                            strategy_allocation_kwargs={'lookback': 21, 'shorts': False},
                            )
    # ####################################################################################################
    #     # Strategy 3 - minimumn correlation strategy
    #     s3 = StrategyParameters(context, ID='s3',
    #                     portfolios=[symbols( 'IVV', 'IJH', 'IJR', 'VEA',
    #                                         'VWO', 'VNQ', 'AGG')],
    #                     portfolio_allocation_modes=['MIN_CORRELATION'],
    #                     portfolio_allocation_kwargs=[{'lookback': 21, 'risk_adjusted': True}],
    #                     protection_modes=['BY_RULE'],
    #                     protection_rules=['smma_rule'],
    #                     protection_formulas=None,
    #                     cash_proxies=[symbol('SHY')],
    #                     strategy_allocation_mode='EW'
    #                    )
    # ####################################################################################################
    #     # sdp_1 - downside protection strategy based on Alpha Architect DPM Rule: 50% TMOM, 50% MA
    #     # http://blog.alphaarchitect.com/2015/08/13/avoiding-the-big-drawdown-downside-protection-investment-strategies/#gs.qtrlStY
    #     sdp_1 = StrategyParameters(context, ID='sdp_1',
    #                      portfolios=[symbols( 'XLY', 'XLF', 'XLK', 'XLE', 'XLV',  'XLI',
    #                                          'XLP', 'XLB', 'XLU')],
    #                      portfolio_allocation_modes=['EW'],
    #                      protection_modes=['RAA'],
    #                      # protection_modes=['BY_RULE'],
    #                      # protection_rules=['smma_rule'],
    #                      # protection_rules=['momentum_rule'],
    #                      cash_proxies=[symbol('SHY')],
    #                     strategy_allocation_mode='EW'
    #                     )
    # ####################################################################################################
    #     # RS with downside protection, single portfolio, EtfReplay-like ranking formula
    #     rs_1 = StrategyParameters(context, ID='rs_1',
    #                     portfolios=[symbols( 'MDY', 'EFA')],
    #                     portfolio_allocation_modes=['EW'],
    #                     security_scoring_methods=['RS'],
    #                     security_scoring_factors=[{'+mom_A': 0.65, '+mom_B' : 0.35, '-vol_C' : 0.}],
    #                     security_n_tops=[1],
    #                     protection_modes=['BY_RULE'],
    #                     protection_rules=['smma_rule'],
    #                     cash_proxies=[symbol('TLT')],
    #                     strategy_allocation_mode='EW'
    #                    )
    # ####################################################################################################
    #     # RS with 2 portfolios based on EtfReplay ranking model
    #     rs_2 = StrategyParameters(context, ID='rs_2',
    #                     portfolios=[symbols( 'MDY', 'EFA'), symbols('IHF', 'EFA')],
    #                     portfolio_allocation_modes=['EW', 'FIXED'],
    #                     security_weights=[None, [0.8, 0.2]],
    #                     security_scoring_methods=['RS', 'RS'],
    #                     security_scoring_factors=[{'+mom_A': 0.65, '+mom_B' : 0.35, '-vol_C' : 0.},
    #                                               {'+mom_A': 0.65, '+mom_B' : 0.35, '-vol_C' : 0.}],
    #                     security_n_tops=[1, 2],
    #                     protection_modes=['BY_RULE', 'BY_RULE'],
    #                     protection_rules=['smma_rule', 'smma_rule'],
    #                     cash_proxies=[symbol('TLT'), symbol('TLT')],
    #                     strategy_allocation_mode='FIXED',
    #                     portfolio_weights=[0.6, 0.4]
    #                    )
    # ####################################################################################################
    #     # EAA - Elastic Asset Allocation
    #     # http://indexswingtrader.blogspot.co.za/2015/01/a-primer-on-elastic-asset-allocation.html
    #     eaa_1 = StrategyParameters (context, ID='eaa_1',
    #                     portfolios=[symbols('EEM', 'IEF', 'IEV', 'MDY', 'QQQ', 'TLT', 'XLV')],
    #                     portfolio_allocation_modes=['PROPORTIONAL'],
    #                     security_scoring_methods=['EAA'],
    #                     # Golden Defensive EAA: wi ~ zi = squareroot( ri * (1-ci) )
    #                     security_scoring_factors = [{'R': 1.0, 'C' : 1.0, 'V' : 0.0, 'S' : 0.5, 'eps' : 1e-6}],
    #                     protection_modes=['BY_FORMULA'], protection_rules=['EAA_rule'],
    #                     protection_formulas=['DPF'], cash_proxies=[symbol('TLT')], strategy_allocation_mode='EW')
    # ####################################################################################################
    #     # Risk_on Risk_off
    #     roo_1 = StrategyParameters(context, ID='roo_1',
    #                      portfolios=[symbols('SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM',
    #                                          'IYR', 'GSG', 'GLD'), symbols('TLT', 'TIP', 'LQD', 'SHY')],
    #                      portfolio_allocation_modes=['EW', 'EW'],
    #                      security_scoring_methods=['RS', 'RS'],
    #                      security_scoring_factors=[{'+smma': 1}, {'+smma': 1}],
    #                      security_n_tops=[3, 1],
    #                      protection_modes=['BY_RULE', None],
    #                      protection_rules=['momentum_rule1', None],
    #                      cash_proxies=[symbol('IEF'), symbol('SHY')], strategy_allocation_mode='EW')
    # ####################################################################################################
    #     # Adaptive Asset Allocation
    #     aaa_1 = StrategyParameters(context, ID='aaa_1',
    #                     portfolios=[symbols( 'SPY', 'IWM', 'EFA', 'EEM', 'VNQ', 'GLD', 'GSG',
    #                                         'JNK', 'AGG', 'TIP', 'IEF', 'TLT')],
    #                     portfolio_allocation_modes=['VOLATILITY_WEIGHTED'],
    #                     security_scoring_methods=['RS'],
    #                     security_scoring_factors=[{'+mom': 1.0}],
    #                     security_n_tops=[3],
    #                     protection_modes=['BY_RULE'],
    #                     protection_rules=['smma_rule'],
    #                     cash_proxies=[symbol('TLT')],
    #                     strategy_allocation_mode='EW')
    ####################################################################################################
    # Protective Asset Allocation
    # http://indexswingtrader.blogspot.co.za/2016/04/introducing-protective-asset-allocation.html
    # paa_1 = StrategyParameters(context, ID='paa_1',
    #                  portfolios=[symbols('SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM',
    #                                      'IYR', 'GSG', 'GLD', 'LQD', 'TLT', 'HYG'),
    #                              symbols('IEF', 'TLT')],
    #                  portfolio_allocation_modes=['EW', 'EW'],
    #                  security_scoring_methods=['RS', 'RS'],
    #                  security_scoring_factors=[{'+mom': 1}, {'+mom': 1}],
    #                  security_n_tops=[3, 1],
    #                  protection_modes=['BY_RULE', None],
    #                  protection_rules=['paa_rule', None],
    #                  cash_proxies=[symbol('TLT'), symbol('TLT')],
    #                  strategy_allocation_mode='BY_FORMULA',
    #                  strategy_allocation_formula='PAA',
    #                  strategy_allocation_rule='paa_filter',
    #                  strategy_allocation_kwargs={'protection_factor': 1})
    ####################################################################################################
    #     brs_1 = StrategyParameters(context, ID='brs_1',
    #                     portfolios=[symbols('CWB', 'JNK'), symbols('CWB', 'JNK'), symbols('CWB', 'JNK'),
    #                                 symbols('CWB', 'PCY'), symbols('CWB', 'PCY'), symbols('CWB', 'PCY'),
    #                                 symbols('CWB', 'TLT'), symbols('CWB', 'TLT'), symbols('CWB', 'TLT'),
    #                                 symbols('JNK', 'PCY'), symbols('JNK', 'PCY'), symbols('JNK', 'PCY'),
    #                                 symbols('JNK', 'TLT'), symbols('JNK', 'TLT'), symbols('JNK', 'TLT'),
    #                                 symbols('PCY', 'TLT'), symbols('PCY', 'TLT'), symbols('PCY', 'TLT')],
    #                     portfolio_allocation_modes=['FIXED', 'FIXED', 'FIXED',
    #                                                 'FIXED', 'FIXED', 'FIXED',
    #                                                 'FIXED', 'FIXED', 'FIXED',
    #                                                 'FIXED', 'FIXED', 'FIXED',
    #                                                 'FIXED', 'FIXED', 'FIXED',
    #                                                 'FIXED', 'FIXED', 'FIXED'],
    #                     security_weights=[[0.6, 0.4], [0.5, 0.5], [0.4, 0.6],
    #                                       [0.6, 0.4], [0.5, 0.5], [0.4, 0.6],
    #                                       [0.6, 0.4], [0.5, 0.5], [0.4, 0.6],
    #                                       [0.6, 0.4], [0.5, 0.5], [0.4, 0.6],
    #                                       [0.6, 0.4], [0.5, 0.5], [0.4, 0.6],
    #                                      [0.6, 0.4], [0.5, 0.5], [0.4, 0.6]],
    #                     cash_proxies=[symbol('TLT'), symbol('TLT'), symbol('TLT'),
    #                                   symbol('TLT'), symbol('TLT'), symbol('TLT'),
    #                                   symbol('TLT'), symbol('TLT'), symbol('TLT'),
    #                                   symbol('TLT'), symbol('TLT'), symbol('TLT'),
    #                                   symbol('TLT'), symbol('TLT'), symbol('TLT'),
    #                                   symbol('TLT'), symbol('TLT'), symbol('TLT')],
    #                     strategy_allocation_mode='BRUTE_FORCE_SHARPE',
    #                     strategy_allocation_kwargs={'lookback' : 73})
    # ####################################################################################################
    #     brs_2 = StrategyParameters(context, ID='brs_2',
    #                     portfolios=[symbols('CWB', 'JNK'), symbols('CWB', 'PCY'), symbols('CWB', 'TLT'),
    #                                 symbols('JNK', 'PCY'), symbols('JNK', 'TLT'), symbols('PCY', 'TLT')],
    #                     portfolio_allocation_modes=['MAX_SHARPE', 'MAX_SHARPE', 'MAX_SHARPE',
    #                                                 'MAX_SHARPE', 'MAX_SHARPE', 'MAX_SHARPE'],
    #                     portfolio_allocation_kwargs=[
    #                                {'lookback' : 73, 'shorts' : False},{'lookback' : 73, 'shorts' : False},
    #                                {'lookback' : 73, 'shorts' : False},{'lookback' : 73, 'shorts' : False},
    #                                {'lookback' : 73, 'shorts' : False},{'lookback' : 73, 'shorts' : False}],
    #                     cash_proxies=[symbol('TLT'), symbol('TLT'), symbol('TLT'),
    #                                   symbol('TLT'), symbol('TLT'), symbol('TLT')],
    #                     strategy_allocation_mode='BRUTE_FORCE_SHARPE',
    #                     strategy_allocation_kwargs={'lookback' : 73, 'SD_factor' : 2})
    ####################################################################################################
    # context.strategy_parameters = [s1, s2, s3, sdp_1, rs_1, rs_2, eaa_1, roo_1, aaa_1, paa_1, brs_1, brs_2]
    context.strategy_parameters = [s2]

    return context.strategy_parameters


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def set_algo_parameters(context, strategies):
    # UNCOMMENT ONLY ONE ALGO BELOW
    ###############################
    # simple downside protection algorithm
    # http://blog.alphaarchitect.com/2015/08/13/avoiding-the-big-drawdown-downside-protection-investment-            strategies/#gs.qtrlStY

    # strategy_ID = 'sdp_1'

    # algo = Algo (context, [s for s in strategies if s.ID == strategy_ID],
    #              allocation_model=AllocationModel(context, mode='EW', weights=None, formula=None),
    #             )
    ###############################
    # EAA - Elastic Asset Allocation
    # http://indexswingtrader.blogspot.co.za/2015/01/a-primer-on-elastic-asset-allocation.html

    # strategy_ID = 'eaa_1'

    # algo = Algo (context, [s for s in strategies if s.ID == strategy_ID],
    #              allocation_model=AllocationModel(context, mode='EW', weights=None, formula=None),
    #             )
    ###############################
    # multiple strategies, equally weighted

    # list of strategies by ID
    # strategy_IDs = ['s1', 's2', 's3', 'sdp_1']

    # algo = Algo (context, strategies=[s for s in strategies if s.ID in strategy_IDs],
    #              allocation_model=AllocationModel(context, mode='EW', weights=None, formula=None),
    #             )
    ###############################
    # run all uncommented strategies (other than regime-switching strategies)

    algo = Algo(context, strategies=[s for s in strategies],
                allocation_model=AllocationModel(context, mode='EW'), scoring_model=None,
                # allocation_model=AllocationModel(context, mode='RISK_PARITY', kwargs={'lookback':21}),     scoring_model=ScoringModel(context, method='RS', factors={'+EMOM':1.}, n_top=1),
                regime=None,
                )
    ########################
    # 2 regimes: riskon riskoff RS ; riskon=market_proxy price > sma, riskoff=market_proxy price <= sma
    # algo = Algo (context, [s for s in strategies if s.ID == 'roo_1'],
    #              allocation_model=AllocationModel(context, mode='REGIME_EW'),
    #              regime=Regime( transitions={'1' : ('riskon', ['roo_1_p1']),
    #                                          '0' : ('riskoff', ['roo_1_p2']),
    #                                   }
    #                            )
    #             )
    ########################
    # 3 regimes : 'bull', 'bear', 'neutral'
    # strategy_IDs = ['rs_2', 'eaa_1']
    # algo = Algo (context, strategies = [s for s in strategies if s.ID in strategy_IDs],
    #              allocation_model=AllocationModel(context, mode='REGIME_EW', weights=None, formula=None),
    #              regime=Regime(
    #                                   transitions={'0' : ('neutral', ['eaa_1']),
    #                                   '1' : ('bull', ['rs_2_p1']),
    #                                   '-1' : ('bear', ['rs_2_p2', 'eaa_1'])
    #                                          }
    #                                  )
    #             )
    ############################
    # AAA - Adaptive Asset Allocation
    # http://papers.ssrn.com/sol3/papers.cfm?abstract_id=2359011

    # algo = Algo (context, strategies = [s for s in strategies if s.ID == 'aaa_1']
    #              allocation_model=AllocationModel(context, mode='EW'),
    #             )
    ############################
    # PAA - Protective Asset Allocation
    # http://indexswingtrader.blogspot.co.za/2016/04/introducing-protective-asset-allocation.html
    # algo = Algo (context, strategies = [s for s in strategies if s.ID == 'paa_1'],
    #              allocation_model=AllocationModel(context, mode='EW'),
    #             )
    ############################
    # BRS - Bond Rotation Strategy
    # https://logical-invest.com/portfolio-items/bond-rotation-sleep-well/
    # https://www.quantopian.com/posts/the-logical-invest-enhanced-bond-rotation-strategy

    # Algo-specific parameters
    # context.calculate_SR = True
    # context.SR_lookback = 73
    # context.SD_factor = 2
    # algo = Algo (context, strategies = [s for s in strategies if s.ID == 'brs_1'],
    #              allocation_model=AllocationModel(context, mode='EW'),
    #             )
    ###############################

    return algo


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# dummy logger
class Log():
    pass

    def info(self, s):
        print('{} INFO : {}'.format(get_datetime().tz_convert('US/Eastern'), s))
        pass

    def debug(self, s):
        print('{} DEBUG : {}'.format(get_datetime().tz_convert('US/Eastern'), s))
        pass

    def warn(self, s):
        print('{} WARNING : {}'.format(get_datetime().tz_convert('US/Eastern'), s))
        pass




# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
############################################
# HELPER FUNCTIONS
##################
# NOTE: as pandas panel has been deprecated, need to fix this!!
# THE ALGORITHM PARAMETERS ARE DEFINED IN THIS SECTION:

# ENVIRONMENT can be set for 'ZIPLINE', 'RESEARCH' or 'IDE'
ENVIRONMENT = 'ZIPLINE'

# the following 3 lines must be commented out for use with RESEARCH or IDE
if ENVIRONMENT == 'ZIPLINE' and ENVIRONMENT != 'IDE':
    from zipline.api import symbol, symbols


#     from zipline.utils.factory import load_bars_from_yahoo, load_from_yahoo
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# this routine will not used for ENVIRONMENT == 'IDE'

# def get_data(ENVIRONMENT, tickers, start, end, benchmark, risk_free, cash_proxy):
#     if ENVIRONMENT == 'ZIPLINE':
#         benchmark_symbol = benchmark
#         cash_proxy_symbol = cash_proxy
#         risk_free_symbol = risk_free
#     elif ENVIRONMENT == 'RESEARCH':
#         if benchmark is None:
#             benchmark_symbol = None
#         else:
#             benchmark_symbol = symbols(benchmark)
#         if cash_proxy is None:
#             cash_proxy_symbol = None
#         else:
#             cash_proxy_symbol = symbols(cash_proxy)
#         if risk_free is None:
#             risk_free_symbol = None
#         else:
#             risk_free_symbol = symbols(risk_free)
#
#     # data is a Panel of DataFrames, one for each security
#     if ENVIRONMENT == 'ZIPLINE':
#         stocks = list(set(tickers + [benchmark_symbol, cash_proxy_symbol, risk_free_symbol]))
#         stocks = [s for s in stocks if s != None]
#         #         data = load_bars_from_yahoo(
#         #             stocks,
#         #             start = start,
#         #             end = end,
#         #             adjusted=False).transpose(2,1,0)
#
#         # User pandas_reader.data.DataReader to load the desired data. As simple as that.
#         d = web.DataReader(stocks, "yahoo", start, end)
#         data = pd.DataFrame(columns=['high', 'low', 'price', 'volume', 'open'], index=d.index)
#         data.high = d.High.copy()
#         data.low = d.Low.copy()
#         data.price = d['Adj Close'].copy()  # use this for comparing to Quantopian 'get_pricing'
#         data.volume = d.Volume.copy()
#         data.open = d.Open.copy()
#
#     elif ENVIRONMENT == 'RESEARCH':
#         stocks = set([symbols(t) for t in tickers] + [benchmark_symbol, cash_proxy_symbol, risk_free_symbol])
#         stocks = [s for s in stocks if s != None]
#         data = get_pricing(
#             stocks,
#             start_date=start,
#             end_date=end,
#             frequency='daily'
#         )
#
#         # repair unusable data
#     # BE CAREFUL!! dropna doesn't change the Panel's Major Index, so NA may still remain!
#     # safer to use ffill
#
#     #     for security in data.transpose(2,1,0):
#     #         data.transpose(2,1,0)[security] = data.transpose(2,1,0)[security].ffill()
#
#     # for
#
#     if benchmark is None:
#         stocks = []
#     else:
#         stocks = [benchmark_symbol]
#
#     if ENVIRONMENT == 'ZIPLINE':
#         stocks = stocks + [cash_proxy_symbol]
#         other_data = load_bars_from_yahoo(
#             stocks=stocks,
#             start=start,
#             end=end,
#             adjusted=False)  # use this for comparing to Quantopian 'get_pricing'
#         other_data.transpose(2, 1, 0).price = other_data.transpose(2, 1,
#                                                                    0).close  # use this for comparing to Quantopian 'get_pricing'
#     elif ENVIRONMENT == 'RESEARCH':
#         other_data = get_pricing(
#             stocks + [cash_proxy_symbol],
#             fields='price',
#             start_date=data.major_axis[0],
#             end_date=data.major_axis[-1],
#             frequency='daily',
#         )
#
#     other_data = other_data.ffill()
#
#     if benchmark is not None:
#         # need to add benchmark (eg SPY) and cash proxy to data panel
#         benchmark = other_data[benchmark_symbol]
#         benchmark_rets = benchmark.pct_change().dropna()
#
#         benchmark2 = other_data[cash_proxy_symbol]
#         benchmark2_rets = benchmark2.pct_change().dropna()
#
#     # make sure we have all the data we need
#     inception_dates = pd.DataFrame([data.transpose(2, 1, 0)[security].dropna().index[0].date() \
#                                     for security in data.transpose(2, 1, 0)], \
#                                    index=data.transpose(2, 1, 0).items, columns=['inception'])
#     if benchmark is not None:
#         inception_dates.loc['benchmark'] = benchmark.index[0].date()
#         inception_dates.loc['benchmark2'] = benchmark2.index[0].date()
#     print(inception_dates)
#
#     # check that the end dates coincide
#     end_dates = pd.DataFrame([data.transpose(2, 1, 0)[security].dropna().index[-1].date() \
#                               for security in data.transpose(2, 1, 0)], \
#                              index=data.transpose(2, 1, 0).items, columns=['end_date'])
#     if benchmark is not None:
#         end_dates.loc['benchmark'] = benchmark.index[-1].date()
#         end_dates.loc['benchmark2'] = benchmark2.index[-1].date()
#     print(end_dates)
#
#     # this will ensure that the strat and end dates are aligned
#     data = data[:, inception_dates.values.max(): end_dates.values.min(), :]
#     if benchmark is not None:
#         benchmark_rets = benchmark_rets[inception_dates.values.max(): end_dates.values.min()]
#         benchmark2_rets = benchmark2_rets[inception_dates.values.max(): end_dates.values.min()]
#
#     print('\n\nBACKTEST DATA IS FROM {} UNTIL {} \n*************************************************'
#           .format(inception_dates.values.max(), end_dates.values.min()))
#
#     # DATA FROM ZIPLINE LOAD_YAHOO_BARS DIFFERS FROM RESEARCH ENVIRONMENT!
#     data.items = ['open_price', 'high', 'low', 'close_price', 'volume', 'price']
#
#     print('\n\n{}'.format(data))
#
#     return data


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# symbol_set =['SPY', 'QQQ', 'IWM', 'VGK', 'EWJ', 'EEM','IYR', 'GSG', 'GLD', 'LQD', 'TLT', 'HYG','IEF', 'TLT','SHY']
# symbol_set = ['MDY', 'EFA','VNQ', 'RWX','GLD', 'AGG','EDV', 'EMB', 'TLT', 'SPY', 'SHY']
# tickers = list(set(symbol_set))

# # # data is a Panel of DataFrames, one for each security
# # data = get_pricing(
# #     tickers,
# #     start_date='2009-12-01',
# #     end_date = '2016-11-1',
# #     frequency='daily'
# # )

# # Define which online source one should use
# data_source = 'yahoo'

# # We would like all available data from 01/01/2000 until today.
# start_date = '2009-12-01'
# end_date = datetime.today().strftime('%Y-%m-%d')

# # User pandas_reader.data.DataReader to load the desired data. As simple as that.
# panel_data = web.DataReader(tickers, data_source, start_date, end_date)
# data = panel_data.sort_index(ascending=True)

# inception_dates = pd.DataFrame([data[ticker].first_valid_index() for ticker in data.columns],
#                                index=data.keys(), columns=['inception'])

# print (inception_dates)

# data = data.ffill()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def initialize(context):
    # this routine should not require changes

    print('PLATFORM  ', get_environment('platform'))

    context.transforms = []
    context.algo_rules = []
    context.max_lookback = 64  # minimum value for max_lookback
    context.outstanding = {}  # orders which span multiple days

    context.raw_data = {}

    context.trading_day_no = 0

    #############################################################
    set_global_parameters(context)
    log.info('GLOBAL PARAMETERS CONFIGURED')
    #############################################################
    context.strategy_parameters = set_strategy_parameters(context)
    # strategy_params = [context.strategy_parameters[p] for p in context.strategy_parameters]
    log.info('STRATEGY PARAMETERS CONFIGURED')
    #############################################################
    # configure strategies
    Configurator(context, strategies=context.strategy_parameters)
    log.info('STRATEGIES CONFIGURED')
    #############################################################
    strategies = [s.strategy for s in context.strategy_parameters]
    algo = set_algo_parameters(context, strategies)
    #############################################################

    print('SET DAILY FUNCTIONS')

    # daily functions to handle GTC orders
    # note: GTC_LIMIT=10 (default) set as global
    schedule_function(algo.check_for_unfilled_orders, date_rules.every_day(), time_rules.market_close())
    schedule_function(algo.fill_outstanding_orders, date_rules.every_day(), time_rules.market_open())

    if context.show_positions:
        schedule_function(algo.show_positions, date_rules.month_start(days_offset=0), time_rules.market_open())

    if context.show_records:
        # show records every day
        # edit the show_records function to include records required
        schedule_function(algo.show_records, date_rules.every_day(), time_rules.market_close())

    if context.rebalance_period == 'A':
        schedule_function(algo.check_signal_trigger, date_rules.every_day(), time_rules.market_open())

    else:
        periods = {'D': date_rules.every_day(),
                   'WS': date_rules.week_start(days_offset=context.days_offset),
                   'WE': date_rules.week_end(days_offset=context.days_offset),
                   'MS': date_rules.month_start(days_offset=context.days_offset),
                   'ME': date_rules.month_end(days_offset=context.days_offset)}

        period = periods[context.rebalance_period]

        if context.on_open:
            time_rule = time_rules.market_open(hours=context.hours, minutes=context.minutes)
        else:
            time_rule = time_rules.market_close(hours=context.hours, minutes=context.minutes)

        schedule_function(algo.rebalance, period, time_rule)

    log.info('REBALANCE INTERVAL = ' + str(period))

    log.info('INITIALIZATION DONE!')
#########################################################################################################
if __name__ == "__main__":

    log = Log()

    start = datetime(2013, 1, 1, 0, 0, 0, 0, pytz.utc)
    #     end = datetime(2014, 1, 10, 0, 0, 0, 0, pytz.utc)
    end = datetime.today().replace(tzinfo=timezone.utc)
    capital_base = 100000

    result = run_algorithm(start=start, end=end, initialize=initialize, \
                           capital_base=capital_base, \
                           before_trading_start=before_trading_start,
                           bundle='etfs_bundle')

    print(result[:3])
