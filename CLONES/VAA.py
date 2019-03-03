from zipline.api import attach_pipeline, pipeline_output, get_datetime
from zipline import run_algorithm
from zipline.api import symbols, symbol, get_datetime, schedule_function, record
from zipline.api import get_open_orders, order_target_percent, order_target_value
from zipline.utils.events import date_rules, time_rules
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.filters import StaticAssets
from datetime import datetime, timezone
import pytz


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


log = Log()

# BREADTH MOMENTUM STRATEGY
# FROM : Breadth Momentum and Vigilant Asset Allocation (VAA);
# Winning More by Losing Less  By Wouter J. Keller and Jan Willem Keuning1 July 14, 2017, v0.99
# https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3002624

# - suitable for IB cash accounts (T+3)
# - prevent trading too much by limiting threshold for orders

# ---------------------------------------------------------------------------------
from collections import defaultdict

TOP = 2  # if there are no X BAD assets how many assets do I invest in
BAD = 1  # How many bad assets are allowed (assets with negative return)

lev = 0.98  # max leverage to use

TplusX = 0  # if you have a T+3 Cash account use 3 else 0 for margin accounts
Month_Offset = 0  # when using TplusX the max month offset can be 3


# ---------------------------------------------------------------------------------
def initialize(context):
    c = context

    # Some Risk-On universes
    core_etf = symbols('QQQ', 'XLP', 'TLT', 'IEF')
    tact_etf = symbols('XLV', 'XLY', 'TLT', 'GLD')

    VAAG12 = symbols('SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'VWO', 'VNQ', 'GSG', 'GLD', 'TLT', 'LQD', 'HYG')
    VAAG4 = symbols('SPY', 'VEA', 'VWO', 'BND')

    c.RISK = VAAG4  # Which Riskasset universe to use?

    alletf = symbols('QQQ', 'XLP', 'TLT', 'IEF', 'SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'VWO',
                     'VNQ', 'GSG', 'GLD', 'TLT', 'LQD', 'HYG', 'XLV', 'XLY', 'GLD', 'VEA')  # 'TLO' repleced by 'TLT'
    # Some Risk Off (Cash) universes
    c.CASH = symbols('SHY', 'IEF', 'LQD')

    schedule_function(define_weights, date_rules.month_end(Month_Offset), time_rules.market_open(minutes=64))
    log.info('define_weights for last trading day of the month')

    schedule_function(trade_sell, date_rules.month_end(Month_Offset), time_rules.market_open(minutes=65))
    log.info('trade_sell for last trading day of the month')

    if TplusX > 0:
        schedule_function(trade_buy, date_rules.month_start(max(0, TplusX - 1 - Month_Offset)),
                          time_rules.market_open(minutes=66))
        log.info('trade_buy for month day ' + str(max(0, TplusX - Month_Offset)))
    else:
        schedule_function(trade_buy, date_rules.month_end(Month_Offset), time_rules.market_open(minutes=66))
        log.info('trade_buy last trading day of the month')

    stocks = core_etf + tact_etf
    context.sell = {}
    context.buy = {}

    for stock in stocks:
        context.sell[stock] = 0
        context.buy[stock] = 0

    context.difference_perc = 0.02

    context.stops = {}
    context.stoploss = 0.30
    context.stoppedout = []


def trade_sell(context, data):
    log.info('\n\n! running Trade Sell ' + str(get_datetime()))

    for stocksymbol in context.sell:
        try:
            if context.sell[stocksymbol] != -1 and data.can_trade(stocksymbol):
                order_target_percent(stocksymbol, context.sell[stocksymbol])
                log.info('-- Stock: ' + str(stocksymbol) + ': Rebalanced so % is now ' + str(
                    context.sell[stocksymbol]) + '%')
            else:
                log.info('!  Stock: ' + str(stocksymbol) + ': No need to sell')
        except:
            log.info("FAIL " + str(stocksymbol))


def trade_buy(context, data):
    log.info('\n\n! running Trade Buy ' + str(get_datetime()))
    print('\n\n  BUY ', context.buy)

    for stocksymbol in context.buy:
        print('\n\n ', data.can_trade(stocksymbol), '    SYMBOL = ', stocksymbol, '      PERCENT = ',
              context.buy[stocksymbol])

        try:
            if context.buy[stocksymbol] != -1 and data.can_trade(stocksymbol):
                order_target_percent(stocksymbol, context.buy[stocksymbol])
                log.info('++ Stock: ' + str(stocksymbol) + ': bought ' + str(context.buy[stocksymbol]) + '%')
            else:
                log.info('!  Stock: ' + str(stocksymbol) + ': No need to buy')
        except:
            log.info("FAIL " + str(stocksymbol))


def define_weights(context, data):
    c = context
    log.info('\n\n! running define_weights ' + str(get_datetime()))

    m = data.history(c.RISK + c.CASH, 'price', 400, '1d')
    p = m.resample('M', closed='right', label='right').last()

    W13612 = 12 * (p.iloc[-1] / p.iloc[-2]) + 4 * (p.iloc[-1] / p.iloc[-4]) + 2 * (p.iloc[-1] / p.iloc[-7]) + 1 * (
                p.iloc[-1] / p.iloc[-13]) - 19

    record(spy=W13612[symbol('SPY')])

    riskon = W13612[W13612.index.isin(c.RISK)]
    riskoff = W13612[W13612.index.isin(c.CASH)]
    pos_mom = riskon[riskon >= 0.0]

    if len(pos_mom) + BAD - 1 >= len(c.RISK):

        # now we are going to find the highest TOP
        log.info('!! Risk ON Universe chosen, top ' + str(TOP))
        universe_to_invest = riskon.sort_values(ascending=False).head(TOP)


        for stock in c.CASH:
            assign_weights(context, stock, 0.0)

        for stock in c.RISK:
            tar_perc = 0.0
            if stock in universe_to_invest:
                tar_perc = lev / len(universe_to_invest)
            assign_weights(context, stock, tar_perc)
    else:

        log.info('!! Risk OFF Universe chosen')
        universe_to_invest = riskoff.sort_values(ascending=False).head(1)

        for stock in c.RISK:
            assign_weights(context, stock, 0.0)
        for stock in c.CASH:
            tar_perc = 0.0
            if stock in universe_to_invest:
                tar_perc = lev / len(universe_to_invest)
            assign_weights(context, stock, tar_perc)


def assign_weights(context, stock, tar_perc):

    if stock not in context.portfolio.positions:
        context.buy[stock] = tar_perc
        if tar_perc != 0.0:
            log.info(stock.symbol + '++ 1st run: current percent 0.00%; target: ' + str(round(tar_perc * 100, 2)) + '%')
        return
    else:

        net_val = context.account.net_liquidation
        stock_owned = context.portfolio.positions[stock]
        cur_perc = (stock_owned.amount * stock_owned.last_sale_price) / net_val
        if cur_perc == tar_perc:
            context.buy[stock_owned.sid] = -1
            context.sell[stock_owned.sid] = -1
            return
        elif cur_perc > tar_perc + context.difference_perc:
            context.sell[stock_owned.sid] = tar_perc
            context.buy[stock_owned.sid] = -1
        elif cur_perc < tar_perc - context.difference_perc:
            context.buy[stock_owned.sid] = tar_perc
            context.sell[stock_owned.sid] = -1
        else:
            context.buy[stock_owned.sid] = -1
            context.sell[stock_owned.sid] = -1
            return
        log.info(stock_owned.asset.symbol + ' current percent ' + str(round(cur_perc * 100, 2)) + '%; target: ' + str(
            round(tar_perc * 100, 2)) + '%')


def before_trading_start(context, data):
    record(leverage=context.account.leverage)


def handle_data(context, data):
    c = context
    for position in c.portfolio.positions.itervalues():
        if position.amount == 0:
            if position.asset.symbol in c.stops: del c.stops[position.asset.symbol]
            continue
        elif position.asset.symbol not in c.stops:
            stoploss = c.stoploss if position.amount > 0 else -c.stoploss
            c.stops[position.asset.symbol] = position.last_sale_price * (1 - stoploss)
            # log.info(' ! I have added '+str(position.asset.symbol)+' to Stops @ '+str((position.last_sale_price)*(1-stoploss)))
        elif c.stops[position.asset.symbol] > position.last_sale_price and position.amount > 0:
            # sell
            log.info(
                ' ! ' + str(position.asset.symbol) + '- (Long) has hit stoploss @ ' + str(position.last_sale_price))
            if not get_open_orders(position.sid):
                order_target_value(position.sid, 0.0)
                c.stoppedout.append(position.asset.symbol)
                del c.stops[position.asset.symbol]
        elif c.stops[position.asset.symbol] < position.last_sale_price and position.amount < 0:
            # sell
            log.info(
                ' ! ' + str(position.asset.symbol) + '- (Short) has hit stoploss @ ' + str(position.last_sale_price))
            if not get_open_orders(position.sid):
                order_target_value(position.sid, 0.0)
                c.stoppedout.append(position.asset.symbol)
                del c.stops[position.asset.symbol]
        elif c.stops[position.asset.symbol] < position.last_sale_price * (1 - c.stoploss) and position.amount > 0:
            c.stops[position.asset.symbol] = position.last_sale_price * (1 - c.stoploss)
            # log.info(' ! I have updated '+str(position.asset.symbol)+'- (Long) to stop @ '+str((position.last_sale_price)*(1- c.stoploss)))
        elif c.stops[position.asset.symbol] > position.last_sale_price * (1 + c.stoploss) and position.amount < 0:
            c.stops[position.asset.symbol] = position.last_sale_price * (1 + c.stoploss)
        # log.info(' ! I have updated '+str(position.asset.symbol)+'- (Short) to stop @ '+str((position.last_sale_price)*(1+ c.stoploss)))


capital_base = 10000
start = datetime(2012, 1, 1, 0, 0, 0, 0, pytz.utc)
end = datetime(2019, 1, 1, 0, 0, 0, 0, pytz.utc)

result = run_algorithm(start=start, end=end, initialize=initialize, \
                       capital_base=capital_base, \
                       before_trading_start=before_trading_start,
                       bundle='etfs_bundle')