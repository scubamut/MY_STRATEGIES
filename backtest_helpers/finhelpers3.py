#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime as dt
from pandas_datareader import data
from math import sqrt
import math


#############################
# FINANCIAL HELPER ROUTINES #
#############################

def compute_nyears(x) :
    return np.double((x.index[-1] - x.index[0]).days) / 365.

def compute_cagr(equity) :
    return np.double((equity.ix[-1] / equity.ix[0]) ** (1. / compute_nyears(equity)) - 1)

def compute_annual_factor(equity) :
    # possible_values = [252,52,26,13,12,6,4,3,2,1]
    # L = pd.Series(len(equity) / compute_nyears(equity) - possible_values)
    # return [possible_values[i] for i in range(len(L)) if L[i] == L.min()][0]
    return len(equity) / compute_nyears(equity)

def compute_sharpe(equity, risk_free=0) :
    rets = equity.pct_change(1)
    factor = compute_annual_factor(rets)
    if isinstance(risk_free, pd.core.frame.DataFrame) :
        risk_free_rets = risk_free[risk_free.columns[0]].pct_change(1)
    elif isinstance(risk_free, pd.core.series.Series) :
        risk_free_rets = risk_free.pct_change(1)
    else :
        risk_free_rets = pd.Series(risk_free, index=rets.index)

    rets = (rets - risk_free_rets[rets.index])[1:]

    return sqrt(factor) * rets.mean()/rets.std()

def compute_DVR(equity):
    return compute_sharpe(equity) * compute_R2(equity) 

def compute_drawdown(x) :
    return (x - x.expanding(min_periods=1).max())/x.expanding(min_periods=1).max()

def compute_max_drawdown(x):
    return compute_drawdown(x).min()

def compute_rolling_drawdown(equity) :
    rolling_dd = pd.rolling_apply(equity, 252, compute_max_drawdown, min_periods=0)
    df = pd.concat([equity, rolling_dd], axis=1)
    df.columns = ['x', 'rol_dd_10']
    #plt.plot(df)
    #plt.grid()

def compute_avg_drawdown(x) :
    drawdown = compute_drawdown(x).shift(-1)
    drawdown[-1]=0.
    drawdown[0] = 0
    dend = [drawdown.index[i] for i in range(len(drawdown)) if drawdown[i] == 0 and drawdown[i-1] != 0]
    dstart = [drawdown.index[i] for i in range(len(drawdown)-1) if drawdown[i] == 0 and drawdown[i+1] != 0]
    f = pd.DataFrame(columns=['dstart', 'dend'])
    f.dstart = dstart
    f.dend = dend
    f['drawdown'] = [drawdown[f['dstart'][i]:f['dend'][i]].min() for i in range(len(f))]
    return f.drawdown.mean()

def compute_calmar(x) :
    return compute_cagr(x) / compute_max_drawdown(x)

def compute_R2(equity) :
    x = pd.DataFrame(equity)
    x.columns=[0]
    x[1]=[equity.index[i].toordinal() for i in range(len(equity))]
    return x[0].corr(x[1]) ** 2

def compute_volatility(x) :
    temp = compute_annual_factor(x)
    return sqrt(temp) * x.std(ddof=0)

def compute_var(d_returns, probs=0.05) :
    return d_returns.quantile(probs)

def compute_cvar(d_returns, probs=0.05) :
    return d_returns[ d_returns < d_returns.quantile(probs) ].mean()

def compute_percent_positive_months(equity):
    p_returns = equity.pct_change(periods=1)
    m_returns = (1 + p_returns).resample('M').prod() - 1
    return np.sum([1 if r > 0 else 0 for r in m_returns]) / len(m_returns) * 100
    
    m_rets = (1 + p_returns).resample('M').prod() - 1

def print_stats(equity, risk_free=0) :
    print ('**** STATISTICS ****')
    print ('====================\n')
    print ('n_years        : ', compute_nyears(equity))
    print ('cagr        : ', compute_cagr(equity) * 100, '%')
    d_returns = equity.pct_change()
    print ('annual_factor    : ', compute_annual_factor(equity))
    print ('sharpe        : ', compute_sharpe(equity, risk_free))
    compute_drawdown(d_returns)
    print ('max_drawdown    : ', compute_max_drawdown(equity) * 100, '%')
    print ('avg_drawdown    : ', compute_avg_drawdown(equity) * 100, '%')
    print ('calmar        : ', compute_calmar(equity))
    print ('R-squared    : ', compute_R2(equity))
    print ('DVR        : ', compute_DVR(equity))
    print ('volatility    : ', compute_volatility(d_returns))

    #print ('exposure    : ', compute_exposure(models$equal_weight))
    print ('VAR 5%       : ', compute_var(d_returns[1:]))
    print ('CVAR 5%       : ', compute_cvar(d_returns[1:]))
    print ('% POSITIVE MONTHS       : ', compute_percent_positive_months(equity))

#from zipline.utils import tradingcalendar

def endpoints(start=None, end=None, period='M', trading_days=None) :
    
    if trading_days is not None:
        dates = trading_days
# the following 2 lines cause python 3.4.2 to crash, so removed them
#    elif start is not None and end is not None:
#        dates = tradingcalendar.get_trading_days(start, end)
    else:
        print ('\n** ERROR : must either provide pandas series (or df) of trading days \n')
        print ('           or a start and end date\n')
    
    if isinstance(period, int) :
        dates = [dates[i] for i in range(period, len(dates) - period, period)]
    else :    
        if period == 'M' : months = 1
        elif period == 'Q' : months = 3
        elif period == 'B' : months = 6
        elif period == 'Y' : months = 12
            
        e_dates = [dates[i] for i in range(len(dates)-1)\
                          if dates[i].month < dates[i+1].month\
                          or dates[i].year < dates[i+1].year ] + list([dates[-1]])
        dates = [e_dates[i] for i in range(0,len(e_dates),months)]
    
    return dates

def add_portfolio(name, portfolios) :
    return dict(list(portfolios.items()) + list({name : {}}.items()))

# topn 
def ntop(prices, n) :
    weights = pd.DataFrame(0., index=prices.index, columns=prices.columns)
    for i in range(len(prices)) :
        n_not_na = prices.ix[i].count()
        n_row = min(n, n_not_na) 
        for s in prices.columns :
            if prices.ix[i][s] <= n :
                weights.ix[i][s] = 1. / n_row
            else :
                weights.ix[i][s] = 0.
    
    return weights

def monthly_return_table (daily_prices) :
    monthly_returns = daily_prices.resample('M').last().pct_change()
    df = pd.DataFrame(monthly_returns.values, columns=['Data'])
    df['Month'] = monthly_returns.index.month
    df['Year']= monthly_returns.index.year
    table = df.pivot_table(index='Year', columns='Month').fillna(0).round(4) * 100
    annual_returns = daily_prices.resample('12M').last().pct_change()[1:].values.round(4) * 100
    if len(table) > len(annual_returns) :
        table = table[1:]
    table['Annual Returns'] = annual_returns
    return table
    
def generate_orders(transactions, prices) :
    orders = pd.DataFrame()
    for i in range(len(transactions)):
        for j in range(len(transactions.columns)):
            t = transactions.ix[i]
            qty = abs(t[j])
            if qty >= 1.:
                if transactions.ix[i][j] < 0 :
                    orders = orders.append([[t.name.date().year, t.name.date().month, t.name.date().day, t.index[j],\
                                             'Sell', abs(t[j]), prices.ix[t.name][t.index[j]]]])
                if transactions.ix[i][j] > 0 :
                    orders = orders.append([[t.name.date().year, t.name.date().month, t.name.date().day, t.index[j],\
                                             'Buy', abs(t[j]), prices.ix[t.name][t.index[j]]]])
    orders.columns = ['Year', 'Month', 'Day', 'Symbol', 'Action', 'Qty', 'Price']
    orders
    return orders


def save_portfolio_metrics (portfolios, portfolio_name, period_ends, prices, \
                            p_value, p_weights, p_holdings, path=None, risk_free=0) :
        
    rebalance_qtys = (p_weights.ix[period_ends] / prices.ix[period_ends]) * p_value.ix[period_ends]
    #p_holdings = rebalance_qtys.align(prices)[0].shift(1).ffill().fillna(0)
    transactions = (p_holdings - p_holdings.shift(1).fillna(0))
    transactions = transactions[transactions.sum(1) != 0]
    
    p_returns = p_value.pct_change(periods=1)
    p_index = np.cumproduct(1 + p_returns)
    
    m_rets = (1 + p_returns).resample('M').prod() - 1
    
    portfolios[portfolio_name]['equity'] = p_value
    portfolios[portfolio_name]['ret'] = p_returns
    portfolios[portfolio_name]['cagr'] = compute_cagr(p_value) * 100
    portfolios[portfolio_name]['sharpe'] = compute_sharpe(p_value, risk_free)
    portfolios[portfolio_name]['weight'] = p_weights
    portfolios[portfolio_name]['transactions'] = transactions
    portfolios[portfolio_name]['period_return'] = 100 * (p_value.ix[-1] / p_value[0] - 1)
    portfolios[portfolio_name]['avg_monthly_return'] = p_index.resample('BM').last().pct_change().mean() * 100
    portfolios[portfolio_name]['monthly_return_table'] = monthly_return_table(m_rets)
    portfolios[portfolio_name]['drawdowns'] = compute_drawdown(p_value).dropna()
    portfolios[portfolio_name]['max_drawdown'] = compute_max_drawdown(p_value) * 100
    portfolios[portfolio_name]['max_drawdown_date'] = p_value.index[compute_drawdown(p_value)==compute_max_drawdown(p_value)][0].date().isoformat()
    portfolios[portfolio_name]['avg_drawdown'] = compute_avg_drawdown(p_value) * 100
    portfolios[portfolio_name]['calmar'] = compute_calmar(p_value)
    portfolios[portfolio_name]['R_squared'] = compute_calmar(p_value)
    portfolios[portfolio_name]['DVR'] = compute_DVR(p_value)
    portfolios[portfolio_name]['volatility'] = compute_volatility(p_returns)
    portfolios[portfolio_name]['VAR'] = compute_var(p_returns)
    portfolios[portfolio_name]['CVAR'] = compute_cvar(p_returns)
    portfolios[portfolio_name]['rolling_annual_returns'] = p_returns.rolling(252).apply(np.sum)
    portfolios[portfolio_name]['%_positive_months'] = compute_percent_positive_months(p_value) 
    portfolios[portfolio_name]['p_holdings'] = p_holdings
    portfolios[portfolio_name]['transactions'] = np.round(transactions[transactions.sum(1)!=0], 0)
    portfolios[portfolio_name]['share'] = p_holdings
    portfolios[portfolio_name]['orders'] = generate_orders(transactions, prices)
    portfolios[portfolio_name]['best'] = max(p_returns)
    portfolios[portfolio_name]['worst'] = min(p_returns)
    portfolios[portfolio_name]['trades'] = len(portfolios[portfolio_name]['orders'])

    if path != None :
        portfolios[portfolio_name].equity.to_csv(path + portfolio_name + '_equity.csv')
        portfolios[portfolio_name].weight.to_csv(path + portfolio_name + '_weight.csv')
        portfolios[portfolio_name].share.to_csv(path + portfolio_name + '_share.csv')
        portfolios[portfolio_name].transactions.to_csv(path + portfolio_name + '_transactions.csv')
        portfolios[portfolio_name].orders.to_csv(path + portfolio_name + '_orders.csv')
        
    return


# THIS ONE MATCHES PV
# SEE PV backtest :https://goo.gl/lBR4K9
# AND spreadsheet : https://goo.gl/8KGp58
# and Quantopian backtest : https://goo.gl/xytT5L

def backtest(prices, weights, capital, offset=1, commission=0.) :
    rebalance_dates = weights.index
    buy_dates = [prices.index[d + offset] for d in range(len(prices.index)-1) if prices.index[d] in rebalance_dates ]
    print ('FIRST BUY DATE = {}\n'.format(buy_dates[0]))
    p_holdings = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    cash = 0.
    for i, date in enumerate(prices.index):
        if date in rebalance_dates :
#             print ('--------------------------------------------------------------------') 
            new_weights = weights.loc[date]
            p_holdings.iloc [i] = p_holdings.iloc [i - 1]
        if date in buy_dates :           
            if date == buy_dates[0] :
                p_holdings.loc[date] = (capital * weights.iloc[0] / prices.loc[date])
#                 print ('INIT', cash, p_holdings.iloc[i-1],prices.loc[date], new_weights)
            else :
                portfolio_value = cash + (p_holdings.iloc[i - 1] * prices.loc[date]).sum() * new_weights
                p_holdings.iloc[i] = (portfolio_value / prices.loc[date]).fillna(0)
#                 print ('{} BUY \n{}\n{}\n{}\n{}\n{}\nHOLDINGS\n{}\n'.format(date,cash,portfolio_value,p_holdings.iloc[i-1],
#                                                                     prices.loc[date],new_weights,p_holdings.iloc[i]))
                cash = (portfolio_value - p_holdings.iloc[i] * prices.loc[date]).sum()
#                 print ('{}\nPORTFOLIO VALUE\n{}\nCASH = {}'.format(date, portfolio_value,cash))
        else :
            p_holdings.iloc [i] = p_holdings.iloc [i - 1]
            #print ('{} HOLDINGS UNCHANGED'.format(date))

    p_value = (p_holdings * prices).sum(1)[p_holdings.index>=buy_dates[0]]
#     print(p_holdings, )
    p_weights = p_holdings.mul(prices).div(p_holdings.mul(prices).sum(axis=1), axis=0).fillna(0)
    
    return p_value, p_holdings, p_weights

# note: hist_returns are CONTINUOUSLY COMPOUNDED RETURNS
# ie R = e ** hist_returns
def create_historical_ia(symbols, hist_returns, annual_factor=252) :
    
    ia = {}
    ia['n'] = len(symbols)
    ia['annual_factor'] = annual_factor
    ia['symbols'] = hist_returns.columns
    ia['symbol_names'] = hist_returns.columns
    ia['hist_returns'] = hist_returns[symbols]
#    ret = hist_returns[symbols].apply(lambda x: (e ** x) -1.)
    ia['arithmetic_return'] = hist_returns[symbols].mean()
    ia['geometric_return'] = hist_returns[symbols].apply(lambda x: np.prod(1. + x) ** (1. / len(x)) -1.)
    ia['std_deviation'] = hist_returns[symbols].std()
    ia['correlation'] = hist_returns[symbols].corr()
    ia['arithmetic_return'] = ia['arithmetic_return'] * ia['annual_factor']
    ia['geometric_return'] = (1. + ia['geometric_return']) ** ia['annual_factor'] - 1.
    ia['risk'] = sqrt(ia['annual_factor']) * ia['std_deviation']
    for i in range(len(ia['risk'])):
        if ia['risk'][i].round(6) == 0.0 : ia['risk'][i] = 0.0000001
    ia['cov'] = ia['correlation'] * (ia['risk'].dot(ia['risk'].T))
    ia['expected_return'] = ia['arithmetic_return']
    return(ia)

def market_sim(orders, prices, capital, commission=0.):
    
    orders.index = [i for i in range(len(orders))]

    # empty dfs for holdings and port_value
    holdings = pd.DataFrame(0, columns=prices.columns, index=prices.index)
    port_value = pd.DataFrame(0, columns=['Cash', 'Value', 'Total'], index=prices.index)
    port_value['Cash'][0] += capital

    for i in range(len(orders)):
        date = dt.datetime(orders.ix[i].Year,orders.ix[i].Month,orders.ix[i].Day)

        if orders.ix[i]['Action'] == 'Buy':
            holdings.ix[date][orders.Symbol[i]] += orders['Qty'][i]
            port_value.ix[date]['Cash'] -= orders['Qty'][i] * prices.ix[date][orders.Symbol[i]]
        elif orders.ix[i]['Action'] == 'Sell':
            holdings.ix[date][orders.Symbol[i]] -= orders['Qty'][i]
            port_value.ix[date]['Cash'] +=  orders['Qty'][i] * prices.ix[date][orders.Symbol[i]]
        else:
            print ('Bad order')
            raise

    port_value = port_value.cumsum()
    holdings = holdings.cumsum()
    port_value['Value'] = (prices * holdings).sum(axis=1)
    port_value['Total'] = port_value.sum(axis=1)
    
    return port_value, holdings

iif = lambda a,b,c: (b,c)[not a]
def ifna(x,y) :
    return(iif(math.isnan(x)(x) or math.isinf(x), y, x))

def get_history(symbols, start, end, data_path, visible=False):

    from pandas_datareader import data as y_data
    
    """ to get Yahoo data from saved csv files. If the file does not exist for the symbol, 
    data is read from Yahoo finance and the csv saved.
    symbols: symbol list
    start, end : datetime start/end dates
    data_path : datapath for csv files - use double \\ and terminate path with \\
    """  

    symbols_ls = list(symbols)
    for ticker in symbols:
        print (ticker,' ',end="")
        try:
            #see if csv data available
            data = pd.read_csv(data_path + ticker + '.csv', index_col='Date', parse_dates=True)
        except:
            #if no csv data, create an empty dataframe
            data = pd.DataFrame(data=None, index=[start])

        #check if there is data for the start-end data range

        if start.toordinal() < data.index[0].toordinal() \
                             or end.toordinal() > data.index[-1].toordinal():

            if visible:
            	print ('Refresh data.. ',)
            try:
                new_data = y_data.get_data_yahoo(ticker, start, end)

                if new_data.empty==False:
                    if data.empty==False:
                        try:
                            ticker_data = data.append(new_data).groupby(level=0, by=['rownum']).last()
                        except:
                            print ('Merge failed.. ')
                    else:
                        ticker_data = new_data
                    try:
                        ticker_data.to_csv(data_path + ticker + '.csv')
                        if visible:
                        	print (' UPDATED.. ')
                    except:
                        print ('Save failed.. ')
                else:
                    if visible:
                    	print ('No new data.. ')
            except:
                print ('Download failed.. ')
                # remove symbol from list
                symbols_ls.remove(ticker)
        else:
            if visible:
            	print ('OK.. ')
        pass

    pdata = pd.Panel(dict((symbols_ls[i], pd.read_csv(data_path + symbols_ls[i] + '.csv',\
                     index_col='Date', parse_dates=True).sort_index(ascending=True)) for i in range(len(symbols_ls))) )


    return pdata.ix[:, start:end, :]

def get_trading_dates(start, end, offset=0):
    
    ''' to create a list of trading dates (timestamps) for use with Zipline or Quantopian.
         offset = 0 -> 1st trading day of month, offset = -1 -> last trading day of month.
         start, end are datetime.dates'''

    trading_dates = list([])

    trading_days= tradingcalendar.get_trading_days(start, end)

    month = trading_days[0].month
    for i in range(len(trading_days)) :
        if trading_days[i].month != month :
            try :
                trading_dates = trading_dates + list([trading_days[i + offset]])
            except :
                raise

            month = trading_days[i].month

    return trading_dates

# for using Quantopian lectures
def get_pricing(symbols, start_date='2013-01-03', end_date='2014-01-03', symbol_reference_date=None, 
                frequency='daily', fields=None, handle_missing='raise') :
    
    '''
    Load a table of historical trade data.

    Parameters
    ----------
    symbols : Object (or iterable of objects) convertible to Asset
        Valid input types are Asset, Integral, or basestring.  In the case that
        the passed objects are strings, they are interpreted
        as ticker symbols and resolved relative to the date specified by
        symbol_reference_date.

    start_date : str or pd.Timestamp, optional
        String or Timestamp representing a start date for the returned data.
        Defaults to '2013-01-03'.

    end_date : str or pd.Timestamp, optional
        String or Timestamp representing an end date for the returned data.
        Defaults to '2014-01-03'.

    symbol_reference_date : str or pd.Timestamp, optional
        String or Timestamp representing a date used to resolve symbols that
        have been held by multiple companies. Defaults to the current time.

    frequency : {'daily', 'minute'}, optional
        Resolution of the data to be returned.

    fields : str or list, optional
        String or list drawn from {'price', 'open_price', 'high', 'low',
        'close_price', 'volume'}.  Default behavior is to return all fields.

    handle_missing : {'raise', 'log', 'ignore'}, optional
        String specifying how to handle unmatched securities.
        Defaults to 'raise'.

    Returns
    -------
    pandas Panel/DataFrame/Series
        The pricing data that was requested.  See note below.

    Notes
    -----
    If a list of symbols is provided, data is returned in the form of a pandas
    Panel object with the following indices::

        items = fields
        major_axis = TimeSeries (start_date -> end_date)
        minor_axis = symbols

    If a string is passed for the value of `symbols` and `fields` is None or a
    list of strings, data is returned as a DataFrame with a DatetimeIndex and
    columns given by the passed fields.

    If a list of symbols is provided, and `fields` is a string, data is
    returned as a DataFrame with a DatetimeIndex and a columns given by the
    passed `symbols`.

    If both parameters are passed as strings, data is returned as a Series.
    File:      /build/src/qexec_repo/qexec/research/_api.py
    Type:      function
    Search Dogpile  Search Bing Search Yahoo
    '''
    
    import os
    os.chdir('C:\\Users\\scuba\\Google Drive\\NOTEBOOKS')
    from finhelpers3 import get_history
    
    import datetime as dt
    import numpy as np
    
    def date_as_datetime (date='') :
        date = [np.int(d) for d in date.split('-')]
        (y, m, d) = [date[0], date[1], date[2]]
        return dt.datetime(y, m, d)
    
    if isinstance(symbols, str) :
        symbols = [symbols]
        
    start = date_as_datetime(start_date)
    end = date_as_datetime(end_date)    
    
    data = get_history(symbols, start, end, data_path='G:\\Python Resources\\Data\\').transpose(2,1,0)
    data.items = [u'open_price', u'high', u'low', u'close_price', u'volume', u'price']

    if fields==None :
        return data
    else :
        return data[fields]

def highlight_pos_neg (strategy_value) :
    is_positive = strategy_value > 0
    return ['background-color : rgb(127,255,0)' if v else 'background-color : rgb(255,99,71)' for v in is_positive]

def show_return_table(strategy_value):
    df = monthly_return_table (strategy_value)
    return df.style.apply(highlight_pos_neg)

def show_annual_returns(strategy_value) :
    df = monthly_return_table (strategy_value)
    frame = df['Annual Returns'].to_frame()
    frame['positive'] = df['Annual Returns'] >= 0
    return frame['Annual Returns'].plot(figsize=(15, 10), kind='bar', color=frame.positive.map({True: 'g', False: 'r'}), grid=True)
