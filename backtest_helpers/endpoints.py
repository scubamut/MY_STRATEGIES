def endpoints(start=None, end=None, period='m', trading_days=None) :

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
        dates = [e_dates[i] for i in range(0,len(e_dates), months)]

    return dates