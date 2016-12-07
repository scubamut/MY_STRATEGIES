def endpoints(start=None, end=None, period='M', trading_days=None):
    if trading_days is not None:
        dates = trading_days
    else:
        print('\n** ERROR : must either provide pandas series (or df) of trading days \n')
        print('           or a start and end date\n')

    if isinstance(period, int):
        dates = [dates[i] for i in range(0, len(dates), period)]
    else:
        if period == 'M':
            months = 1
        elif period == 'Q':
            months = 3
        elif period == 'B':
            months = 6
        elif period == 'Y':
            months = 12

        e_dates = [dates[i - 1] for i in range(1, len(dates)) \
                   if dates[i].month > dates[i - 1].month \
                   or dates[i].year > dates[i - 1].year] + list([dates[-1]])
        dates = [e_dates[i] for i in range(0, len(e_dates), months)]

    return dates