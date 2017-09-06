from datetime import datetime

def set_start_end(start=None, end=None):
    def valid_date(date, proxy_date):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
        try:
            dateparse(date)
        except:
            date = proxy_date
        return date

    start = valid_date(start, '1986-01-01')
    end = valid_date(end, datetime.today().strftime('%Y-%m-%d'))

    if start < end:
        return start, end
    else:
        print("start must be < end : ", start, ' >= ', end)