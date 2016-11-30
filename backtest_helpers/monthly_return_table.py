import pandas as pd

def monthly_return_table (daily_prices) :
    monthly_returns = daily_prices.resample('M').last().pct_change()
    # monthly_returns = daily_prices.resample('M', how='last').pct_change()
    df = pd.DataFrame(monthly_returns.values, columns=['Data'])
    df['Month'] = monthly_returns.index.month
    df['Year']= monthly_returns.index.year
    table = df.pivot_table(index='Year', columns='Month').fillna(0).round(4) * 100
    annual_returns = daily_prices.resample('12M').last().pct_change()[1:].values.round(4) * 100
    # annual_returns = daily_prices.resample('12M', how='last').pct_change()[1:].values.round(4) * 100
    if len(table) > len(annual_returns) :
        table = table[1:]
    table['Annual Returns'] = annual_returns
    return table