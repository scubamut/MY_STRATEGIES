# THIS ONE MATCHES PV
# SEE PV backtest :https://goo.gl/lBR4K9
# AND spreadsheet : https://goo.gl/8KGp58
# and Quantopian backtest : https://goo.gl/xytT5L



def backtest(prices, weights, capital, offset=1, commission=0.):

    import pandas as pd

    rebalance_dates = weights.index
    buy_dates = [prices.index[d + offset] for d in range(len(prices.index) - 1) if prices.index[d] in rebalance_dates]
    print('FIRST BUY DATE = {}\n'.format(buy_dates[0]))
    p_holdings = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    cash = 0.
    for i, date in enumerate(prices.index):
        if date in rebalance_dates:
            #             print ('--------------------------------------------------------------------')
            new_weights = weights.loc[date]
            p_holdings.iloc[i] = p_holdings.iloc[i - 1]
        if date in buy_dates:
            if date == buy_dates[0]:
                p_holdings.loc[date] = (capital * weights.iloc[0] / prices.loc[date])
            # print ('INIT', cash, p_holdings.iloc[i-1],prices.loc[date], new_weights)
            else:
                portfolio_value = cash + (p_holdings.iloc[i - 1] * prices.loc[date]).sum() * new_weights
                p_holdings.iloc[i] = (portfolio_value / prices.loc[date]).fillna(0)
        else:
            p_holdings.iloc[i] = p_holdings.iloc[i - 1]
            # print ('{} HOLDINGS UNCHANGED'.format(date))

    p_value = (p_holdings * prices).sum(1)[p_holdings.index >= buy_dates[0]]
    #     print(p_holdings, )
    p_weights = p_holdings.mul(prices).div(p_holdings.mul(prices).sum(axis=1), axis=0).fillna(0)

    return p_value, p_holdings, p_weights