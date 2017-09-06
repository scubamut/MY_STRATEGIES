class Parameters():

    def __init__(self, parameters):

        if 'symbols' in parameters:
            self.symbols = parameters['symbols']
        if 'prices' in parameters:
            self.prices = parameters['prices']
        if 'start' in parameters:
            self.start = parameters['start']
        if 'end' in parameters:
            self.end = parameters['end']
        if 'risk_free' in parameters:
            self.risk_free = parameters['risk_free']
        if 'cash_proxy' in parameters:
            self.cash_proxy = parameters['cash_proxy']
        if 'rs_lookback' in parameters:
            self.rs_lookback = parameters['rs_lookback']
        if 'risk_lookback' in parameters:
            self.risk_lookback = parameters['risk_lookback']
        if 'n_top' in parameters:
            self.n_top = parameters['n_top']
        if 'frequency' in parameters:
            self.frequency = parameters['frequency']
        if 'allocations' in parameters:
            self.allocations = parameters['allocations']

