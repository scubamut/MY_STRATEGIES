# https://www.quantopian.com/posts/forward-filling-nans-in-pipeline-custom-factors

def nanfill(arr):
    nan_num = np.count_nonzero(np.isnan(arr))
    if nan_num:
        log.info(nan_num)
        log.info(str(arr))
    mask = np.isnan(arr)
    idx  = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]
    if nan_num:
        log.info(str(arr))
    return arr

##############################################################################
class Quality(CustomFactor):
    inputs = [Fundamentals.total_revenue]
    window_length = 24
    def compute(self, today, assets, out, total_revenue):
        total_revenue = nanfill(total_revenue)
        out[:] = total_revenue

def nanfill(arr):
    mask = np.isnan(arr)
    idx  = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]
    return arr