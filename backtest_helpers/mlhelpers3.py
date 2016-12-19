 #!/usr/bin/python
# -*- coding: iso-8859-15 -*-

import pandas as pd
import numpy as np

#Routine to find pivots of degree k
def pivots(s, k):
    '''Routine to find pivots of degree k in 1D numpy array'''
    peaks = []
    troughs = []
    # is there peak or trough in the first k values?
    values = list(s[:k+1].values)
    if values.index(max(values)) < k:
       peaks.append(values.index(max(values)))
    if values.index(min(values)) < k:
       troughs.append(values.index(min(values)))

    for i in range(k,len(s) - k):
        if max([s[j] for j in range(i-k,i+k+1)]) == s[i]:
            peaks.append(i)
        elif min([s[j] for j in range(i-k,i+k+1)]) == s[i]:
            troughs.append(i)

    # is there peak or trough in the last k values?
    values = list(s[-k:].values)
    if values.index(max(values)) < k:
        peaks.append(len(s) -len(values) + values.index(max(values)))
    if values.index(min(values)) < k:
        troughs.append(len(s) - len(values) + values.index(min(values)))
    
    return peaks, troughs

def impact(s, peaks, troughs, down_threshold=0., up_threshold=0.):
    
    '''Routine to calculate the directional impact of peaks and troughs,
    given a sequence (pandas Series) and lists of indices of peaks and troughs.
    Returns indices and magnitudes of impacts > threshold. Threshold may be set
    independently for upward and downward impacts'''

    impacts_p = []
    impacts_t = []
    impacts_p_idx = []
    impacts_t_idx = []
    
    for p in peaks:
        #print ('p = ', p, s[p], s[p:].max())
        if s[p] == s[p:].max() :
            #print ('global max = ', s[p], min(s[p:]) / s[p])
            impact_p = (1. - min(s[p:]) / s[p])
        else :
            next_index = p + next(x[0] for x in enumerate(s[p + 1:])if x[1] > s[p])
            impact_p = (1. - min(s[p : next_index + 1]) / s[p])
        # only include peak if the downward threshold is exceeded
        if impact_p > down_threshold:
            impacts_p.append(impact_p)
            impacts_p_idx.append(p)
        #print (p, impact_p)
            
    for t in troughs:
        #print ('t = ', t, s[t], s[t:].min())
        if s[t] == s[t:].min() :
            #print ('global min = ', s[t], max(s[t:]) / s[t])
            impact_t = (max(s[t:]) / s[t] - 1.)
        else :
            next_index = t + next(x[0] for x in enumerate(s[t + 1:])if x[1] < s[t])
            impact_t = (max(s[t : next_index + 1]) / s[t] - 1.)    
        # only include trough if the upward threshold is exceeded
        if impact_t > up_threshold:
            impacts_t.append(impact_t)
            impacts_t_idx.append(t)
        #print (t, impact_t)
            
    return impacts_p_idx, impacts_p, impacts_t_idx, impacts_t

def momentum(s, peaks, troughs, w, down_threshold=0., up_threshold=0.):
    
    '''Routine to calculate the momentum, with respect to a lookahead window of length w > 0,
    of a peaks or troughs, given a sequence (pandas Series) and lists of indices of peaks
    and troughs. Returns indices and magnitudes of momentum > threshold. Threshold may be set
    independently for upward and downward momentum'''
    
    momentum_p = []
    momentum_t = []
    momentum_p_idx = []
    momentum_t_idx = []
    
    for p in peaks[:-1] :
        momentum = s[p] / min(s[p + 1 : p + w + 1]) - 1.
        if momentum > down_threshold:
            momentum_p.append(momentum)
            momentum_p_idx.append(p)
        
    for t in troughs[:-1] :
        momentum = max(s[t + 1 : t + w + 1] ) / s[t] - 1.
        if momentum > up_threshold:
            momentum_t.append(momentum)
            momentum_t_idx.append(t)
        
    return momentum_p_idx, momentum_p, momentum_t_idx, momentum_t 

def alternate_pivots(s, peaks, troughs): 
    
    ''' Given a timeseries s and 2 lists of turning points, defined by their indices in s,
        this routine will return new sets of alternating peaks and troughs (new_peaks,
        new_troughs). With the exception of the ﬁrst and last elements, every trough 
        will correspond to a global minimum in the time interval deﬁned by the pair of peaks 
        surrounding it, and vice versa – every peak is a global maximum in the time interval 
        deﬁned by the troughs surrounding it.

        calling: new_peaks, new_troughs = alternate_pivots(s, peaks, troughs)
    ''' 

    if peaks == [] or troughs == []:
        raise ValueError ('*** Error : list of peaks or troughs empty')
    
    idxs = sorted(peaks + troughs)
    vals = [s[i] for i in idxs]
    pts = [1 if idxs[i] in peaks else 0 for i in range(len(idxs))]

    current_val = vals[0]
    current_idx = idxs[0]
    new_idxs = []
    new_peaks = []
    new_troughs = []

    for i in range(1,len(idxs)):

        #print (i, current_idx, idxs[i])

        if pts[i] != pts[i - 1]:
            if pts[i-1] == 0:
                new_troughs.append(current_idx)
            else:
                new_peaks.append(current_idx)
            new_idxs.append(current_idx)
            current_val = vals[i]
            current_idx = idxs[i]


        elif pts[i] == pts[i - 1] & pts[i] == 0:    # another trough
            if vals[i] < current_val:
                current_val = vals[i]
                current_idx = idxs[i]

        elif pts[i] == pts[i - 1] & pts[i] == 1:    # another peak
            if vals[i] > current_val:
                current_val = vals[i]
                current_idx = idxs[i]

        if i == len(idxs) - 1:
            #print ('last one', current_idx)
            if pts[i] == 0:
                new_troughs.append(current_idx)
            else:
                new_peaks.append(current_idx)
            new_idxs.append(current_idx)
            
    return new_peaks, new_troughs
        
def optimum_tps(s, peaks, troughs):
    
    '''Routine to determine min/max troughs/peaks, given a sequence (pandas Series) and lists of indices of peaks
    and troughs. Returns indices of the optimum turning points'''
    
    optimum_p = []
    optimum_t = []
    # include first peak/trough
    if peaks[0] < troughs[0]:
        optimum_p.append(peaks[0])
    else:
        optimum_t.append(troughs[0]) 

    for i in range(len(troughs) - 1):
        # find s_index for max between troughs[i] and troughs[i+1]
        maximum = max (s[troughs[i]:troughs[i+1]])
        #print (troughs[i], troughs[i+1], [j for j in range(troughs[i],troughs[i+1]) if s[j] == maximum])
        optimum_p.append([j for j in range(troughs[i],troughs[i+1]) if s[j] == maximum][0])
    for i in range(len(peaks) - 1):
        # find s_index for min between peaks[i] and peaks[i+1]
        minimum = min (s[peaks[i]:peaks[i+1]])
        #print (peaks[i], peaks[i+1], [j for j in range(peaks[i],peaks[i+1]) if s[j] == minimum])
        optimum_t.append([j for j in range(peaks[i],peaks[i+1]) if s[j] == minimum][0])
        
    # include last and peak/trough
    if peaks[-1] > troughs[-1]:
        optimum_p.append(peaks[-1])
    else:
        optimum_t.append(troughs[-1])
        
    return optimum_p, optimum_t

def create_tposcillator(s, peaks, troughs):

    tps = sorted(peaks + troughs)
    TPOscillator = []
    index = []

    # TPOscillator = pd.Series(index=s.index)

    for i in range(len(s)):
        if i < min(min(peaks), min(troughs)) or i > max(max(peaks), max(troughs)):   
            TPOsc = np.nan    # TPOsc undefined
        elif i in troughs:
            TPOsc = 0.
        elif i in peaks:
            TPOsc = 1.
        else:
            idx_left = max([idx for idx in tps  if idx < i])
            idx_right = min([idx for idx in tps  if idx > i])
            if idx_left in peaks:
                Pt = s[idx_left]
                Tt = s[idx_right]
            else :
                Tt = s[idx_left]
                Pt = s[idx_right]
          
            TPOsc = (s[i] - Tt) / (Pt - Tt)
            
            if TPOsc < 0. or TPOsc > 1.:
                raise ValueError ('INVALID TP_OSCILLATOR VALUE')

        # if TPOsc != 999:
        #     #print (i, TPOsc)
        TPOscillator.append(TPOsc)
        index.append(s.index[i])
            
    return pd.Series(TPOscillator, index=index)

def avg_periods(peaks, troughs):

    ''' Routine to calculate average duration p-t, t-p
    '''
    
    p_to_t = []
    t_to_p = []
    
    if min(peaks + troughs) in peaks:
         for i in range(len(peaks)):
            try:
                p_to_t.append(troughs[i] - peaks[i])
            except:
                pass
            try:           
                t_to_p.append(peaks[i+1] - troughs[i])
            except:
                pass
    else:
         for i in range(len(troughs)):
            try:
                t_to_p.append(peaks[i] - troughs[i])
            except:
                pass
            try:           
                p_to_t.append(troughs[i+1] - peaks[i])
            except:
                pass

    return np.average(p_to_t), np.average(t_to_p), p_to_t, t_to_p

def plot_turning_points(s, peaks, troughs, visible=False) :
    import matplotlib.pylab as plt
    fig = plt.figure(figsize=(15,10))
    s.plot(grid=True)

    for p in peaks:
        plt.plot(s.index[p], s[p], 'ro', markersize=8)
        if visible:
            plt.text(s.index[p], s[p], str(p), fontsize=10)
    for t in troughs:
        plt.plot(s.index[t], s[t], 'go', markersize=8)
        if visible:
            plt.text(s.index[t], s[t], str(t), fontsize=10)

def momentum_turning_points(s, pivot_k=1, up_momentum=0.01, down_momentum=0.01, lookahead_window=20, visible=False, optimize=True):

    peaks, troughs = pivots(s, pivot_k)

    momentum_p_idx, momentum_p, momentum_t_idx, momentum_t = momentum(s, peaks, troughs, w=lookahead_window, 
                                                                      down_threshold=down_momentum, up_threshold=up_momentum)

    if len(momentum_p) == 0 or len(momentum_t) == 0:
        raise ValueError ('NO PEAKS/TROUGHS WITH SPECIFIED MOMENTUM: TRY INCREASNG PIVOT_K OR DECREASING UP/DOWN MOMENTUM %')    
    
    peaks, troughs = alternate_pivots(s, momentum_p_idx, momentum_t_idx)

    if optimize:
        peaks, troughs = optimum_tps(s, peaks, troughs)

    plot_turning_points(s, peaks, troughs, visible)

    return peaks, troughs

def impact_turning_points(s, pivot_k=1, up_impact=0.01, down_impact=0.01, visible=False, optimize=True):

    import matplotlib.pylab as plt

    peaks, troughs = pivots(s, pivot_k)
    impacts_p_idx, impacts_p, impacts_t_idx, impacts_t = impact(s, peaks, troughs)

    if len(impacts_p) == 0 or len(impacts_t) == 0:
        raise ValueError ('NO PEAKS/TROUGHS WITH SPECIFIED IMPACT: TRY INCREASNG PIVOT_K OR DECREASING UP/DOWN IMPACT %')     

    new_peaks = []
    new_troughs = []

    for i, p in enumerate(impacts_p_idx):
        if impacts_p[i] >= down_impact :
            new_peaks.append(p)
    for i, t in enumerate(impacts_t_idx):
        if impacts_t[i] >= up_impact :
            new_troughs.append(t)
            
    peaks, troughs = alternate_pivots(s, new_peaks, new_troughs)

    if optimize:
        peaks, troughs = optimum_tps(s, peaks, troughs)

    plot_turning_points(s, peaks, troughs, visible) 

    return peaks, troughs
        
def plot_buy_sell(s, sell_signals, buy_signals, visible=False) :
    import matplotlib.pylab as plt
    from matplotlib import pyplot
    
    buy_and_sell = buy_signals + sell_signals
    buy_and_sell.sort()
    
    fig = plt.figure(figsize=(15,10))
    s.plot(grid=True)


    buy_signal = u'$\u25B2$'
    sell_signal = u'$\u25BC$'

    for p in sell_signals:
        plt.plot(s.index[p], s[p], 'ro', markersize=8)
        plt.plot(s.index[p], s[p] * 1.01, 'r', linestyle='none', marker=sell_signal, markersize=10)
        if visible:
            plt.text(s.index[p], s[p], str(p), fontsize=10)
    for t in buy_signals:
        plt.plot(s.index[t], s[t], 'go', markersize=8)
        plt.plot(s.index[t], s[t] * 0.99, 'g', linestyle='none', marker=buy_signal, markersize=10)
        if visible:
            plt.text(s.index[t], s[t], str(t), fontsize=10)
            
    # join buy pts to sell pts
    for b in buy_signals :
        if b < np.max(sell_signals) :
            ss = [i for i in sell_signals if i > b][0]
            plt.plot([s.index[b], s.index[ss]], [s[b], s[ss]], color='k', linestyle='-', linewidth=1)

