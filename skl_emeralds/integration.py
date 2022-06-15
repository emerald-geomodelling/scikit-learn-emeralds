import numpy as np
import scipy.interpolate
import scipy.integrate

def indexof(a, b):
    """Return the indexes in b for each value in a also found in b,
    and -1 for any value missing from b."""
    sorter = np.argsort(b)
    indexes = np.searchsorted(b, a, sorter=sorter)
    sorter = np.concatenate((sorter, [-1]))
    res = sorter[indexes]
    res[a != b[res]] = -1
    return res

def integrate_to(x, y, to_x):
    """Integrate the function y(x) dx from minus infinity to each
    point in to_x. y(x) is assumed to be 0 outside of the range of x:s
    provided."""

    f = scipy.interpolate.interp1d(x, y)
    ax = np.unique(np.concatenate((x, to_x)))
    ax = ax[~np.isnan(ax) & (ax >= np.nanmin(x)) & (ax <= np.nanmax(x))]
    
    ay = f(ax)

    asy = scipy.integrate.cumulative_trapezoid(ay, ax, initial = 0)
    to_x_idx = indexof(to_x, ax)
    return np.where(to_x_idx >= 0,
                    asy[to_x_idx],
                    np.where(to_x < np.nanmin(x),
                             0,
                             asy[-1]))
                    
def window_avg(x, y, window=6):
    """Integrate y(x) from xi - window/2 to x+window/2 for each xi in x,
    then divide by window."""
    window = window / 2
    return ((integrate_to(x, y, x + window) - integrate_to(x, y, x - window)) / 
            (np.where(x + window <= np.nanmax(x), x + window, np.nanmax(x))
             - np.where(x - window >= np.nanmin(x), x - window, np.nanmin(x))))
