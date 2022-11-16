import pandas as pd
import numpy as np
import scipy.signal

def group_arange(groups):
    """Creates an integer index for each row inside a group, for all groups. Input is series of group ids."""
    return groups.groupby(groups).apply(lambda a: pd.Series(np.arange(len(a)), index=a.index))

def dfargrelextrema(data, op=np.greater, ffill=True, **kw):
    """Find the local maxima along the second axis of data using
    scipy.signal.argrelextrema(), and returns a dataframe of the
    indexes of the local extrema. Rows in the returned df corresponds
    to rows of data. Each collumn contains the next local maxima if
    there are any more or else <NA>, or a copy of the last one of that
    row if ffill==True."""

    y, x = scipy.signal.argrelextrema(
        data.values, op, axis=1, **kw)

    maxids = pd.DataFrame({"x": x, "y": y})
    maxids["layer"] = group_arange(maxids.y)
    
    numlayers = maxids.y.value_counts().max()
    layers = np.full((data.shape[0], numlayers), -1)

    for layer in range(numlayers):
        layermaxids = maxids.loc[maxids.layer == layer]
        layers[layermaxids["y"].values,layer] = layermaxids["x"].values

    res = pd.DataFrame(layers, index=data.index).astype(pd.Int64Dtype()).replace(-1, np.nan)
    if ffill:
        res = res.T.fillna(method="ffill").T
        
    return res
