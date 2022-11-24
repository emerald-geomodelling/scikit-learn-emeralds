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

def dfextrema_to_numpy(layers):
    l = layers.fillna(-100).astype(int).values
    return np.where(l < 0, np.broadcast_to((np.arange(l.shape[1]) + 1) *  -100, l.shape), l)
    

def dfextrema_connectivity(layers):
    l = dfextrema_to_numpy(layers)

    ls = np.broadcast_to(l.reshape((l.shape[0], 1, l.shape[1])), (l.shape[0], l.shape[1], l.shape[1]))
    nsl = np.broadcast_to(l.reshape(l.shape + (1,)), ls.shape)

    connectivity = np.argmin(np.abs(nsl[1:,:,:] - ls[:-1,:,:]), axis=2)
    connectivity = np.concatenate((np.arange(l.shape[1]).reshape(1, l.shape[1]), connectivity))

    changeovers = np.where(
           (connectivity != np.broadcast_to([np.arange(connectivity.shape[1])],
                                            connectivity.shape)
           ).max(axis=1)
        | np.concatenate(([False], ((l[:-1] < 0) != (l[1:,:] < 0)).max(axis=1)))
    )[0]
    
    return connectivity, changeovers

def dfextrema_to_surfaces(layers, maxchange = None):
    """Takes the output from dfargrelextrema and connects up extrema
    points from consecutive rows, in such a way as to generate as
    contiguous surfaces as possible. If maxchange is specified, a
    surface is broken in two if the extrema positions differ more than
    maxchange.
    """
    
    l = dfextrema_to_numpy(layers)
    connectivity, changeovers = dfextrema_connectivity(layers)
    
    surfaces = []
    current_surfaces = {}
    last_changeover = 0
    for changeover in changeovers:
        for layeridx in range(0, l.shape[1]):
            if (l[last_changeover:changeover, layeridx] >= 0).sum() == 0:
                if layeridx in current_surfaces:
                    surfaces.append(current_surfaces.pop(layeridx))
                continue
            if layeridx not in current_surfaces:
                current_surfaces[layeridx] = {
                    "start": last_changeover,
                    "layers" : []
                }
            current_surfaces[layeridx]["layers"].append(l[last_changeover:changeover, layeridx])
        old_surfaces = current_surfaces
        current_surfaces = {}
        for layeridx in range(0, l.shape[1]):
            oldlayeridx = connectivity[changeover,layeridx]
            if oldlayeridx in old_surfaces:
                if maxchange is None or np.abs(l[changeover-1, oldlayeridx] - l[changeover, layeridx]) < maxchange:
                    current_surfaces[layeridx] = old_surfaces.pop(oldlayeridx)
        surfaces.extend(old_surfaces.values())
        last_changeover = changeover
    for layeridx in range(0, l.shape[1]):
        if layeridx not in current_surfaces:
            current_surfaces[layeridx] = {
                "start": last_changeover,
                "layers" : []
            }
        current_surfaces[layeridx]["layers"].append(l[last_changeover:, layeridx])
    surfaces.extend(current_surfaces.values())

    for surface in surfaces:
        surface["layers"] = np.concatenate(surface["layers"])


    return pd.concat([
        pd.DataFrame({"layers": surface["layers"],
                      "idx": surface["start"] + np.arange(surface["layers"].shape[0]),
                      "surface": idx})
        for idx, surface in enumerate(surfaces)])
