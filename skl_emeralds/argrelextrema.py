import pandas as pd
import numpy as np
import scipy.signal
import warnings

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

    if not len(y):
        return pd.DataFrame(index=data.index)
        
    maxids = pd.DataFrame({"x": x, "y": y})
    maxids["layer"] = group_arange(maxids.y)
    
    numlayers = maxids.y.value_counts().max()
    layers = np.full((data.shape[0], numlayers), -1)

    for layer in range(numlayers):
        layermaxids = maxids.loc[maxids.layer == layer]
        layers[layermaxids["y"].values,layer] = layermaxids["x"].values

    res = pd.DataFrame(layers, index=data.index).astype(pd.Int64Dtype()).replace(-1, np.nan)
    if ffill:
        res = res.T.ffill().T
        
    return res

def dfextrema_to_numpy(layers):
    l = layers.fillna(-100).astype(int).values
    return np.where(l < 0, np.broadcast_to((np.arange(l.shape[1]) + 1) *  -100, l.shape), l)
    

def dfextrema_connectivity(layers):
    l = dfextrema_to_numpy(layers)

    if l.shape[1] == 0:
        return pd.DataFrame(index=layers.index).values, pd.DataFrame().values

    ls = np.broadcast_to(l.reshape((l.shape[0], 1, l.shape[1])), (l.shape[0], l.shape[1], l.shape[1]))
    nsl = np.broadcast_to(l.reshape(l.shape + (1,)), ls.shape)

    connectivity = np.argmin(np.abs(nsl[1:,:,:] - ls[:-1,:,:]), axis=2)
    connectivity = np.concatenate((np.arange(l.shape[1]).reshape(1, l.shape[1]), connectivity))

    changeovers = (
        (connectivity != np.broadcast_to([np.arange(connectivity.shape[1])],
                                            connectivity.shape)).max(axis=1)
        | np.concatenate(([False], ((l[:-1] < 0) != (l[1:,:] < 0)).max(axis=1))))
    
    return connectivity, changeovers

def dfextrema_to_surfaces(layers, layers_groups=None, start_new_surface = None, maxchange = None):
    """Takes the output from dfargrelextrema and connects up extrema
    points from consecutive rows, in such a way as to generate as
    contiguous surfaces as possible. If maxchange is specified, a
    surface is broken in two if the extrema positions differ more than
    maxchange.


    :param layers: pandas.DataFrame containing the layer indices where a sufficiently high extrema point was located.
        Shape of layers:
            number of rows = number of locations being evaluated, with row order assumed to be ordered by adjcent locations
            number of coolumns= maximum number of extrema points encountered in any location
    :param layers_groups: I'm afraid I don't remember, I'll have to ask redhog if he recalls
    :param start_new_surface: pandas.Series with the same index (number of rows) as layers. Indicates where there should
        be breaks in the surfaces (i.e., consecutive rows are not adjacent points or there's a large gap in the line)
    :param maxchange: maximum number of layer indices to allow until extrema points are no longer connected on the same
        surface. For example, if an extrema point is detected at layer index 4 in one location and at layer index 10 at
        the next location, they will not be connected unless maxchange = .... 5 or 6? I don't recall the behaviour if
        the difference is exactly equal to maxchange. Anyway, use this parameter to reduce the number of large magnitude
        steep jumps
    :return:
    """

    if layers_groups is None:
        layers_groups = pd.DataFrame(index=layers.index, columns=layers.columns)
        layers_groups.values[:,:] = 0
    if not isinstance(layers_groups, pd.DataFrame):
        if isinstance(layers_groups, np.ndarray):
            warnings.warn(f'layers_group was provided as a numpy.ndarray, not as a pandas.DataFrame. Coverting to '
                          f'a DataFrame...')
            layers_groups = pd.DataFrame(layers_groups,index=layers.index, columns=layers.columns)
        else:
            raise TypeError(f'layers_groups must be a pandas.DataFrame or a numpy.ndarray that can be converted to one. '
                            f'In your case, type(layers_groups) ={type(layers_groups)}.')

    l = dfextrema_to_numpy(layers)
    connectivity, changeovers = dfextrema_connectivity(layers)

    disconnect = np.zeros(l.shape, dtype=bool)
    if start_new_surface is not None:
        disconnect[start_new_surface, :] = True

    if maxchange is not None:
        for layeridx in range(0, l.shape[1]):
            oldlayeridx = connectivity[1:,layeridx]

            above = np.concatenate(([False], np.abs(l[1:, layeridx] - l[np.arange(len(oldlayeridx)), oldlayeridx]) >= maxchange))
            disconnect[above, layeridx] = True

    if layers_groups is not None:
        for layeridx in range(0, l.shape[1]):
            oldlayeridx = connectivity[1:,layeridx]

            g = layers_groups.values[1:, layeridx]
            go = layers_groups.values[np.arange(len(oldlayeridx)), oldlayeridx]
            
            diff = np.concatenate(([False], g != go))
            disconnect[diff, layeridx] = True
            
    changeovers = changeovers | disconnect.max(axis=1)    
    changeovers = np.where(changeovers)[0]
    
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
                if not disconnect[changeover, layeridx]:
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

    if not len(surfaces):
        return pd.DataFrame(columns=["layers", "idx", "surface"]).astype({"layers": int, "idx": int, "surface": int})
    return pd.concat([
        pd.DataFrame({"layers": surface["layers"],
                      "idx": surface["start"] + np.arange(surface["layers"].shape[0]),
                      "surface": idx})
        for idx, surface in enumerate(surfaces)], ignore_index=True)

def filter_dfextrema(layers, layer_filter):
    """Filters the output of dfargrelextrema by a boolean dataframe of the
    same shape as the data input to dfargrelextrema, only keeping
    extremas where the boolean filter is True.
    """

    layers_melt = layers.reset_index(names="sounding_idx").melt(id_vars="sounding_idx", var_name="extema_idx", value_name="layer")
    layers_melt = layers_melt.loc[~layers_melt.layer.isna()].astype(int).reset_index(drop=True)

    layers_melt = layers_melt.loc[
        layer_filter[layers_melt.sounding_idx.values, layers_melt.layer.values]
    ].astype(pd.Int64Dtype())

    return layers[[]].join(layers_melt.pivot("sounding_idx", "extema_idx", "layer"))
