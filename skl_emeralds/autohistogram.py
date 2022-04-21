import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib

def _auto_bins_histogram(X, bins):
    bin_heights, bin_edges = np.histogram(X, bins=bins)
    return {
        "bins": bins,
        "bin_heights": bin_heights,
        "bin_edges": bin_edges,
        "error": ((bin_heights[1:] > 0.0) & (bin_heights[:-1] == 0.0)).sum() / len(bin_heights)}

    
def auto_bins(X, error=0.05):
    best = {"bins": 0}
    steps = 10
    while best["bins"] < len(X):
        attempt = _auto_bins_histogram(X, best["bins"] + steps)
        if attempt["error"] > error:
            break
        best = attempt
        steps *= 2
    steps //= 2
    while steps >= 10:
        attempt = _auto_bins_histogram(X, best["bins"] + steps)
        if attempt["error"] <= error:
            best = attempt
        steps //= 2
    return best

def auto_histogram(X, bins=None, error=0.05, hist_smoothing = 5, order=1):
    X = X[~np.isnan(X)]
    
    if bins is None:
        auto = auto_bins(X, error=error)
        bin_edges = auto['bin_edges']
        bin_heights = auto['bin_heights']
    else:
        bin_heights, bin_edges = np.histogram(X, bins=bins)
    bin_heights_smooth = scipy.ndimage.gaussian_filter1d(bin_heights, hist_smoothing)

    bin_centers = bin_edges_to_centers(bin_edges)
    minima_idx = local_minima(bin_heights_smooth, order)
    minima_idx = merge_plateaus(minima_idx, bin_centers)
    minima_idx = merge_tails(minima_idx, bin_heights_smooth)

    w, il, ir = calculate_peak_widths(minima_idx, bin_heights_smooth)
    prominence = callculate_prominences(minima_idx, bin_heights_smooth)
    
    return {
        "minima_idx": minima_idx,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "bin_heights": bin_heights,
        "bin_heights_smooth": bin_heights_smooth,
        "w": w,
        "il": il,
        "ir": ir,
        "prominence": prominence,
    }

def bin_edges_to_centers(bin_edges):
    return (bin_edges[1:] + bin_edges[:-1]) / 2

def local_minima(y, order=1):
    if order < 1:
        order = order * len(y)
    return scipy.signal.argrelextrema(y, lambda a, b: a <= b, order=order)[0]

def merge_plateaus(minima_idx, bin_centers):
    xm = bin_centers[minima_idx]
    minsep = (xm[1:] - xm[:-1]).min()
    return minima_idx[xm - np.concatenate(([-1], xm[:-1])) > minsep * 2]

def merge_tails(minima_idx, bin_heights_smooth):
    def get_i(j):
        if j < 0:
            return 0
        elif j >= len(minima_idx):
            return len(bin_heights_smooth)
        return minima_idx[j]
    return [
        minima_idx[j] for j in range(len(minima_idx))
        if not (   (bin_heights_smooth[:get_i(j+1)] < np.mean(bin_heights_smooth)).min()
                or (bin_heights_smooth[get_i(j-1):] < np.mean(bin_heights_smooth)).min())]

def callculate_prominences(minima_idx, y):
    return scipy.signal.peak_prominences(-y, minima_idx)[0]

def calculate_peak_widths(minima_idx, bin_heights_smooth):
    w = scipy.signal.peak_widths(-bin_heights_smooth, minima_idx)[0]
    il = (minima_idx - w / 2).astype(int)
    ir = (minima_idx + w / 2).astype(int)
    return w, il, ir


def plot_peak_lines(il, ir, bin_centers, bin_heights_smooth, ax=None, **kw):
    if ax is None: ax = plt.gca()
    l = bin_centers[il]
    r = bin_centers[ir]
    lh = bin_heights_smooth[il]
    rh = bin_heights_smooth[ir]
    lines = np.zeros((len(l), 2, 2))
    lines[:,0,0] = l
    lines[:,0,1] = lh
    lines[:,1,0] = r
    lines[:,1,1] = rh
    ax.add_collection(matplotlib.collections.LineCollection(lines, **kw))

def plot_autohistogram(X=None, autohist=None, ax = None, **kw):
    if ax is None: ax = plt.gca()

    if isinstance(X, dict):
        autohist = X
        X = None
    if autohist is None:
        autohist = auto_histogram(X, **kw)
    
    minima_idx=autohist["minima_idx"]
    bin_edges = autohist["bin_edges"]
    bin_heights = autohist["bin_heights"]
    bin_heights_smooth = autohist["bin_heights_smooth"]
    w = autohist["w"]
    il = autohist["il"]
    ir = autohist["ir"]
    prominence = autohist["prominence"]
    bin_centers = autohist.get("bin_centers", None)

    ax.set_xlim((np.min(bin_edges), np.max(bin_edges)))
        
    ax.plot(bin_centers, bin_heights, c="blue", alpha=0.2, label="Histogram")
    ax.plot(bin_centers, bin_heights_smooth, c="red", label="Smoothed histogram")
    ax.scatter(bin_centers[minima_idx], bin_heights_smooth[minima_idx], s=400, c="purple")

    ax.set_ylim((0, bin_heights_smooth.max()))

    plot_peak_lines(il, ir, bin_centers, bin_heights_smooth, ax=ax, colors="black", linewidths=2)
    
    ax.scatter(
        bin_centers[minima_idx], prominence,
        s=25, c="green", label="Prominence")

    ax2 = plt.gca().secondary_xaxis("top")
    ax2.set_xticks(bin_centers[minima_idx])

    handles1, labels1 = ax.get_legend_handles_labels()
    ax.legend(handles1, labels1)

    ax.tick_params(axis='y', colors='red')
    ax.yaxis.label.set_color('red')
    ax.set_ylabel("Count")

def make_labelmap(labels):
    labels = np.array(labels)
    def labelmap(values):
        single = False
        if isinstance(values, float):
            single = True
            values = [values]
        indexes = (np.array(values) * (len(labels) - 1)).round().astype(int)
        ret = labels[indexes]
        if single:
            ret = ret[0]
        return ret
    return labelmap

default_labelmap = ["XSoft", "Soft", "SSoft", "Medium", "SHard", "Hard", "XHard"]

def bins_to_data_ranges(autohist):
    return np.concatenate((
        [autohist["bin_edges"].min()],
        autohist["bin_centers"][autohist["minima_idx"]],
        [autohist["bin_edges"].max()]))

def plot_data_split(X, weights=None, autohist=None, value_based_cmap=True, cmap="rainbow", labelmap=default_labelmap, ax=None, **kw):
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    if not callable(labelmap):
        labelmap = make_labelmap(labelmap)
    if ax is None:
        ax = plt.gca()
        
    if autohist is None:
        autohist = auto_histogram(X, **kw)
        
    cutoffs = bins_to_data_ranges(autohist)
    
    if value_based_cmap:
        split_centers = ((cutoffs[1:] + cutoffs[:-1]) / 2) / autohist["bin_edges"].max()
    else:
        split_centers = np.linspace(0, 1, len(cutoffs) - 1)
    labels = labelmap(split_centers)
    colors = cmap(split_centers)
        
    n, bins, patches = ax.hist(X, weights=weights, bins=cutoffs, rwidth=0.9)
    for patch, color in zip(patches, colors):
        patch.set_color(color)

    if hasattr(X, "name"):
        ax.set_xlabel(X.name)
    if weights is None:
        ax.set_ylabel("Count")
    else:
        if hasattr(weights, "name"):
            ax.set_ylabel(weights.name)
        
    ax2 = ax.twiny()
    ax2.xaxis.set_ticks_position("bottom")
    ax2.set_xticks((cutoffs[:-1] + cutoffs[1:])/2)
    ax2.set_xticklabels(labels)
    ax2.set_xlim(ax.get_xlim())
    ax.set_frame_on(False)
    ax.spines["bottom"].set_position(("axes", -0.03))
    
    return n, bins, patches
    
def auto_split_filter(X, autohist = None, **kw):
    """Splits a numpy array into ranges of values based in its histogram,
    separating each mode of the distribution. Returns a list of tripplets:

    (start, end, mask_array)

    where the mask_array is a binary array of the same shape as X
    filtering for where start <= X <= end.
    """
    if autohist is None:
        autohist = auto_histogram(X, **kw)
    cutoffs = np.concatenate(([np.nanmin(X)], autohist["bin_centers"][autohist["minima_idx"]], [np.nanmax(X)]))
    ranges = np.column_stack((cutoffs[:-1], cutoffs[1:]))
    return [(start, end, (X >= start) & (X <= end))
            for start, end in ranges]

def auto_split_data(X, midpoints=False, **kw):
    """Like auto_split_filter but returns filtered data arrays instead
    of boolean filter arrays.

    If midpoints is True, replace data values with the center value of
    each interval ((start+end)/2).
    """
    if midpoints:
        return [(start, end, np.where(filt, (start + end) / 2, np.nan))
                for start, end, filt in auto_split_filter(X, **kw)]
    else:
        return [(start, end, np.where(filt, X, np.nan))
                for start, end, filt in auto_split_filter(X, **kw)]

