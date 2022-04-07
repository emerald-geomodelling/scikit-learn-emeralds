import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib

def auto_histogram(X, bins=1000, hist_smoothing = 5):
    bin_heights, bin_edges = np.histogram(X, bins=bins)
    bin_heights_smooth = scipy.ndimage.gaussian_filter1d(bin_heights, hist_smoothing)

    bin_centers = bin_edges_to_centers(bin_edges)
    minima_idx = local_minima(bin_heights_smooth)
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

def local_minima(y):
    return scipy.signal.argrelextrema(y, lambda a, b: a <= b, order=20)[0]

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

def plot_autohistogram(minima_idx, bin_edges, bin_heights, bin_heights_smooth, w, il, ir, prominence, ax = None, bin_centers=None):
    if ax is None: ax = plt.gca()
    if bin_centers is None: bin_centers = bin_edges_to_centers(bin_edges)

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

def auto_split_filter(data, **kw):
    """Splits a numpy array into ranges of values based in its histogram,
    separating each mode of the distribution. Returns a list of tripplets:

    (start, end, mask_array)

    where the mask_array is a binary array of the same shape as data
    filtering for where start <= data <= end.
    """
    autohist = auto_histogram(data[~np.isnan(data)], **kw)
    cutoffs = np.concatenate(([np.nanmin(data)], autohist["bin_centers"][autohist["minima_idx"]], [np.nanmax(data)]))
    ranges = np.column_stack((cutoffs[:-1], cutoffs[1:]))
    return [(start, end, (data >= start) & (data <= end))
            for start, end in ranges]

def auto_split_data(data, midpoints=False, **kw):
    """Like auto_split_filter but returns filtered data arrays instead
    of boolean filter arrays.

    If midpoints is True, replace data values with the center value of
    each interval ((start+end)/2).
    """
    if midpoints:
        return [(start, end, np.where(filt, (start + end) / 2, np.nan))
                for start, end, filt in auto_split_filter(data, **kw)]
    else:
        return [(start, end, np.where(filt, data, np.nan))
                for start, end, filt in auto_split_filter(data, **kw)]

