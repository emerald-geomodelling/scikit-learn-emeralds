import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib

# class AutoHistogram(object):
#     def __init__(self, bins=1000, hist_smoothing = 5):
#         self.hist_smoothing = hist_smoothing
#         self.bins=bins

#     def fit(self, X):
#         self.X = X
#         self._compute_smooth_histogram()


#     def _compute_smooth_histogram(self):
#         self.h, self.b = np.histogram(self.X, bins=self.bins)
#         self.hs = scipy.ndimage.gaussian_filter1d(self.h, self.hist_smoothing)

def auto_histogram(X, bins=1000, hist_smoothing = 5):
    h, b = np.histogram(X, bins=bins)
    hs = scipy.ndimage.gaussian_filter1d(h, hist_smoothing)

    im = local_minima(b, hs)
    im = merge_plateaus(im, b)
    im = merge_tails(im, b, hs)
    x = bin_centers(b)

    w, il, ir = calculate_peak_widths(im, x, hs)
    prominence = callculate_prominences(im, hs)
    
    return {
        "im": im,
        "b": b,
        "h": h,
        "hs": hs,
        "w": w,
        "il": il,
        "ir": ir,
        "prominence": prominence,
    }

def bin_centers(x):
    return (x[1:] + x[:-1]) / 2

def local_minima(x, y):
    im = scipy.signal.argrelextrema(y, lambda a, b: a <= b, order=20)[0]
    return im

def merge_plateaus(im, x):
    xm = bin_centers(x)[im]
    minsep = (xm[1:] - xm[:-1]).min()
    pim = im[xm - np.concatenate(([-1], xm[:-1])) > minsep * 2]
    return pim

def merge_tails(im, x, y):
    def get_i(j):
        if j < 0:
            return 0
        elif j >= len(im):
            return len(y)
        return im[j]
    pim = [
        im[j] for j in range(len(im))
        if not (   (y[:get_i(j+1)] < np.mean(y)).min()
                or (y[get_i(j-1):] < np.mean(y)).min())]
    #pxm = bin_centers(x)[pim]
    return pim


def callculate_prominences(im, y):
    return scipy.signal.peak_prominences(-y, im)[0]

def calculate_peak_widths(i, x, y):
    w = scipy.signal.peak_widths(-y, i)[0]
    il = (i - w / 2).astype(int)
    ir = (i + w / 2).astype(int)
    return w, il, ir


def plot_peak_lines(il, ir, x, y, ax=None, **kw):
    if ax is None: ax = plt.gca()
    l = x[il]
    r = x[ir]
    lh = y[il]
    rh = y[ir]
    lines = np.zeros((len(l), 2, 2))
    lines[:,0,0] = l
    lines[:,0,1] = lh
    lines[:,1,0] = r
    lines[:,1,1] = rh
    ax.add_collection(matplotlib.collections.LineCollection(lines, **kw))



def plot_autohistogram(im, b, h, hs, w, il, ir, prominence, ax = None):
    if ax is None: ax = plt.gca()
    x = bin_centers(b)

    ax.set_xlim((np.min(b), np.max(b)))
        
    ax.plot(x, h, c="blue", alpha=0.2, label="Histogram")
    ax.plot(x, hs, c="red", label="Smoothed histogram")
    ax.scatter(x[im], hs[im], s=400, c="purple")

    ax.set_ylim((0, hs.max()))

    plot_peak_lines(il, ir, x, hs, ax=ax, colors="black", linewidths=2)
    
    ax.scatter(
        x[im], prominence,
        s=25, c="green", label="Prominence")

    ax2 = plt.gca().secondary_xaxis("top")
    ax2.set_xticks(x[im])

    handles1, labels1 = ax.get_legend_handles_labels()
    ax.legend(handles1, labels1)

    ax.tick_params(axis='y', colors='red')
    ax.yaxis.label.set_color('red')
    ax.set_ylabel("Count")
