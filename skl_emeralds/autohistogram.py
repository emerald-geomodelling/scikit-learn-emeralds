import numpy as np
import scipy.ndimage
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib

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

class Autohistogram(object):
    def __init__(self, bins=None, error=0.05, hist_smoothing = 5, order=1):
        self.bins = bins
        self.error = error
        self.hist_smoothing = hist_smoothing
        self.order = order

    def fit(self, X):
        X = X[~np.isnan(X)]

        if self.bins is None:
            self.hist_bin_heights, self.hist_bin_edges = self._auto_bins(X)
        else:
            self.hist_bin_heights, self.hist_bin_edges = np.histogram(X, bins=self.bins)
        self.hist_bin_heights_smooth = scipy.ndimage.gaussian_filter1d(self.hist_bin_heights, self.hist_smoothing)

        self._calculate_local_minima()
        self._merge_plateaus()
        self._merge_tails()
        self._calculate_peak_widths()
        self._callculate_prominences()        
        
    def predict(self, X):
        shape = X.shape
        X = X.flatten()
        
        cutoffs = self._cutoffs(X)
        
        X_p, lower_p = np.meshgrid(X, cutoffs[:-1], indexing="ij")
        X_p, upper_p = np.meshgrid(X, cutoffs[1:], indexing="ij")

        mask = (X_p > lower_p) & (X_p <= upper_p)
        
        idx, cls = np.nonzero(mask)
        data_cls = np.full(X.shape, -1)
        data_cls[idx] = cls

        return data_cls.reshape(shape)

    def predict_ranges_filt(self, X):
        """Returns tripplets (start, end, mask_array) for each data range
        where the mask_array is a binary array of the same shape as X
        filtering for where start <= X <= end.
        """

        pred = self.predict(X)
        cutoffs = self._cutoffs(X)
        ranges = np.column_stack((cutoffs[:-1], cutoffs[1:]))
        
        return [(start, end, pred == idx)
                for idx, (start, end) in enumerate(ranges)]

    def predict_ranges_midpoints(self, X):
        ranges = self.predict_ranges_filt(X)
        return [(start, end, np.where(filt, (start + end) / 2, np.nan))
                for start, end, filt in ranges]
    
    def predict_ranges_values(self, X):
        ranges = self.predict_ranges_filt(X)
        return [(start, end, np.where(filt, X, np.nan))
                for start, end, filt in ranges]
    
    @property
    def hist_bin_centers(self):
        return (self.hist_bin_edges[1:] + self.hist_bin_edges[:-1]) / 2

    @property
    def bin_edges(self):
        return np.concatenate((
            [self.hist_bin_edges.min()],
            self.hist_bin_centers[self.minima_idx],
            [self.hist_bin_edges.max()]))
        
    def plot_peak_lines(self, ax=None, **kw):
        if ax is None: ax = plt.gca()
        l = self.hist_bin_centers[self.il]
        r = self.hist_bin_centers[self.ir]
        lh = self.hist_bin_heights_smooth[self.il]
        rh = self.hist_bin_heights_smooth[self.ir]
        lines = np.zeros((len(l), 2, 2))
        lines[:,0,0] = l
        lines[:,0,1] = lh
        lines[:,1,0] = r
        lines[:,1,1] = rh
        ax.add_collection(matplotlib.collections.LineCollection(lines, **kw))

    def plot(self, ax = None, **kw):
        if ax is None: ax = plt.gca()

        minima_idx = self.minima_idx
        bin_edges = self.hist_bin_edges
        bin_heights = self.hist_bin_heights
        bin_heights_smooth = self.hist_bin_heights_smooth
        w = self.w
        il = self.il
        ir = self.ir
        prominence = self.prominence
        hist_bin_centers = self.hist_bin_centers

        ax.set_xlim((np.min(bin_edges), np.max(bin_edges)))

        ax.plot(hist_bin_centers, bin_heights, c="blue", alpha=0.2, label="Histogram")
        ax.plot(hist_bin_centers, bin_heights_smooth, c="red", label="Smoothed histogram")
        ax.scatter(hist_bin_centers[minima_idx], bin_heights_smooth[minima_idx], s=400, c="purple")

        ax.set_ylim((0, bin_heights_smooth.max()))

        self.plot_peak_lines(ax=ax, colors="black", linewidths=2)

        ax.scatter(
            hist_bin_centers[minima_idx], prominence,
            s=25, c="green", label="Prominence")

        ax2 = plt.gca().secondary_xaxis("top")
        ax2.set_xticks(hist_bin_centers[minima_idx])

        handles1, labels1 = ax.get_legend_handles_labels()
        ax.legend(handles1, labels1)

        ax.tick_params(axis='y', colors='red')
        ax.yaxis.label.set_color('red')
        ax.set_ylabel("Count")

    def plot_predict(self, X, weights=None, value_based_cmap=True, cmap="rainbow", labelmap=default_labelmap, ax=None):
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        if not callable(labelmap):
            labelmap = make_labelmap(labelmap)
        if ax is None:
            ax = plt.gca()

        cutoffs = self.bin_edges

        if value_based_cmap:
            split_centers = ((cutoffs[1:] + cutoffs[:-1]) / 2) / self.hist_bin_edges.max()
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


    def _cutoffs(self, X):
        return np.concatenate((
            [np.nanmin(X) - 1],
            self.hist_bin_centers[self.minima_idx],
            [np.nanmax(X) + 1]))    
    
    def _auto_bins(self, X):
        best = {"bins": 0}
        steps = 10
        while best["bins"] < len(X):
            attempt = self._auto_bins_histogram(X, best["bins"] + steps)
            if attempt["error"] > self.error:
                break
            best = attempt
            steps *= 2
        steps //= 2
        while steps >= 10:
            attempt = self._auto_bins_histogram(X, best["bins"] + steps)
            if attempt["error"] <= self.error:
                best = attempt
            steps //= 2
        return best['bin_heights'], best['bin_edges']

    def _auto_bins_histogram(self, X, bins):
        bin_heights, bin_edges = np.histogram(X, bins=bins)
        return {
            "bins": bins,
            "bin_heights": bin_heights,
            "bin_edges": bin_edges,
            "error": ((bin_heights[1:] > 0.0) & (bin_heights[:-1] == 0.0)).sum() / len(bin_heights)}
    
    def _calculate_local_minima(self):
        y = self.hist_bin_heights_smooth
        order = self.order
        if order < 1:
            order = order * len(y)
        self.minima_idx = scipy.signal.argrelextrema(y, lambda a, b: a <= b, order=order)[0]

    def _merge_plateaus(self):
        xm = self.hist_bin_centers[self.minima_idx]
        minsep = (xm[1:] - xm[:-1]).min()
        self.minima_idx = self.minima_idx[xm - np.concatenate(([-1], xm[:-1])) > minsep * 2]

    def _merge_tails(self):
        def get_i(j):
            if j < 0:
                return 0
            elif j >= len(self.minima_idx):
                return len(self.hist_bin_heights_smooth)
            return self.minima_idx[j]
        self.minima_idx = [
            self.minima_idx[j] for j in range(len(self.minima_idx))
            if not (   (self.hist_bin_heights_smooth[:get_i(j+1)] < np.mean(self.hist_bin_heights_smooth)).min()
                    or (self.hist_bin_heights_smooth[get_i(j-1):] < np.mean(self.hist_bin_heights_smooth)).min())]

    def _calculate_peak_widths(self):
        self.w = scipy.signal.peak_widths(-self.hist_bin_heights_smooth, self.minima_idx)[0]
        self.il = (self.minima_idx - self.w / 2).astype(int)
        self.ir = (self.minima_idx + self.w / 2).astype(int)

    def _callculate_prominences(self):
        self.prominence = scipy.signal.peak_prominences(-self.hist_bin_heights_smooth, self.minima_idx)[0]
    
