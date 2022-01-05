import numpy as np


def cdf(ax, values, bins='exact', range=None, **kwargs):
    """Plot the empirical CDF.

    Parameters
    ----------
    ax : matplotlib.pyplot.Axes
        The axis to plot on
    Values : array-like
       The data to be plotted.
    bins
        See matplotlib.pyplot.hist documentation.
        Can also be 'exact' to calculate the exact empirical CDF
    range
        See matplotlib.pyplot.hist documentation.
    **kwargs
        Any additional keyword arguments are passed to the plotting function.

    """
    if bins == 'exact':
        bins = np.unique(np.sort(values))
        if len(bins) == 1:
            return None, None
    hist_counts, hist_bins = np.histogram(values, bins=bins, range=range)

    cum_counts = np.cumsum(hist_counts)
    cdf = cum_counts * 1.0 / cum_counts[-1]

    # Want to plot each value at the right side of the bin, but then also put
    # back in value for the beginning of the first bin
    cdf_zero = np.sum(values <= hist_bins[0]) * 1.0 / cum_counts[-1]
    cdf = np.hstack([cdf_zero, cdf])

    ax.plot(hist_bins, cdf, **kwargs)

    ax.set_ylim((0, 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='x', direction='out')
    ax.tick_params(axis='y', direction='out')
    ax.set_ylabel('Cumulative probability')

    return hist_bins, cdf