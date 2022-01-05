import numpy as np
import pandas as pd

import itertools as it
from itertools import count, izip


def silent_cell_percentage(expt_grp, metric='binary_spike_frequency',
                           interval=None, label=None, channel='Ch2',
                           roi_filter=None):

    data_list = []

    for expt in expt_grp:
        spike_freq = calc_activity(
            expt, metric, interval=interval,
            channel=channel, label=label, roi_filter=roi_filter)

        n_silent_cells = np.sum(spike_freq == 0)
        percent_silent_cells = n_silent_cells / float(len(spike_freq))

        data_list.append({'expt': expt,
                          'value': percent_silent_cells})

    return pd.DataFrame(data_list, columns=['expt', 'value'])


def population_activity_new(
        expt_grp, stat, channel='Ch2', label=None, roi_filter=None,
        interval=None, **imaging_kwargs):
    """Calculates various activity metrics on each cell.

    Parameters
    ----------
    stat : str
        Metric to calculate. See lab.analysis.calc_activity for details.
    interval : dict of lab.classes.interval.Interval
        Dictionary by experiment of Interval objects corresponding to frames
        to include.
    **imaging_kwargs
        Additional arguments are passed to expt.imagingData.

    Returns
    -------
    pd.DataFrame
        Returns DataFrame with 'trial', 'roi', and 'value' as columns.

    """

    activity = []
    for expt in expt_grp:
        if interval is None:
            expt_interval = None
        elif isinstance(interval, basestring):
            expt_interval = interval
        else:
            expt_interval = interval[expt]

        if label is None:
            try:
                label = expt_grp.args['imaging_label']
            except (KeyError, AttributeError):
                pass

        if hasattr(expt_grp, 'args'):
            signal = expt_grp.args['signal']
        else:
            signal = None

        expt_activity = calc_activity(expt, 
            signal=signal, channel=channel, 
            label=label, roi_filter=roi_filter,
            method=stat, interval=expt_interval, **imaging_kwargs)

        expt_rois = expt.rois(
            channel=channel, label=label, roi_filter=roi_filter)

        assert expt_activity.shape[0] == len(expt_rois)
        assert expt_activity.shape[1] == len(expt.findall('trial'))

        for trial_idx, trial in enumerate(expt.findall('trial')):
            for roi_activity, roi in it.izip(expt_activity, expt_rois):
                activity.append(
                    {'trial': trial, 'roi': roi,
                     'value': roi_activity[trial_idx]})

    return pd.DataFrame(activity, columns=['trial', 'roi', 'value'])


def calc_activity(
        experiment, method, interval=None, dF='from_file', channel='Ch2',
        label=None, signal=None, roi_filter=None, demixed=False, running_kwargs=None,
        trans_threshold=95):

    """Calculate various population statistics on each ROI

    Takes an BehavioralAnalysis.Experiment object and calculates various
    statistics on the imaging data for each ROI, returning a population vector
    of the desired activity measure. Each cycle is analyzed individually and an
    interval can be passed in to select which frames to include.

    Parameters
    ----------
    experiment : behavior_analysis.Experiment
        Experiment object to analyze
    method : string
        Calculation to perform on each ROI
    interval : boolean array or start/stop frames, optional
        Boolean array of imaging frames to include in analysis, defaults to all
        frames
        Can have a unique interval for each ROI or cycle, automatically
        expanded if a single interval is passed in
    df : string, optional
        dF/F algorithm to run on imaging data, passed to
        behavior_analysis.Experiment.imagingData as 'dFOverF' argument
    average_trials : bool, optional
        If True, average metric across trials

    Returns
    -------
    a : ndarray
        Returns a ndarray of shape (nROIS, nCycles)
    """
    # im_shape = experiment.imaging_shape(
    #    channel=channel, label=label, roi_filter=roi_filter)
    # if im_shape[0] == 0:
    #     return np.empty((0, im_shape[2]))
    data = None

    if interval is None:
        # If no interval passed in, look at the entire imaging sequence
        data = experiment.spikes(roi_filter=roi_filter, channel=channel, label=label)[..., np.newaxis]
        interval = np.ones(data.shape, 'bool')
    elif interval == 'running':
        if running_kwargs:
            interval = np.array(experiment.runningIntervals(
                returnBoolList=True, **running_kwargs))
        else:
            interval = np.array(experiment.runningIntervals(
                returnBoolList=True))
    elif interval == 'non-running':
        if running_kwargs:
            interval = ~np.array(experiment.runningIntervals(
                returnBoolList=True, **running_kwargs))
        else:
            interval = ~np.array(experiment.runningIntervals(
                returnBoolList=True))
    elif interval.dtype is not np.dtype('bool'):
        # If interval is not boolean, assume start/stop times and convert
        # Must pass in a tuple/list/array of exactly 2 elements
        data = experiment.imagingData(
            dFOverF=dF, roi_filter=roi_filter, channel=channel, label=label,
            demixed=demixed)
        inter = np.zeros((data.shape[1], 1), 'bool')
        inter[interval[0]:interval[1] + 1] = True
        interval = np.tile(inter, (data.shape[0], 1, data.shape[2]))

    # If input interval is smaller than shape of data, expand it
    if interval.ndim == 1:
        data = experiment.spikes(roi_filter=roi_filter, channel=channel, label=label)[..., np.newaxis]

        interval = np.reshape(interval, (-1, 1))
        interval = np.tile(interval, (data.shape[0], 1, data.shape[2]))
    elif interval.ndim == 2 and \
            (interval.shape[0] == 1 or interval.shape[1] == 1):
        data = experiment.spikes(roi_filter=roi_filter, channel=channel, label=label)[..., np.newaxis]

        interval = np.reshape(interval, (-1, 1))
        interval = np.tile(interval, (data.shape[0], 1, data.shape[2]))
    elif interval.ndim == 2:
        data = experiment.spikes(roi_filter=roi_filter, channel=channel, label=label)[..., np.newaxis]
        interval = interval[:, :, np.newaxis]
        interval = np.tile(interval, (1, 1, data.shape[2]))

    #
    # Begin calculations
    #
    if method == 'mean':
        # Mean value of signal during interval
        if data is None:
            data = experiment.imagingData(
                dFOverF=dF, roi_filter=roi_filter, channel=channel,
                label=label, demixed=demixed)

        metric = np.zeros((data.shape[0], data.shape[2]))
        for roi_idx, roi_data, roi_int in izip(count(), data, interval):
            for cycle_idx, cycle_data, cycle_int in izip(
                    count(), roi_data.T, roi_int.T):
                metric[roi_idx, cycle_idx] = np.nanmean(
                    cycle_data[cycle_int])

    elif method == 'time active':
        # Percentage of the interval the cell is active
        active = ia.isActive(
            experiment, conf_level=trans_threshold, roi_filter=roi_filter, channel=channel,
            label=label, demixed=demixed)

        metric = np.sum(active & interval, axis=1) / \
            np.sum(interval, axis=1).astype('float')

    elif method in 'binary_spike_frequency':
        period = experiment.frame_period()

        binary = ('binary' in method)

        spikes = experiment.spikes(roi_filter=roi_filter, channel=channel,
                                   label=label, binary=binary)

        spikes = spikes[..., np.newaxis]
        # spikes = spikes[:, interval[0, :, 0], :]
        spikes[~interval] = np.nan

        n_spikes = np.nansum(spikes, axis=1)

        metric = n_spikes / (np.sum(interval, axis=1) * period)

    elif method == 'is place cell':
        with open(experiment.placeFieldsFilePath(channel=channel, signal=signal), 'rb') as f:
            pfs = pkl.load(
                f)[label]['demixed' if demixed else 'undemixed']['pfs']
        inds = experiment._filter_indices(
            roi_filter, channel=channel, label=label)

        pfs = np.array(pfs)[np.array(inds)]

        pc = []
        for roi in pfs:
            if len(roi):
                pc.append(1)
            else:
                pc.append(0)
        metric = np.array(pc).astype('int')[:, np.newaxis]

    else:
        raise ValueError('Unrecognized method: ' + str(method))

    return metric
