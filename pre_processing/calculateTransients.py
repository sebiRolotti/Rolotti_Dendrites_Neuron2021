"""Script to automatically detect significant calcium transients"""

import sys
import re
import os
import argparse
from datetime import datetime
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import json
import numpy.ma as ma
from scipy.optimize import curve_fitf
from scipy.interpolate import interp1d
try:
    from bottleneck import nanstd
except ImportError:
    from numpy import nanstd
import cPickle as pickle
import itertools as it
import warnings

import lab_repo.classes.classes as cla
import lab.classes.exceptions as exc
from lab_repo.classes.dbclasses import dbExperimentSet


class configs:
    if (os.path.exists('configs.json')):
        settings = json.load(
            open('configs.json')).get('calculateTransients', {})
    else:
        settings = {}

    # keep first value .01 for initial pruning and baseline recalculation
    P_VALUES = settings.get('P_VALUES', [.01, .05])

    ##
    # Transient identification parameters
    ##
    ON_THRESHOLD = settings.get('ON_THRESHOLD', 2)  # threshold for starting an event (in sigmas)
    OFF_THRESHOLD = settings.get('OFF_THRESHOLD', 0.5)  # end of event (in sigmas)
    # for which to calculate parameters (all events greater are in final bin)
    MAX_SIGMA = settings.get('MAX_SIGMA', 5)
    MIN_DURATION = settings.get('MIN_DURATION', 0.5) # for which to calculate parameters
    MAX_DURATION = settings.get('MAX_DURATION', 5) # for which to calculate parameters
    N_BINS_PER_SIGMA = settings.get('N_BINS_PER_SIGMA', 2)
    N_BINS_PER_SEC = settings.get('N_BINS_PER_SEC', 4)
    # to iteratively identify events and re-estimate noise
    N_ITERATIONS = settings.get('N_ITERATIONS', 3)

    parameters = {
        'on_threshold': ON_THRESHOLD, 'off_threshold': OFF_THRESHOLD,
        'max_sigma': MAX_SIGMA, 'min_duration': MIN_DURATION,
        'max_duration': MAX_DURATION, 'n_bins_per_sigma': N_BINS_PER_SIGMA,
        'n_bins_per_sec': N_BINS_PER_SEC, 'n_iterations': N_ITERATIONS}

    parameters_text = 'ON_THRESHOLD = %.2f, OFF_THRESHOLD = %.2f, \
                       MAX_SIGMA = %.2f, MIN_DURATION = %.2f, \
                       MAX_DURATION = %.2f, N_BINS_PER_SIGMA = %d, \
                       N_BINS_PER_SEC = %d, N_ITERATIONS = %d' % \
        (float(ON_THRESHOLD), float(OFF_THRESHOLD), float(MAX_SIGMA),
         float(MIN_DURATION), float(MAX_DURATION), N_BINS_PER_SIGMA,
         N_BINS_PER_SEC, N_ITERATIONS)

    # one extra bin for events > the max time/sigma
    # parameters above must be chosen such that nBins are integers

    nTimeBins = int((MAX_DURATION - MIN_DURATION) * N_BINS_PER_SEC + 1)
    nSigmaBins = int((MAX_SIGMA - ON_THRESHOLD) * N_BINS_PER_SIGMA + 1)


def channelsToProcess(expt):
    channels_to_process = []
    for channel in expt.imaging_dataset().channel_names:
        try:
            expt.imagingData(dFOverF='from_file', channel=channel,
                             trim_to_behavior=False)
        except (exc.NoDfofTraces, exc.NoSignalsData):
            pass
        else:
            channels_to_process.append(channel)
    return channels_to_process


def labelsToProcess(exptGrp, channel, demixed, overwrite=False):
    """
    Returns the list of labels for which there exists an experiment in the
    group that has either an old or absent transients file relative to
    dfof_traces.

    Also returns a dictionary by dfof_traces labels as keys of the experiments
    that have dfof traces for each label (needs to be recalculated and
    reapplied for all)

    """

    labels_to_process = []
    grp_indices = {}  # dictionary of expt inds for each label
    for expt in exptGrp:
        try:
            with open(expt.dfofFilePath(channel=channel), 'rb') as f:
                dfof = pickle.load(f)
        except (exc.NoSimaPath, exc.NoTSeriesDirectory, IOError,
                pickle.UnpicklingError):
            continue

        try:
            with open(expt.transientsFilePath(channel=channel), 'rb') as f:
                transients = pickle.load(f)
        except (exc.NoSimaPath, exc.NoTSeriesDirectory, IOError,
                pickle.UnpicklingError):

            for label in dfof:
                traces_key = 'demixed_traces' if demixed else 'traces'
                if traces_key in dfof[label]:
                    if label in grp_indices:
                        grp_indices[label].append(expt)
                    else:
                        grp_indices[label] = [expt]
                    labels_to_process.append(label)
        else:
            for label in dfof:
                traces_key = \
                    'demixed_traces' if demixed else 'traces'

                transients_key = \
                    'demixed_transients' if demixed else 'transients'

                if traces_key in dfof[label]:
                    if label in grp_indices:
                        grp_indices[label].append(expt)
                    else:
                        grp_indices[label] = [expt]
                    if overwrite or label not in transients or \
                            transients_key not in transients[label] or \
                            dfof[label]['timestamp'] > transients[label][
                                'timestamp']:

                        labels_to_process.append(label)
    return set(labels_to_process), grp_indices


def estimate_noise(expt, exclude_transients=None, channel='Ch2', label=None,
                   demixed=False):
    """
    Estimate the noise of each cell in the experiment.  If no transients have
    been identified, the noise is coarsely estimated to be the std of the trace
    (concatenated across cycles), a valid assumption for infrequently spiking
    pyramidal cells.  If transients are available, these epochs are excluded
    from the concatenated traces before calculating the std

    exclude_transients -- a transients structure (np record array)
    """
    # concatentate across imaging cycles
    imData = expt.imagingData(dFOverF='from_file', channel=channel,
                              label=label, demixed=demixed,
                              trim_to_behavior=False)

    nCycles = imData.shape[2]
    # start_idx = expt.imagingIndex(10)
    # end_idx = expt.imagingIndex(45) - 1
    # imData = imData[:, start_idx:end_idx]
    concatenated_data = imData[:, :, 0]
    for i in range(nCycles - 1):
        concatenated_data = np.concatenate(
            (concatenated_data, imData[:, :, i + 1]), axis=1)

    if exclude_transients is not None:

        activity = np.zeros(imData.shape, 'bool')

        if activity.ndim == 2:
            activity = activity.reshape(activity.shape[0],
                                        activity.shape[1], 1)

        for cell_index, cell in enumerate(exclude_transients):
            for cycle_index, cycle in enumerate(cell):
                starts = cycle['start_indices']
                ends = cycle['end_indices']
                for start, end in zip(starts, ends):
                    activity[cell_index, start:end + 1, cycle_index] = True

        concatenated_activity = activity[:, :, 0]
        for i in range(nCycles - 1):
            concatenated_activity = np.concatenate(
                (concatenated_activity, activity[:, :, i + 1]), axis=1)

        # nan mask
        nan_mask = np.zeros(concatenated_activity.shape, dtype=bool)
        nan_mask[np.where(np.isnan(concatenated_data))] = True
        concatenated_activity = concatenated_activity | nan_mask

        masked_imData = ma.array(concatenated_data, mask=concatenated_activity)
        noise = masked_imData.std(axis=1).data

    else:
        noise = nanstd(concatenated_data, axis=1)

    return noise


def identify_events(data):
    # generator function which yields the start and stop frames for putative
    # events (data[start, stop] yields the full event)

    # accepts data in terms of sigmas
    L = len(data)
    start_index = 0

    while start_index < L:
        starts = np.where(data[start_index:] > configs.ON_THRESHOLD)[0].tolist()
        if starts:
            # start is the frame right before it crosses ON_THRESHOLD
            # (inclusive)
            abs_start = np.max([0, start_index + starts[0] - 1])
        else:
            break

        ends = np.where(data[abs_start + 1:] < configs.OFF_THRESHOLD)[0].tolist()
        if ends:
            # end is the first frame after it crosses OFF_THRESHOLD (event is
            # inclusive of abs_end -- need to add 1 to include this frame when
            # slicing)
            abs_end = abs_start + ends[0]
        else:
            break

        start_index = abs_end + 1

        yield abs_start, abs_end


def add_events_to_histogram(data, sigma, frame_period, direction, counter):
    """
    Given a timeseries (data), identify putative events and add them to counter
    """

    if direction is 'negative':
        data = -1 * data / sigma
    else:
        data = data / sigma

    for abs_start, abs_end in identify_events(data):

        dur = (abs_end - abs_start) * frame_period
        amp = np.nanmax(data[abs_start:abs_end + 1])

        if np.isnan(amp):
            continue

        if dur >= configs.MIN_DURATION:
            # bins are of the form [start, end)
            # add one for slicing
            sigma_bin_ind = int(np.floor(configs.N_BINS_PER_SIGMA * amp) -
                                configs.ON_THRESHOLD * configs.N_BINS_PER_SIGMA + 1)
            if sigma_bin_ind > configs.nSigmaBins:
                sigma_bin_ind = configs.nSigmaBins
            # add one for slicing
            time_bin_ind = int(np.floor(configs.N_BINS_PER_SEC * dur) -
                               configs.MIN_DURATION * configs.N_BINS_PER_SEC + 1)
            if time_bin_ind > configs.nTimeBins:
                time_bin_ind = configs.nTimeBins

            counter[:sigma_bin_ind, :
                    time_bin_ind] += np.ones([sigma_bin_ind, time_bin_ind])

    return counter


def calculate_event_histograms(experimentList, exclude_transients=None,
                               channel='Ch2', label=None, demixed=False):
    """
    Recursively calls add_events_to_histogram in order to pool event across
    cells across experiments
    """

    negative_event_counter = np.zeros((configs.nSigmaBins, configs.nTimeBins))
    positive_event_counter = np.zeros((configs.nSigmaBins, configs.nTimeBins))
    noise_dict = {}

    for expt in experimentList:
        if exclude_transients is None:
            exclusion = None
        else:
            exclusion = exclude_transients[expt]

        # valid_filter = expt.validROIs(
        #     fraction_isnans_threshold=0, contiguous_isnans_threshold=0,
        #     dFOverF='from_file', channel=channel, label=label, demixed=demixed)
        valid_filter = None

        frame_period = expt.frame_period()

        imData = expt.imagingData(dFOverF='from_file', channel=channel,
                                  label=label, demixed=demixed,
                                  roi_filter=valid_filter,
                                  trim_to_behavior=False)

        noise_dict[expt] = estimate_noise(expt, exclude_transients=exclusion,
                                          channel=channel, label=label,
                                          demixed=demixed)

        valid_indices = expt._filter_indices(
            valid_filter, channel=channel, label=label)

        for cell_data, sigma in zip(imData, noise_dict[expt][valid_indices]):
            for cycle_data in cell_data.T:
                positive_event_counter = add_events_to_histogram(
                    cycle_data, sigma, frame_period, 'positive',
                    positive_event_counter)
                negative_event_counter = add_events_to_histogram(
                    cycle_data, sigma, frame_period, 'negative',
                    negative_event_counter)

    return positive_event_counter, negative_event_counter, noise_dict


def calculate_transient_thresholds(experimentList, p=[.05],
                                   fit_type='pw_linear',
                                   exclude_transients=None, channel='Ch2',
                                   label=None, demixed=False):

    if fit_type is None:
        fit_type = 'pw_linear'

    positive_events, negative_events, noise = calculate_event_histograms(
        experimentList, exclude_transients=exclude_transients, channel=channel,
        label=label, demixed=demixed)

    fpr = negative_events / positive_events

    d = np.linspace(configs.MIN_DURATION, configs.MAX_DURATION, configs.nTimeBins)
    d_fit_axis = np.linspace(configs.MIN_DURATION, configs.MAX_DURATION, 10000)
    thresholds = np.empty([configs.nSigmaBins, len(p)])


    def exp_fit_func(x, a, b, c):
        return a * np.expt(-b * x) + c

    labels_list = [[] for x in range(configs.nSigmaBins)]
    for sigma_ind, sigma_fpr in enumerate(fpr):
        # pull out the monotonically decreasing part of the fpr curves --
        # currently not used for anything
        monotonic_inds = np.where(np.diff(sigma_fpr) > 0)[0] + 1
        if len(monotonic_inds) > 0:
            d1 = d[:monotonic_inds[0]]
            sigma_fpr1 = sigma_fpr[:monotonic_inds[0]]
            for fpr_ind in range(monotonic_inds[0], len(sigma_fpr)):
                if sigma_fpr[fpr_ind] < sigma_fpr1[-1]:
                    d1 = np.append(d1, d[fpr_ind])
                    sigma_fpr1 = np.append(sigma_fpr1, sigma_fpr[fpr_ind])
        else:
            # all monotonic
            d1 = d
            sigma_fpr1 = sigma_fpr

        sigma_level = np.around(
            configs.ON_THRESHOLD + sigma_ind * (1. / configs.N_BINS_PER_SIGMA), decimals=3)
        labels_list[sigma_ind] = '%s Sigma -- ' % (str(sigma_level))

        fit = None  # curve to plot if interpolation is performed
        for p_ind, p_val in enumerate(p):

            if sigma_fpr[0] < p_val:
                threshold = 0
            elif len(np.where(sigma_fpr < p_val)[0]) == 0:
                threshold = np.nan
            else:
                if fit_type is 'exponential':
                    popt, pcov = curve_fit(exp_fit_func, d, sigma_fpr)
                    fit = exp_fit_func(d_fit_axis, popt[0], popt[1], popt[2])
                    threshold = -1 * np.log(
                        (p_val - popt[2]) / popt[0]) / popt[1]
                    if threshold < 0:
                        threshold = 0

                elif fit_type is 'polynomial':
                    z = np.polyfit(d, sigma_fpr, 4)
                    f = np.poly1d(z)
                    fit = f(d_fit_axis)

                    roots = (f - p_val).r
                    real_positive_roots = roots[
                        np.isreal(roots) * roots.real > 0]
                    if len(real_positive_roots) > 0:
                        threshold = np.amin(real_positive_roots)
                    else:
                        threshold = np.nan

                elif fit_type is 'pw_linear':
                    f = interp1d(sigma_fpr1[::-1], d1[::-1],
                                 kind='linear', bounds_error=False)

                    d_fit_axis = d1
                    fit = sigma_fpr1

                    threshold = f(p_val)

            thresholds[sigma_ind, p_ind] = threshold

            threshold_str = str(
                np.around(thresholds[sigma_ind, p_ind], decimals=3))
            if p_ind == len(p) - 1:
                labels_list[
                    sigma_ind] += 'p=%s: %s' % (str(p_val), threshold_str)
            else:
                labels_list[
                    sigma_ind] += 'p=%s: %s, ' % (str(p_val), threshold_str)

    return thresholds, noise


def identify_transients(experimentList, thresholds, noise=None, channel='Ch2',
                        label=None, demixed=False):

    transients = {}
    for expt in experimentList:

        imData = expt.imagingData(dFOverF='from_file', channel=channel,
                                  label=label, demixed=demixed,
                                  trim_to_behavior=False)
        (nCells, _, nCycles) = imData.shape

        t = np.empty(
            (nCells, nCycles), dtype=[
                ('sigma', object), ('start_indices', object),
                ('end_indices', object), ('max_amplitudes', object),
                ('durations_sec', object), ('max_indices', object)])

        if noise is None:
            exp_noise = estimate_noise(expt, exclude_transients=None)
        else:
            exp_noise = noise[expt]

        frame_period = expt.frame_period()
        for cell_idx, cell_data, sigma in it.izip(
                it.count(), imData, exp_noise):

            for cycle_idx, cycle_data in it.izip(it.count(), cell_data.T):
                t[cell_idx][cycle_idx]['sigma'] = sigma
                t[cell_idx][cycle_idx]['start_indices'] = []
                t[cell_idx][cycle_idx]['end_indices'] = []
                t[cell_idx][cycle_idx]['max_amplitudes'] = []
                t[cell_idx][cycle_idx]['durations_sec'] = []
                t[cell_idx][cycle_idx]['max_indices'] = []
                for start, stop in identify_events(cycle_data / sigma):
                    amp = np.nanmax(cycle_data[start:stop + 1])
                    if np.isnan(amp):
                        continue
                    dur = (stop - start) * frame_period

                    sigma_bin_ind = int(np.floor(configs.N_BINS_PER_SIGMA *
                                        (amp / sigma)) - configs.ON_THRESHOLD *
                                        configs.N_BINS_PER_SIGMA)
                    if sigma_bin_ind > configs.nSigmaBins - 1:
                        sigma_bin_ind = configs.nSigmaBins - 1

                    if dur > thresholds[sigma_bin_ind] and \
                            dur > configs.MIN_DURATION:
                        t[cell_idx][cycle_idx]['start_indices'].extend([start])
                        t[cell_idx][cycle_idx]['end_indices'].extend([stop])
                        t[cell_idx][cycle_idx]['max_amplitudes'].extend([amp])
                        t[cell_idx][cycle_idx]['durations_sec'].extend([dur])

                        rel_max_ind = np.where(
                            cycle_data[start:stop + 1] == amp)[0].tolist()[0]
                        t[cell_idx][cycle_idx]['max_indices'].extend(
                            [start + rel_max_ind])

                for field in t.dtype.names:
                    t[cell_idx][cycle_idx][field] = np.array(
                        t[cell_idx][cycle_idx][field])

        transients[expt] = t
    return transients


def main(argv=[]):

    """
    CALCTRANSIENTS Calculates the minimum duration for transients of
    a given amplitude to reach statistical significance at 95% and 99%
    confidence.  This function follows the method outlined in:

    Dombeck, D. a, Khabbaz, A. N., Collman, F., Adelman, T. L. & Tank, D. W.
    Imaging large-scale neural activity with cellular resolution in awake,
    mobile mice. Neuron 56, 4357 (2007).

    This method relies on the assumption that any negative deflection in the
    df/f curve is attributable to motion out of the z-plane.   Because it is
    equally likely that a cell will move into the plane of view as it is that
    a cell will move out of the plane of view, positive deflections
    attributable to this motion should occur at the same frequency as negative
    deflections.  Consequently it is possible to calculate a false positive
    rate for deflections of a given magnitude and duration by dividing the
    number of negative deflections at that amplitude and duration by the
    number of positive deflections of that magnitude and duration.

    The noise and transietns of each cell are estimated iteratively -- the
    noise (sigma) is originally taken to be the 45prctile of the trace (valid
    for sparse activity), transients are conservatively identified at p=.01,
    and the noise is then re-estimated as the 45prctile excluding these events.

    Experiment imaging summaries are created at the end of the __main__ loop

    """

    parameters = configs.parameters
    parameters_text = configs.parameters_text

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-m", "--mice", nargs='+', type=str,
        help="List of mouseIDs over which to pool data and calc transients")
    argParser.add_argument(
        "-a", "--applyList", nargs='+', type=str,
        help="List of mouseIDs to which the thresholds will be applied")
    argParser.add_argument(
        "-l", "--layers", nargs='+', type=str,
        help="Analyze only experiments with these imagingLayers")
    argParser.add_argument(
        "-f", "--fitType", type=str,
        help="Specify fit type for false positive rate curves \
              (default=piecewise linear interpolation): 'pw_linear', \
              'exponential', 'polynomial'")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Overwrite existing analysis")
    argParser.add_argument(
        "-d", "--directory", action="store", type=str, default='',
        help="Process any experiment that has a tSeriesDirectory containing \
              'directory'")
    argParser.add_argument(
        "-s", "--sima_folder", action="store", type=str,
        help="Separate sima folder to save signals out to")
    argParser.add_argument(
        "xml", action='store', type=str, default='behavior.xml',
        help="name of xml file to parse")
    argParser.add_argument(
        "-c", "--channelNames", nargs='+', type=str, default=['Ch1', 'Ch2'],
        help="List of channel names to be used to find signal data. \
              Script defaults to search for 'Ch1'and 'Ch2'")
    argParser.add_argument(
        "-t", "--experimentType", action="store", type=str, default='',
        help="Calculate transients filtered by experiment type")
    argParser.add_argument(
        "-L", "--labels", action="store", type=str, nargs='+', default='',
        help="List of roiList labels to process")
    args = argParser.parse_args()

    if not args.mice and not args.directory:
        print('Must pass in a mouse or directory')
        exit(0)

    # if you pass in a mouse list, it will analyze all of them together to
    # calculate parameters (not separately)

    mouseApplyList = None
    if (re.match('.*sql$', args.xml)):
        experimentSet = dbExperimentSet(args.xml)
        if args.mice:
            param_mice = experimentSet.fetchMice(mouse_name=args.mice)
            if args.applyList:
                mouseApplyList = experimentSet.fetchMice(
                    mouse_name=args.applyList)
        else:
            param_mice = experimentSet.fetchMice()
    else:
        print('Must use SQL database')
        exit(0)

        if args.mice:
            param_mice = [[experimentSet.grabMouse(x) for x in args.mice]]
            if args.applyList:
                mouseApplyList = [experimentSet.grabMouse(x)
                                  for x in args.applyList]
        else:
            param_mice = experimentSet.root.findall('mouse')

    for mice in param_mice:
        # mice is a list of mice on which to calculate the thresholds
        # typically only one mouse
        if isinstance(mice, cla.Mouse):
            mice = [mice]

        # in cases where datasets and rois/signals are stored in a local folder
        # make sure to associate experiment with that .sima before filtering
        if args.sima_folder:
            for m in mice:
                for e in m.findall('experiment'):
                    if e.get('tSeriesDirectory') == args.directory:
                        e.set_sima_path(args.sima_folder)

        if args.layers:
            layers = args.layers
        else:
            layers = list(set([e.get('imagingLayer') for m in mice
                               for e in m.imagingExperiments(
                               channels=args.channelNames)]))
            for x in [None, '']:
                if x in layers:
                    warnings.warn(
                        "Imaging experiments with no layer information " +
                        "will be skipped.")
                    layers.remove(x)

        for layer in layers:
            if mouseApplyList is None:
                expApplyList = cla.ExperimentGroup(
                    [e for m in mice for e in m.imagingExperiments(
                        channels=args.channelNames) if
                     e.get('imagingLayer') == layer and args.directory in
                     e.get('tSeriesDirectory', '') and args.experimentType in
                     e.get('experimentType', '')],
                    label='Thresholds applied')
            else:
                expApplyList = cla.ExperimentGroup(
                    [e for m in mouseApplyList for e in m.imagingExperiments(
                        channels=args.channelNames)
                     if e.get('imagingLayer') == layer and args.directory in
                     e.get('tSeriesDirectory', '') and args.experimentType in
                     e.get('experimentType', '')],
                    label='Thresholds applied')

            if len(expApplyList) == 0:
                continue

            expApplyList = [ex for ex in expApplyList if 'apical' not in ex.get('tSeriesDirectory')]

            # take consensus of recorded channels to process per the apply list
            channels_to_process = []
            for expt in expApplyList:
                channels_to_process.extend(channelsToProcess(expt))
            channels_to_process = list(set(channels_to_process))

            for channel in channels_to_process:
                for demix in (False, True):
                    labels_to_process, apply_lists = labelsToProcess(
                        expApplyList, channel=channel, demixed=demix,
                        overwrite=args.overwrite)

                    if args.labels:
                        labels = args.labels
                    else:
                        labels = labels_to_process

                    for label in labels:
                        try:
                            applyList = apply_lists[label]
                        except KeyError:
                            continue
                        # assemble the calc list
                        calcList = []
                        for e in [e for m in mice for e in
                                  m.imagingExperiments(
                                      channels=channels_to_process)
                                  if e.get('imagingLayer') == layer and
                                  args.directory in
                                  e.get('tSeriesDirectory', '') and
                                  args.experimentType in e.get(
                                      'experimentType', '')]:
                            try:
                                e.imagingData(dFOverF='from_file',
                                              channel=channel, label=label,
                                              demixed=demix,
                                              trim_to_behavior=False)

                            except exc.NoDfofTraces:
                                continue
                            else:
                                calcList.append(e)

                        transients_mask = None
                        for idx in range(configs.N_ITERATIONS):
                            if idx == configs.N_ITERATIONS - 1:
                                thresholds, noise, param_figs = \
                                    calculate_transient_thresholds(
                                        calcList, p=configs.P_VALUES,
                                        fit_type=args.fitType,
                                        exclude_transients=transients_mask,
                                        channel=channel, label=label,
                                        demixed=demix)
                            else:
                                thresholds, noise = \
                                    calculate_transient_thresholds(
                                        calcList, p=configs.P_VALUES,
                                        fit_type=args.fitType,
                                        exclude_transients=transients_mask,
                                        channel=channel, label=label,
                                        demixed=demix)

                                transients_mask = identify_transients(
                                    calcList, thresholds[:, 0], noise=noise,
                                    channel=channel, label=label,
                                    demixed=demix)

                        transients_by_p_val = [[] for x in
                                               range(len(configs.P_VALUES))]
                        for p_ind in range(len(configs.P_VALUES)):
                            transients_by_p_val[p_ind] = identify_transients(
                                applyList, thresholds[:, p_ind], noise=noise,
                                channel=channel, label=label, demixed=demix)

                        for i, p in enumerate(configs.P_VALUES):
                            parameters['thresholds_p' + str(p)] = \
                                thresholds[:, i]

                        for expt in applyList:
                            demix_key = 'transients' if not demix \
                                else 'demixed_transients'
                            try:
                                with open(expt.transientsFilePath(
                                        channel=channel), 'rb') as f:

                                    transients = pickle.load(f)

                            except (IOError, pickle.UnpicklingError):
                                transients = {}
                                transients[label] = {}
                                transients[label][demix_key] = {}
                            else:
                                if label not in transients:
                                    transients[label] = {}
                                transients[label][demix_key] = {}

                            for p_ind, p_val in enumerate(configs.P_VALUES):
                                transients[label][demix_key][
                                    p_val] = transients_by_p_val[p_ind][expt]
                            transients[label][demix_key]['parameters'] = \
                                parameters

                            transients[label]['timestamp'] = datetime.strftime(
                                datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')

                            with open(expt.transientsFilePath(
                                    channel=channel), 'w') as f:

                                pickle.dump(transients, f,
                                            protocol=pickle.HIGHEST_PROTOCOL)
                            text = 'Transients file created for ' + \
                                'Mouse {}, Experiment {}'
                            print text.format(expt.parent.get('mouseID'),
                                expt.get('startTime'))


if __name__ == '__main__':
    main(argv=sys.argv[1:])
