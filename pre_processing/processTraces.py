"""
for experiments with signals.mat, calculate dfof_traces.pkl in the sima folder
"""

import os
import argparse
import numpy as np
import numpy.ma as ma
try:
    from bottleneck import nanmedian
except ImportError:
    from numpy import nanmedian
import scipy.stats.mstats as mstats
import cPickle as pickle
from datetime import datetime
import itertools as it
import pandas as pd
import traceback
import sys
import re

from lab_repo.classes.dbclasses import dbExperimentSet
import lab_repo.classes.exceptions as exc


def dfofFileCheck(expt):
    """return True if target does not exist or if target is older than ref"""
    try:
        target = expt.dfof_tracesFilePath()
        ref = expt.signalsFilePath()
    except exc.NoSignalsData:
        return False

    if not os.path.exists(target):
        return True
    target_mod_time = os.path.getmtime(target)
    ref_mod_time = os.path.getmtime(ref)
    if target_mod_time < ref_mod_time:
        return True
    return False


def channelsToProcess(expt):
    channels_to_process = []
    try:
        imaging_dataset = expt.imaging_dataset()
    except (exc.NoTSeriesDirectory, exc.NoSimaPath):
        return []
    for channel in imaging_dataset.channel_names:
        if len(expt.imaging_dataset().signals(channel=channel)):
            channels_to_process.append(channel)
    return channels_to_process


def labelsToProcess(expt, channel, overwrite=False):

    try:
        with open(expt.signalsFilePath(channel=channel), 'rb') as f:
            signals = pickle.load(f)
    except (IOError, pickle.UnpicklingError):
        return []

    try:
        with open(expt.dfofFilePath(channel=channel), 'rb') as f:
            dfof = pickle.load(f)
    except (IOError, pickle.UnpicklingError):
        return signals.keys()

    labels_to_process = []
    for key in signals:
        if key not in dfof or overwrite:
            labels_to_process.append(key)
        elif signals[key]['timestamp'] > dfof[key]['timestamp']:
            labels_to_process.append(key)
    return labels_to_process


def calc_baseline(data, method, **kwargs):
    """
    Calculate the baseline of the experiment's imaging data using either a
    static or adaptive method. The static method defines the baseline as the
    nth percentile of the entire trace. The adaptive method first smooths the
    raw data and then uses a sliding window to define the baseline as the
    nth percentile of the preceding 'size' frames

    method:             'adaptive', 'static'
    prctile:            the percentile to use as the baseline
    size:               size in frames of the baseline window
    """

    baseline = np.empty(data.shape)
    (nCells, nFrames, nCycles) = data.shape

    if method == 'static':
        concatenate_cycles = kwargs.get('concatenate_cycles')
        prctile = kwargs.get('prctile')

        if concatenate_cycles:
            concatenated_data = data[:, :, 0]
            for i in range(nCycles - 1):
                concatenated_data = np.concatenate(
                    (concatenated_data, data[:, :, i + 1]), axis=1)

            for cell_idx, cell in it.izip(it.count(), concatenated_data):
                baseline[cell_idx, :, :] = mstats.scoreatpercentile(
                    concatenated_data[cell_idx, :], per=prctile).data
        else:
            for cell_idx, cell in it.izip(it.count(), data):
                for cycle_idx, cycle in it.izip(it.count(), cell.T):
                    baseline[cell_idx, :, cycle_idx] = \
                        mstats.scoreatpercentile(
                            data[cell_idx, :, cycle_idx], per=prctile).data

    if method == 'adaptive':
        size = kwargs.get('size')
        prctile = kwargs.get('prctile')

        t1 = np.amax((1, int(size / 100)))
        for cycle in range(nCycles):

            cycle_frame = pd.DataFrame(data[:, :, cycle])

            # first smooth with rolling boxcar
            cycle_frame = cycle_frame.rolling(
                t1, min_periods=t1 / 2, center=True, win_type='boxcar',
                axis=1).mean()

            # now calculating rolling quantile
            baseline[:, :, cycle] = cycle_frame.rolling(
                size, min_periods=size / 2, center=True, axis=1).quantile(
                    prctile / 100).values

    if method == 'jia':
        # t1 : (int)
        #     number of frames to use for initial boxcar smoothing
        # t2 : (int)
        #     number of centered frames to consider in defining baseline

        t1 = kwargs.get('t1')
        t2 = kwargs.get('t2')

        for cycle in range(nCycles):
            cycle_frame = pd.DataFrame(data[:, :, cycle])

            # Apply boxcar averaging of t1 centered frames
            cycle_frame = cycle_frame.rolling(
                t1, min_periods=t1 / 3, center=True, win_type='boxcar',
                axis=1).mean()

            # Take rolling minimum of t2 centered frames
            baseline[:, :, cycle]  = cycle_frame.rolling(
                t2, min_periods=t2 / 3, center=True, axis=1).apply(np.nanmin,
                                                                   raw=True)

    if method == 'hist':
        # implements method introduced in Dorostkar et al., J Neurosci (2010)
        for cycle in range(nCycles):

            # background = data[-1,:,cycle]
            # data = data[:-1, :, cycle]
            # need map( - background, baseline)?

            T = data.shape[1]
            n = round(np.log2(T) + 1)  # prescribed bin number

            hists = [np.histogram(d[~np.isnan(d)], n) for d in data[:, :, cycle]]
            # returns center of highest frequency fluorescence
            # bin from histogram tuple
            maxbin_center = lambda x: np.mean([x[1][np.argmax(x[0])],
                                              x[1][np.argmax(x[0])+1]])
            baseline_values = map(maxbin_center, hists)  # static baseline value

            # repeat and shape baseline values
            baseline[:, :, cycle] = np.tile(baseline_values, [T,1]).T

    if method == "stimulus":
        blStart = kwargs.get("blStart")
        blEnd = kwargs.get("blEnd")

        baseline = np.nanmedian(data[:, blStart:blEnd, :], 1)
        baseline = np.reshape(baseline, (nCells, 1, nCycles))
        baseline = np.repeat(baseline, nFrames, axis=1)


    return baseline


def smooth_data(data, method='exp', t0=1):
    """
    smooth the data.  Currently the only method is 'exp', though we might want
    to add more later. t0 is the exponential time constant expressed in terms
    of frames
    """

    nan_mask = np.isnan(data)
    data = ma.masked_array(data, mask=nan_mask)

    result = np.empty(data.shape)
    if method == 'exp':
        tau = np.arange(0, np.ceil(3 * t0) + 1)
        w0 = np.exp(-tau / float(t0))
        w0 = np.repeat(w0.reshape(1, w0.shape[0]), data.shape[0], axis=0)
        w0 = np.repeat(
            w0.reshape(w0.shape[0], w0.shape[1], 1), data.shape[2], axis=2)

        result[:, 0, :] = data[:, 0, :]
        for t in range(1, data.shape[1]):
            d = data[:, np.amax([0, t + 1 - w0.shape[1]]):t + 1, :]
            w = w0[:, :np.amin([d.shape[1], w0.shape[1]]), :]
            w = ma.masked_array(w, mask=d.mask)

            num = np.trapz(d[:, ::-1, :] * w, axis=1)
            den = np.trapz(w, axis=1)
            result[:, t, :] = num / den
    result[nan_mask] = np.nan
    return result


def calc_slow_changes(data, window_size=15, prctile=50,
                      exclude_transients=False,
                      experimenter=None):
    """
    Subtract the 50th percentile in a window around each time point
    window:     size in frames of the window centered about each time point
                from which to calculate the 45th percentile
    """

    nCycles = data.shape[2]
    result = np.empty(data.shape)

    for cycle in range(nCycles):
        cycle_frame = pd.DataFrame(data[:, :, cycle])

        # require 95% of frames observed to calculate baseline
        if exclude_transients and experimenter == 'Mohsin':
            percent_needed = 0.50
        else:
            percent_needed = 0.95

        b = cycle_frame.rolling(
            window_size, min_periods=int(percent_needed * window_size),
            center=True, axis=1).apply(np.nanmedian, raw=True)

        b.fillna(method='ffill', axis=1, inplace=True)
        b.fillna(method='bfill', axis=1, inplace=True)

        result[:, :, cycle] = b

    return result


def calc_dfof(expt, label=None,
              baseline_method=None, bl_method_arg1=None, bl_method_arg2=None,
              smoothing='exp', smoothing_t0=0.2, save=True, slow_window=15.,
              exclude_transients=False, channel='Ch2', dFOnly=False):
    """
    baseline_method:       'adaptive' for jia (nature protocols implementation),
                           'static' for constant baseline
    baseline_size:         length (in sec) of baseline imaging
    smoothing_t0:          time constant (in sec) of exponential smoothing
                           function
    baseline_percentile:   default is 45th percentile
    smoothing:             method to use for smoothing of df/f result
    slow_window:           centered window (in sec)
    """

    print "Processing signals for Experiment: \n {} \nChannel: {} \nLabel: {}, ".format(expt, channel, label)
    if save:
        try:
            with open(expt.dfofFilePath(channel=channel), 'rb') as f:
                dfof_data = pickle.load(f)
        except (IOError, pickle.UnpicklingError):
            dfof_data = {}
    else:
        dfof_data = {}

    if 'demixed_raw' in expt.imaging_dataset().signals(channel=channel)[label]:
        demixers = [True, False]
    else:
        demixers = [False]

    for demix in demixers:
        imData = expt.imagingData(label=label, channel=channel,
                                  demixed=demix, trim_to_behavior=False)

        imPeriod = expt.frame_period()
        nCells, nFrames, nCycles = imData.shape

        experimenter = expt.get('experimenter')

        if baseline_method is None:
            if nFrames < 400:
                baseline_method = 'static'
            else:
                baseline_method = 'jia'


        if baseline_method == 'static':
            method_kwargs = {'concatenate_cycles': False,
                             'prctile': 50}
            if bl_method_arg1 is not None:
                method_kwargs["concatenate_cycles"] = bl_method_arg1 > 0
            if bl_method_arg2 is not None:
                method_kwargs["prctile"]=bl_method_arg2

        elif baseline_method == 'jia':
            method_kwargs = {'t1': int(3 / imPeriod),
                             't2': int(60 / imPeriod)}
            if(bl_method_arg1 is not None):
                method_kwargs["t1"]=int(bl_method_arg1/imPeriod);
            if(bl_method_arg2 is not None):
                method_kwargs["t2"]=int(bl_method_arg2/imPeriod);

        elif baseline_method == 'adaptive':
            method_kwargs = {'size': 60,
                             'prctile': 50}
            if(bl_method_arg1 is not None):
                method_kwargs["size"]=int(bl_method_arg1);
            if(bl_method_arg2 is not None):
                method_kwargs["prctile"]=bl_method_arg2;

        elif baseline_method == "stimulus":
            try:
                stimTime = expt.stimulusTime()
            except AttributeError:
                print("you are trying to use stimulus method on an experiment that is not of type SalienceExperiment")
                raise AttributeError
            method_kwargs = {"blStart" : int(0 / imPeriod),
                           "blEnd" : int(stimTime / imPeriod)}
            if(bl_method_arg1 is not None):
                method_kwargs["blStart"] = int((stimTime - bl_method_arg1) / imPeriod)
                if(method_kwargs["blStart"] < 0):
                    method_kwargs["blStart"] = 0;

            if(bl_method_arg2 is not None):
                method_kwargs["blEnd"] = int((stimTime - bl_method_arg2) / imPeriod)
                if(method_kwargs["blEnd"] < 0):
                    method_kwargs["blEnd"] = stimTime

        else:
            method_kwargs = {}

        print "Calculating baseline..."

        baseline_data = imData.copy()

        if exclude_transients:
            exclude_mask = np.zeros(imData.shape, dtype=bool)
            for cycle in range(baseline_data.shape[2]):
                trans = expt.transientsData(channel=channel,
                                            label=label,
                                            behaviorSync=False)[:, cycle]
                for roi_idx, roi_trans in enumerate(trans):
                    for start, stop in it.izip(roi_trans['start_indices'],
                                               roi_trans['end_indices']):
                        exclude_mask[roi_idx, start: stop + 1, cycle] = True
            baseline_data[exclude_mask] = np.nan

        baseline = calc_baseline(
            baseline_data, baseline_method, **method_kwargs)

        print "Calculating df/f..."
        dfof = imData - baseline
        if not dFOnly:
            dfof = dfof / baseline
        if smoothing is not None:
            print "Smoothing df/f..."

            dfof = smooth_data(
                dfof, method='exp', t0=float(smoothing_t0 / imPeriod))

        if slow_window > 0:
            print "Removing slow timescale changes..."
            slow_baseline = dfof.copy()
            if exclude_transients:
                slow_baseline[exclude_mask] = np.nan
            slow_baseline = calc_slow_changes(
                slow_baseline, window_size=int(slow_window / float(imPeriod)),
                exclude_transients=exclude_transients,
                experimenter=experimenter)
            dfof -= slow_baseline

        if label not in dfof_data:
            dfof_data[label] = {}

        dfof_data[label]['traces' if not demix else 'demixed_traces'] = dfof
        dfof_data[label]["baseline"] = baseline

    timestamp = datetime.strftime(datetime.now(), '%Y-%m-%d-%Hh%Mm%Ss')
    dfof_data[label]['timestamp'] = timestamp

    parameters = {
        'baseline_method': baseline_method,
        'baseline_kwargs': method_kwargs,
        'smoothing': smoothing,
        'smoothing_t0': smoothing_t0,
        'remove_slow_changes': slow_window > 0,
        'slow_window': slow_window
    }
    dfof_data[label]['parameters'] = parameters

    if save:
        print "Saving df/f traces..."
        with open(expt.dfofFilePath(channel=channel), 'wb') as f:
            pickle.dump(dfof_data, f, pickle.HIGHEST_PROTOCOL)

    return dfof_data[label]

def main(argv):
    """
    pass in command line arguments to process a single mouse, experiment, or the
    entire experiment tree and save result in dfof_traces.pkl in the sima folder
    """

    ###############SCRIPT STARTS HERE########################
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-m", "--mouse", type=str, help="mouseID to process")
    argParser.add_argument(
        "-e", "--experiment", type=str,
        help="start time of experiment to process")
    argParser.add_argument(
        "-l", "--labels", type=str, nargs='+',
        help="list of labels to calculate dfof for")
    argParser.add_argument(
        "-a", "--adaptive_baseline", action="store_true",
        help="Use an adaptive baseline, parameters: \n\
        -m1 : (int) size of the rolling window\n\
        -m2 : precentile, between 0 and 100")
    argParser.add_argument(
        "-s", "--static_baseline", action="store_true",
        help="Use a static baseline, parameters: \n\
        -m1 : concatenate_cycles, 0 for False, 1 for True \n\
        -m2 : precentile, between 0 and 100")
    argParser.add_argument(
        "-stim", "--stimulus_baseline", action="store_true",
        help="for salience experiments only, use median of pre-stimulus \
            activity as baseline, parameters: \n\
        -m1 : start of the baseline trace, as time (in seconds) before the \
            start of stimulus \n\
        -m2 : end of the baseline trace, as time (in seconds) before the start \
            of stimulus")
    argParser.add_argument(
        "-j", "--jia_baseline", action="store_true",
        help="Use baseline as defined in Jia, et al. Nature Protocols 2012, \
            parameters: \n\
        -m1 : (int) number of frames to use for initial boxcar smoothing \n\
        -m2 : (int) number of centered frames to consider in defining baseline")
    argParser.add_argument(
        "-b", "--hist_baseline", action="store_true",
        help="use baseline and background as defined in Dorostkar, et al. \
            J Neurosci 2010, \n\
        -m1 and -m2 are ignored")
    argParser.add_argument(
        "-m1", "--method_arg1", type=float, action="store", default=float("nan"),
        help="optional parameter 1 for baseline method, depending on the method")
    argParser.add_argument(
        "-m2", "--method_arg2", type=float, action="store", default=float("nan"),
        help="optional parameter 2 for baseline method, depending on the method")
    argParser.add_argument(
        "-t", "--exclude_transients", action="store_true",
        help="Exclude transients in the baseline calculation")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="overwrite existing calculated dF/F traces")
    argParser.add_argument(
        "-dF", "--dF_only", action="store_true",
        help="only calculate dF, and not dF/F")
    argParser.add_argument(
        "-nS", "--no_smoothing", action="store_true",
        help="don't do exponential smoothing on the dF/F")
    argParser.add_argument(
        "-f", "--sima_folder", action="store", type=str,
        help="Separate sima folder where signals are stored to save dFoF out to")
    argParser.add_argument(
        "-d", "--directory", action="store", type=str, default='',
        help="Process any experiment that has a tSeriesDirectory containing 'directory'")
    argParser.add_argument(
        "-y", "--experimentType", action="store", type=str, default='',
        help="Filter experiments on the experimentType xml/sql parameter")
    argParser.add_argument(
        "-S", "--slow_window", action="store", type=float, default=15.,
        help="Window size for remove slow changes, <0 will turn off")
    argParser.add_argument(
        "xml", action='store', type=str, default='behavior.xml',
        help="name of xml file to parse")
    args = argParser.parse_args(argv)

    if args.directory:
        args.directory = os.path.normpath(args.directory)

    if not args.mouse and not args.directory:
        print('Must pass in a mouse or directory')
        exit(0)
    #args.jia_baseline=True
    if args.adaptive_baseline and args.static_baseline:
        raise ValueError('Cannot force both adaptive and static baseline')

    smoothingParam = "exp"
    if args.no_smoothing:
        smoothingParam = None

    exptsToProcess = []
    if (re.match('.*sql$',args.xml)):
        experimentSet = None
        exptType = None
        if args.experimentType == '':
            exptType = None
        exptsToProcess = dbExperimentSet.FetchTrials(
            project_name=args.xml.split('.sql')[0], mouse_name=args.mouse,
            tSeriesDirectory=args.directory, experimentType=exptType)
    else:
        raise('Must use SQL database')
        exit(0)

    exptsToProcess = [ex for ex in exptsToProcess if 'GOL' not in ex.get('tSeriesDirectory')]

    method_arg1 = None
    method_arg2 = None
    if (not np.isnan(args.method_arg1)):
        method_arg1 = args.method_arg1
    if (not np.isnan(args.method_arg2)):
        method_arg2 = args.method_arg2

    if args.static_baseline:
        baseline_method = 'static'
    elif args.adaptive_baseline:
        baseline_method = 'adaptive'
    elif args.jia_baseline:
        baseline_method = 'jia'
    elif args.hist_baseline:
        baseline_method = 'hist'
    elif args.stimulus_baseline:
        baseline_method = "stimulus"
    else:
        baseline_method = None

    if experimentSet is not None:
        exptsToProcess = []
        if args.mouse:
            mouse = experimentSet.grabMouse(args.mouse)
            if args.experiment:
                exptsToProcess.append(
                    experimentSet.grabExpt(args.mouse, args.experiment))
            else:
                for expt in mouse.findall('experiment'):
                    exptsToProcess.append(expt)
        else:
            for mouse in experimentSet.root.findall('mouse'):
                for expt in mouse.findall('experiment'):
                    exptsToProcess.append(expt)

    # in cases where datasets and rois/signals are stored in a local folder
    # make sure to associate experiment with that .sima before filtering
    if args.sima_folder:
        for e in exptsToProcess:
            if e.get('tSeriesDirectory') == args.directory:
                e.set_sima_path(args.sima_folder)

    for expt in exptsToProcess:
        # TODO: is this clause correct?
        if args.directory not in \
                os.path.normpath(expt.get('tSeriesDirectory', '')) or \
                args.experimentType not in expt.get('experimentType', ''):
            continue
        for channel in channelsToProcess(expt):
            if args.labels:
                labelsToProcess_list = args.labels #.split()
            else:
                labelsToProcess_list = labelsToProcess(expt,
                                                       channel, args.overwrite)

            for label in labelsToProcess_list:
                try:
                    calc_dfof(expt, label, baseline_method,
                              bl_method_arg1=method_arg1,
                              bl_method_arg2=method_arg2,
                              smoothing=smoothingParam, smoothing_t0=0.2,
                              save=True, channel=channel,
                              exclude_transients=args.exclude_transients,
                              slow_window=args.slow_window,
                              dFOnly=args.dF_only)
                except:
                    traceback.print_exc(file=sys.stdout)
                    print "failed on ", expt


if __name__ == "__main__":
    main(sys.argv[1:])
