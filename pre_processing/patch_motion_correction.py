"""Set of tools for correcting artifacts introduced during motion correction

To print results to file, run:

python -u patch_motion_correction /path/to/directory >> 'logfile.log'
"""

import os
import sys
from sys import path
import argparse
import itertools as it
import matplotlib
import pandas as pd

if os.environ.get('DISPLAY') is None:
    __DISPLAY__ = False
    matplotlib.use('Agg')
else:
    __DISPLAY__ = True
    matplotlib.use('QT4Agg')

import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle

import gc
from matplotlib.backends.backend_pdf import PdfPages

from sima import ImagingDataset
from sima.motion import HiddenMarkov2D, ResonantCorrection, \
    MotionEstimationStrategy
from sima.sequence import _MotionCorrectedSequence

from signal_extraction import datasets_to_extract
from fix_dataset import make_nonnegative, backup_sima_dir, max_displacements, \
    recalc_averages
from lab_repo.misc.misc import suppress_stdout_stderr
import lab_repo.misc.infer_resonant_phase as irp
from lab_repo.classes.dbclasses import dbExperimentSet
import datetime
#import pdb

def _patch_displacements(mc_seq, method, patch_slice, buffer_slice=None):
    """
    Calculate new displacements for some part of the sequence.

    Parameters
    ----------
    method : sima.motion.MotionEstimationStrategy
        The method for estimating the motion artifacts.
    patch_slice : slice
        The slice of frames that you wish to be patched with new
        displacements.
    buffer_slice : slice
        A larger slice including the patch_slice. The motion correction
        method will be applied to this buffer_slice, and the buffer slice
        will also be used to estimate the offset difference between the
        original and the newly calculated displacements.

    Effects
    -------
    Changes the Sequences displacements for frames within patch_slice.
    This method does NOT save the resulting dataset, so you can check
    whether you like the patch and then decide whether do save or just
    delete and reload the dataset that you are patching.

    """
    if buffer_slice is None:
        buffer_slice = patch_slice

    raw_seq = mc_seq._base

    new_displacements = method.estimate(
        ImagingDataset([raw_seq[buffer_slice]], None))[0]

    # indices for accessing from the original data
    all_indices = list(range(*buffer_slice.indices(len(raw_seq))))
    patch_indices = list(range(*patch_slice.indices(len(raw_seq))))
    buffer_indices = sorted(set(all_indices).difference(patch_indices))

    # indices for accessing from the new displacements
    buffer_idxs = [all_indices.index(i) for i in buffer_indices]
    patch_idxs = [all_indices.index(i) for i in patch_indices]

    # determine the calibration between the new and old displacements
    if len(buffer_idxs):
        orig_displacements = mc_seq.displacements[buffer_indices]
        calibration_displacements = new_displacements[buffer_idxs]
    else:
        orig_displacements = mc_seq.displacements
        calibration_displacements = new_displacements
    shift = np.mean(np.mean(np.mean(
        orig_displacements - calibration_displacements, axis=0),
        axis=0), axis=0)

    # modify the displacements (i.e. apply the patch)
    mc_seq.displacements[patch_indices] = \
        new_displacements[patch_idxs] + np.round(shift)


def define_patch_models(dataset, n_processes=1):

    patch_models = \
        [
           # HiddenMarkov2D(
           #     num_states_retained=150, max_displacement=[800, 200],
           #     n_processes=n_processes, restarts=None),
           # HiddenMarkov2D(
           #     num_states_retained=200, max_displacement=[30, 100],
           #     n_processes=n_processes, restarts=None),
           # HiddenMarkov2D(
           #     num_states_retained=150, max_displacement=[500, 125],
           #     n_processes=n_processes, restarts=None),
           # HiddenMarkov2D(
           #     num_states_retained=150, max_displacement=[8, 50],
           #     n_processes=n_processes, restarts=None)
        ]

    # Shortcut to not calc resonant phase if we're not going to try to re-run
    if not len(patch_models):
        return patch_models

    # Check if this was resonant data...
    dirname = os.path.dirname(dataset.savedir)
    xml = os.path.basename(dirname) + '.xml'
    is_resonant = irp.is_resonant_data(os.path.join(dirname, xml))
    # if it is, calc the phase offset
    if is_resonant:
        h5_path = get_h5_seq(sequence)._path
        # res_phase = 0
        res_phase = irp.identify_resonant_phase_offset(h5_path)
        new_models = [ResonantCorrection(model, offset=res_phase)
                      for model in patch_models]
        return new_models
    return patch_models


# https://github.com/joferkington/oost_paper_code/blob/master/utilities.py
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.nanmedian(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.nanmedian(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def evaluate_results(
        corrected_seq, time_average=None, buffer_slice=None, patch_slice=None,
        ch_idx=-1, window_size=1, percentile=None):
    # If a time average is passed in, buffer_slice must be None, as it is
    # assumed the time_average was calculated on the unsliced sequence
    if time_average is not None:
        assert buffer_slice is None
    if not isinstance(corrected_seq, _MotionCorrectedSequence):
        mc_seq = get_mc_seq(corrected_seq)
    else:
        mc_seq = corrected_seq

    def prep(arr1, arr2):
        arr1 = arr1.copy()
        arr2 = arr2.copy()

        arr1_nans = np.where(np.isnan(arr1))[0]
        arr2_nans = np.where(np.isnan(arr2))[0]

        arr1[arr2_nans] = np.nan
        arr2[arr1_nans] = np.nan

        return \
            np.nan_to_num(arr1 - np.nanmean(arr1)), \
            np.nan_to_num(arr2 - np.nanmean(arr2))

    if buffer_slice is None:
        buffer_slice = slice(0, corrected_seq.shape[0])
    if patch_slice is None:
        patch_slice = slice(0, corrected_seq.shape[0])

    # mc_seq = get_mc_seq(corrected_seq)
    if np.amin(mc_seq.displacements[buffer_slice, ...]) < 0:
        new_displacements = np.array(
            MotionEstimationStrategy._make_nonnegative([mc_seq.displacements]))
        mc_seq.displacements = new_displacements[0]
        max_disp = max_displacements(mc_seq.displacements)
        mc_seq._frame_shape_zyx = tuple(mc_seq._base.shape[1:4] + max_disp)

    imset = ImagingDataset([corrected_seq], None)
    imset = imset[0, buffer_slice, ...]
    if time_average is None:
        time_average = imset.time_averages

    if time_average.ndim == 4:
        time_average = time_average[..., ch_idx]

    time_avgs = []
    masks = []
    for plane in time_average:
        if percentile:
            masks.append(np.where(plane.flatten() >
                                  np.percentile(plane, percentile)))
            time_avgs.append(plane.flatten()[masks[-1]])
        else:
            time_avgs.append(plane.flatten())
            masks.append(np.ones(time_avgs[-1].shape, dtype=bool))
    assert imset.sequences[0].shape[1] == len(time_avgs)

    corrected_patch_slice = slice(patch_slice.start - buffer_slice.start,
                                  patch_slice.stop - buffer_slice.start, None)

    result = [list([]) for _ in xrange(len(time_avgs))]

    def window(seq, n):
        """Return a sliding window (of width n) over data from the iterable."""
        it_seq = iter(seq)
        result = tuple(it.islice(it_seq, n))
        if len(result) == n:
            yield result
        for elem in it_seq:
            result = result[1:] + (elem,)
            yield result

    if window_size == 1:
        # If window size is trivial, use default behavior

        for frame in imset.sequences[0][corrected_patch_slice, ...]:
            for plane_idx, plane, time_avg, mask in it.izip(
                    it.count(), frame, time_avgs, masks):

                result[plane_idx].append(
                    np.corrcoef(prep(plane[:, :, -1].flatten()[mask],
                                     time_avg))[0, 1])

    else:
        # Otherwise take average over sliding window to boost signal
        # before doing comparison to time average
        for frames in window(imset.sequences[0][corrected_patch_slice, ...],
                             window_size):
            frame = np.mean(frames, 0)
            for plane_idx, plane, time_avg in it.izip(
                    it.count(), frame, time_avgs):

                result[plane_idx].append(
                    np.corrcoef(prep(plane[:, :, -1].flatten(),
                                     time_avg))[0, 1])

    return [np.array(x) for x in result]


def plot_artifact_analysis(corrs, patch_intervals, cutOff, Zthresh, title=''):

    n = int(np.ceil(np.sqrt(len(corrs))))
    fig, axs = plt.subplots(n, n, sharex=True, figsize=(12, 12), squeeze=False)
    axs = axs.flatten()

    if type(Zthresh) is not list:
        Zthresh = [Zthresh] * len(corrs)

    for idx, c, th, zth, intervals, ax in it.izip(
            it.count(), corrs, cutOff, Zthresh, patch_intervals, axs):
        ax.plot(c, lw=0.3)
        plt.axes(ax)
        matplotlib.rcParams.update({'font.size': 12})
        for (start, stop) in it.izip(intervals[0], intervals[1]):
            for x in xrange(start, stop):
                ax.vlines(x, ymin=0, ymax=1, colors='r', linestyles='dotted')
        meanC = np.nanmean(c)
        cvC = np.nanstd(c) / meanC
        total_num_frames = len(c)

        if len(intervals[0]) == 0:
            num_corrected_frames = 0
        else:
            num_corrected_frames = np.sum(intervals[1] - intervals[0])

        perc_corrected = \
            (100 * float(num_corrected_frames)) / float(total_num_frames)
        titleIn = 'Plane {}'.format(idx) + ', Mean r = ' + "%1.4f" % meanC + \
            ', CV = ' + "%1.4f" % cvC + '\n' + 'Th = ' + "%1.1f" % zth + \
            ', ' + str(num_corrected_frames) + '/' + str(total_num_frames) + \
            ' (%1.4f' % perc_corrected + '%) Patched'
        ax.set_title(titleIn, fontsize=12)
        ax.set_ylim(0, 1)
        plt.ylabel('Corr. (r)', fontsize=12)
        plt.xlabel('Frame #', fontsize=12)

        ax.plot(range(1, total_num_frames+1), th, color='g', linestyle='--')

    fig.suptitle(title, fontsize=12)
    return fig


def print_artifact_analysis(corrs, patch_intervals, filename='', thresh='nan'):
    n_rows = 4
    n_cols = 4
    while (n_cols-1)*n_rows >= len(corrs):
        n_cols -= 1
    while (n_rows-1)*n_cols >= len(corrs):
        n_rows -= 1

    n_figs = int(np.ceil(len(corrs) / float(n_rows * n_cols)))

    figs = []

    for fig_idx in xrange(n_figs):

        fig, axs = plt.subplots(n_rows, n_cols, sharex=True, squeeze=False)

        for row_idx in xrange(n_rows):
            for col_idx in xrange(n_cols):

                plane_idx = fig_idx * n_rows * n_cols + \
                    row_idx * n_cols + col_idx

                if plane_idx >= len(corrs):
                    break

                else:
                    c = corrs[plane_idx]
                    intervals = patch_intervals[plane_idx]
                    ax = axs[row_idx, col_idx]

                    ax.plot(c, lw=0.3)
                    for (start, stop) in it.izip(intervals[0], intervals[1]):
                        for x in xrange(start, stop):
                            ax.vlines(x, ymin=0, ymax=1, colors='r',
                                      linestyles='dotted')

                    ax.set_title('Plane {}'.format(plane_idx), fontsize=12)
                    ax.set_ylim(0, 1)

        figs.append(fig)

    pp = PdfPages(filename)
    for fig in figs:
        pp.savefig(fig)
        fig.clf()
    for ax in axs.flatten():
        ax.clear()
    pp.close()
    plt.close('all')
    del figs, axs, pp
    gc.collect()


def get_h5_seq(sequence):
    base = sequence
    while True:
        try:
            base = base._base
        except AttributeError:
            break
    return base


def get_mc_seq(sequence):
    base = sequence
    while True:
        if base.__class__ == _MotionCorrectedSequence:
            return base
        try:
            base = base._base
        except AttributeError:
            return None


def slidingStats(sequence, window): #this definitely works now. 
    ds = pd.Series(sequence)    
    moveMean = ds.rolling(window, min_periods=1).mean()
    moveSTD = ds.rolling(window, min_periods=1).std()
    return moveMean, moveSTD


def identify_patch_regions(correlations, buffer, thresholds, window):

    if type(thresholds) is not list:
        thresholds = [thresholds] * len(correlations)
    else:
        assert(len(thresholds) == len(correlations))

    regions = []
    cutoffs = []
    for corrs, thresh in zip(correlations, thresholds): 

        if window >1 and len(corrs) > 1:
            moveMean, moveSTD = slidingStats(np.array(corrs), window)
            absCutOff = np.subtract(moveMean, moveSTD*thresh)

            #initial sliding values will be unstable so patch with val @ .25*window
            firstRealInd = int(round(window*.25))
            initVal = absCutOff[firstRealInd]
            initThresh = [initVal] * (firstRealInd -1)
            for i, x in zip(range(firstRealInd),initThresh): 
                absCutOff[i] = x

        else:
            absCutOff = [np.nanmean(corrs) - thresh * np.nanstd(corrs)] * len(corrs)
            
        outliers = np.where(corrs < absCutOff)[0]
        outliers = np.hstack([outliers, np.where(np.isnan(corrs))[0]])
        outliers = np.sort(outliers)

        cutoffs.append(absCutOff)

        if not len(outliers):
            regions.append([[], []])
            continue

        starts = np.hstack(
            (0, np.where(np.diff(outliers) > 5)[0] + 1))
        stops = np.hstack(
            (np.where(np.diff(outliers) > 5)[0] + 1, len(outliers)))
        o = [outliers[s:e] for s, e in zip(starts, stops)]

        regions.append(np.array([(x[0], x[-1] + 1) for x in o]).T)

    return regions, cutoffs


def mask_frames(sequence, frames, plane=None):
    if plane is not None:
        starts = np.hstack(
            (0, np.where(np.diff(outliers) > 5)[0] + 1))
        stops = np.hstack(
            (np.where(np.diff(outliers) > 5)[0] + 1, len(outliers)))
        o = [outliers[s:e] for s, e in zip(starts, stops)]

        result.append(np.array([(x[0], x[-1] + 1) for x in o]).T)
    return result


def mask_frames(sequence, frames, plane=None):
    if plane is not None:
        plane = [plane]
    return sequence.mask([(frames, plane, None, None)])


def patch_plane_intervals(
        starts, stops, sequence, plane, buffer_size, patch_models, ch_idx,
        verbose=True):

    if verbose:
        print 'Plane {}:'.format(plane)

    intervals_to_patch, new_displacements, problem_frames = [], [], []

    # Shortcut out if we aren't going to attempt to re-run MC
    if not len(patch_models):
        return [], [], zip(starts, stops)

    corrected = get_mc_seq(sequence)

    for start, stop in zip(starts, stops):
        if verbose:
            print 'Frames {} - {}'.format(str(start), str(stop - 1))
            print '-' * (10 + len(str(start)) + len(str(stop)))

        patch_slice = slice(
            start, stop)
        buffer_slice = slice(
            max(0, start - buffer_size),
            min(sequence.shape[0], stop + buffer_size))

        original_quality = np.mean(
            evaluate_results(
                corrected[:, plane, ...], buffer_slice=buffer_slice,
                patch_slice=patch_slice, ch_idx=ch_idx,
                window_size=args.window))

        if verbose:
            print 'ORIGINAL CORRELATION = {}'.format(str(original_quality)[:6])

        model_displacements = []
        results = []
        for i, patch_model in enumerate(patch_models):

            c_dict = corrected._todict()
            c_dict.pop('__class__')
            corrected1 = _MotionCorrectedSequence._from_dict(
                c_dict)

            # patch_displacements method modifies in place
            with suppress_stdout_stderr():
                _patch_displacements(
                    corrected1, patch_model, patch_slice, buffer_slice)
            patch_displacements = corrected1.displacements[
                patch_slice, plane, ...].copy()
            new_quality = np.mean(
                evaluate_results(
                    corrected1[:, plane, ...], buffer_slice=buffer_slice,
                    patch_slice=patch_slice, ch_idx=ch_idx,
                    window_size=args.window))

            if verbose:
                print 'Model {}: {}'.format(i + 1, str(new_quality)[:6])
            model_displacements.append(patch_displacements)
            results.append(new_quality)

        if verbose:
            print '\n'

        if results and max(results) > 1.2 * original_quality:
            intervals_to_patch.append(patch_slice)
            new_displacements.append(
                model_displacements[np.argmax(results)])
        else:
            problem_frames.append([start, stop])

    return intervals_to_patch, new_displacements, problem_frames

def main(argv):

    argParser = argparse.ArgumentParser()
    
    argParser.add_argument(
        "directory", action="store", type=str, default=os.curdir,
        help="Locate all datasets below this folder")
    argParser.add_argument(
        "-b", "--buffer", action="store", type=int, default=10,
        help="Number of buffer frames on each side of the patch slice")
    argParser.add_argument(
        "-t", "--threshold", action="store", type=float, default=10,
        help="Modified z-score to use for detecting problem frames.  Lower \
        values results in inclusion of more frames")
    argParser.add_argument(
        "-c", "--trim_criterion", action="store", type=float, default=0.95,
        help="Minimum occupancy used for creating new indexed sequence on top \
        of motion corrected sequence")
    argParser.add_argument(
        "-p", "--processes", action="store", type=int, default=1,
        help="Number of processes to use for motion correction. " +
             "Be considerate of others.")
    argParser.add_argument(
        "-m", "--mask", nargs='+', type=int,
        help="Apply mask to list of frame indices.  Note this applies the \
        mask to the top layer of the sequence list")
    argParser.add_argument(
        "-s", "--sequence", action="store", type=int, default=0,
        help="If mask frames are passed in, apply to this sequence index")
    argParser.add_argument(
        "-z", "--z_plane", action="store", type=int, default=None,
        help="If mask frames are passed in, only mask this plane")
    argParser.add_argument(
        "-M", "--mask_failures", action="store_true",
        help="Automatically mask all frames that were not successfully \
        patched")
    argParser.add_argument(
        "-l", "--load_corrs", action="store_true",
        help="Load corrs.pkl file instead of re-calculating")
    argParser.add_argument(
        "-d", "--dynamic_plot", action="store_true",
        help="Plot like in do-nothing condition, but  with option to update \
        threshold, re-plot, and then run patch on detected segments")
    argParser.add_argument(
        "-C", "--channel", action="store", type=str, default="Ch2",
        help="Channel to calculate frame-to-average correlation")
    argParser.add_argument(
        "--backup", action="store_true",
        help="Create a backup of the original .sima directory before patching")
    argParser.add_argument(
        "-B", "--batch_corr", action="store_true",
        help="Batch calc correlations mode")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="overwite corrs.pkl in batch calc correlations mode")
    argParser.add_argument(
        "-r", "--range", action="store", type=int, nargs=2, default=None,
        help="Give a start and end frame (inclusive) to mask between")
    argParser.add_argument(
        "-P", "--pdf", action="store_true",
        help="Print to pdf instead of plotting figure")
    argParser.add_argument(
        "-wr", "--r_window", action="store", type=int, default=1,
        help="Rolling window of frames to average over before calculating \
        r values to time average")
    argParser.add_argument(
        "-wz", "--z_window", action="store", type=int, default=1,
        help="Rolling window of frame-to-frame correlations to calculate \
        Z-score. Can address slow decay of r in long sessions due to bleaching")
    argParser.add_argument(
        "--pct", action="store", type=float, default=None,
        help="Only include pixels above this percentile in each time average \
                      plane when calculating correlations")
    argParser.add_argument(
        "-A", "--skip_avg_recalc", action="store_true",
        help="Skip recalculation of time_avgs after patch correction.")

    args = argParser.parse_args(argv)

    # dsets = [ImagingDataset.load(args.directory)]
    dsets = [dset for dset in datasets_to_extract(args.directory)]
    if args.range:
        args.mask = range(args.range[0], args.range[1] + 1)

    if args.mask: #not the case anymore so ignore this block
        assert len(dsets) == 1

        if args.backup:
            backup_sima_dir(os.path.join(dsets[0].savedir, 'dataset.pkl'))

        new_sequences = dsets[0].sequences[:]
        seq_to_mask = new_sequences[args.sequence]
        new_sequences[args.sequence] = mask_frames(
            seq_to_mask, args.mask, args.z_plane)
        dsets[0].sequences = new_sequences
        dsets[0].save()

    else:
        for dset in dsets:

            ch_idx = dset._resolve_channel(args.channel)
            print 'Processing dataset: {}'.format(dset.savedir)
            print '=' * (20 + len(dset.savedir))

            if args.backup:
                # backup_sima_dir(os.path.join(dset.savedir, 'dataset.pkl'))
                backup_sima_dir(dset.savedir)

            if args.batch_corr:
                corrs_path = os.path.join(dset.savedir, 'corrs.pkl')
                already_calculated = False
                if os.path.exists(corrs_path):
                    with open(corrs_path, 'rb') as rp:
                        corrs_check = pickle.load(rp)
                    if isinstance(corrs_check, dict):
                        if args.r_window in corrs_check.keys():
                            already_calculated = True
                    elif args.r_window == 1:
                        already_calculated = True

                if args.overwrite or not already_calculated:
                    corrs = []  # list by sequence (one element per sequence)
                    try:
                        for seq_idx, sequence in enumerate(dset.sequences):
                            corrs.append(evaluate_results(
                                sequence,
                                time_average=dset.time_averages, ch_idx=ch_idx,
                                window_size=args.r_window, percentile=args.pct))
                    except AttributeError:
                        continue
                    else:
                        print 'Done calculating correlations'

                    if os.path.exists(corrs_path):
                        with open(corrs_path, 'rb') as rp:
                            corrs_dict = pickle.load(rp)
                        if not isinstance(corrs_dict, dict):
                            corrs_dict = {1: corrs_dict}
                    else:
                        corrs_dict = {}

                    corrs_dict[args.r_window] = corrs
                    with open(corrs_path, 'wb') as wp:
                        pickle.dump(corrs_dict, wp)
                else:
                    print 'Stored correlations found\n'

                if args.pdf:
                    with open(os.path.join(dset.savedir,
                                           "corrs.pkl"), 'rb') as rp:
                        corrs = pickle.load(rp)

                    for seq_idx, sequence in enumerate(dset.sequences):
                        if isinstance(corrs, dict):
                            corrs = corrs[args.r_window][seq_idx]
                        else:
                            corrs = corrs[seq_idx]

                        patch_intervals, absCutOff = identify_patch_regions(
                            corrs, args.buffer, args.threshold, args.z_window)
                        print_artifact_analysis(
                            corrs, patch_intervals,
                            dset.savedir + '/Sequence_{}'.format(seq_idx) + \
                                '_patch_result.pdf',
                            args.threshold)
                continue

            seq_intervals = []
            seq_displacements = []
            seq_problem_frames = []
            for seq_idx, sequence in enumerate(dset.sequences):
                print 'Sequence {}'.format(str(seq_idx))
                print '==========='
                # a list by plane
                if args.load_corrs:
                    with open(os.path.join(dset.savedir,
                                           "corrs.pkl"), 'rb') as rp:
                        corrs = pickle.load(rp)

                    if isinstance(corrs, dict):
                        corrs = corrs[args.r_window][seq_idx]
                    else:
                        corrs = corrs[seq_idx]

                    print 'Done loading correlations'
                else:
                    corrs = evaluate_results(
                        sequence,
                        time_average=dset.time_averages, ch_idx=ch_idx,
                        window_size=args.r_window)
                    print 'Done calculating correlations'\

                patch_intervals, absCutOff = identify_patch_regions(
                    corrs, args.buffer, args.threshold, args.z_window)

                if args.pdf:
                    print_artifact_analysis(
                        corrs, patch_intervals,
                        dset.savedir + '/Sequence_{}'.format(seq_idx) + '_patch_result.pdf',
                        args.threshold)

                next_sequence_flag = False
                if args.dynamic_plot:
                    if not __DISPLAY__:
                        raise Exception(
                            "Dynamic mode requires DISPLAY to be set")
                    while True:
                        plot_artifact_analysis(
                            corrs, patch_intervals, absCutOff, args.threshold, 
                            dset.savedir + '\n Sequence {}'.format(seq_idx))
                        plt.show(block=False)

                        command = raw_input(
                            "Input [n]ew threshold (or [l]ist), [c]ontinue, or [q]uit " +
                            "sequence:")

                        if command == 'n':
                            try:
                                new_thresh = float(
                                    raw_input('Input threshold level: '))
                            except ValueError:
                                print 'Invalid value, must enter float'
                                continue
                            args.threshold = new_thresh
                            patch_intervals, absCutOff = identify_patch_regions(
                                corrs, args.buffer, new_thresh, args.z_window)
                        elif command == 'l':
                            prompt = 'Previous values: {}\n'.format(args.threshold) + \
                                     'Input new values for each plane: '
                            new_thresh = raw_input(prompt)
                            args.threshold = [float(x) for x in new_thresh.split()]
                            try:
                                assert(len(args.threshold) == len(corrs))
                            except AssertionError:
                                print '{} values given but there are {} planes'.format(
                                      len(args.threshold), len(corrs))
                                continue
                            patch_intervals, absCutOff = identify_patch_regions(
                                corrs, args.buffer, args.threshold, args.z_window)
                        elif command == 'c':
                            break
                        elif command == 'q':
                            next_sequence_flag = True
                            break
                        else:
                            print 'Unknown command, choose n, c, or q'

                # If there's nothing to patch, move along
                if not any([len(x) for x in patch_intervals]) or \
                        next_sequence_flag:
                    seq_intervals.append([[]])
                    seq_displacements.append([[]])
                    seq_problem_frames.append([[]])
                    continue

                patch_models = define_patch_models(
                    dset, n_processes=args.processes)

                intervals_to_patch = []
                new_displacements = []
                problem_frames = []

                for plane, [starts, stops] in enumerate(patch_intervals):

                    plane_intervals_to_patch, plane_new_displacements, \
                        plane_problem_frames = patch_plane_intervals(
                            starts, stops, sequence, plane, args.buffer,
                            patch_models, ch_idx, verbose=True)

                    intervals_to_patch.append(plane_intervals_to_patch)
                    new_displacements.append(plane_new_displacements)
                    problem_frames.append(plane_problem_frames)
                seq_intervals.append(intervals_to_patch)
                seq_displacements.append(new_displacements)
                seq_problem_frames.append(problem_frames)
            if args.batch_corr:
                continue

            if any([len(x) for seq in seq_problem_frames for x in seq]):
                there_are_problem_frames = True
                print '********** PROBLEM FRAMES ***********'
            else:
                there_are_problem_frames = False

            orig_displacements = []
            something_corrected = False
            for seq_idx, seq, plane_intervals, plane_disps, plane_problem_frames \
                    in it.izip(it.count(), dset.sequences, seq_intervals,
                               seq_displacements, seq_problem_frames):

                corrected = get_mc_seq(seq)
                print 'Sequence {}:'.format(str(seq_idx))
                print '-' * 10
                plane_something_corrected = False
                for plane_idx, intervals, displacements, problem_frames in \
                        it.izip(
                            it.count(), plane_intervals, plane_disps,
                            plane_problem_frames):

                    for interval, d in it.izip(intervals, displacements):
                        corrected.displacements[interval, plane_idx] = d
                        plane_something_corrected = True
                        something_corrected = True

                    s = ''
                    for (start, stop) in problem_frames:
                        for f in np.arange(start, stop):
                            s += str(f) + ' '
                    print 'Plane {}: '.format(str(plane_idx)) + s[:-1]

                if plane_something_corrected:
                    orig_displacements.append(corrected.displacements.copy())
                    max_disp = max_displacements(corrected.displacements)
                    corrected._frame_shape_zyx = tuple(
                        corrected._base.shape[1:4] + max_disp)

            if something_corrected:
                print "Making displacements non-negative"
                dset.save()
                with suppress_stdout_stderr():
                    make_nonnegative(os.path.join(dset.savedir, 'dataset.pkl'),
                                     trim_criterion=args.trim_criterion,
                                     recalc_averages=False)
                new_displacements = []
                for seq in dset.sequences:
                    mc_seq = get_mc_seq(seq)
                    new_displacements.append(mc_seq.displacements)

                mnn_shift = np.median([np.median(np.median(np.median(
                    new_disp - disp, 0), 0), 0) for new_disp, disp in it.izip(
                        new_displacements, orig_displacements)], 0)
                if np.any(mnn_shift):
                    new_seqs = []
                    for seq in dset.sequences:
                        new_seqs.append(
                            seq[:, :, mnn_shift[0]:, mnn_shift[1]:])
                    dset.sequences = new_seqs
                    dset.save()

            if args.mask_failures and there_are_problem_frames:
                print "Masking remaining bad frames"
                imset = ImagingDataset.load(dset.savedir)
                new_sequences = []
                for sequence, plane_problem_frames in it.izip(
                        imset.sequences, seq_problem_frames):
                    seq_frames_to_mask = []  # list by plane
                    for plane_idx, problem_frames in it.izip(
                            it.count(), plane_problem_frames):

                        frames_to_mask = []
                        for (start, stop) in problem_frames:
                            frames_to_mask.extend(
                                np.arange(start, stop).tolist())
                        seq_frames_to_mask.append(frames_to_mask)
                    mask = [(f, [idx], None, None) for idx, f in
                            enumerate(seq_frames_to_mask)]
                    new_sequences.append(sequence.mask(mask))
                imset.sequences = new_sequences
                something_corrected = True
                imset.save()

            if something_corrected:
                # if this sima folder is store in the database log the threshold
                # used
                try:
                    expts = dbExperimentSet.FetchTrials(
                        tSeriesDirectory=os.path.dirname(dset.savedir))
                except:
                    pass
                else:
                    if len(expts) == 1:
                        expts[0].patch_threshold = args.threshold
                        expts[0].save(store=True)
                imset = ImagingDataset.load(dset.savedir)
                if not(args.skip_avg_recalc):
                    print "Re-calcing time averages"
                    recalc_averages(imset)

            if args.dynamic_plot or something_corrected:
                print "Saving log figure"
                logFigName1 = 'patchMotionLogFig_' + \
                    datetime.datetime.now().strftime("%m-%d-%Y_%H_%M") + \
                    '.pdf'
                logFigName1 = os.path.join(dset.savedir, logFigName1)
                fig1 = plot_artifact_analysis(
                    corrs, patch_intervals, absCutOff, args.threshold, 
                    dset.savedir + '\n Sequence {}'.format(seq_idx))
                fig1.savefig(logFigName1, format='pdf')

if __name__ == '__main__':
    main(sys.argv[1:])
