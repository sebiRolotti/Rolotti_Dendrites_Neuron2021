import numpy as np
import pandas as pd
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean
from scipy.ndimage.filters import gaussian_filter1d

import lab_repo.classes.exceptions as exc

from pycircstat.descriptive import _complex_mean

def calcCentroids(data, pfs, returnAll=False, return_pfs=False):
    """
    Input:
        data (df/f by position)
        pfs  (not-normalized place field output from identifyPlaceFields)
        returnAll
            -False --> for cells with >1 place field, take the one with bigger
                peak
            -True  --> return centroids for each place field ordered by peak
                df/f in the pf
        return_pfs : If True, returns pfs matching centroids

    Output:
    centroids: nROIs length list, each element is either an empty list or a
    list containing the bin (not rounded) of the centroid

    """

    # if np.all(np.array(list(flatten(pfs)))<1):
    #    raise TypeError('Invalid argument, must pass in non-normalized pfs')

    centroids = [[] for x in range(len(pfs))]
    pfs_out = [[] for x in range(len(pfs))]
    # peaks_out  = [[] for x in range(len(pfs))]
    for pfIdx, roi, pfList in it.izip(it.count(), data, pfs):
        if len(pfList) > 0:
            peaks = []
            roi_centroids = []
            for pf in pfList:
                if pf[0] < pf[1]:
                    pf_data = roi[pf[0]:pf[1] + 1]
                    peaks.append(np.amax(pf_data))
                    roi_centroids.append(pf[0] + np.sum(
                        pf_data * np.arange(len(pf_data))) / np.sum(pf_data))
                else:
                    pf_data = np.hstack([roi[pf[0]:], roi[:pf[1] + 1]])
                    peaks.append(np.amax(pf_data))
                    roi_centroids.append((pf[0] + np.sum(
                        pf_data * np.arange(len(pf_data))) / np.sum(pf_data))
                        % data.shape[1])
            # sort the pf peaks in descending order
            order = np.argsort(peaks)[::-1]
            if returnAll:
                centroids[pfIdx] = [roi_centroids[x] for x in order]
                pfs_out[pfIdx] = [pfList[x] for x in order]
                # peaks_out[pfIdx] = [peaks[x] for x in order]
            else:
                centroids[pfIdx] = [roi_centroids[order[0]]]
                pfs_out[pfIdx] = [pfList[order[0]]]
                # peaks_out[pfIdx] = [peaks[order[0]]]
            assert not np.any(np.isnan(centroids[pfIdx]))
    if return_pfs:
        return centroids, pfs_out  # , peaks_out
    return centroids


def calc_activity_centroids(exptGrp, roi_filter=None):
    """
    Output:
        list = (nROIs,)
    """

    bins = 2 * np.pi * np.arange(0, 1, 1. / exptGrp.args['nPositionBins'])

    result = {}
    for expt in exptGrp:
        expt_result = []
        for tuning_curve in exptGrp.data_raw(roi_filter=roi_filter)[expt]:
            finite_idxs = np.where(np.isfinite(tuning_curve))[0]
            p = _complex_mean(bins[finite_idxs], tuning_curve[finite_idxs])
            expt_result.append(p)
        result[expt] = expt_result
    return result


def get_heatmap(sig, running, abs_pos, n_bins, fp=1,
                normalized=False):
    n_laps = int(abs_pos[-1]) + 1

    heatmap = np.zeros((n_laps, n_bins))
    counts = np.zeros((n_laps, n_bins))

    for i in xrange(len(sig)):

        if not running[i]:
            continue
        if np.isnan(sig[i]):
            continue

        lap = int(abs_pos[i])
        posbin = int(np.mod(abs_pos[i] * n_bins, n_bins))

        heatmap[lap, posbin] += sig[i]
        counts[lap, posbin] += fp

    heatmap /= counts

    # Normalized
    if normalized and (np.nanmax(heatmap) > 0):
        heatmap = heatmap / np.nanmax(heatmap)

    return heatmap

def sensitivity(
        exptGrp, roi_filter=None, includeFrames='running_only'):
    """
    Fraction of complete forward passes through the place field that trigger
    a significant calcium transient

    returns a place_cell_df
    """

    pc_filter = exptGrp.pcs_filter(roi_filter=roi_filter)
    pfs_n = exptGrp.pfs(roi_filter=pc_filter)

    data_list = []
    for expt in exptGrp:
        tid = expt.trial_id
        mouse_name = expt.parent.mouse_name
        fov = expt.get('uniqueLocationKey')

        pfs = pfs_n[expt]
        if includeFrames == 'running_only':
            imaging_label = exptGrp.args['imaging_label']
            if imaging_label is None:
                imaging_label = expt.most_recent_key(
                    channel=exptGrp.args['channel'])
        for trial_idx, trial in enumerate(expt.findall('trial')):
            position = ba.absolutePosition(trial, imageSync=True)
            running_kwargs = {'min_duration': 1.0, 'min_mean_speed': 0,
                      'end_padding': 0, 'stationary_tolerance': 0.5,
                      'min_peak_speed': 5, 'direction': 'forward'}

            running_frames = expt.runningIntervals(returnBoolList=True, **running_kwargs)[0]
            rois = expt.rois(
                roi_filter=pc_filter, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'])

            trans_dict = expt.transientsData(roi_filter=pc_filter, label=exptGrp.args['imaging_label'])
            trans = np.zeros((len(rois), len(position)))
            for ri, rtrans in enumerate(trans_dict):
                for start, stop in zip(rtrans['start_indices'][0], rtrans['end_indices'][0]):
                    trans[ri, start] += 1
            spikes = trans

            assert len(rois) == len(pfs)
            assert len(rois) == len(spikes)

            for roi_spikes, roi_pfs, roi in it.izip(spikes, pfs, rois):

                pf_idx = np.zeros((100,))
                for roi_pf in roi_pfs:

                    if roi_pf[0] < roi_pf[1]:
                        pf_idx[roi_pf[0]:roi_pf[1] + 1] = 1.
                    else:
                        pf_idx[roi_pf[0]:] = 1.
                        pf_idx[:roi_pf[1]] = 1.
                pf_idx = pf_idx.astype(bool)

                heatmap = get_heatmap(roi_spikes, running_frames, position, 100)

                passes = heatmap.shape[0]
                hits = 0
                for lap in heatmap:
                    if np.nansum(lap[pf_idx]) > 0:
                        hits += 1

                data_dict = {'expt_id': tid,
                             'mouse': mouse_name,
                             'FOV': fov,
                             'roi': roi.label,
                             'value': float(hits) / passes}
                data_list.append(data_dict)

    return pd.DataFrame(data_list, columns=['expt_id', 'mouse', 'FOV', 'roi', 'value'])

def specificity(
        exptGrp, roi_filter=None, includeFrames='running_only', ignore_start=False):
    """
    Fraction of transient onsets that occur in a place field
    """
    pc_filter = exptGrp.pcs_filter(roi_filter=roi_filter)
    pfs_n = exptGrp.pfs(roi_filter=pc_filter)

    data_list = []
    for expt in exptGrp:
        tid = expt.trial_id
        mouse_name = expt.parent.mouse_name
        fov = expt.get('uniqueLocationKey')

        pfs = pfs_n[expt]
        if includeFrames == 'running_only':
            imaging_label = exptGrp.args['imaging_label']
            if imaging_label is None:
                imaging_label = expt.most_recent_key(
                    channel=exptGrp.args['channel'])
        for trial_idx, trial in enumerate(expt.findall('trial')):
            abspos = ba.absolutePosition(trial, imageSync=True)
            position = (abspos % 1) * 100
            running_kwargs = {'min_duration': 1.0, 'min_mean_speed': 0,
                      'end_padding': 0, 'stationary_tolerance': 0.5,
                      'min_peak_speed': 5, 'direction': 'forward'}

            running_frames = expt.runningIntervals(returnBoolList=True, **running_kwargs)[0]
            rois = expt.rois(
                roi_filter=pc_filter, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'])

            trans_dict = expt.transientsData(roi_filter=pc_filter, label=exptGrp.args['imaging_label'])
            trans = np.zeros((len(rois), len(position)))
            for ri, rtrans in enumerate(trans_dict):
                for start, stop in zip(rtrans['start_indices'][0], rtrans['end_indices'][0]):
                    trans[ri, start] += 1
            spikes = trans

            laps = abspos.astype(int)

            if ignore_start:
                interval = np.where(running_frames & (laps >= 6))[0]
            else:
                interval = np.where(running_frames)[0]

            

            assert len(rois) == len(pfs)
            assert len(rois) == len(spikes)

            position = position[interval]
            spikes = spikes[:, interval]

            for roi_spikes, roi_pfs, roi in it.izip(spikes, pfs, rois):

                pf_bins = []
                for roi_pf in roi_pfs:

                    pf_idx = np.zeros(roi_spikes.shape)

                    if roi_pf[0] < roi_pf[1]:
                        pf_idx = (position > roi_pf[0]) & (position <= roi_pf[1])
                    else:
                        pf_idx = (position > roi_pf[0]) | (position <= roi_pf[1])

                    pf_bins.append(pf_idx)
                pf_idx = reduce(np.logical_or, pf_bins)

                n_pf_spikes = np.nansum(roi_spikes[pf_idx])
                n_spikes = np.nansum(roi_spikes)

                data_dict = {'expt_id': tid,
                             'mouse': mouse_name,
                             'FOV': fov,
                             'roi': roi.label,
                             'value': n_pf_spikes / n_spikes}
                data_list.append(data_dict)
    return pd.DataFrame(data_list, columns=['expt_id', 'mouse', 'FOV', 'roi', 'value'])


def within_session_correlation(exptGrp, roi_filter=None, n_splits=100):

    pc_filter = exptGrp.pcs_filter(roi_filter=roi_filter)

    data_list = []
    for expt in exptGrp:
        tid = expt.trial_id
        mouse_name = expt.parent.mouse_name
        fov = expt.get('uniqueLocationKey')

        position = ba.absolutePosition(expt.find('trial'), imageSync=True)
        running_kwargs = {'min_duration': 1.0, 'min_mean_speed': 0,
                      'end_padding': 0, 'stationary_tolerance': 0.5,
                      'min_peak_speed': 5, 'direction': 'forward'}

        running_frames = expt.runningIntervals(returnBoolList=True, **running_kwargs)[0]
        rois = expt.rois(
            roi_filter=pc_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])

        trans_dict = expt.transientsData(roi_filter=pc_filter, label=exptGrp.args['imaging_label'])
        trans = np.zeros((len(rois), len(position)))
        for ri, rtrans in enumerate(trans_dict):
            for start, stop in zip(rtrans['start_indices'][0], rtrans['end_indices'][0]):
                trans[ri, start] += 1
        spikes = trans

        for roi, roi_spikes in zip(rois, spikes):

            heatmap = get_heatmap(roi_spikes, running_frames, position, 100)
            n_laps = heatmap.shape[0]
            half_laps = int(n_laps / 2)

            # Generate random splits
            corrs = []
            if n_splits == 1:
                tc1 = np.nan_to_num(np.nanmean(heatmap[::2, :], axis=0))
                tc2 = np.nan_to_num(np.nanmean(heatmap[1::2, :], axis=0))

                tc1 = gaussian_filter1d(tc1, 3, mode='wrap')
                tc2 = gaussian_filter1d(tc2, 3, mode='wrap')

                corrs.append(pearsonr(tc1, tc2)[0])

            else:
                for i in xrange(n_splits):
                    permutation = np.random.permutation(n_laps)

                    tc1 = np.nan_to_num(np.nanmean(heatmap[permutation[:half_laps], :], axis=0))
                    tc2 = np.nan_to_num(np.nanmean(heatmap[permutation[half_laps:], :], axis=0))

                    tc1 = gaussian_filter1d(tc1, 3, mode='wrap')
                    tc2 = gaussian_filter1d(tc2, 3, mode='wrap')

                    corrs.append(pearsonr(tc1, tc2)[0])

            data_dict = {'expt_id': tid,
                         'mouse': mouse_name,
                         'FOV': fov,
                         'roi': roi.label,
                         'value': np.nanmean(corrs),
                         'median_value': np.nanmedian(corrs)}
            data_list.append(data_dict)

    return pd.DataFrame(data_list, columns=['expt_id', 'mouse', 'FOV', 'roi', 'value', 'median_value'])

