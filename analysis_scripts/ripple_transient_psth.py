import matplotlib.pyplot as plt
import seaborn as sns

from lab_repo.classes.dbclasses import dbExperimentSet
from lab_repo.classes import ExperimentGroup

import lab_repo.misc.lfp_helpers as lfp

import numpy as np

import pudb
import cPickle as pkl
from scipy.ndimage.filters import gaussian_filter1d


def ripple_responders(ripple_start, ripple_stop,
    transient_times, pre_tolerance, post_tolerance):

    ripple_range = range(ripple_start - pre_tolerance, ripple_stop + post_tolerance)
    
    idx = []
    for i, trans_time in enumerate(transient_times):
        for x in ripple_range:
            if x in trans_time:
                idx.append(i)
                break

    return np.array(idx, dtype=int)

exptSet = dbExperimentSet(project_name='sebi')

path = '/home/sebi/grps/dend_grp.json'
grp = ExperimentGroup.from_json(path, exptSet)
grp = ExperimentGroup([e for e in grp if (e.parent.mouse_name != 'svrExp19_3') and e.condition == 1])

## FOR DEBUGGING
# grp = ExperimentGroup([grp[0]])
## FOR DEBUGGING

UP_SAMPLE = 1.

zscore = True

DEND = True

im_period = grp[0].frame_period()
super_period = im_period / UP_SAMPLE

PRE = 1.0
POST = 3

pre = int(PRE / super_period) + 1
post = int(POST / super_period) + 1


ripple_pre = int(0.3 / im_period)
ripple_post = int(0.3 / im_period)


Fs = 1250.
n_planes = 3

soma_filter = lambda x: '_' not in x.label
if DEND:
    roi_filter = lambda x: '_' in x.label
else:
    roi_filter = soma_filter

psths = []

for expt in grp:


    imdata = expt.imagingData(dFOverF=None, label='mergedmerged', roi_filter=roi_filter)[:,:,0]

    soma_trans = expt.transientsData(label='mergedmerged', roi_filter=soma_filter)
    soma_starts = [soma_trans['start_indices'][i][0] for i in xrange(len(soma_trans))]

    # imdata = expt.spikes(roi_filter=roi_filter)
    # imdata = np.where(imdata > 0, 1, 0)
    ripples = expt.ripple_times()
    stop_times = expt.ripple_times(trigger='tend')

    ft = expt.lfp_frames()
    ripple_starts = lfp.closest_idx(ft, ripples)
    ripple_stops = lfp.closest_idx(ft, stop_times)

    ripple_starts = [int(r / n_planes) for r in ripple_starts]
    ripple_stops = [int(r / n_planes) for r in ripple_stops]

    running = expt.runningIntervals(returnBoolList=True)[0]

    if zscore:
        means = np.nanmean(imdata[:, ~running], axis=1, keepdims=True)
        stds = np.nanstd(imdata[:, ~running], axis=1, keepdims=True)
        imdata = (imdata - means) / stds

    ft = expt.lfp_frames()

    rois = expt.rois(label='mergedmerged', roi_filter=roi_filter)
    roi_planes = [r.polygons[0].exterior.coords[0][2] for r in rois]


    if DEND:
        soma_rois = expt.rois(label='mergedmerged', roi_filter=soma_filter)
        soma_labels = [x.label for x in soma_rois]

        trans_starts = []
        for roi in rois:
            try:
                soma_idx = soma_labels.index(roi.label.split('_')[0])
            except ValueError:
                trans_starts.append([])
            else:
                trans_starts.append(soma_starts[soma_idx])
    else:
        trans_starts = soma_starts

    plane_idxs = []
    for i in xrange(n_planes):
        plane_idxs.append(np.where(np.array([plane == i for plane in roi_planes]))[0])

    psth = np.full((len(ripples), len(rois), pre + post + 1), np.nan)

    for ri, ripple in enumerate(ripples):

        if (ripple / Fs) > expt.duration().seconds:
            continue

        closest_idx = lfp.closest_idx(ft, [ripple])
        closest_plane = np.mod(closest_idx, n_planes)
        # slice for imdata: relative frame indices centered on closest imaging frame

        # Ensure no running happens within window (+ up to a frame)
        im_frame_idx = int(closest_idx / n_planes)
        pre_imaging = int(PRE / im_period) + 1
        post_imaging = int(POST / im_period) + 2
        im_window_idx = np.arange(im_frame_idx - pre_imaging, im_frame_idx + post_imaging)

        # Exclude ripples with any running in window
        valid_indices = np.where((im_window_idx >= 0) & (im_window_idx < imdata.shape[1]))[0]
        if np.any(running[im_window_idx[valid_indices]]):
            continue

        # Only include rois with transient near this ripple
        # Will have to make new "soma trans" that is n_dends long
        roi_idx = ripple_responders(ripple_starts[ri], ripple_stops[ri], trans_starts, ripple_pre, ripple_post)
        bad_idx = np.array([i for i in xrange(len(rois)) if i not in roi_idx])

        # difference between ripple and plane/frame onset in s
        time_diff = (ripple - ft[closest_idx]) / Fs

        # difference between ripple and all relevant imaging frames in this plane
        # diffs = relative_im_times + time_diff

        # psth_idx = (diffs / super_period).astype(int) + pre

        # psth = np.full((len(rois), pre + post + 1), np.nan)

        for plane in xrange(n_planes):

            plane_diff = time_diff + (plane - closest_plane) * im_period / n_planes
            # plane_diffs = relative_im_times + plane_diff
            plane_diffs = np.hstack([np.arange(plane_diff, -1*PRE, -1*im_period)[::-1],
                                     np.arange(plane_diff, POST, im_period)[1:]])

            plane_psth_idx = np.floor(plane_diffs / super_period).astype(int) + pre
            plane_im_idx = np.floor(plane_diffs / im_period).astype(int) + im_frame_idx
            assert(np.min(np.diff(plane_im_idx)) == 1)
            assert(np.max(np.diff(plane_im_idx)) == 1)

            # Only use part of window falling within imaging session
            valid_indices = np.where((plane_im_idx >= 0) & (plane_im_idx < imdata.shape[1]))[0]
            plane_psth_idx = plane_psth_idx[valid_indices]
            plane_im_idx = plane_im_idx[valid_indices]

            psth_i, psth_j = np.meshgrid(plane_idxs[plane], plane_psth_idx)
            imdata_i, imdata_j = np.meshgrid(plane_idxs[plane], plane_im_idx)

            # psth[psth_i, psth_j] = imdata[imdata_i, imdata_j]
            psth[ri, psth_i, psth_j] = imdata[imdata_i, imdata_j]

        psth[ri, bad_idx, :] = np.nan

    psths.append(np.nanmean(psth, axis=0))

T = np.arange(-pre, post + 1) * super_period


# Note: We added 1 to pre and post above to account for convolution
# We thus trim first and last point of all signals after smoothing

psths = np.vstack(psths)
with open('/home/sebi/psth_dend_trans_1.pkl', 'wb') as fw:
    T, pkl.dump(psths, fw)
