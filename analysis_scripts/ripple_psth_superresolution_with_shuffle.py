import matplotlib.pyplot as plt
import seaborn as sns

from lab_repo.classes.dbclasses import dbExperimentSet
from lab_repo.classes import ExperimentGroup

import lab_repo.misc.lfp_helpers as lfp

import numpy as np

import pudb
import cPickle as pkl
from scipy.ndimage.filters import gaussian_filter1d

def expt_responses(ripples, plane_idxs, imdata, ft, pre, post, im_period, super_period, Fs=1250.):

    psth = np.full((len(ripples), imdata.shape[0], pre + post + 1), np.nan)

    n_planes = len(plane_idxs)

    for ri, ripple in enumerate(ripples):

        closest_idx = lfp.closest_idx(ft, [ripple])
        closest_plane = np.mod(closest_idx, n_planes)

        im_frame_idx = int(closest_idx / n_planes)
        # slice for imdata: relative frame indices centered on closest imaging frame

        # difference between ripple and plane/frame onset in s
        time_diff = (ripple - ft[closest_idx]) / Fs

        for plane in xrange(n_planes):

            plane_diff = time_diff + (plane - closest_plane) * im_period / n_planes
            plane_diffs = np.hstack([np.arange(plane_diff, -1*PRE, -1*im_period)[::-1],
                                     np.arange(plane_diff, POST, im_period)[1:]])

            plane_psth_idx = np.floor(plane_diffs / super_period).astype(int) + pre
            plane_im_idx = np.floor(plane_diffs / im_period).astype(int) + im_frame_idx
            assert(np.min(np.diff(plane_im_idx)) == 1)

            # Only use part of window falling within imaging session
            valid_indices = np.where((plane_im_idx >= 0) & (plane_im_idx < imdata.shape[1]))[0]
            plane_psth_idx = plane_psth_idx[valid_indices]
            plane_im_idx = plane_im_idx[valid_indices]

            psth_i, psth_j = np.meshgrid(plane_idxs[plane], plane_psth_idx)
            imdata_i, imdata_j = np.meshgrid(plane_idxs[plane], plane_im_idx)

            psth[ri, psth_i, psth_j] = imdata[imdata_i, imdata_j]

    return psth



exptSet = dbExperimentSet(project_name='sebi')

path = '/home/sebi/grps/dend_grp.json'
grp = ExperimentGroup.from_json(path, exptSet)
# Exclude experiments due to frame rate differences
grp = ExperimentGroup([e for e in grp if (e.parent.mouse_name != 'svrExp19_3') and e.condition == 1])

UP_SAMPLE = 4.

zscore = True

n_shuffles = 1000

im_period = grp[0].frame_period()
super_period = im_period / UP_SAMPLE

PRE = 1.5
POST = 2
WIN = 0.5

pre = int(PRE / super_period) + 1
post = int(POST / super_period) + 1

calc_window = int(WIN / super_period)
calc_frames = int(WIN / im_period)

pre_frames = int(PRE / im_period) + 1
post_frames = int(POST / im_period) + 2

Fs = 1250.
n_planes = 3

roi_filter = lambda x: '_' in x.label
# roi_filter = lambda x: '_' not in x.label

psths = []
shuffle_vals = [[] for i in xrange(n_shuffles)]

for expt in grp:

    imdata = expt.imagingData(dFOverF=None, label='mergedmerged', roi_filter=roi_filter)[:,:,0]
    n_frames = imdata.shape[1]

    running = expt.runningIntervals(returnBoolList=True)[0]

    # Filter ripples that occur beyond end time and those that occur during running
    ripples = expt.ripple_times()
    ripples = [x for x in ripples if (x / Fs) <= expt.duration().seconds]

    ft = expt.lfp_frames()
    ripple_imaging_frames = lfp.closest_idx(ft, ripples)
    ripple_imaging_frames = [int(r/n_planes) for r in ripple_imaging_frames]

    ripples = [x for x,y in zip(ripples, ripple_imaging_frames) if ~np.any(running[max(y-pre_frames, 0):min(y+post_frames, n_frames)])]

    if zscore:
        means = np.nanmean(imdata[:, ~running], axis=1, keepdims=True)
        stds = np.nanstd(imdata[:, ~running], axis=1, keepdims=True)
        imdata = (imdata - means) / stds

    rois = expt.rois(label='mergedmerged', roi_filter=roi_filter)
    roi_planes = [r.polygons[0].exterior.coords[0][2] for r in rois]
    plane_idxs = []
    for i in xrange(n_planes):
        plane_idxs.append(np.where(np.array([plane == i for plane in roi_planes]))[0])

    psth = expt_responses(ripples, plane_idxs, imdata, ft, pre, post, im_period, super_period, Fs)

    # For "all" and not "by cell", just extend list with psth, don't take mean
    psths.append(np.nanmean(psth, axis=0))

    # Now do the same with random times
    # First find all valid frame indices to subselect from
    valid_frames = range(n_frames)
    exclude_ripple_frames = []
    for x in ripple_imaging_frames:
        exclude_ripple_frames.extend(range(x-calc_frames, x+calc_frames))
    # Exclude based on running (same as above)
    valid_frames = [x for x in valid_frames if ~np.any(running[max(x-pre_frames, 0):min(x+post_frames, n_frames)])]
    # Exclude actual ripple times
    valid_frames = [x for x in valid_frames if x not in exclude_ripple_frames]
    # Now use this to find valid lfp samples
    valid_idx = []
    for x in valid_frames:
        # LFP samples associated with this valid imaging frame range from those a half plane prior to first plane acquisition
        # to those a half plane past last plane acquisition
        try:
            start = ft[x] - int(0.5 * (ft[x] - ft[x-1]))
        except KeyError:
            start = 0

        try:
            stop = ft[x + n_planes - 1] + int(0.5 * (ft[x + n_planes] - ft[x + n_planes - 1]))
        except KeyError:
            stop = ft[-1]

        valid_idx.extend(range(start, stop))

    # Now sample same number of pseudo-ripple times, calculate psth, calc diff and store per ROI for each shuffle
    n_ripples = len(ripples)
    for i in xrange(n_shuffles):

        rand_idx = np.random.choice(valid_idx, (n_ripples,))
        shuffle_psth = expt_responses(rand_idx, plane_idxs, imdata, ft, pre, post, im_period, super_period, Fs)

        shuffle_psth = np.nanmean(shuffle_psth, axis=0)
        for p in shuffle_psth:
            shuffle_vals[i].append(np.nanmean(p[pre:pre+calc_window]) - np.nanmean(p[pre-calc_window:pre]))



psths = np.vstack(psths)
with open('/home/sebi/psth_dend_all_4_by_cell.pkl', 'wb') as fw:
    pkl.dump(psths, fw)

with open('/home/sebi/psth_dend_all_4_by_cell_shuffles.pkl', 'wb') as fw:
    pkl.dump(shuffle_vals, fw)

