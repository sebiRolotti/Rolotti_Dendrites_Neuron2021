import matplotlib.pyplot as plt
import seaborn as sns

from lab.classes.dbclasses import dbExperimentSet
from lab.classes import ExperimentGroup

import lab.misc.lfp_helpers as lfp
import lab.analysis.imaging_analysis as ia
import lab.analysis.dendrites_analysis as da

import numpy as np

import pudb
import cPickle as pkl
from scipy.ndimage.filters import gaussian_filter1d
from lab.misc import lfp_helpers


def psth(exptGrp, roi_filter=None, pre=2, post=4, **interval_kwargs):

    if not hasattr(exptGrp, '_cells'):
        exptGrp._cells = da.CellSet(exptGrp)
    cell_set = exptGrp._cells.filter(roi_filter)

    im_period = exptGrp[0].frame_period()
    pre = int(pre / im_period)
    post = int(post / im_period)

    psths = []

    for cell in cell_set:

        interval = da.intervals(cell, cell.parent_experiment,
                             pf=None, **interval_kwargs)

        n_frames = cell.parent_experiment.num_frames()

        soma_starts = cell.soma.signal()['start_indices']
        amps = cell.soma.signal()['max_amplitudes']

        include_idx = [i for i, x in enumerate(soma_starts) if interval[x] and (x > pre) and (x < n_frames - post)]
        soma_starts = [soma_starts[i] for i in include_idx]
        amps = [amps[i] for i in include_idx]

        dendrites = cell.dendrites

        cell_psths = np.zeros((len(soma_starts), len(dendrites), pre + post + 1))

        for di, dendrite in enumerate(dendrites):

            dend_sigs = dendrite.imaging_data()

            for si, (a, s) in enumerate(zip(amps, soma_starts)):

                cell_psths[si, di, :] = dend_sigs[s - pre: s + post + 1] / a

        psths.extend(np.nanmean(cell_psths, axis=0))

    T = np.arange(-pre, post + 1) * im_period

    return T, psths


exptSet = dbExperimentSet(project_name='sebi')

path = '/home/sebi/grps/dend_grp.json'
grp = ExperimentGroup.from_json(path, exptSet)
grp = ExperimentGroup([e for e in grp if (e.parent.mouse_name != 'svrExp19_3') and e.condition == 1])


T, psths = psth(grp, running_only=True)
psths = np.vstack(psths)

with open('/home/sebi/psth_running.pkl', 'wb') as fw:
    pkl.dump([T, psths], fw)

T, psths = psth(grp, non_running_only=True, non_ripple_only=True)
psths = np.vstack(psths)

with open('/home/sebi/psth_nonrunning.pkl', 'wb') as fw:
    pkl.dump([T, psths], fw)

T, psths = psth(grp, non_running_only=True, ripple_only=True)
psths = np.vstack(psths)

with open('/home/sebi/psth_swr.pkl', 'wb') as fw:
    pkl.dump([T, psths], fw)
