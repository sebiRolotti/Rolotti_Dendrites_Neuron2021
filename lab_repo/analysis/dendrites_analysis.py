"""
Sparse GCaMP labeling analysis.

Nomenclature conventions:
-------------------------
    Soma: 'sf0' or 'deep0'
    1st order basal branch: '0_0'
    2nd order apical branch: '0_0_0'

"""

import pandas as pd

import numpy as np
import random
from scipy import stats, linalg
from scipy.signal import convolve2d, correlate

import lab.analysis.behavior_analysis as ba
import lab.analysis.ensemble_analysis as ea
from lab.plotting.plotting_helpers import prepare_dataframe
from lab.misc.progressbar import ProgressBar
from lab.misc.lfp_helpers import closest_idx
from lab.misc.stats import nanzscore

import os
import cPickle as pkl

import itertools as it


class Compartment(object):
    """Abstract compartment class underlying Dendrite and Soma."""

    def __init__(self, roi, parent_cell, data, pfs):
        """Initialize Compartment."""
        self.parent = parent_cell
        self.roi = roi
        self.label = roi.label

        self._data = data
        self._pfs = pfs

    def data(self):
        """Return tuning curve."""
        return self._data

    def pfs(self):
        return self._pfs

    def imaging_data(self):
        """Return dFoF."""
        if not hasattr(self, '_imaging_data'):
            idx = [x.label for x in self.parent.parent_experiment.rois()].index(self.roi.label)
            # idx = self.parent.parent_experiment.rois().index(self.roi)
            self._imaging_data = self.parent.parent_experiment.\
                imagingData(dFOverF='from_file')[idx, :, 0]
        return self._imaging_data

    def transients(self, threshold=95):
        """Return all transient info."""
        if not hasattr(self, '_transients'):
            idx = [x.label for x in self.parent.parent_experiment.rois()].index(self.roi.label)
            # idx = self.parent.parent_experiment.rois().index(self.roi)
            self._transients = self.parent.parent_experiment.\
                transientsData(threshold=threshold, label='mergedmerged')[idx, 0]

        return self._transients

    def spikes(self):
        """Return deconvolved spikes."""
        if not hasattr(self, '_spikes'):

            cell_spikes = self.parent.spikes(label='mergedmerged')
            idx = [x.label for x in self.parent.parent_experiment.rois()].index(self.roi.label)
            # idx = self.parent.parent_experiment.rois().index(self.roi)
            # Assumes first cycle
            self._spikes = cell_spikes[idx, 0]

        return self._spikes

    def signal(self):

        if self.parent._signal == 'transients':
            return self.transients()
        elif self.parent._signal == 'spikes':
            return self.spikes()

    def parent_compartment(self):

        parent_name = '_'.join(self.label.split('_')[:-1])
        if not parent_name:
            return None

        if '_' not in parent_name:
            return self.parent.soma

        dends = self.parent.dendrites
        dend_labels = [d.label for d in dends]

        try:
            dend_idx = dend_labels.index(parent_name)
        except ValueError:
            return None
        else:
            return dends[dend_idx]

    def children(self):

        dends = self.parent.dendrites

        child_name_len = len(self.label) + 2

        kids = [d for d in dends if d.label.startswith(self.label) and
                    (len(d.label) == child_name_len)]

        return kids

    def siblings(self):

        dends = self.parent.dendrites

        name_len = len(self.label)
        sib_name_prefix = '_'.join(self.label.split('_')[:-1])

        if not sib_name_prefix:
            return []

        sibs = [d for d in dends if d.label.startswith(sib_name_prefix) and
                (len(d.label) == name_len) and (d.label != self.label)]

        return sibs

    def neighbors(self):

        neighbor_list = []

        parent = self.parent_compartment()
        if parent:
            neighbor_list.append(parent)

        sibs = self.siblings()
        kids = self.children()

        neighbor_list.extend(sibs)
        neighbor_list.extend(kids)

        return neighbor_list


class Dendrite(Compartment):
    """Dendrites have an order, length, and distance from soma."""

    def __init__(self, roi, parent_cell, data, pfs):
        Compartment.__init__(self, roi, parent_cell, data, pfs)
        self._parse_tags()

    def __repr__(self):
        return '<Dendrite: label={label}, order={order}>'.format(
            label=self.label, order=self.order)

    def _parse_tags(self):
        tags = self.roi.tags

        o = [x for x in tags if x.startswith('o')]
        assert len(o) == 1
        self.order = int(o[0].split('_')[1])

        l = [x for x in tags if x.startswith('l')]
        assert len(l) == 1
        self.length = float(l[0].split('_')[1])

        d = [x for x in tags if x.startswith('d')]
        assert len(d) == 1
        self.distance_to_soma = float(d[0].split('_')[1])


class Soma(Compartment):
    def __init__(self, roi, parent_cell, data, pfs):
        Compartment.__init__(self, roi, parent_cell, data, pfs)

    def __repr__(self):
        return '<Soma: {label}>'.format(label=self.label)


class Cell:
    def __init__(self, rois, experiment, data, pfs):
        self.parent_experiment = experiment
        self.initialize_soma(rois, data, pfs)

        try:
            with open(os.path.join(experiment.sima_path(), 'dend_filter.pkl'), 'rb') as fp:
                filter_dict = pkl.load(fp)

            with open(experiment.transientsFilePath(), 'rb') as fp:
                trans_time = pkl.load(fp)['mergedmerged']['timestamp']

            bad_dendrites = filter_dict['roi_ids']

        except IOError:
            bad_dendrites = []

        bad_dendrites = []

        def dend_filter(x):
            return x.label not in bad_dendrites

        self.dendrites = []
        # rois = sorted(rois, key=lambda x: x.label)
        for roi, d, pf in zip(rois, data, pfs):
            if roi is self.soma.roi:
                continue
            elif dend_filter(roi):
                self.add_dendrite(roi, d, pf)
        # Cell is labeled by soma
        self.label = self.soma.roi.label

        # Pre-set signal to transients
        self._signal = 'transients'

    def __repr__(self):
        return '<Cell: mouseID={m}, experiment={e}, label={l}>'.format(
            m=self.parent_experiment.parent.get('mouseID'),
            e=self.parent_experiment.get('startTime'),
            l=self.label)

    def initialize_soma(self, rois, data, pfs):
        somata_idx = [i for i, x in enumerate(rois) if '_' not in x.label]
        if len(somata_idx) != 1:
            raise('Could not initialize soma. ' +
                  'One somatic ROI should be defined per cell')
        soma_idx = somata_idx[0]
        self.soma = Soma(rois[soma_idx], self, data[soma_idx], pfs[soma_idx])

    def add_dendrite(self, roi, data, pfs):
        self.dendrites.append(Dendrite(roi, self, data, pfs))

    def set_signal(self, signal):
        if signal == 'spikes':
            self._signal = 'spikes'
        else:
            self._signal = 'transients'

    def spikes(self):

        if not hasattr(self, '_spikes'):
            self._spikes = self.parent_experiment.spikes(trans_like=True)

        return self._spikes


class CellSet(list):
    """A list-like container for storing Cells
    """

    def __init__(self, experiment_group):
        if isinstance(experiment_group, list):
            cells = experiment_group
        else:
            data = experiment_group.data()
            pfs = experiment_group.pfs_n()
            cells = []
            for experiment in experiment_group:
                if 'mergedmerged' not in experiment.imaging_dataset().ROIs.keys():
                    continue

                edata = data[experiment]
                epfs = pfs[experiment]
                rois = experiment.rois(label='mergedmerged')
                for soma_label in [x.label for x in rois
                                   if '_' not in x.label]:

                    cell_idx = [i for i, x in enumerate(rois)
                                if x.label.split('_')[0].lower() == soma_label.lower()]

                    cell_rois = [rois[i] for i in cell_idx]
                    cell_pfs = [epfs[i] for i in cell_idx]
                    cell_data = [edata[i] for i in cell_idx]
                    cells.append(Cell(cell_rois, experiment, cell_data, cell_pfs))
        list.__init__(self, cells)


    def filter(self, filter_fn):
        if filter_fn is None:
            return self

        return CellSet([cell for cell in self if filter_fn(cell.soma.roi)])


def in_somatic_window(soma_start, dend_start,
                      pre_tolerance, post_tolerance):

    if dend_start >= soma_start:
        if dend_start - soma_start < post_tolerance:
            return True
    else:
        if soma_start - dend_start < pre_tolerance:
            return True

    return False


def prep_df(df, grp, include_columns=None):
    ''' Add cell info to dfs that only have an ROI col.
    '''
    if include_columns is None:
        return df

    if not hasattr(grp, '_cells'):
        grp._cells = CellSet(grp)
    cell_set = grp._cells

    first_include = [x for x in include_columns if 'cell' not in x]

    df = prepare_dataframe(df, include_columns=first_include)

    def return_cell(row, prefix):
        for cell in cell_set:
            if cell.parent_experiment == row[prefix + 'expt']:
                if row[prefix + 'roi'].label.startswith(cell.label):
                    return cell

    if any(['cell' in col_list for col_list in include_columns]):

        for prefix in ('first_', 'second_', ''):

            # If 'cell' is already reported, nothing to do
            if prefix + 'cell' in df.columns:
                continue

            if prefix + 'expt' in df.columns:

                df[prefix + 'cell'] = df.apply(return_cell, args=(prefix,), axis=1)

    return df


def in_pf(pos, pf):
            if pf[0] < pf[1]:
                if (pos >= pf[0]) and (pos <= pf[1]):
                    return True
                else:
                    return False
            else:
                if (pos >= pf[0]) or (pos <= pf[1]):
                    return True
                else:
                    return False


def interval_filter(cell, expt, event_times,
                    running_only=False, non_running_only=False,
                    ripple_only=False, non_ripple_only=False,
                    in_field_only=False, out_field_only=False,
                    reward_only=False, non_reward_only=False,
                    **kwargs):

    assert(not (running_only and non_running_only))
    assert(not (in_field_only and out_field_only))
    # assert(not (ripple_only and non_ripple_only))

    individual_filters = []

    if running_only:
        # Assume cycle = 0
        running = expt.runningIntervals(
            returnBoolList=True)[0]
        include_idxs = [x for x, y in
                        enumerate(event_times) if running[y]]

        individual_filters.append(include_idxs)

    elif non_running_only:
        # Assume cycle = 0
        running = expt.runningIntervals(
            returnBoolList=True)[0]
        include_idxs = [x for x, y in
                        enumerate(event_times) if not running[y]]

        individual_filters.append(include_idxs)

    elif ripple_only:
        min_ripple_dur = kwargs.get('min_ripple_dur')
        ripples = expt.rippleIntervals(
            returnBoolList=True, min_ripple_dur=min_ripple_dur)
        include_idxs = [x for x, y in
                        enumerate(event_times) if ripples[y]]

        individual_filters.append(include_idxs)

    elif non_ripple_only:

        ripples = expt.rippleIntervals(
            returnBoolList=True)
        include_idxs = [x for x, y in
                        enumerate(event_times) if not ripples[y]]

        individual_filters.append(include_idxs)

    if in_field_only:

        # pf must be passed in as a kwarg
        pf = kwargs['pf']
        position = ba.absolutePosition(expt.find('trial'), imageSync=True) % 1

        include_idxs = [x for x, y in
                        enumerate(event_times) if in_pf(position[y], pf)]

        individual_filters.append(include_idxs)

    if out_field_only:

        # pf must be passed in as a kwarg
        pf = kwargs['pf']
        position = ba.absolutePosition(expt.find('trial'), imageSync=True) % 1

        include_idxs = [x for x, y in
                        enumerate(event_times) if not in_pf(position[y], pf)]

        individual_filters.append(include_idxs)

    if reward_only or non_reward_only:

        reward_pos = expt.rewardPositions(units='normalized')[0]

        belt_len = expt.belt().length(units='mm')

        window_len = expt.reward_parameters()['window_length']
        if isinstance(window_len, list):
            window_len = window_len[0]
        window_len /= belt_len
        window_radius = window_len / 2.

        reward_start = reward_pos - 1.5 * window_radius
        reward_stop = reward_pos + 1.5 * window_radius

        position = ba.absolutePosition(expt.find('trial'), imageSync=True) % 1

        in_reward_zone = (position >= reward_start) & (position <= reward_stop)

        if reward_only:
            include_idxs = [x for x, y in
                            enumerate(event_times) if in_reward_zone[y]]

        else:
            include_idxs = [x for x, y in
                            enumerate(event_times) if not in_reward_zone[y]]

        individual_filters.append(include_idxs)

    # If no filtering occurred, return all indices
    if not individual_filters:
        return range(len(event_times))

    # Take intersection of idxs included from each individual filter
    include_idxs = set.intersection(*map(set, individual_filters))

    return sorted(include_idxs)


def intervals(cell, expt,
            running_only=False, non_running_only=False,
            ripple_only=False, non_ripple_only=False,
            in_field_only=False, out_field_only=False,
            reward_only=False, non_reward_only=False,
            **kwargs):

    assert(not (running_only and non_running_only))
    assert(not (in_field_only and out_field_only))
    # assert(not (ripple_only and non_ripple_only))

    interval_so_far = np.ones((expt.num_frames(),), dtype=bool)
    if running_only:
        running = expt.runningIntervals(
            returnBoolList=True)[0]
        
        interval_so_far = np.logical_and(running, interval_so_far)

    elif non_running_only:
        running = expt.runningIntervals(
            returnBoolList=True)[0]

        interval_so_far = np.logical_and(~running, interval_so_far)

    if ripple_only:
        min_ripple_dur = kwargs.get('min_ripple_dur')
        ripples = expt.rippleIntervals(
            returnBoolList=True, min_ripple_dur=min_ripple_dur)
        
        interval_so_far = np.logical_and(ripples, interval_so_far)

    elif non_ripple_only:

        ripples = expt.rippleIntervals(
            returnBoolList=True)

        interval_so_far = np.logical_and(~ripples, interval_so_far)

    if in_field_only:

        # pf must be passed in as a kwarg
        pf = kwargs['pf']
        position = ba.absolutePosition(expt.find('trial'), imageSync=True) % 1

        if pf[0] < pf[1]:
            in_pf = (position >= pf[0]) & (position < pf[1])
        else:
            in_pf = (position >= pf[0]) | (position < pf[1])

        interval_so_far = np.logical_and(in_pf, interval_so_far)

    elif out_field_only:

        # pf must be passed in as a kwarg
        pf = kwargs['pf']
        position = ba.absolutePosition(expt.find('trial'), imageSync=True) % 1

        if pf[0] < pf[1]:
            in_pf = (position >= pf[0]) & (position < pf[1])
        else:
            in_pf = (position >= pf[0]) | (position < pf[1])

        interval_so_far = np.logical_and(~in_pf, interval_so_far)

    return interval_so_far


def branch_recruitment(
        exptGrp, roi_filter=None, max_order=None, tolerance=0.5,
        signal=None, min_somatic_amp=None, max_somatic_amp=None,
        **interval_kwargs):

    # The percentage of somatic transients that each branch participated in

    # init the cell set if necessary and filter
    if not hasattr(exptGrp, '_cells'):
        exptGrp._cells = CellSet(exptGrp)
    cell_set = exptGrp._cells.filter(roi_filter)

    if interval_kwargs.get('in_field_only', False) \
            or interval_kwargs.get('out_field_only'):
        pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)

    data_list = []
    for cell in cell_set:

        cell.set_signal(signal)

        soma_starts = cell.soma.signal()['start_indices']
        soma_amps = cell.soma.signal()['max_amplitudes']

        if min_somatic_amp:
            soma_starts = [x for x, y in zip(soma_starts, soma_amps)
                           if y >= min_somatic_amp]
        if max_somatic_amp:
            soma_starts = [x for x, y in zip(soma_starts, soma_amps)
                           if y <= max_somatic_amp]

        tol = int(tolerance / cell.parent_experiment.frame_period())

        if interval_kwargs.get('in_field_only', False) \
                or interval_kwargs.get('out_field_only'):
            pfs = pfs_n[cell.parent_experiment]
            idx = [x.label for x in cell.parent_experiment.rois(roi_filter=roi_filter)].index(cell.soma.label)

            if not len(pfs[idx]):
                continue
            else:
                pf = pfs[idx][0]
        else:
            pf = None

        interval = intervals(cell, cell.parent_experiment,
                             pf=pf, **interval_kwargs)

        soma_starts = [s for s in soma_starts if interval[s]]

        if len(soma_starts) == 0:
            continue

        if max_order:
            dendrites = [d for d in cell.dendrites if d.order <= max_order]
        else:
            dendrites = cell.dendrites

        # Get Experiment Info
        mouse_name = cell.parent_experiment.parent.mouse_name
        fov = cell.parent_experiment.get('uniqueLocationKey')
        day = cell.parent_experiment.get('day')
        condition = cell.parent_experiment.get('condition')
        session = cell.parent_experiment.get('session')
        tid = cell.parent_experiment.trial_id

        for dend in dendrites:

            dend_starts = dend.signal()['start_indices']
            dend_starts = [s for s in dend_starts if interval[s]]

            times_recruited = [start for start in soma_starts if
                               len([t for t in dend_starts
                                    if abs(start - t) <= tol])]

            br = len(times_recruited) / float(len(soma_starts))

            data_list.append({'cell': cell.label,
                              'roi': dend.label,
                              'distance_to_soma': dend.distance_to_soma +
                              dend.length / 2.,
                              'order': dend.order,
                              'expt': tid,
                              'mouse_name': mouse_name,
                              'fov': fov,
                              'day': day,
                              'condition': condition,
                              'session': session,
                              'value': br,
                              'n_coactive': len(times_recruited),
                              'n_isolated': len(dend_starts) - len(times_recruited)})

    df = pd.DataFrame(data_list, columns=['cell', 'roi', 'mouse_name', 'fov', 'day', 'condition',
                        'session', 'distance_to_soma', 'order', 'expt', 'value', 'n_coactive', 'n_isolated'])

    if df.empty:
        return df.astype(float)
    else:
        return df


def branch_impact(exptGrp, roi_filter=None, tolerance=0.5,
                  signal=None, **interval_kwargs):

    # Percentage of branch spikes that coincided with a somatic transient

    # init the cell set if necessary and filter
    if not hasattr(exptGrp, '_cells'):
        exptGrp._cells = CellSet(exptGrp)
    cell_set = exptGrp._cells.filter(roi_filter)

    if interval_kwargs.get('in_field_only', False) \
            or interval_kwargs.get('out_field_only'):
        pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)

    data_list = []
    for cell in cell_set:

        mouse_name = cell.parent_experiment.parent.mouse_name
        fov = cell.parent_experiment.get('uniqueLocationKey')
        day = cell.parent_experiment.get('day')
        condition = cell.parent_experiment.get('condition')
        session = cell.parent_experiment.get('session')
        tid = cell.parent_experiment.trial_id

        cell.set_signal(signal)

        soma_starts = cell.soma.signal()['start_indices']
        tol = int(tolerance / cell.parent_experiment.frame_period())

        if interval_kwargs.get('in_field_only', False) \
                or interval_kwargs.get('out_field_only'):
            pfs = pfs_n[cell.parent_experiment]
            idx = [x.label for x in cell.parent_experiment.rois(roi_filter=roi_filter)].index(cell.soma.label)

            if not len(pfs[idx]):
                continue
            else:
                pf = pfs[idx][0]
        else:
            pf = None

        interval = intervals(cell, cell.parent_experiment,
                             pf=pf, **interval_kwargs)

        interval_dur = np.sum(interval) * cell.parent_experiment.frame_period()

        n_soma_trans = len([x for x in soma_starts if interval[x]])

        for dend in cell.dendrites:

            dend_starts = dend.signal()['start_indices']

            dend_starts = [s for s in dend_starts if interval[s]]

            if len(dend_starts) == 0:
                continue

            times_coincident = [start for start in dend_starts if
                                len([t for t in soma_starts if
                                     abs(start - t) <= tol])]

            bi = len(times_coincident) / float(len(dend_starts))

            data_list.append({'cell': cell.label,
                              'roi': dend.label,
                              'distance_to_soma': dend.distance_to_soma +
                              dend.length / 2.,
                              'order': dend.order,
                              'expt': tid,
                              'mouse_name': mouse_name,
                              'day': day,
                              'condition': condition,
                              'session': session,
                              'fov': fov,
                              'value': bi,
                              'iso_value': 1 - bi,
                              'n_trans': len(dend_starts),
                              'n_coactive': len(times_coincident),
                              'n_solo': len(dend_starts) - len(times_coincident),
                              'freq_coactive': len(times_coincident) / interval_dur,
                              'freq_solo': (len(dend_starts) - len(times_coincident)) / interval_dur,
                              'n_soma_trans': n_soma_trans})

    df = pd.DataFrame(data_list, columns=['cell', 'roi',
                        'distance_to_soma', 'order', 'expt', 'mouse_name', 'day', 'condition', 'session', 'fov',
                        'value', 'iso_value', 'n_trans', 'n_cooactive', 'n_solo',
                        'freq_coactive', 'freq_solo', 'n_soma_trans'])

    if df.empty:
        return df.astype(float)
    else:
        return df


def branch_spike_prevalence(
        exptGrp, roi_filter=None, dend_filter=None,
        min_order=None, max_order=None,
        pre=0.5, post=0.75, store_pos=False,
        min_distance=0, max_distance=np.inf,
        include_failures=True,
        signal=None, **interval_kwargs):
    """Calculate the branch spike prevalence.

    This is done on a per transient basis
    as defined in Sheffield and Dombeck (Nature, 2015).

    Parameters
    ----------
    cell_set : CellSet
        A CellSet instance to analyze
    min_order : int
        Only consider branches equal or higher than this degree (inclusive)
    max_order : int
        Only consider branches of this degree and lower (inclusive) in
        calculating the BSP.  By default, all branches are included.
    tolerance: float
        Absolute time in seconds that an roi's transient time can differ
        from a somal transient to be declared as a backprop of that transient

    Output
    ------
    bsp : pd.DataFrame
        A Pandas dataframe with one entry per somatic transient

    """

    # init the cell set if necessary and filter
    if not hasattr(exptGrp, '_cells'):
        exptGrp._cells = CellSet(exptGrp)

    cell_set = exptGrp._cells.filter(roi_filter)

    if interval_kwargs.get('in_field_only') \
            or interval_kwargs.get('out_field_only'):
        pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)

    data_list = []

    for cell in cell_set:

        if store_pos:
            pos = ((ba.absolutePosition(cell.parent_experiment.find('trial'), imageSync=True) % 1) * 100).astype(int)
            reward = int(cell.parent_experiment.rewardPositions(units='normalized')[0] * 100)
            shift = reward - 50
            pos = pos - shift
            pos[pos < 0] = pos[pos < 0] + 100
            pos[pos > 99] = pos[pos > 99] - 100


        cell.set_signal(signal)

        soma_starts = cell.soma.signal()['start_indices']
        amps = cell.soma.signal()['max_amplitudes']
        durs = cell.soma.signal()['durations_sec']

        pre_tol = int(pre / cell.parent_experiment.frame_period())
        post_tol = int(post / cell.parent_experiment.frame_period())

        if interval_kwargs.get('in_field_only') \
                or interval_kwargs.get('out_field_only'):
            pfs = pfs_n[cell.parent_experiment]
            idx = [x.label for x in cell.parent_experiment.rois(roi_filter=roi_filter)].index(cell.soma.label)

            if not len(pfs[idx]):
                continue
            else:
                pf = pfs[idx][0]
        else:
            pf = None

        include_idxs = interval_filter(cell, cell.parent_experiment, soma_starts,
                                       pf=pf, **interval_kwargs)

        soma_starts = [soma_starts[x] for x in include_idxs]
        amps = [amps[x] for x in include_idxs]
        durs = [durs[x] for x in include_idxs]

        # Filter dendrites
        dendrites = cell.dendrites

        if max_order:
            dendrites = [d for d in dendrites if d.order <= max_order]
        if min_order:
            dendrites = [d for d in dendrites if min_order <= d.order]

        dendrites = [d for d in dendrites if
                     min_distance <= (d.distance_to_soma + 0.5 * d.length) <= max_distance]

        if dend_filter:
            dendrites = [d for d in dendrites if dend_filter(d)]

        if dendrites:

            # Get Experiment Info
            mouse_name = cell.parent_experiment.parent.mouse_name
            fov = cell.parent_experiment.get('uniqueLocationKey')
            day = cell.parent_experiment.get('day')
            condition = cell.parent_experiment.get('condition')
            session = cell.parent_experiment.get('session')
            tid = cell.parent_experiment.trial_id

            for i, trans_start in enumerate(soma_starts):

                spiking_branches = [d for d in dendrites if
                                    len([t for t in d.signal()['start_indices']
                                         if in_somatic_window(
                                         trans_start, t, pre_tol, post_tol)])]

                if (len(spiking_branches) == 0) and not include_failures:
                    continue

                bsp = len(spiking_branches) / float(len(dendrites))

                if store_pos:
                    savepos = pos[trans_start]
                else:
                    savepos = np.nan

                data_list.append({'roi': cell.soma.roi.label,
                                  'expt': tid,
                                  'mouse_name': mouse_name,
                                  'fov': fov,
                                  'day': day,
                                  'condition': condition,
                                  'session': session,
                                  'amplitude': amps[i],
                                  'duration': durs[i],
                                  'event_idx': i,
                                  'n_dends': len(dendrites),
                                  'value': bsp,
                                  'position': savepos})

    df = pd.DataFrame(data_list, columns=['roi', 'expt', 'mouse_name', 'fov', 'day', 'condition',
                                          'session', 'amplitude', 'duration', 'event_idx', 'n_dends', 'value', 'position'])

    if df.empty:
        return df.astype(float)
    else:
        return df


def branch_coactivation(
        exptGrp, roi_filter=None, dend_filter=None,
        tolerance=0.5,
        signal=None, **interval_kwargs):
    """Calculate the branch spike prevalence.

    This is done on a per transient basis
    as defined in Sheffield and Dombeck (Nature, 2015).

    Parameters
    ----------
    cell_set : CellSet
        A CellSet instance to analyze
    min_order : int
        Only consider branches equal or higher than this degree (inclusive)
    max_order : int
        Only consider branches of this degree and lower (inclusive) in
        calculating the BSP.  By default, all branches are included.
    tolerance: float
        Absolute time in seconds that an roi's transient time can differ
        from a somal transient to be declared as a backprop of that transient

    Output
    ------
    bsp : pd.DataFrame
        A Pandas dataframe with one entry per somatic transient

    """

    # init the cell set if necessary and filter
    if not hasattr(exptGrp, '_cells'):
        exptGrp._cells = CellSet(exptGrp)

    cell_set = exptGrp._cells.filter(roi_filter)

    if interval_kwargs.get('in_field_only') \
            or interval_kwargs.get('out_field_only'):
        pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)

    data_list = []

    for cell in cell_set:

        cell.set_signal(signal)

        soma_starts = cell.soma.signal()['start_indices']
        amps = cell.soma.signal()['max_amplitudes']
        durs = cell.soma.signal()['durations_sec']

        tol = int(tolerance / cell.parent_experiment.frame_period())

        if interval_kwargs.get('in_field_only') \
                or interval_kwargs.get('out_field_only'):
            pfs = pfs_n[cell.parent_experiment]
            idx = [x.label for x in cell.parent_experiment.rois(roi_filter=roi_filter)].index(cell.soma.label)
            # idx = cell.parent_experiment.rois(roi_filter=roi_filter).index(cell.soma.roi)

            if not len(pfs[idx]):
                continue
            else:
                pf = pfs[idx][0]
        else:
            pf = None


        interval = intervals(cell, cell.parent_experiment,
                             pf=pf, **interval_kwargs)

        soma_starts = [s for s in soma_starts if interval[s]]

        n_soma_events = len(soma_starts)

        # Get trial info
        mouse_name = cell.parent_experiment.parent.mouse_name
        fov = cell.parent_experiment.get('uniqueLocationKey')
        day = cell.parent_experiment.get('day')
        condition = cell.parent_experiment.get('condition')
        session = cell.parent_experiment.get('session')
        tid = cell.parent_experiment.trial_id

        # Record all times each dendrite was coactive with soma
        # As well as just all times dendrite was active in interval
        soma_coincidence_times = []
        dend_on_times = []
        dendrites = cell.dendrites
        for dend in dendrites:

            times_recruited = [start for start in soma_starts if
                               len([t for t in dend.signal()['start_indices']
                                    if abs(start - t) <= tol])]

            soma_coincidence_times.append(times_recruited)

            dend_starts = dend.signal()['start_indices']
            dend_on_times.append([s for s in dend_starts if interval[s]])

        # Now for every pair of dendrites, count how many times they both coincided with soma,
        # as well as how many times the coincided with eachother (how to normalize this?)
        for di in range(1, len(dendrites)):
            for dj in range(di):

                # First compute conditioned on soma being active
                n_both_coactive_with_soma = len([x for x in soma_coincidence_times[di]
                                  if x in soma_coincidence_times[dj]])

                n_either_coactive_with_soma = len(set(soma_coincidence_times[di]).union(soma_coincidence_times[dj]))

                if n_soma_events == 0:
                    soma_fraction = np.nan
                else:
                    soma_fraction = float(n_both_coactive_with_soma) / n_soma_events

                if n_either_coactive_with_soma == 0:
                    dend_fraction = np.nan
                else:
                    dend_fraction = float(n_both_coactive_with_soma) / n_either_coactive_with_soma


                # Now just compute based on dendritic event times
                n_coactive = 0
                for x in dend_on_times[di]:
                    for y in dend_on_times[dj]:
                        if abs(x-y) <= tol:
                            n_coactive += 1


                total_active = len(dend_on_times[di]) + len(dend_on_times[dj])

                if total_active == 0:
                    coactivity = np.nan
                else:
                    coactivity = 2 * float(n_coactive) / total_active

                # Also just calculate raw correlation
                corr = sig_corr(dendrites[di].imaging_data()[interval],
                                dendrites[dj].imaging_data()[interval])

                # Also calculate partial correlation (control for soma)
                partial = partial_corr(dendrites[di].imaging_data()[interval],
                                       dendrites[dj].imaging_data()[interval],
                                       cell.soma.imaging_data()[interval])

                data_list.append({'roi_pair': (dendrites[di].roi.label, dendrites[dj].roi.label),
                                  'expt': tid,
                                  'mouse_name': mouse_name,
                                  'fov': fov,
                                  'day': day,
                                  'condition': condition,
                                  'session': session,
                                  'n_soma_events': n_soma_events,
                                  'n_both_coactive_with_soma': n_both_coactive_with_soma,
                                  'soma_fraction': soma_fraction,
                                  'dend_fraction': dend_fraction,
                                  'coactivity': coactivity,
                                  'corr': corr,
                                  'partial_corr': partial})

    df = pd.DataFrame(data_list, columns=['roi_pair', 'expt', 'mouse_name', 'fov', 'day', 'condition',
                                          'session', 'n_soma_events', 'n_both_coactive_with_soma', 'soma_fraction', 'dend_fraction',
                                          'coactivity', 'corr', 'partial_corr'])

    if df.empty:
        return df.astype(float)
    else:
        return df


def sig_corr(a, b):

    # Pearson correlation of two signals after removing nan time points

    b_nans = np.where(np.isnan(b))[0]

    a[b_nans] = np.nan

    good_idx = np.where(~np.isnan(a))

    return stats.pearsonr(a[good_idx], b[good_idx])[0]


def partial_corr(a, b, c):

    # Pearson correlation of two signals after removing nan time points
    # and controlling for effect of c

    b_nans = np.where(np.isnan(b))[0]
    c_nans = np.where(np.isnan(c))[0]

    a[b_nans] = np.nan
    a[c_nans] = np.nan

    good_idx = np.where(~np.isnan(a))

    anew = (a[good_idx] - np.mean(a[good_idx])) / np.std(a[good_idx])
    bnew = (b[good_idx] - np.mean(b[good_idx])) / np.std(b[good_idx])
    cnew = (c[good_idx] - np.mean(c[good_idx])) / np.std(c[good_idx])

    cnew = cnew[:, np.newaxis]

    # Inflate

    # Calc least squares fit for c
    beta_a = linalg.lstsq(cnew, anew)[0]
    beta_b = linalg.lstsq(cnew, bnew)[0]

    # Calc residuals after removing linear contribution of c
    res_a = anew - cnew.dot(beta_a)
    res_b = bnew - cnew.dot(beta_b)
    
    # Correlate residuals
    corr = stats.pearsonr(res_a, res_b)[0]

    return corr

def swr_activity(exptGrp, roi_filter=None, signal=None, window=0.3):

    if not hasattr(exptGrp, '_cells'):
        exptGrp._cells = CellSet(exptGrp)
    cell_set = exptGrp._cells.filter(roi_filter)
    data_list = []

    for cell in cell_set:

        cell.set_signal(signal)

        ripples = cell.parent_experiment.ripple_frames(doublets=False)
        win = int(window / cell.parent_experiment.frame_period())

        dendrites = cell.dendrites
        d_starts = [d.signal()['start_indices'] for d in dendrites]
        soma_starts = cell.soma.signal()['start_indices']

        soma_active = np.zeros((len(ripples)))
        dends_active = np.zeros((len(dendrites), len(ripples)))

        # Get Experiment Info
        mouse_name = cell.parent_experiment.parent.mouse_name
        fov = cell.parent_experiment.get('uniqueLocationKey')
        day = cell.parent_experiment.get('day')
        condition = cell.parent_experiment.get('condition')
        session = cell.parent_experiment.get('session')
        tid = cell.parent_experiment.trial_id

        for i, ripple in enumerate(ripples):

            rstart = np.max([ripple - win, 0])
            rstop = np.min([ripple + win, cell.parent_experiment.num_frames()])

            in_ripple = range(rstart, rstop + 1)

            soma_active[i] = np.any([x in in_ripple for x in soma_starts])

            for di, dend in enumerate(dendrites):
                if np.any([x in in_ripple for x in d_starts[di]]):
                    dends_active[di, i] = 1

        for di, dendrite in enumerate(dendrites):

            data_list.append({'expt': tid,
                              'mouse_name': mouse_name,
                              'fov': fov,
                              'condition': condition,
                              'session': session,
                              'day': day,
                              'soma': cell.soma.roi.label,
                              'roi': dendrite.roi.label,
                              'n_ripples': len(ripples),
                              'n_active': sum(dends_active[di, :]),
                              'n_coactive': len([i for i in xrange(len(ripples)) if dends_active[di, i] and soma_active[i]]),
                              'n_solo': len([i for i in xrange(len(ripples)) if dends_active[di, i] and not soma_active[i]]),
                              'n_missed': len([i for i in xrange(len(ripples)) if not dends_active[di, i] and soma_active[i]])})

    return pd.DataFrame(data_list)


def branch_spikes(
        exptGrp, roi_filter=None, dend_filter=None, include_failures=False,
        normalize=False, max_order=None, pre=0.5, post=0.75,
        signal=None, **interval_kwargs):
    """Return the amplitude of backpropagated dendritic transients.

    Parameters
    ----------
    exptGrp : ExperimentGroup
        An ExperimentGroup instance to analyze
    roi_filter : filter fn
        roi_filter
    include_failures: boolean
        Whether or not to include somatic transients that failed to
        backpropagate at all (will result in many 0's in the df)
    normalize : boolean
        Whether or not to normalize the amplitude of the dendritic
        transient to that of the corresponding somatic transient
    bsp_kwargs: kwargs
        Passed into the branch_spike_prevalence function

    Output
    ------
    pd.DataFrame
        A Pandas dataframe with one entry per dendritic transient
    """
    # init the cell set if necessary and filter
    if not hasattr(exptGrp, '_cells'):
        exptGrp._cells = CellSet(exptGrp)
    cell_set = exptGrp._cells.filter(roi_filter)

    if interval_kwargs.get('in_field_only') \
        or interval_kwargs.get('out_field_only'):
        pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)

    data_list = []
    for cell in cell_set:

        cell.set_signal(signal)

        soma_starts = cell.soma.signal()['start_indices']
        soma_stops = cell.soma.signal()['end_indices']
        amps = cell.soma.signal()['max_amplitudes']
        durs = cell.soma.signal()['durations_sec']

        cell.set_signal(signal)

        if interval_kwargs.get('in_field_only') \
                or interval_kwargs.get('out_field_only'):
            pfs = pfs_n[cell.parent_experiment]
            idx = [x.label for x in cell.parent_experiment.rois(roi_filter=roi_filter)].index(cell.soma.label)

            if not len(pfs[idx]):
                continue
            else:
                pf = pfs[idx][0]
        else:
            pf = None

        include_idxs = interval_filter(cell, cell.parent_experiment, soma_starts,
                                       pf=pf, **interval_kwargs)

        soma_starts = [soma_starts[x] for x in include_idxs]
        soma_stops = [soma_stops[x] for x in include_idxs]
        amps = [amps[x] for x in include_idxs]
        durs = [durs[x] for x in include_idxs]

        pre_tol = int(pre / cell.parent_experiment.frame_period())
        post_tol = int(post / cell.parent_experiment.frame_period())

        # Filter dendrites
        dendrites = cell.dendrites

        if max_order:
            dendrites = [d for d in dendrites if d.order <= max_order]

        if dend_filter:
            dendrites = [d for d in dendrites if dend_filter(d)]

        mouse_name = cell.parent_experiment.parent.mouse_name
        fov = cell.parent_experiment.get('uniqueLocationKey')
        day = cell.parent_experiment.get('day')
        condition = cell.parent_experiment.get('condition')
        session = cell.parent_experiment.get('session')
        tid = cell.parent_experiment.trial_id

        for i, trans_start, trans_stop in zip(it.count(), soma_starts, soma_stops):

            spiking_branches = [d for d in dendrites if
                                len([t for t in d.signal()['start_indices']
                                     if in_somatic_window(trans_start, t, pre_tol, post_tol) ])]

            if not len(spiking_branches) and not include_failures:
                continue

            for dendrite in dendrites:

                dspike_amp = np.nanmax(dendrite.imaging_data()[trans_start:trans_stop])
                if np.isnan(dspike_amp):
                    continue

                if normalize:
                    dspike_amp /= amps[i]

                data_list.append(
                    {
                    # 'expt': cell.parent_experiment,
                     'expt': tid,
                     'mouse_name': mouse_name,
                     'fov': fov,
                     'day': day,
                     'condition': condition,
                     'session': session,
                     'soma': cell.soma.roi.label,
                     'roi': dendrite.roi.label,
                     'distance_to_soma': dendrite.distance_to_soma +
                        dendrite.length / 2.,
                     'order': dendrite.order,
                     'soma_spike_amp': amps[i],
                     'soma_event_idx': i,
                     'soma_start_frame': trans_start,
                     'value': dspike_amp,
                     'normalized_value': dspike_amp / amps[i]})

    df = pd.DataFrame(data_list, columns=['expt', 'mouse_name', 'fov', 'day', 'condition',
                        'soma', 'roi', 'session',
                        'distance_to_soma', 'order', 'soma_spike_amp', 'soma_event_idx', 'soma_start_frame', 'value', 'normalized_value'])

    if df.empty:
        return df.astype(float)
    else:
        return df


def all_branch_spikes(
        exptGrp, roi_filter=None, dend_filter=None,
        pre=0.5, post=0.5,
        signal=None, **interval_kwargs):
    """Return the amplitude of backpropagated dendritic transients.

    Parameters
    ----------
    exptGrp : ExperimentGroup
        An ExperimentGroup instance to analyze
    roi_filter : filter fn
        roi_filter
    include_failures: boolean
        Whether or not to include somatic transients that failed to
        backpropagate at all (will result in many 0's in the df)
    normalize : boolean
        Whether or not to normalize the amplitude of the dendritic
        transient to that of the corresponding somatic transient
    bsp_kwargs: kwargs
        Passed into the branch_spike_prevalence function

    Output
    ------
    pd.DataFrame
        A Pandas dataframe with one entry per dendritic transient
    """
    # init the cell set if necessary and filter
    if not hasattr(exptGrp, '_cells'):
        exptGrp._cells = CellSet(exptGrp)
    cell_set = exptGrp._cells.filter(roi_filter)

    if interval_kwargs.get('in_field_only') \
        or interval_kwargs.get('out_field_only'):
        pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)

    data_list = []
    for cell in cell_set:

        cell.set_signal(signal)

        soma_starts = cell.soma.signal()['start_indices']
        soma_stops = cell.soma.signal()['end_indices']
        amps = cell.soma.signal()['max_amplitudes']
        durs = cell.soma.signal()['durations_sec']

        cell.set_signal(signal)

        if interval_kwargs.get('in_field_only') \
                or interval_kwargs.get('out_field_only'):
            pfs = pfs_n[cell.parent_experiment]
            idx = [x.label for x in cell.parent_experiment.rois(roi_filter=roi_filter)].index(cell.soma.label)

            if not len(pfs[idx]):
                continue
            else:
                pf = pfs[idx][0]
        else:
            pf = None

        include_idxs = interval_filter(cell, cell.parent_experiment, soma_starts,
                                       pf=pf, **interval_kwargs)

        soma_starts = [soma_starts[x] for x in include_idxs]
        soma_stops = [soma_stops[x] for x in include_idxs]
        soma_on = np.zeros((cell.parent_experiment.num_frames(),))
        for start, stop in zip(soma_starts, soma_stops):
            soma_on[start:stop] = 1
        amps = [amps[x] for x in include_idxs]
        durs = [durs[x] for x in include_idxs]

        tol = int(pre / cell.parent_experiment.frame_period())

        # Filter dendrites
        dendrites = cell.dendrites

        mouse_name = cell.parent_experiment.parent.mouse_name
        fov = cell.parent_experiment.get('uniqueLocationKey')
        day = cell.parent_experiment.get('day')
        condition = cell.parent_experiment.get('condition')
        session = cell.parent_experiment.get('session')
        tid = cell.parent_experiment.trial_id

        for dend in cell.dendrites:

            dend_starts = dend.signal()['start_indices']
            dend_stops = dend.signal()['end_indices']
            dend_amps = dend.signal()['max_amplitudes']
            dend_durs = dend.signal()['durations_sec']
            dend_sigma = dend.signal()['sigma']


            include_idxs = interval_filter(cell, cell.parent_experiment, dend_starts,
                                           pf=pf, **interval_kwargs)

            dend_starts = [dend_starts[x] for x in include_idxs]
            dend_stops = [dend_stops[x] for x in include_idxs]
            dend_amps = [dend_amps[x] for x in include_idxs]
            dend_durs = [dend_durs[x] for x in include_idxs]

            if len(dend_starts) == 0:
                continue

            # indices of dendritic events with start with tol of soma start
            coincident_idx = [i for i,start in enumerate(dend_starts) if
                                len([t for t in soma_starts if
                                     abs(start - t) <= tol])]

            # indices of dendritic events that have any overlap with soma event
            on_idx = [i for i, start, stop in zip(it.count(), dend_starts, dend_stops) if 
                        any([soma_on[x] for x in range(start, stop)])]

            for i in xrange(len(dend_starts)):

                data_list.append({'expt': tid,
                                  'mouse_name': mouse_name,
                                  'fov': fov,
                                  'day': day,
                                  'condition': condition,
                                  'session': session,
                                  'cell': cell.label,
                                  'roi': dend.label,
                                  'distance_to_soma': dend.distance_to_soma +
                                  dend.length / 2.,
                                  'order': dend.order,
                                  'event_idx': i,
                                  'amp': dend_amps[i],
                                  'zamp': dend_amps[i] / dend_sigma,
                                  'dur': dend_durs[i],
                                  'coincident': i in coincident_idx,
                                  'overlap': i in on_idx})


    df = pd.DataFrame(data_list, columns=['expt', 'mouse_name', 'fov', 'day', 'condition',
                        'cell', 'roi', 'session',
                        'distance_to_soma', 'order', 'event_idx', 'amp', 'zamp', 'dur', 'coincident', 'overlap'])

    if df.empty:
        return df.astype(float)
    else:
        return df


def frequency(exptGrp, signal='transients', roi_filter=None, **interval_kwargs):

    if not hasattr(exptGrp, '_cells'):
        exptGrp._cells = CellSet(exptGrp)

    cell_set = exptGrp._cells.filter(roi_filter)

    if interval_kwargs.get('in_field_only') \
            or interval_kwargs.get('out_field_only'):
        pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)

    data_list = []

    for cell in cell_set:

    # Get Experiment Info
        mouse_name = cell.parent_experiment.parent.mouse_name
        fov = cell.parent_experiment.get('uniqueLocationKey')
        day = cell.parent_experiment.get('day')
        condition = cell.parent_experiment.get('condition')
        tid = cell.parent_experiment.trial_id

        cell.set_signal(signal)

        if interval_kwargs.get('in_field_only') \
                or interval_kwargs.get('out_field_only'):
            pfs = pfs_n[cell.parent_experiment]
            idx = [x.label for x in cell.parent_experiment.rois(roi_filter=roi_filter)].index(cell.soma.label)
            # idx = cell.parent_experiment.rois(roi_filter=roi_filter).index(cell.soma.roi)

            if not len(pfs[idx]):
                continue
            else:
                pf = pfs[idx][0]
        else:
            pf = None

        interval = intervals(cell, cell.parent_experiment,
                             pf=pf, **interval_kwargs)

        interval_dur = np.sum(interval) * cell.parent_experiment.frame_period()

        soma_starts = cell.soma.transients()['start_indices']
        soma_starts = [x for x in soma_starts if interval[x]]

        soma_freq = len(soma_starts) / interval_dur

        data_list.append({'roi': cell.soma.roi.label,
                          'expt': tid,
                          'mouse_name': mouse_name,
                          'fov': fov,
                          'day': day,
                          'condition': condition,
                          'frequency': soma_freq,
                          'norm_frequency': np.nan,
                          'order': 0,
                          'soma': True})

        for dendrite in cell.dendrites:

            dend_starts = [x for x in dendrite.signal()['start_indices'] if interval[x]]

            freq = len(dend_starts) / interval_dur

            data_list.append({'roi': dendrite.roi.label,
                          'expt': tid,
                          'mouse_name': mouse_name,
                          'fov': fov,
                          'day': day,
                          'condition': condition,
                          'frequency': freq,
                          'norm_frequency': freq / soma_freq,
                          'order': dendrite.order,
                          'soma': False})

    df = pd.DataFrame(data_list, columns=['roi', 'expt', 'mouse_name', 'fov', 'day', 'condition',
                                          'norm_frequency', 'frequency', 'order', 'soma'])

    if df.empty:
        return df.astype(float)
    else:
        return df


def interval_durs(exptGrp):

    data_list = []

    for expt in exptGrp:

        tid = expt.trial_id

        expt_dur = expt.duration().seconds
        fp = expt.frame_period()

        # Running
        interval = intervals(None, expt,pf=None, running_only=True)
        run_dur = np.sum(interval) * fp
        run_frac = run_dur / expt_dur

        # Non-Run
        interval = intervals(None, expt,pf=None, non_running_only=True, non_ripple_only=True)
        nonrun_dur = np.sum(interval) * fp
        nonrun_frac = nonrun_dur / expt_dur

        # SWR
        interval = intervals(None, expt,pf=None, non_running_only=True, ripple_only=True)
        swr_dur = np.sum(interval) * fp
        swr_frac = swr_dur / expt_dur

        data_list.append({'expt':tid,
                          'run_dur': run_dur,
                          'run_frac': run_frac,
                          'nonrun_dur': nonrun_dur,
                          'nonrun_frac': nonrun_frac,
                          'swr_dur': swr_dur,
                          'swr_frac': swr_frac})

    df = pd.DataFrame(data_list)

    return df
