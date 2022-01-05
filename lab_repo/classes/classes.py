"""Experiment subclasses"""

import warnings
import numpy as np
from xml.etree import ElementTree
import cPickle as pkl
import os
import glob
import time
import itertools as it
from copy import copy, deepcopy
import pandas as pd
import h5py
import json
from datetime import datetime

import sima
from sima.ROI import ROI, ROIList

import lab_repo.classes.exceptions as exc
import lab_repo.analysis.behavior_analysis as ba
import lab_repo.analysis.imaging_analysis as ia

from collections import defaultdict


def parseTime(timeStr):
    """Parses a time string from the xml into a datetime object"""
    # Check for sql format
    if ':' in timeStr:
        return datetime.datetime.strptime(timeStr, '%Y-%m-%d %H:%M:%S')
    else:
        return datetime.datetime.strptime(timeStr, '%Y-%m-%d-%Hh%Mm%Ss')


class Experiment(ElementTree.Element):
    """An object representing a single recording session.

    Experiment instances comprise an ExperimentSet and are instantiated upon
    initialization of an ExperimentSet.  They are not directly initialized.

    Example
    -------

    >>> from lab import ExperimentSet
    >>> expt_set = ExperimentSet(
        '/analysis/experimentSummaries/.clean_code/experiments/behavior.xml')
    >>> experiment = expt_set.grabExpt('mouseID', 'startTime')

    Note
    ----
    The Experiment class inherits from the ElementTree.Element class, and as
    such, it retains the hierarchical organization of the .xml file.
    Experiment.parent returns the mouse object associated with that experiment,
    and Experiment.findall('trial') returns the list of trials associated with
    the Experiment.

    """

    def __init__(self):
        """Initialization occurs upon initialization of an ExperimentSet."""
        self._rois = {}

    #
    # Defines simple relationship operations for convenience
    #
    def __lt__(self, other):
        """Sortable by startTime."""
        return self.get('startTime') < other.get('startTime')

    def __eq__(self, other):
        """Check the equivalence of two experiments.

        Experiments are equivalent if they have the same associated mouse
        and startTime
        """
        try:
            self_mouse = self.parent.get('mouseID', np.nan)
            other_mouse = other.parent.get('mouseID', np.nan)
            self_time = self.get('startTime', np.nan)
            other_time = other.get('startTime', np.nan)
        except AttributeError:
            return False
        return (type(self) == type(other)) and (self_mouse == other_mouse) \
            and (self_time == other_time)

    def __sub__(self, other):
        """Take the difference of two experiment startTimes."""
        try:
            return self.startTime() - other.startTime()
        except AttributeError:
            raise NotImplemented

    def totuple(self):
        """Return a unique tuple representation of the experiment."""
        return (self.parent.get('mouseID'), self.get('startTime'))

    def getparent(self):
        """Simple getter function to return mouse. Same as self.parent,
        used for compatibility with lxml etree

        """
        return self.parent

    @property
    def trial_id(self):
        try:
            return int(self.get('trial_id'))
        except TypeError:
            pass

    @property
    def contexts(self):
        if 'contexts' in self.attrib:
            return self.attrib['contexts']

        self.attrib['contexts'] = \
            self.behaviorData().get(
                '__trial_info', {}).get('contexts', {}).values()
        return self.attrib['contexts']

    def field_tuple(self):
        return (self.parent.get('mouseID'), self.get('uniqueLocationKey'))

    def _filter_indices(self, roi_filter, channel='Ch2', label=None):
        if roi_filter is not None:
            indices = [i for i, r in enumerate(
                self.rois(channel=channel, label=label)) if roi_filter(r)]
        else:
            indices = np.arange(len(self.rois(channel=channel, label=label)))
        return indices

    def frame_shape(self):
        # (z, y, x, c)
        return self.imaging_dataset().frame_shape

    def num_frames(self, trim_to_behavior=True, channel='Ch2', label=None):
        if not trim_to_behavior:
            return self.imaging_dataset().num_frames

        try:
            min_time = self.imaging_dataset().num_frames
        except exc.NoTSeriesDirectory():
            min_time = np.inf

        for trial in self.findall('trial'):
            trial_duration = trial.behaviorData()['recordingDuration']
            trial_duration = int(trial_duration / self.frame_period())
            min_time = min(min_time, trial_duration)
        assert np.isfinite(min_time)
        return min_time

    def num_rois(self, label=None, roi_filter=None):
        return len(self.rois(label=label, roi_filter=roi_filter))

    def imaging_shape(
            self, channel='Ch2', label=None, roi_filter=None,
            trim_to_behavior=True):
        """Lazy stores and returns the shape of imaging data.
        (n_rois, n_frames, n_cycles)

        """

        # channel is unnecessary here, remove eventually

        if not hasattr(self, '_imaging_shape'):
            self._imaging_shape = {}
        if (channel, label, roi_filter, trim_to_behavior) not in \
                self._imaging_shape:
            self._imaging_shape[
                (channel, label, roi_filter, trim_to_behavior)] = \
                self.imagingData(
                    channel=channel, label=label, roi_filter=roi_filter,
                    trim_to_behavior=trim_to_behavior).shape
        return self._imaging_shape[
            (channel, label, roi_filter, trim_to_behavior)]

    def frame_period(self, round_result=True):
        """Returns the time between zyxc volumes.

        This is complicated by Prairie's inconsistent handling of planes/cycles
        in the xml.

        """
        if not hasattr(self, '_frame_period'):
            # If recorded with prairie, don't trust the framePeriod in the
            # imaging parameters
            try:
                xml_path = self.prairie_xml_path()
            except:
                self._frame_period = self.imagingParameters(
                    required_params=('framePeriod',))['framePeriod']
            else:
                # First make sure there is only 1 element in the tSeries, if
                # there's more than 1 this might still work, but check...
                et = ElementTree.parse('%s.env' % os.path.splitext(xml_path)[0])
                n_elements = len(et.find('./TSeries'))
                if n_elements != 1:
                    raise ValueError(
                        'Invalid experiment type, unable to determine frame ' +
                        'period')
                # If this time seems variable, adding more might even this out,
                # but it seems reliable enough to just take the first 2 frames
                n_seqs = 2
                seqs = []
                _iter = ElementTree.iterparse(xml_path)
                while len(seqs) < n_seqs:
                    try:
                        _, elem = _iter.next()
                    except StopIteration:
                        break
                    if elem.tag == 'Sequence':
                        seqs.append(elem)

                if len(seqs) > 1:
                    times = [float(s.find('Frame').get('absoluteTime'))
                             for s in seqs]
                else:
                    times = [float(frame.get('absoluteTime'))
                             for frame in seqs[0].findall('Frame')[:n_seqs]]

                self._frame_period = np.diff(times).mean()
                if seqs[0].get('bidirectionalZ', 'False') == 'True':
                    self._frame_period *= 2.0

        if round_result:
            return np.around(self._frame_period, 6)

        return self._frame_period

    def imaging_dataset(self, dataset_path=None, reload_dataset=False):
        if dataset_path is None:
            dataset_path = self.sima_path()
        if reload_dataset or not hasattr(self, '_dataset'):
            self._dataset = sima.ImagingDataset.load(dataset_path)
        return self._dataset

    def most_recent_key(self, channel='Ch2'):
        if not hasattr(self, '_most_recent_key'):
            self._most_recent_key = {}
        if channel not in self._most_recent_key:
            try:
                self._most_recent_key[channel] = sima.misc.most_recent_key(
                    self.imaging_dataset().signals(channel=channel))
            except ValueError:
                raise exc.NoSignalsData('No signals for channel {}'.format(
                    channel))
        return self._most_recent_key[channel]

    def rois(self, channel='Ch2', label=None, roi_filter=None):
        if label is None:
            label = self.most_recent_key(channel=channel)
        if label not in self._rois:
            # If the desired label is not extracted, return the original ROIs
            # TODO: Should we catch errors here?
            #
            # TODO: Do we really want to return the original rois?
            # All the analysis assumes that the ROIs are from the extracted
            # data, I don't think this is a good idea.
            #
            try:
                signals = self.imaging_dataset().signals(
                    channel=channel)[label]
            except KeyError:
                warnings.warn(
                    "LABEL MISSING FROM SIGNALS FILE, LOADING rois.pkl ROIS")
                with open(self.roisFilePath(), 'rb') as f:
                    roi_list = pkl.load(f)[label]['rois']
            else:
                roi_list = signals['rois']
            self._rois[label] = [ROI(**roi) for roi in roi_list]
            # Fill in empty IDs with what should be a unique id
            id_str = '_{}_{}_'.format(self.parent.get('mouseID'),
                                      self.get('startTime'))
            id_idx = 0
            for roi in self._rois[label]:
                if roi.id is None:
                    roi.id = id_str + str(id_idx)
                    id_idx += 1
                roi.expt = self
        if roi_filter is None:
            return ROIList(self._rois[label])

        return ROIList([roi for roi in self._rois[label] if roi_filter(roi)])

    def duration(self):
        try:
            return parseTime(self.get('stopTime')) - \
                parseTime(self.get('startTime'))
        except:
            return None

    def prairie_xml_path(self):
        root_dir = os.path.join(self.parent.parent.dataPath,
                                self.get('tSeriesDirectory').lstrip('/'))
        xml_files = glob.glob(os.path.join(root_dir, '*.xml'))
        if len(xml_files) == 1:
            return xml_files[0]
        elif len(xml_files) > 1:
            raise ValueError(
                "Unable to determine Prairie xml path: too many xml files")
        else:
            raise ValueError(
                "Unable to determine Prairie xml path: no xml files found")

    def imagingParameters(self, param=None,
            required_params=('channel_names', 'micronsPerPixel',
                             'pixelsPerLine', 'linesPerFrame')):

        if not hasattr(self, '_imagingParameters') or \
                len(set(required_params)-set(self._imagingParameters.keys()))  :
            try:
                attrs = {k:v for k, v in
                    self.imaging_dataset().sequences[0]._file['/imaging'].attrs.items()}
            except:
                pass
            else:
                add_attrs = attrs.get('__attrs__')
                if add_attrs is not None:
                    add_attrs = json.loads(add_attrs)

                    add_attrs.update(attrs)
                    attrs = add_attrs

            if len(set(required_params)-set(attrs.keys())):
                try:
                    prairie_xml = self.prairie_xml_path()
                except ValueError:
                    pass
                else:
                    attrs.update(
                        sima.imaging_parameters.prairie_imaging_parameters(
                            self.prairie_xml_path()))

            if len(set(required_params)-set(attrs.keys())):
                raise ValueError(
                    'Required Parameter(s) [%s] not found' % ', '.join(
                        set(required_params)-set(attrs.keys())))

            self._imagingParameters = attrs
        if param is None:
            return self._imagingParameters
        else:
            warnings.warn('param argument is deprecated, index the dictionary',
                          DeprecationWarning)
            return self._imagingParameters[param]

    def sima_path(self):
        """Returns the path to the .sima folder
        If > 1 .sima directory exists, raise an ambiguity error
        """
        if hasattr(self, '_sima_path'):
            return self._sima_path

        if not self.get('tSeriesDirectory'):
            raise exc.NoTSeriesDirectory()

        if self.get('sima_path'):
            return self.get('sima_path')

        tSeriesDirectory = os.path.normpath(
            os.path.join(self.parent.parent.dataPath,
                         self.get('tSeriesDirectory').lstrip('/')))
        sima_dirs = glob.glob(os.path.join(tSeriesDirectory, '*.sima'))

        if len(sima_dirs) == 1:
            return sima_dirs[0]
        elif len(sima_dirs) > 1:
            raise exc.NoSimaPath(
                'Multiple .sima directories contained in t-series directory')
        else:
            raise exc.NoSimaPath('Unable to locate .sima directory')

    def signalsFilePath(self, channel='Ch2'):
        signals_path = os.path.normpath(os.path.join(
            self.sima_path(), 'signals_{}.pkl'.format(
                self.imaging_dataset()._resolve_channel(channel))))
        return signals_path

    def transientsFilePath(self, channel='Ch2'):
        transients_path = os.path.normpath(os.path.join(
            self.sima_path(), 'transients_{}.pkl'.format(
                self.imaging_dataset()._resolve_channel(channel))))
        return transients_path

    def dfofFilePath(self, channel='Ch2'):
        dfof_path = os.path.normpath(os.path.join(
            self.sima_path(), 'dFoF_{}.pkl'.format(
                self.imaging_dataset()._resolve_channel(channel))))
        return dfof_path

    def placeFieldsFilePath(self, channel='Ch2', signal=None):
        if not signal:
            signal = 'transients'
        if signal == 'transients':
            place_path = os.path.normpath(os.path.join(
                self.sima_path(), 'place_fields_{}.pkl'.format(
                    self.imaging_dataset()._resolve_channel(channel))))
        elif signal == 'spikes':
            place_path = os.path.normpath(os.path.join(
                self.sima_path(), 'place_fields_{}_spikes.pkl'.format(
                    self.imaging_dataset()._resolve_channel(channel))))
        return place_path

    def spikesFilePath(self, channel='Ch2'):
        spikes_path_pkl = os.path.normpath(os.path.join(
            self.sima_path(), 'spikes_{}.pkl'.format(
                self.imaging_dataset()._resolve_channel(channel))))
        if os.path.exists(spikes_path_pkl):
            return spikes_path_pkl

        spikes_path = os.path.normpath(os.path.join(
            self.sima_path(), 'spikes.h5'))

        return spikes_path

    def roisFilePath(self):
        return os.path.normpath(os.path.join(self.sima_path(), 'rois.pkl'))

    def hasSpikesFile(self, channel='Ch2'):
        try:
            with open(self.spikesFilePath(channel=channel)) as _:
                return True
        except:
            return False

    def hasSignalsFile(self, channel='Ch2'):
        try:
            with open(self.signalsFilePath(channel=channel)) as _:
                return True
        except:
            return False

    def hasTransientsFile(self, channel='Ch2'):
        """Return whether there are transients saved for the experiment"""
        try:
            with open(self.transientsFilePath(channel=channel)) as _:
                return True
        except:
            return False

    def hasDfofTracesFile(self, channel='Ch2'):
        """Return whether dFoF traces have been saved for the experiment"""
        try:
            with open(self.dfofFilePath(channel=channel)) as _:
                return True
        except:
            return False

    def hasPlaceFieldsFile(self, channel='Ch2', signal=None):
        """Return whether place fields have been saved for the experiment"""
        try:
            with open(self.placeFieldsFilePath(channel=channel, signal=signal)) as _:
                return True
        except:
            return False

    def belt(self):
        """Returns a Belt object corresponding to the beltID in the experiment
        attributes

        """

        class belt:

            def __init__(self, track_length):
                self.track_length = float(track_length)

            def length(self, units='cm'):
                """Return length of the belt in cm."""
                if units == 'cm':
                    return self.track_length / 10.
                if units == 'mm':
                    return self.track_length
                raise ValueError('Unrecognized units.')

        self._belt = belt(self.track_length)

        return self._belt

    def imagingData(
            self, dFOverF=None, demixed=False, roi_filter=None,
            linearTransform=None, window_width=100, dFOverF_percentile=8,
            removeNanBoutons=False, channel='Ch2', label=None,
            trim_to_behavior=True, dataframe=False, dFOFTraceType="dFOF"):
        """Return a 3D array (with axes for ROI, time, and cycle number) of
        imaging data for the experiment.

        Parameters
        ----------
        dFOverF : {'from_file', 'mean', 'median', 'sliding_window', None}
            Method for converting to dFOverF
        demixed : bool
            whether to use ICA-demixed imaging data
        roi_filter : func
            Filter for selecting ROIs
        linearTransform : np.array
            Array specifying a linear transform to be performed on the input
            arguments, e.g. a matrix from PCA
        window_width : int
            Number of surrounding frames from which to calculate dFOverF using
            the sliding_baseline method
        dFOverF_percentile : int
            Percentile of the window to take as the baseline
        trim_to_behavior : bool
            If True, trim imaging data down to the length of
            the recorded behavior data
        dFOFTraceType : {"dFOF", "baseline", None}
            type of traces to get if dFoverF is "from_file", does not apply otherwise
            defaults to "dFOF"

        Returns
        -------
        imaging_data : np.array
            3D array (ROI, time, trial) of imaging data

        """

        if self.get('ignoreImagingData'):
            raise Exception('Imaging data ignored.')

        if label is None:
            label = self.most_recent_key(channel=channel)

        if trim_to_behavior:
            trimmed_frames = self.num_frames()

        if dFOverF == 'from_file':
            if removeNanBoutons:
                warnings.warn(
                    "NaN boutons not removed when dF method is 'from_file'")
            path = self.dfofFilePath(channel=channel)
            try:
                with open(path, 'rb') as f:
                    dfof_traces = pkl.load(f)
            except (IOError, pkl.UnpicklingError):
                raise exc.NoDfofTraces('No dfof traces')


            if(dFOFTraceType == "baseline"):
                traceKey = "baseline"
            else:
                traceKey = "traces" if not demixed else "demixed_traces"

            try:
                traces = dfof_traces[label][traceKey]
#                 ['traces' if not demixed else 'demixed_traces']
            except KeyError:
                raise exc.NoDfofTraces('This label does not exist in dfof file')

            if trim_to_behavior:
                traces = traces[:, :trimmed_frames, :]

            if roi_filter is None:
                imData = traces
            else:
                try:
                    imData = traces[self._filter_indices(
                        roi_filter, channel=channel, label=label)]
                except KeyError:
                    raise exc.NoDfofTraces(
                        'No signals found for ch: {}, label: '.format(
                            channel, label))

        else:
            signals = self.imaging_dataset().signals(channel=channel)
            if len(signals) == 0:
                raise exc.NoSignalsData
            try:
                imData = signals[label]
            except KeyError:
                raise exc.NoSignalsData
            else:
                imData = imData['demixed_raw' if demixed else 'raw']
                # Check to make sure that all cycles have the same number of
                # frames, trim off any that don't match the first cycle
                frames_per_cycle = [cycle.shape[1] for cycle in imData]
                cycle_iter = zip(it.count(), frames_per_cycle)
                cycle_iter.reverse()
                for cycle_idx, cycle_frames in cycle_iter:
                    if cycle_frames != frames_per_cycle[0]:
                        warnings.warn(
                            'Dropping cycle with non-matching number of ' +
                            'frames: cycle_0: {}, cycle_{}: {}'.format(
                                frames_per_cycle[0], cycle_idx,
                                frames_per_cycle[cycle_idx]))
                        imData.pop(cycle_idx)
                imData = np.array(imData)
                imData = np.rollaxis(imData, 0, 3)

            if imData.ndim == 2:
                # Reshape data to always contain 3 dimensions
                imData = imData.reshape(imData.shape + (1,))

            if trim_to_behavior:
                imData = imData[:, :trimmed_frames, :]

            if roi_filter is None:
                indices = np.arange(imData.shape[0])
            else:
                try:
                    indices = self._filter_indices(
                        roi_filter, channel=channel, label=label)
                except KeyError:
                    raise exc.NoDfofTraces(
                        'No signals found for ch: {}, label: '.format(
                            channel, label))

            if removeNanBoutons:
                nan_indices = np.nonzero(np.any(np.isnan(
                    imData.reshape([imData.shape[0], -1], order='F')),
                    axis=1))[0]
                indices = list(set(indices).difference(nan_indices))

            # filter if filter passed in
            imData = imData[indices, :, :]

            # perform a linear transformation on the space of ROIs
            if linearTransform is not None:
                imData = np.tensordot(linearTransform, imData, axes=1)

            if dFOverF == 'mean':
                imData = dff.mean(imData)

            elif dFOverF == 'median':
                imData = dff.median(imData)

            elif dFOverF == 'sliding_window':
                imData = dff.sliding_window(
                    imData, window_width, dFOverF_percentile)

            elif dFOverF == 'sliding_window2':
                t0 = 2.  # exponential decay constant for df/f smoothing
                t1 = 8.  # size of sliding window for smoothing
                t2 = 400.  # size of baseline
                baselinePercentile = 5

                imData = dff.sliding_window_jia(
                    imData, t0, t1, t2, baselinePercentile,
                    self.frame_period())

            elif dFOverF == 'non-running-baseline':
                imData = dff.non_running_baseline(imData,
                                                  self.runningIntervals())

        if dataframe:
            data_list = []
            rois = self.rois(channel=channel, label=label,
                             roi_filter=roi_filter)
            for trial_idx, trial in enumerate(self.findall('trial')):
                for roi_idx, roi in enumerate(rois):
                    data = {"roi": roi,
                            "trial": trial,
                            "im_data": imData[roi_idx, :, trial_idx]
                            }
                    data_list.append(data)
            imData = pd.DataFrame(data_list)

        return imData


    def spikes(self, channel='Ch2', label=None, trans_like=False,
               roi_filter=None, binary=False):
        """Return spike signal"""
        if not self.get('tSeriesDirectory'):
            raise exc.NoTSeriesDirectory
        path = self.spikesFilePath(channel=channel)

        if label is None:
            try:
                label = self.most_recent_key(channel=channel)
            except exc.NoSignalsData:
                raise exc.NoTransientsData(
                    'No signals for channel \'{}\''.format(channel))

        if roi_filter is not None:
            try:
                indices = self._filter_indices(
                    roi_filter, channel=channel, label=label)
            except KeyError:
                raise exc.NoTransientsData(
                    'No signals found for ch: {}, label: '.format(
                        channel, label))
        else:
            indices = slice(0, None)

        # if not hasattr(self, '_spikes'):
        if os.path.splitext(path)[1] == '.h5':
            channel_ = self.imaging_dataset().channel_names[
               self.imaging_dataset()._resolve_channel(channel)]
            with h5py.File(path, 'r') as f:
                spikes = np.array(f[str(channel_)][label])
        else:
            try:
                with open(path, 'rb') as file:
                    spikes = pkl.load(file)[label]['spikes']
            except (IOError, pkl.UnpicklingError):
                    raise exc.NoSpikesData
        spikes = deepcopy(spikes)[indices, ...]

        if binary:
            spikes[spikes > 0] = 1


        if trans_like:

            trans_spikes = np.empty(
                (spikes.shape[0], 1), dtype=[
                    ('start_indices', object),
                    ('end_indices', object), ('max_amplitudes', object),
                    ('durations_sec', object), ('max_indices', object)])

            frame_period = self.frame_period()

            for i, roi_spikes in enumerate(spikes):

                trans_spikes[i][0]['start_indices'] = []
                trans_spikes[i][0]['end_indices'] = []
                trans_spikes[i][0]['max_amplitudes'] = []
                trans_spikes[i][0]['durations_sec'] = []
                trans_spikes[i][0]['max_indices'] = []

                spike_idx = np.where(roi_spikes > 0)[0]

                runs = consecutive_integers(spike_idx)

                for run in runs:

                    amp = np.max(roi_spikes[run])
                    rel_max_ind = np.where(roi_spikes[run] == amp)[0].tolist()[0]
                    dur = (run[-1] - run[0] + 1) * frame_period

                    trans_spikes[i][0]['start_indices'].append(run[0])
                    trans_spikes[i][0]['end_indices'].append(run[-1])
                    trans_spikes[i][0]['max_amplitudes'].append(amp)
                    trans_spikes[i][0]['durations_sec'].append(dur)
                    trans_spikes[i][0]['max_indices'].append(run[0] + rel_max_ind)

                for field in trans_spikes.dtype.names:
                    trans_spikes[i][0][field] = np.array(trans_spikes[i][0][field])

            return trans_spikes

        return spikes

    def runningIntervals(self, **kwargs):
        """ Returns a list (one entry per cycle) of np arrays, each containing
        the starting and stopping imaging index for each running interval.

        See Trial.runningIntervals for details.

        """

        result = []
        for cycle in self.findall('trial'):
            result.append(ba.runningIntervals(cycle, **kwargs))
        return result

    def velocity(self, **kwargs):
        """ Returns a list (one entry per cycle) of np arrays, each containing
        the velocity of the mouse at the current imaging frame

        See Trial.velocity for details.

        """

        result = []
        for cycle in self.findall('trial'):
            result.append(ba.velocity(cycle, **kwargs))
        return result


class Trial(ElementTree.Element):

    def __lt__(self, other):
        return self.startTime() < other.startTime()

    def __eq__(self, other):
        try:
            self_expt = self.parent
            other_expt = other.parent
            self_time = self.get('time', np.nan)
            other_time = other.get('time', np.nan)
        except AttributeError:
            return False
        return isinstance(other, Trial) and (self_expt == other_expt) \
            and (self_time == other_time)

    def startTime(self):
        return parseTime(self.get('time'))

    def image_sync_behavior_length(self):
        """Returns the number of frames of image sync'd behaviorData
        Can be less than number of imaging frames
        """
        if not hasattr(self, '_image_sync_behavior_length'):
            self._image_sync_behavior_length = len(
                self.behaviorData(imageSync=True)['treadmillTimes'])
        return self._image_sync_behavior_length

    def behaviorDataPath(self):
        """Returns the path to the behaviorData pkl file"""
        if 'filename' not in self.keys():
            raise exc.MissingBehaviorData(
                'Missing filename field, no behavior data recorded')
        return normpath(join(self.parent.parent.parent.behaviorDataPath,
                             self.get('filename').replace('.csv', '.pkl')))


    def behavior_sampling_interval(self):
        """Shortcut method to just return the behavior data sampling_interval.
        Saves it as well, so it's only loaded once."""
        if not hasattr(self, '_behavior_sampling_interval'):
            bd = pickle.load(open(self.behaviorDataPath(), 'r'))
            try:
                self._behavior_sampling_interval = float(bd['samplingInterval'])
            except KeyError:
                self._behavior_sampling_interval = 0.01

        return self._behavior_sampling_interval

    def _resample_position(self, positions, sampling_interval=None):
        rate = np.min(np.diff(positions[:, 0]))
        gaps = np.where(np.diff(positions[:,0]) > rate*2)[0]
        for gap in gaps:
            positions = np.insert(positions, gap+1, [positions[gap][0]+rate,
                positions[gap][1]], axis=0)
            gaps += 1

        lap_times = np.where(np.diff(positions[:,1]) < -0.5)[0]
        for ti in lap_times:
            positions[ti+1:, 1] += 1.0

        lap_times = np.where(np.diff(positions[:,1]) > 0.5)[0]
        for ti in lap_times:
            positions[ti+1:, 1] -= 1.0

        if sampling_interval is not None:
            rate = sampling_interval

        position_func = scipy.interpolate.interp1d(
            positions[:, 0], positions[:, 1])
        new_times = np.arange(0, max(positions[:, 0]), rate)
        new_positions = position_func(new_times) % 1.0

        return np.vstack(([new_times], [new_positions])).T, rate

    def behaviorData(self, imageSync=False, sampling_interval=None, discard_initial=False,
                     use_rebinning=False):
        """Return a dictionary containing the the behavioral data.

        Parameters
        ----------
        imageSync : bool
             If False, the structure will represent the sparse times at
            which the stimuli/behavioral variables changed.
            If True, the structure will contain a boolean array corresponding
            to the stimulus intervals, with timepoints separated by the
            'framePeriod' up to the length of the imaging data.
            For 'treadmillTimes' the structure will contain the number of beam
            breaks within each sampling interval.
            For 'lapCounter' the times will of each marker will be converted to
            frame numbers for imageSync.
            The default 'lapCounter' format is an Nx2 array of times for each
            marker, the first column being the time and the second column being
            the marker number with '1' being the lap start marker.
        sampling_interval : {None, 'actual', float}
             The sampling interval (in seconds) of the output
            data structure or 'actual' to use the sampling interval that the
            data was recorded at or None to return sparse intervals.
        discard_initial : bool
            Discards data from first partial lap and starts at first 0 position
            if True
        Note
        ----
        imageSync=True is mostly the same as
        sampling_interval=self.parent.frame_period(),
        though additionally all arrays are trimmed down to the length of the
        imaging data

        If sampling_interval is not None or imageSync = True, data is converted
        from time (in seconds) to frame (in units of sampling_interval).
        For this conversion, frame n = [n, n+1) so the times of the final
        output arrays is [0, recording_duration)

        """

        # make sure the behavior data is there and load it
        if not hasattr(self, '_behavior_data'):
            try:
                self._behavior_data = {'original': pickle.load(
                    open(self.behaviorDataPath(), 'rb'))}
            except:
                raise exc.MissingBehaviorData('Unable to find behavior data')

        if ((imageSync, sampling_interval) in self._behavior_data) and (not use_rebinning):
            return deepcopy(self._behavior_data[imageSync, sampling_interval])

        dataDict = deepcopy(self._behavior_data['original'])

        # All of these conversions are unnecessary with pickled behaviorData
        # TODO: remove them all?
        try:
            assert discard_initial is True
            d = self._behavior_data['original']
            first_lap_start = d['lapCounter'][d['lapCounter'][:,1] == 1][0,0] / 100.
            dataDict['recordingDuration'] = float(dataDict['recordingDuration']) - first_lap_start
            dataDict['samplingInterval'] = float(dataDict['samplingInterval'])
            dataDict['trackLength'] = float(dataDict['trackLength'])
            dataDict['lapCounter'] = dataDict['lapCounter'][dataDict['lapCounter'][:,0] >= first_lap_start * 100]
            dataDict['licking'] = dataDict['licking'][dataDict['licking'][:, 0] > first_lap_start]
            dataDict['water'] = dataDict['water'][dataDict['water'][:,0] > first_lap_start]
            dataDict['laser'] = dataDict['laser'][dataDict['laser'][:,0] > first_lap_start]
            dataDict['reward'] = dataDict['reward'][dataDict['reward'][:,0] > first_lap_start]
            dataDict['light'] = dataDict['light'][dataDict['light'][:,0] > first_lap_start]
            dataDict['tone'] = dataDict['tone'][dataDict['tone'][:,0] > first_lap_start]
            dataDict['shock'] = dataDict['shock'][dataDict['shock'][:,0] > first_lap_start]
            dataDict['airpuff'] = dataDict['airpuff'][dataDict['airpuff'][:,0] > first_lap_start]
            dataDict['odorA'] = dataDict['odorA'][dataDict['odorA'][:,0] > first_lap_start]
            dataDict['odorB'] = dataDict['odorB'][dataDict['odorB'][:,0] > first_lap_start]
            try:
                dataDict['treadmillPosition'] = dataDict['treadmillPosition'][dataDict['treadmillPosition'][:,0] > first_lap_start]
            except KeyError:
                print """{} doesn't have treadmillPosition??""".format(self.behaviorDataPath())
        except:
            if discard_initial is True:
                print """Bad stuff happened with {}""".format(self.behaviorDataPath())
            try:
                dataDict['recordingDuration'] = \
                    float(dataDict['recordingDuration'])
            except:
                pass
            try:
                dataDict['samplingInterval'] = float(dataDict['samplingInterval'])
            except:
                pass
            try:
                dataDict['trackLength'] = float(dataDict['trackLength'])
            except:
                pass
            # This shouldn't be necessary, what happened that changed this?
            try:
                dataDict['treadmillTimes'] = dataDict['treadmillTimes'].reshape(-1)
            except:
                pass
            try:
                dataDict['treadmillTimes2'] = \
                    dataDict['treadmillTimes2'].reshape(-1)
            except:
                pass

        orig_samp_int = sampling_interval
        if imageSync:
            sampling_interval = self.parent.frame_period()
            nFrames = self.parent.num_frames()

        if 'samplingInterval' not in dataDict:
            if type(sampling_interval) == type(1.0) or \
                    type(sampling_interval) == type(np.float64(1.0)):
                dataDict['samplingInterval'] = sampling_interval
            else:
                dataDict["samplingInterval"] = 0.01

        if sampling_interval == 'actual':
            sampling_interval = dataDict['samplingInterval']

        # If we don't want the data resampled, just return as intervals
        if sampling_interval is None:
            self._behavior_data[(imageSync, sampling_interval)] = dataDict
            return deepcopy(dataDict)

        output_interval = sampling_interval
        uprateFactor = None
        if(use_rebinning):
            uprateFactor = int((1 / dataDict["samplingInterval"]) * 5 / (1 / sampling_interval));
            upRate = uprateFactor * (1 / sampling_interval)
            sampling_interval = 1.0 / upRate

        recordingDuration = dataDict['recordingDuration']

        # Changed 4/6 by Jeff, will now include frames with full behavior data
        # numberBehaviorFrames = \
        #     int(np.ceil(recordingDuration / sampling_interval))
        numberBehaviorFrames = int(recordingDuration / sampling_interval)
        # Return data as boolean array, matched to imaging data
        for stim in dataDict:
            if not stim.startswith('__') and stim not in [
                    'treadmillTimes', 'treadmillTimes2', 'lapCounter',
                    'treadmillPosition', 'samplingInterval',
                    'recordingDuration', 'trackLength', 'lap',
                    'position_lap_reader']:
                out = np.zeros(numberBehaviorFrames, 'bool')

                for i, (start, stop) in enumerate(dataDict[stim]):
                    if np.isnan(start):
                        if i == 0:
                            start = 0
                        else:
                            start = dataDict[stim][i-1, 0]
                        dataDict[stim][i, 0] = start

                    if np.isnan(stop):
                        try:
                            stop = dataDict[stim][np.isfinite(dataDict[stim][:,1])][i,1]
                        except(IndexError):
                            stop = recordingDuration
                        dataDict[stim][i, 1] = stop

                    start_frame = int(start / sampling_interval)
                    # Changed 4/6 by Jeff, this is more correct
                    stop_frame = int(stop / sampling_interval) + 1
                    # stop_frame = int(np.ceil(stop / sampling_interval))
                    out[start_frame:stop_frame] = True

                if(use_rebinning):
                    numFrames = int(numberBehaviorFrames / uprateFactor)
                    out = out[:(numFrames * uprateFactor)]
                    out = np.sum(np.reshape(out, (numFrames, uprateFactor)), axis=1) / float(uprateFactor)

                if imageSync and len(out) > nFrames:
                    out = out[:nFrames]
                dataDict[stim] = out
        # treadmillTimes will be the number of beam breaks in each interval
        for treadmill_times in ['treadmillTimes', 'treadmillTimes2']:
            if treadmill_times in dataDict:
                out = np.zeros(numberBehaviorFrames)
                for tick_time in dataDict[treadmill_times]:
                    behavior_bin = int(tick_time / sampling_interval)
                    if behavior_bin < numberBehaviorFrames:
                        out[behavior_bin] += 1
                    # # Correct for the edge case
                    # if tick_time == recordingDuration:
                    #     tick_time -= np.spacing(1)
                    # out[int(tick_time / sampling_interval)] += 1
                if(use_rebinning):
                    numFrames = int(numberBehaviorFrames / uprateFactor)
                    out = out[:(numFrames * uprateFactor)]
                    out = np.sum(np.reshape(out, (numFrames, uprateFactor)), axis=1)

                if imageSync and len(out) > nFrames:
                    out = out[:nFrames]
                dataDict[treadmill_times] = out

        sampling_interval = output_interval
        # lapCounter will just convert the real times into frame numbers
        if 'lapCounter' in dataDict and len(dataDict['lapCounter']) > 0:
            dataDict['lapCounter'][:, 0] /= sampling_interval
            dataDict['lapCounter'][:, 0] = np.floor(
                dataDict['lapCounter'][:, 0])

        if '__position_updates' in dataDict:
            times = dataDict['__position_updates'][:, 0]
            times /= sampling_interval
            times = np.floor(times)
            position_updates = dataDict['__position_updates'][:, 1:]
            dataDict['__position_updates'] = np.zeros((numberBehaviorFrames, 3))
            for time, update in zip(times.astype(int), position_updates):
                if time < numberBehaviorFrames:
                    dataDict['__position_updates'][time] += update
            #dataDict['__position_updates'][times.astype(int)] = position_updates

        for lap_value in ['lap', 'position_lap_reader']:
            if lap_value in dataDict:
                dataDict[lap_value] = np.ceil(
                    dataDict[lap_value]/sampling_interval)

        # treadmillPosition will be the mean position during each frame
        if 'treadmillPosition' in dataDict:
            # This is a little complicated since, just rounding down the
            # treadmill times to the nearest bin biases the result more for
            # low sampling rates than for high ones.
            # To get around this, always calculate the full position at the
            # original sampling rate, and then downsample from there.

            out = np.empty(numberBehaviorFrames)

            original_sampling_interval = dataDict['samplingInterval']
            assert sampling_interval >= original_sampling_interval

            treadmill_position = dataDict['treadmillPosition']

            assert treadmill_position[0, 0] == 0.
            # Make sure time 0 is in the positions, so the fill will be for
            # times after the last change in position (which will be constant)
            position_interp = scipy.interpolate.interp1d(
                treadmill_position[:, 0], treadmill_position[:, 1],
                kind='zero', bounds_error=False,
                fill_value=treadmill_position[-1, 1])

            times = np.arange(0., recordingDuration,
                              original_sampling_interval)

            position = position_interp(times)

            if sampling_interval == original_sampling_interval:
                out = position[:numberBehaviorFrames]
            else:
                frames = np.arange(numberBehaviorFrames)
                start_frames = np.around(
                    frames * sampling_interval / original_sampling_interval,
                    0).astype(int)
                stop_frames = np.around(
                    (frames + 1) * sampling_interval /
                    original_sampling_interval, 0).astype(int)

                for frame, start_frame, stop_frame in it.izip(
                        it.count(), start_frames, stop_frames):

                    data = position[start_frame:stop_frame]

                    data_sorted = sorted(data)
                    if len(data_sorted) >= 2 and \
                            data_sorted[-1] - data_sorted[0] > 0.9:
                        high_vals = data[data >= 0.5]
                        low_vals = data[data < 0.5]

                        out[frame] = np.mean(
                            np.hstack((high_vals, low_vals + 1))) % 1
                    else:
                        out[frame] = np.mean(data)

            assert np.all(out < 1.)
            assert np.all(out >= 0.)

            # Trim down if behavior data runs longer
            if imageSync and len(out) > nFrames:
                out = out[:nFrames]

            dataDict['treadmillPosition'] = np.round(out, 8)

        if(not use_rebinning):
            self._behavior_data[(imageSync, orig_samp_int)] = dataDict
        return deepcopy(dataDict)

    def trialNum(self):
        """Return the unique sequential number of this trial within
        it's experiment

        """
        return self.parent.findall('trial').index(self)

    def duration(self):
        try:
            return int(self.get('duration'))
        except TypeError:
            raise

    def __repr__(self):
        s = "<  Trial: " + self.parent.parent.get('mouseID', '')
        for key, value in self.attrib.iteritems():
            s = s + ", " + key + ' = ' + str(value)
        s = s + '>'
        return s

    def __str__(self):
        return "<  Trial: " + self.parent.parent.get('mouseID', '') + \
               ", stimulus = " + self.get('stimulus', '') + \
               ", time = " + self.get('time', '') + ">"



class Mouse(ElementTree.Element):

    # Make them sortable by mouseID
    def __lt__(self, other):
        return self.get('mouseID') < other.get('mouseID')

    #
    # Collect information about the experiments within the mouse
    #

    def imagingExperiments(self, channels=['Ch1', 'Ch2']):
        """Returns a list of the experiments with non-empty signals files for
        any of the above channels."""
        expts = []
        for expt in self.findall('experiment'):
            for channel in channels:
                if expt.hasSignalsFile(channel=channel) \
                        and not expt.get('ignoreImagingData'):
                    expts.append(expt)
                    break
        return expts

    #
    # Print more useful string representations of Mouse objects
    #

    def __repr__(self):
        s = "<Mouse: "
        for key, value in self.attrib.iteritems():
            s = s + key + ' = ' + str(value) + ", "
        s = s[:-1] + '>'
        return s

    def __str__(self):
        return "<Mouse: {}, genotype={}, nExpts={}>".format(
            self.get('mouseID', ''), self.get('genotype', ''),
            len(self.findall('experiment')))


class ExperimentGroup(object):
    """Grouping of experiments, e.g. by same location and experiment type.

    Example
    -------
    >>> from lab import ExperimentSet
    >>> expt_set = ExperimentSet(
        '/analysis/experimentSummaries/.clean_code/experiments/behavior_jeff.xml')

    >>> e1 = expt_set.grabExpt('sample_mouse', 'startTime1')
    >>> e2 = expt_set.grabExpt('sample_mouse', 'startTime2')

    >>> expt_grp = ExperimentGroup([e1, e2], label='example_group')

    Parameters
    ----------
    experiment_list : list
        A list of lab.classes.Experiment objects comprising the group.

    label : string
        A string describing the contents of the group.  For example, if you are
        comparing two ExperimentGroups (e.g. WT vs. mutant), you could label
        them as such.
    """

    def __init__(self, experiment_list, label=None, **kwargs):
        """Initialize the group."""
        super(ExperimentGroup, self).__init__(**kwargs)
        self._list = list(experiment_list)
        self._label = label

    """
    CONTAINER TYPE FUNCTIONS
    """
    def __len__(self):
        return len(self._list)

    def __getitem__(self, i):
        return self._list[i]

    def __setitem__(self, i, v):
        self._list[i] = v

    def __delitem__(self, i):
        self._list.__delitem__(i)

    def __iter__(self):
        return self._list.__iter__()

    def __reversed__(self):
        return self._list.__reversed__()

    def __str__(self):
        return "<Experiment group: label={label}, nExpts={nExpts}>".format(
            label=self.label(), nExpts=len(self))

    def __repr__(self):
        return '{}({})'.format(repr(type(self)), repr(self._list))

    def __copy__(self):
        return type(self)(copy(self._list), self._label)

    def __deepcopy__(self):
        return type(self)(deepcopy(self._list), self._label)

    def index(self, expt):
        return self._list.index(expt)

    def remove(self, expt):
        self._list.remove(expt)

    def append(self, expt):
        self._list.append(expt)

    def extend(self, expt_grp):
        self._list.extend(expt_grp)

    def label(self, newLabel=None):
        """Return or set label for the ExperimentGroup.

        Parameters
        ----------
        newLabel : None or string, optional
            If not None, change the label to 'newLabel'. Otherwise, just return
            the label.

        Returns
        -------
        label : string
            The label of the ExperimentGroup.

        """
        if newLabel is not None:
            self._label = str(newLabel)
        return self._label


    def to_json(self, path=None):
        """Dump an ExperimentGroup to a JSON representation that can be used
        to re-initialize the same experiments.

        Parameters
        ----------
        path : string, optional
            The full file path of the output .json file. If 'None', prints
            JSON.

        Notes
        -----
        The keys of the output JSON are mouseIDs and the elements are lists of
        experiment startTimes (as indicated in the .xml / SQL database).

        Returns
        -------
        None

        """

        grp_dict = defaultdict(list)
        for expt in self:
            grp_dict[expt.parent.get('mouseID')].append(expt.get('startTime'))

        for mouse, expt_list in grp_dict.iteritems():
            grp_dict[mouse] = sorted(expt_list)

        save_dict = {'experiments': grp_dict}
        if path is None:
            print json.dumps(save_dict, sort_keys=True, indent=4)
        else:
            with open(path, 'wb') as f:
                json.dump(save_dict, f, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, path, expt_set, **kwargs):
        """Initializes a new ExperimentGroup with the experiments from the JSON.

        Parameters
        ----------
        path : string
            Path to the JSON file from which to load the experiments

        expt_set : lab.classes.ExperimentSet
            An instance of the ExperimentSet class containing the experiments
            of interest.

        Returns
        -------
        ExperimentGroup

        """

        expts = json.load(open(path, 'r'))

        expt_list = []
        for mouseID, mouse_expts in expts['experiments'].iteritems():
            for startTime in mouse_expts:
                expt_list.append(expt_set.grabExpt(mouseID, startTime))

        return cls(expt_list, **kwargs)

  
    def rois(self, channel='Ch2', label=None, roi_filter=None):
        rois = {}
        for expt in self:
            # If the expt wasn't imaged, wasn't MC'd, wasn't extracted,
            # or is missing the desired label, rois[expt] = None
            try:
                rois[expt] = expt.rois(
                    channel=channel, label=label, roi_filter=roi_filter)
            except (exc.NoTSeriesDirectory, exc.NoSignalsData, exc.NoSimaPath, KeyError):
                rois[expt] = None
        return rois

    def roi_ids(self, **kwargs):

        ids = {}
        for expt in self:
            try:
                ids[expt] = expt.roi_ids(**kwargs)
            except (exc.NoTSeriesDirectory, exc.NoSignalsData):
                ids[expt] = None
        return ids

    def allROIs(self, channel='Ch2', label=None, roi_filter=None):
        """Return a dictionary containing all the ROIs in the experiment group.
        The keys are a tuple of the format (mouse, uniqueLocationKey, roi_id)
        The values are a list of tuples of the format (experiment, roi_number)
        for all the experiments that contain the ROI

        """

        if not hasattr(self, '._all_rois'):
            self._all_rois = {}

        if (channel, label, roi_filter) in self._all_rois:
            # Check if exptGrp has changed
            expts = self._all_rois[(channel, label, roi_filter)]['expts']
            same_check = all(
                [a == b for a, b in it.izip_longest(self, expts)])
        else:
            same_check = False
        if not same_check:
            rois = defaultdict(list)
            for expt in self:
                for roi_idx, roi in enumerate(self.roi_ids(
                        channel=channel, label=label,
                        roi_filter=roi_filter)[expt]):
                    key = (expt.parent, expt.get('uniqueLocationKey'), roi)
                    value = (expt, roi_idx)
                    rois[key].append(value)
            self._all_rois[(channel, label, roi_filter)] = {}
            self._all_rois[(channel, label, roi_filter)]['expts'] = copy(self)
            self._all_rois[(channel, label, roi_filter)]['rois'] = dict(rois)
        return self._all_rois[(channel, label, roi_filter)]['rois']


class pcExperimentGroup(ExperimentGroup):
    """Place cell experiment group"""

    def __init__(self, experiment_list, nPositionBins=100,
                 channel='Ch2', imaging_label=None, demixed=False,
                 pf_subset=None, signal=None, **kwargs):

        super(pcExperimentGroup, self).__init__(experiment_list, **kwargs)

        # Store all args as a dictionary
        self.args = {}
        self.args['nPositionBins'] = nPositionBins
        self.args['channel'] = channel
        self.args['imaging_label'] = imaging_label
        self.args['demixed'] = demixed
        self.args['pf_subset'] = pf_subset
        self.args['signal'] = signal

        self._data, self._data_raw, self._pfs, self._std, self._circ_var, \
            self._circ_var_p, self._info, self._info_p = \
            {}, {}, {}, {}, {}, {}, {}, {}

    def __repr__(self):
        return super(pcExperimentGroup, self).__repr__() + \
            '(args: {})'.format(repr(self.args))

    def __copy__(self):
        return type(self)(
            copy(self._list), label=self._label, **copy(self.args))

    def __deepcopy__(self):
        return type(self)(
            deepcopy(self._list), label=self._label, **deepcopy(self.args))

    def __delitem__(self, i):
        expt = self[i]
        super(pcExperimentGroup, self).__delitem__(i)
        self._data.pop(expt, 0)
        self._data_raw.pop(expt, 0)
        self._pfs.pop(expt, 0)
        self._std.pop(expt, 0)

    def data(self, roi_filter=None, dataframe=False):
        # tuning curves by experiment

        indices = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])

            if expt not in self._data:
                try:
                    # check for existence of place_fields.pkl
                    with open(expt.placeFieldsFilePath(
                            channel=self.args['channel'],
                            signal=self.args['signal']), 'rb') as f:
                        place_fields = pickle.load(f)
                except IOError:
                    self._data[expt] = None
                    self._data_raw[expt] = None
                else:
                    demixed_key = 'demixed' if self.args['demixed'] \
                        else 'undemixed'
                    imaging_label = self.args['imaging_label'] \
                        if self.args['imaging_label'] is not None \
                        else expt.most_recent_key(channel=self.args['channel'])
                    if self.args['pf_subset']:
                        try:
                            self._data[expt] = place_fields[
                                imaging_label][demixed_key][self.args['pf_subset']][
                                'spatial_tuning_smooth']
                            self._data_raw[expt] = place_fields[
                                imaging_label][demixed_key][self.args['pf_subset']][
                                'spatial_tuning']
                        except KeyError:
                            self._data[expt] = None
                            self._data_raw[expt] = None
                    else:
                        try:
                            self._data[expt] = place_fields[
                                imaging_label][demixed_key][
                                'spatial_tuning_smooth']
                            self._data_raw[expt] = place_fields[
                                imaging_label][demixed_key][
                                'spatial_tuning']
                        except KeyError:
                            self._data[expt] = None
                            self._data_raw[expt] = None

        return_data = [] if dataframe else {}
        if dataframe:
            rois = self.rois(
                roi_filter=roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
        for expt in self:
            try:
                expt_data = self._data[expt][indices[expt], :]
            except (TypeError, IndexError):
                expt_data = None
            if dataframe:
                assert len(rois[expt]) == len(expt_data)
                for roi, dat in zip(rois[expt], expt_data):
                    return_data.append(
                        {'expt': expt, 'roi': roi, 'value': dat})
            else:
                return_data[expt] = expt_data
        if dataframe:
            return pd.DataFrame(return_data)
        return return_data

    def data_raw(self, roi_filter=None, dataframe=False):
        self.data()
        return_data = [] if dataframe else {}
        if dataframe:
            rois = self.rois(
                roi_filter=roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
        for expt in self:
            indices = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            try:
                expt_data = self._data_raw[expt][indices, :]
            except (TypeError, IndexError):
                expt_data = None
            if dataframe:
                assert len(rois[expt]) == len(expt_data)
                for roi, dat in zip(rois[expt], expt_data):
                    return_data.append(
                        {'expt': expt, 'roi': roi, 'value': dat})
            else:
                return_data[expt] = expt_data
        if dataframe:
            return pd.DataFrame(return_data)
        return return_data


    def pfs(self, roi_filter=None):
        indices = {}
        return_data = {}
        for expt in self:
            indices[expt] = expt._filter_indices(
                roi_filter, channel=self.args['channel'],
                label=self.args['imaging_label'])
            if expt not in self._pfs:
                if self.data()[expt] is None:
                    self._pfs[expt] = None
                else:
                    with open(expt.placeFieldsFilePath(
                            channel=self.args['channel'],
                            signal=self.args['signal']), 'rb') as f:
                        result = pickle.load(f)
                    demixed_key = 'demixed' if self.args['demixed'] \
                        else 'undemixed'
                    imaging_label = self.args['imaging_label'] \
                        if self.args['imaging_label'] is not None \
                        else expt.most_recent_key(channel=self.args['channel'])
                    try:
                        self._pfs[expt] = result[
                            imaging_label][demixed_key][self.args['pf_subset']]['pfs']
                    except KeyError:
                        self._pfs[expt] = result[
                            imaging_label][demixed_key]['pfs']
            try:
                return_data[expt] = [
                    self._pfs[expt][idx] for idx in indices[expt]]
            except (TypeError, IndexError):
                return_data[expt] = None
        return return_data

    def pfs_n(self, roi_filter=None):
        pfs = self.pfs(roi_filter=roi_filter)
        pfs_n = {}
        nBins = self.args['nPositionBins']
        for expt in self:
            if pfs[expt] is None:
                pfs_n[expt] = None
            else:
                pfs_n[expt] = []
                for roi in pfs[expt]:
                    roiPfs = []
                    for pf in roi:
                        roiPfs.append(
                            [pf[0] / float(nBins), pf[1] / float(nBins)])
                    pfs_n[expt].append(roiPfs)
        return pfs_n


    def pcs_filter(self, roi_filter=None, circ_var=False):
        pcs = []

        if not circ_var:
            pfs = self.pfs(roi_filter=roi_filter)
        else:
            circular_variance_p = self.circular_variance_p(
                roi_filter=roi_filter)
        for expt in self:
            if circ_var:
                pc_inds = np.where(circular_variance_p[expt] < 0.05)[0]
            else:
                pc_inds = np.where(pfs[expt])[0]
            rois = expt.rois(channel=self.args['channel'],
                             label=self.args['imaging_label'],
                             roi_filter=roi_filter)
            pcs.extend([rois[x] for x in pc_inds])
        pcs = set(pcs)

        def pc_filter(roi):
            return roi in pcs
        return pc_filter

    def running_kwargs(self):
        if not hasattr(self, '_running_kwargs'):
            running_kwargs = []
            for expt in self:
                with open(expt.placeFieldsFilePath(
                        channel=self.args['channel'],
                        signal=self.args['signal']), 'rb') as f:
                    p = pickle.load(f)
                imaging_label = self.args['imaging_label'] \
                    if self.args['imaging_label'] is not None \
                    else expt.most_recent_key(channel=self.args['channel'])
                demixed_key = 'demixed' if self.args['demixed'] \
                    else 'undemixed'
                try:
                    running_kwargs.append(
                        p[imaging_label][demixed_key][self.args['pf_subset']]['running_kwargs'])
                except KeyError:
                    running_kwargs.append(
                        p[imaging_label][demixed_key]['running_kwargs'])
            if np.all(
                [running_kwargs[0].items() == x.items()
                 for x in running_kwargs[1:]]):
                self._running_kwargs = running_kwargs[0]
            else:
                raise(
                    'Place fields calculated with different running kwargs')
        return self._running_kwargs

    def removeDatalessExperiments(self, **kwargs):
        super(pcExperimentGroup, self).removeDatalessExperiments(**kwargs)
        for expt in reversed(self):
            if self.data()[expt] is None:
                self.remove(expt)

    def extend(self, exptGrp):
        """Extend a pcExperimentGroup with another ExperimentGroup"""

        super(pcExperimentGroup, self).extend(exptGrp)

        for expt in exptGrp:
            try:
                if expt in exptGrp._data:
                    self._data[expt] = exptGrp._data[expt]
                if expt in exptGrp._data_raw:
                    self._data_raw[expt] = exptGrp._data_raw[expt]
                if expt in exptGrp._pfs:
                    self._pfs[expt] = exptGrp._pfs[expt]
                if expt in exptGrp._std:
                    self._std[expt] = exptGrp._std[expt]
            except AttributeError:
                return

    def set_args(self, **kwargs):
        # change arguments and clear data/pfs
        for key, value in kwargs.iteritems():
            self.args[key] = value
        self._data, self._data_raw, self._pfs, self._std = {}, {}, {}, {}

    def to_json(self, path=None):
        """Dump an ExperimentGroup to a JSON representation that can be used
        to re-initialize the same experiments.

        Parameters
        ----------
        path : string, optional
            The full file path of the output .json file. If 'None', prints
            JSON.


        Notes
        -----
        The keys of the output JSON are mouseIDs and the elements are lists of
        experiment startTimes (as indicated in the .xml / SQL database).

        Returns
        -------
        None

        """
        grp_dict = defaultdict(list)
        for expt in self:
            grp_dict[expt.parent.get('mouseID')].append(expt.get('startTime'))

        for mouse, expt_list in grp_dict.iteritems():
            grp_dict[mouse] = sorted(expt_list)

        save_dict = {'experiments': grp_dict, 'args': self.args}
        if path is None:
            print json.dumps(save_dict, sort_keys=True, indent=4)
        else:
            with open(path, 'wb') as f:
                json.dump(save_dict, f, sort_keys=True, indent=4)

    @classmethod
    def from_json(cls, path, expt_set, **kwargs):
        """Initializes a new ExperimentGroup with the experiments from the JSON.

        Parameters
        ----------
        path : string
            Path to the JSON file from which to load the experiments

        expt_set : lab.classes.ExperimentSet
            An instance of the ExperimentSet class containing the experiments
            of interest.

        Returns
        -------
        ExperimentGroup

        """

        expts = json.load(open(path, 'r'))
        expt_list = []
        for mouseID, mouse_expts in expts['experiments'].iteritems():
            for startTime in mouse_expts:
                expt_list.append(expt_set.grabExpt(mouseID, startTime))

        try:
            args_dict = expts['args']
        except KeyError:
            return cls(expt_list, **kwargs)
        else:
            args_dict.update(kwargs)
            return cls(expt_list, **args_dict)
