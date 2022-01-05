import os
import re
import copy
import json
import urllib, urllib2
import warnings
import glob

from lab_repo.classes import database
from lab_repo.classes.classes import Experiment, Mouse, Trial


class dbMouse(Mouse):
    _keyDict = {
        'mouseID': 'mouse_name'
    }

    def __init__(self, mouse_id, project_name=None):
        try:
            int(mouse_id)
        except ValueError:
            mouse_id = database.fetchMouseId(mouse_id,
                project_name=project_name)

        self._mouse_id = mouse_id
        self.attrib = database.fetchMouse(mouse_id)
        self.attrib['viruses'] = []
        self.attrib.update(database.fetchAllMouseAttrs(self.mouse_name,
            parse=True))
        self._experiments = [int(r['trial_id']) for r in
            database.fetchMouseTrials(self.mouse_name)]
        self._attrib = copy.deepcopy(self.attrib)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        return self.mouse_id == other.mouse_id

    def __hash__(self):
        return self.__repr__().__hash__()

    def __iter__(self):
        for i in range(len(self._experiments)):
            yield dbExperiment(self._experiments[i])

    def __len__(self):
        return len(self._experiments)

    def __getitem__(self, key):
        return dbExperiment(self._experiments[key])

    def __getattr__(self, item):
        if item == 'attrib' or item[0] == '_':
            return self.__dict__[item]

        item = self._keyDict.get(item, item)
        if '_'+item in self.__dict__.keys():
            return self.__dict__['_'+item]

        if item == 'viruses':
            return self.getViruses()

        return self.attrib[item]

    def __setattr__(self, item, value):
        if item == 'attrib' or item[0] == '_':
            self.__dict__[item] = value
        else:
            item = self._keyDict.get(item, item)
            if '_' + item in self.__dict__.keys():
                raise Exception('{} is not settable'.format(item))
            if item == 'virus' or item == 'viruses':
                self.addVirus(value)
            else:
                self.attrib[item] = value

    def __delattr__(self, item):
        if item == 'attrib' or item[0] == '_':
            del self.__dict__[item]
        else:
            item = self._keyDict.get(item, item)
            if '_' + item in self.__dict__.keys():
                raise Exception('{} is not settable'.format(item))
            elif item in self._attrib:
                self.attrib[item] = None
            else:
                del self.attrib[item]

    @property
    def mouse_id(self):
        return self._mouse_id

    def get(self, arg, default=None):
        try:
            return self.__getattr__(arg)
        except KeyError:
            return default


class dbExperiment(Experiment, Trial):
    _keyDict= {
        'startTime': 'start_time',
        'time': 'start_time',
        'stopTime': 'stop_time',
        'tSeriesDirectory': 'tSeries_path',
        'project_name': 'experiment_group'
    }

    def __init__ (self, trial_id):
        self._rois = {}
        self._props = {}
        self.attrib = database.fetchTrial(trial_id)
        self._props['trial_id'] = trial_id

        if self.attrib is None:
            raise KeyError("Trial ID {} does not exist".format(trial_id))

        self._props['start_time'] = self.attrib['start_time'].__format__('%Y-%m-%d-%Hh%Mm%Ss')
        self.attrib['stop_time'] = \
            self.attrib['stop_time'].__format__('%Y-%m-%d-%Hh%Mm%Ss')
        for key in self.attrib.keys():
            if self.attrib[key] is None:
                del self.attrib[key]

        if self.attrib.get('behavior_file') is not None:
            self.attrib['filename'] = \
                os.path.splitext(self.attrib['behavior_file'])[0] + '.pkl'

        self.attrib.update(database.fetchAllTrialAttrs(self.trial_id,
            parse=True))

        self.attrib['experimentType'] = self.attrib.get('experimentType', '')
        self.attrib['imagingLayer'] = self.attrib.get('imagingLayer', 'unk')
        self.attrib['uniqueLocationKey'] = self.attrib.get(
            'uniqueLocationKey', '')
        self.attrib['experimenter'] = self.attrib.get('experimenter', '')
        self.attrib['belt'] = self.attrib.get('belt', '')
        self.attrib['belt_length'] = self.attrib.get('track_length', None)

        self._trial = dbTrial(self)
        self._attrib = self.attrib.copy()

    def __str__(self):
        return "<dbExperiemnt: trial_id=%d mouse_id=%d experimentType=%s>" % \
            (self.trial_id, self.mouse_id, self.get('experimentType',''))

    def __repr__(self):
        return "<dbExperiemnt: trial_id=%d mouse_id=%d experimentType=%s>" % \
            (self.trial_id, self.mouse_id, self.get('experimentType',''))

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        return self.trial_id == other.trial_id

    def __hash__(self):
        return self.__repr__().__hash__()

    def __iter__(self):
        yield self._trial

    def __len__(self):
        return 1

    def __getitem__(self, key):
        if key == 0:
            return self._trial
        raise KeyError

    def __getattr__(self, item):
        # Hacky way to deal with pandas type checking
        if item == '_typ':
            return None
        if item == 'attrib' or item[0] == '_':
            return self.__dict__[item]

        item = self._keyDict.get(item, item)
        if '_'+item in self.__dict__.keys():
            return self.__dict__['_'+item]

        if item == 'filename':
            behavior_file = self.behavior_file
            if behavior_file is None:
                return None
            return os.path.splitext(behavior_file)[0] + '.pkl'

        return self.attrib[item]

    def __setattr__(self, item, value):
        if item == 'attrib' or item[0] == '_':
            self.__dict__[item] = value
        else:
            item = self._keyDict.get(item, item)
            if '_' + item in self.__dict__.keys() or item in self.props.keys():
                raise Exception('{} is not settable'.format(item))

            self.attrib[item] = value

    def __delattr__(self, item):
        if item == 'attrib' or item[0] == '_':
            del self.__dict__[item]
        else:
            item = self._keyDict.get(item, item)
            if '_' + item in self.__dict__.keys():
                raise Exception('{} is not settable'.format(item))
            elif item in self._attrib:
                self.attrib[item] = None
            else:
                del self.attrib[item]

    @property
    def parent(self):
        return dbMouse(self.mouse_id)

    @property
    def trial_id(self):
        return self._props['trial_id']

    @property
    def start_time(self):
        return self._props['start_time']

    @property
    def archived(self):
        try:
            return self._props['archived']
        except KeyError:
            if self.get('tSeriesDirectory') == None:
                self._props['archived'] = False
            else:
                self._props['archived'] = \
                    len(glob.glob(os.path.join(
                        self.tSeriesDirectory, '*.archive'))) > 0

            return self.archived

    def get(self, item, default=None):
        item = self._keyDict.get(item, item)
        if item == 'filename':
            behavior_file = self.behavior_file
            if behavior_file is None:
                return None
            return os.path.splitext(behavior_file)[0] + '.pkl'

        if item in self.props.keys():
            return self.props[item]
        return self.attrib.get(item, default)

    def findall(self, arg):
        if arg == 'trial':
            return [self._trial]

    def find(self, arg):
        if arg == 'trial':
            return self._trial

    def behaviorData(self, **kwargs):
        return self._trial.behaviorData(**kwargs)

    def save(self, store=False):
        updates = {k:v for k,v in self.attrib.iteritems() if k not in
            self._attrib.keys() or v != self._attrib[k]}

        update_trial = False
        trial_args = ['behavior_file', 'mouse_name', 'start_time', 'stop_time',
            'experiment_group']

        if not store:
            print 'changes to {}: {}'.format(self.trial_id, updates)
        else:
            print 'saving changes to {}: {}'.format(self.trial_id, updates)
            for key, value in updates.iteritems():
                if key == 'trial_id':
                    raise Exception('changing trial_id')
                elif key == 'tSeries_path':
                    database.pairImagingData(value, trial_id=self.trial_id)
                elif key in trial_args and key != 'mouse_name':
                    update_trial = True
                else:
                    if value is None:
                        database.deleteTrialAttr(self.trial_id, key)
                    else:
                        if isinstance(value, dict) or isinstance(value,list):
                            value = json.dumps(value)
                        database.updateTrialAttr(self.trial_id, key, value)

            self._attrib = self.attrib.copy()
            self.attrib['mouse_name'] = \
                database.fetchMouse(self.mouse_id)['mouse_name']

        if update_trial:
            database.updateTrial(*[self.get(k) for k in trial_args],
                trial_id=self.trial_id)

    def delete(self):
        database.deleteTrial(self.trial_id)
        self._trial_id = None
        self.attrib={}
        self._props={}


class dbTrial(Trial):
    def __init__(self, parent):
        self._parent = parent
        self.attrib = parent.attrib

    @property
    def parent(self):
        return self._parent

    def get(self, arg, default=None):
        return self.parent.get(arg, default=default)
