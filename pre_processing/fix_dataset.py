"""Fix dataset.pkl"""

import os
import cPickle as pickle
import argparse
import itertools as it
import numpy as np
import shutil
import datetime
import shapely
import re
import glob

import sima
from sima.sequence import _MotionCorrectedSequence
from sima.motion import MotionEstimationStrategy
from sima.motion.motion import _trim_coords
#
# Fix functions
#

def get_backup_path(path):
    date_stamp = datetime.date.today().strftime('%Y%m%d')
    backup_int = 0
    backup_path = '{}.{}.{}.bak'.format(path, date_stamp, backup_int)

    while os.path.exists(backup_path):
        backup_int += 1
        backup_path = '{}.{}.{}.bak'.format(path, date_stamp, backup_int)

    return backup_path

def backup_sima_dir(sima_dir):
    backup_dir = get_backup_path(sima_dir)

    os.mkdir(backup_dir)
    shutil.copy2(os.path.join(sima_dir, 'dataset.pkl'),
                 os.path.join(backup_dir, 'dataset.pkl'))
    try:
        shutil.copy2(os.path.join(sima_dir, 'time_averages.pkl'),
                     os.path.join(backup_dir, 'time_averages.pkl'))
    except (OSError, IOError):
        pass
    try:
        shutil.copy2(os.path.join(sima_dir, 'sequences.pkl'),
                     os.path.join(backup_dir, 'sequences.pkl'))
    except (OSError, IOError):
        pass
    try:
        shutil.copy2(os.path.join(sima_dir, 'rois.pkl'),
                     os.path.join(backup_dir, 'rois.pkl'))
    except (OSError, IOError):
        pass
    for f in glob.glob(os.path.join(sima_dir, 'time_avg_*.tif')):
        shutil.copy2(
            f, os.path.join(backup_dir, os.path.split(f)[1]))



def make_nonnegative(path, trim_criterion=None, no_action=False, backup=False,
                     recalc_averages=True):
    """Updates displacements of MC sequences to ensure they are non-negative.

    Make sure to recalc averages at some point, if not now. They should
    always match the dataset.

    """

    sima_dir = os.path.split(path)[0]
    dataset = sima.ImagingDataset.load(sima_dir)

    def _make_nonnegative(sequences):
            updated = False
            if all([isinstance(s, _MotionCorrectedSequence)
                    for s in sequences]):
                displacements = [s.displacements for s in sequences]
                if any([np.min(disp) != 0 for disp in displacements]):
                    new_displacements = \
                        MotionEstimationStrategy._make_nonnegative(
                            displacements)
                    max_disp = max_displacements(new_displacements)
                    for sequence, displacement in it.izip(
                            sequences, new_displacements):
                        sequence.displacements = displacement
                        sequence._frame_shape_zyx = tuple(
                            sequence._base.shape[1:4] + max_disp)
                    updated = True
            elif any([isinstance(s, _MotionCorrectedSequence)
                      for s in sequences]):
                print(
                    "Sequences do not line up, unable to update displacements")

            try:
                bases = [s._base for s in sequences]
            except AttributeError:
                return updated
            else:
                return _make_nonnegative(bases) or updated

    updated = _make_nonnegative(dataset.sequences)

    if updated:
        print "Displacements now non-negative:", path

        if trim_criterion is not None:
            remove_old_trim(dataset)
            planes, rows, columns = get_new_slice(dataset, trim_criterion)
            dataset.sequences = [s[:, planes, rows, columns]
                                 for s in dataset.sequences]
            dataset._frame_shape = dataset.sequences[0].shape[1:]
            print "Trim bounds updated:", path

        if not no_action:
            if backup:
                backup_sima_dir(sima_dir)
            dataset.save()
            if recalc_averages:
                recalc_averages(dataset)


def update_time_averages(path):
    """Updates time_averages.pkl. This does not update the time average TIFFs
    files. It will not work if the dataset is a 0.x dataset. There is not a
    'no_action' or 'backup' option. Live with the consequences.

    """

    dataset = sima.ImagingDataset.load(path)
    dataset.time_averages


def export_averages(path, no_action=False):
    """Just re-exports time averages, calcs if needed, but does not re-calc"""

    dataset = sima.ImagingDataset.load(path)
    if not args.no_action:
        dataset.export_averages(
            [os.path.join(path, 'time_avg_' + n + '.tif')
             for n in dataset.channel_names])
    print "Time averages exported: {}".format(path)



#
# Helper functions
#


def max_displacements(displacements):
    disp_dim = displacements[0].shape[-1]
    max_disp = np.max(list(it.chain.from_iterable(d.reshape(-1, disp_dim)
                           for d in displacements)),
                      axis=0)

    if len(max_disp) == 2:  # if 2D displacements
        max_disp = np.array([0, max_disp[0], max_disp[1]])

    return max_disp


def save_dataset(path, dataset, backup=False):

    sima_dir = os.path.split(path)[0]
    if backup:
        backup_sima_dir(sima_dir)

    # Write the new dataset.pkl and update the time_averages
    pickle.dump(dataset, open(path, 'wb'), pickle.HIGHEST_PROTOCOL)
    sima_dataset = sima.ImagingDataset.load(sima_dir)

    recalc_averages(sima_dataset)


def save_sequences(path, sequences, backup=False):

    sima_dir = os.path.split(path)[0]
    if backup:
        backup_sima_dir(sima_dir)

    # Write the new sequences.pkl and update the time_averages
    pickle.dump(sequences, open(os.path.join(sima_dir, 'sequences.pkl', 'wb')),
                pickle.HIGHEST_PROTOCOL)
    sima_dataset = sima.ImagingDataset.load(sima_dir)

    recalc_averages(sima_dataset)


def recalc_averages(dataset):

    sima_dir = dataset.savedir
    try:
        os.remove(os.path.join(sima_dir, 'time_averages.pkl'))
    except OSError:
        pass
    dataset.export_averages(
        [os.path.join(sima_dir, 'time_avg_' + n + '.tif')
         for n in dataset.channel_names])
