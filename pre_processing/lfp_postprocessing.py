from lab_repo.misc.misc import locate
from lab_repo.misc.lfp_helpers import find_frame_times, loadEVT

import numpy as np

import argparse
import os
import cPickle as pkl


def store_frames(dirname):

    eegFile = [x for x in os.listdir(dirname) if x.endswith('eeg')]

    assert(len(eegFile) == 1)

    eegFile = os.path.join(dirname, eegFile[0])

    ft = find_frame_times(eegFile)

    save_name = os.path.join(dirname, 'frame_times.pkl')
    with open(save_name, 'wb') as fw:
        pkl.dump(ft, fw)

    print 'Frame Times Stored for {}'.format(dirname)


def store_ripples(dirname):

    try:
        ripple_file = [x for x in os.listdir(dirname) if x.endswith('rip.evt')][0]
    except IndexError:
        print 'No SWR events found for {}'.format(dirname)
        return

    ripple_file = os.path.join(dirname, ripple_file)

    ripples = loadEVT(ripple_file, 'ripple')

    # Convert ripples from ms to lfp_samples
    lfp_rate = 1250.  # hz
    conversionFactor = (1. / 1000) * lfp_rate

    for key in ripples:

        converted = np.array(ripples[key]) * conversionFactor
        ripples[key] = list(converted)

    save_name = os.path.join(dirname, 'ripples.pkl')
    with open(save_name, 'wb') as fw:
        pkl.dump(ripples, fw)

    print 'Ripples Stored for {}'.format(dirname)


def frame_file_exists(dirname):

    try:
        [x for x in os.listdir(dirname) if x == 'frame_times.pkl'][0]
    except IndexError:
        return False
    else:
        return True


def ripple_file_exists(dirname):

    try:
        [x for x in os.listdir(dirname) if x == 'ripples.pkl'][0]
    except IndexError:
        return False
    else:
        return True


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "directory", action="store", type=str, default='',
        help="Process the t-series folders contained in 'directory'")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Overwrite previous post processing results")
    args = argParser.parse_args()

    dirSet = set([os.path.split(fn)[0] for fn in locate(
        '*.eeg', args.directory)])

    # Find Frames

    for lfp_dir in dirSet:

        print lfp_dir

        if args.overwrite or not frame_file_exists(lfp_dir):

            store_frames(lfp_dir)

        if args.overwrite or not ripple_file_exists(lfp_dir):

            store_ripples(lfp_dir)
