import os
import re
import sys
import numpy as np
import argparse

from sima import ImagingDataset
from sima.motion.frame_align import align_cross_correlation
# a script to register the planes of a multi-plane sima ImagingDatasets which
# have been individually corrected using in-plane, 2D motion correction.

def align_planes(path, channel=None, max_displacement=None):
    ds = ImagingDataset.load(path)

    # use the alignment cross correlation from sima to figure out the
    # between plane corrections
    time_averages = ds.time_averages
    if channel is not None:
        channel = ds._resolve_channel(channel)
        time_averages = time_averages[..., channel]
    corrs = [np.array([0, 0])]
    if max_displacement is not None:
        disp_bounds = (
            [-max_displacement] * 2, [max_displacement] * 2)
    else:
        disp_bounds = None

    cum_disp = np.zeros((2,))

    for i in xrange(len(ds.time_averages) - 1):

        p1 = np.clip(ds.time_averages[i], 0, np.nanpercentile(ds.time_averages[i], 99))
        p2 = np.clip(ds.time_averages[i+1], 0, np.nanpercentile(ds.time_averages[i+1], 99))

        new_disps = align_cross_correlation(p1, p2, displacement_bounds=disp_bounds)[0]

        cum_disp += new_disps

        corrs.append(cum_disp.copy())

    corrs = np.array(corrs)

    # if maximum correlation is at 0, exit without doing any further
    # processing
    if not np.any(corrs):
        print 'no alignment necessary %s' % path
        return

    # shift corrections so all displacements will be > 0
    if np.any(corrs[:,0] < 0):
        corrs[:,0] -= np.min(corrs[:,0])
    if np.any(corrs[:,1] < 0):
        corrs[:,1] -= np.min(corrs[:,1])

    # build (t, z, 2) sized array for storing displacements
    displacements = np.tile(np.expand_dims(corrs, 0), (ds.num_frames, 1, 1))

    # in order to use sequences setter all sequences must be set at the same
    # time. trim pixels to remove the overhaning planes before re setting
    # the sequences
    max_x = int(np.max(displacements[..., 0]))
    max_y = int(np.max(displacements[..., 1]))
    adjusted_sequences = []
    for i in xrange(ds.num_sequences):
        adjusted_sequence = ds.sequences[i].apply_displacements(
            displacements.astype('int64'))
        adjusted_sequences.append(adjusted_sequence)
    ds.sequences = adjusted_sequences

    # delete the old time_averages.pkl file to force recalculation of time
    # average
    try:
        os.remove(os.path.join(ds.savedir, 'time_averages.pkl'))
    except:
        pass

    for i in xrange(ds.num_sequences):
        ds.sequences[i] = ds.sequences[i]
    print 'alignment complete, calculating time average ...'
    new_time_average = ds.time_averages

    ds.save()

    # Export tifs
    ds.export_averages([os.path.join(path, 'time_avg_' + n + '.tif')
                             for n in ds.channel_names])

    print 'complete %s' % path


def main(argv):
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        '-c', '--channel', action='store', type=str, default=None,
        help='channel to perform alignment on, default is to use all channels')
    argParser.add_argument(
        '-d', '--max_displacement', action='store', type=int, default=None,
        help='maximum displacement allowed for alignment')
    argParser.add_argument(
        'sima_path', action='store', type=str,
        help='.sima folder or parent directory of folders to align')
    args = argParser.parse_args(argv)

    directory = args.sima_path
    paths = []
    if not re.search('.sima$', directory):
        for dirpath, dirnames, filenames in os.walk(directory):
            paths.extend(
                map(lambda f: os.path.join(dirpath, f),
                filter(lambda f: re.search('.sima$', f), dirnames)))
    else:
        paths = [directory]

    for path in paths:
        align_planes(path, channel=args.channel, max_displacement=args.max_displacement)

if __name__ == '__main__':
    main(sys.argv[1:])
