import sima

from lab_repo.classes.sima_sequences import _SpatialFilterSequence, _SplicedSequence

import os
import sys
import shutil
import argparse

from h5py import File
import cPickle as pkl

from datetime import datetime

import align_planes

import numpy as np

from lab_repo.misc.misc import locate, locate_dir, createReplacementDirectoryName


def backup_dir(save_dir):

    param_file_path = [f for f in os.listdir(save_dir) if
                       'MCParamLog' in f]

    dir_name = os.path.dirname(save_dir)

    if param_file_path:
        param_file_path = os.path.join(save_dir, param_file_path[0])
        new_dir_name = os.path.join(dir_name,
                                    createReplacementDirectoryName(
                                        param_file_path))
    else:
        new_dir_name = os.path.join(dir_name,
                                    'mc_backup_{}'.format(
                                        datetime.today().strftime(
                                            '%Y-%m-%dT%H-%M-%S')))

    shutil.move(save_dir, new_dir_name)


def strip_sequence(dataset, seqType):

    # Remove sequence layers until we have removed the first
    # layer of the given sequence type

    seq = dataset.sequences[0]
    base = seq
    while not isinstance(base, seqType):
        parent = base
        base = base._base
    parent._base = base._base
    dataset.sequences = [seq]
    dataset.save()

    return dataset


def main(argv):

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "directory", action="store", type=str,
        help="Path to directory in which to find all h5s to correct")
    argParser.add_argument(
        "-x", "--max_x", action="store", type=str, default="",
        help="Maximum displacement in x. If integer, this is the maximum number of pixels, otherwise, if float in (0.0, 1.0], percentage of total number of x-axis pixels.")
    argParser.add_argument(
        "-y", "--max_y", action="store", type=str, default="",
        help="Maximum displacement in y. If integer, this is the maximum number of pixels, otherwise, if float in (0.0, 1.0], percentage of total number of y-axis pixels.")
    argParser.add_argument(
        "-l", "--max_levels", action="store", type=int, default=0,
        help="Number of times to downsample before motion correcting")
    argParser.add_argument(
        '-t', '--trim_criterion', action="store", type=float, default=0.95,
        help="Minimum fraction of frames for which a row/column must be" +
             " imaged to be included in corrected data")
    argParser.add_argument(
        "-i", "--ignore_channel", action="store", type=str,
        help="Channel to ignore for purposes of motion correction, ex. Ch1")
    argParser.add_argument(
        '-a', '--align_planes', action="store_true",
        help="Align the independently motion corrected planes")
    argParser.add_argument(
        '-f', '--spatial_filter', action="store_true",
        help="Gaussian blur and clip each frame before motion correcting")
    argParser.add_argument(
        "-p", "--n_processes", action="store", type=int, default=1,
        help="Number of processes to use (be considerate!)")
    argParser.add_argument(
        "-w", "--wrap", action="store_true",
        help="MC wrap existing sima sequence")
    argParser.add_argument(
        "-P", "--intensity_percentile", action="store", type=float, default=None,
        help="First only motion correct this percentile brightest" +
             " frames to build template")
    argParser.add_argument(
        "-g", "--template_weight", action="store", type=int, default=100,
        help="How heavily to weight the first-pass template")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Redo MC for previously corrected files (history is saved)")

    args = argParser.parse_args()

    if args.wrap:
        dirs = [fn for fn in locate_dir(
                '*.sima', args.directory, ignore=None)]
    else:
        dirs = [os.path.split(fn)[0] for fn in locate(
                '*.h5', args.directory, ignore=None)]

    for tseries_dir in dirs:

        print tseries_dir

        # Load sequences
        if args.wrap:
            ds = sima.ImagingDataset.load(tseries_dir)
            sequences = ds.sequences

            ch_names = ds.channel_names

            save_dir = tseries_dir

        else:

            save_dir = os.path.join(tseries_dir,
                                    os.path.basename(tseries_dir) + '.sima')

            if os.path.isdir(save_dir) and not args.overwrite:
                continue

            h5_paths = [f for f in os.listdir(tseries_dir) if f.endswith('.h5')]
            assert(len(h5_paths) == 1)
            h5_path = os.path.join(tseries_dir, h5_paths[0])

            try:

                sequence = sima.Sequence.create(
                    'HDF5', h5_path, 'tzyxc',
                    group='/', key='imaging')[:, :, :, :]

            except IOError:
                print 'Problem opening {}, skipping...'.format(h5_path)
                continue

            if sequence.shape[0] == 1:
                print 'Single Frame, skipping...'
                continue

            with File(h5_path, 'r') as h5:
                # Get channel names
                f_dataset = h5['imaging']
                ch_names = f_dataset.attrs['channel_names'].tolist()

                # Possibly mask frames
                if '/bad_frames' in h5:
                    if len(h5['/bad_frames'][0]) == 1:
                        masks = [(b[0], None, None, None)
                                 for b in h5['/bad_frames']]
                    else:
                        masks = [(b[0], b[1], None, b[2])
                                 for b in h5['/bad_frames']]
                    new_sequence = sequence.mask(masks)
                else:
                    new_sequence = sequence

            sequences = [new_sequence]

        # Always backup
        if os.path.isdir(save_dir):
            backup_dir(save_dir)

        if args.ignore_channel:
            ch_idx = [idx for idx, ch in enumerate(ch_names)
                      if ch != args.ignore_channel]
        else:
            ch_idx = range(len(ch_names))

        # set max displacements
        if args.max_x and args.max_y:
            # switch between maximum displacement type by trying to convert to int; if fails, try float, if still fails, generate error
            try:
                max_x = int(args.max_x)
            except ValueError:
                max_x = int(float(args.max_x)*sequences[0].shape[3])
            try:
                max_y = int(args.max_y)
            except ValueError:
                max_y = int(float(args.max_y)*sequences[0].shape[2])
            max_displacement = [max_y, max_x]
        elif args.max_x or args.max_y:
            raise Exception("Both maximum displacements must be specified or ommited.")
        else:
            max_displacement = None
        
        translation = sima.motion.PlaneTranslation2D(max_displacement=max_displacement,
                                                     max_levels=args.max_levels,
                                                     n_processes=args.n_processes)


        if args.spatial_filter:
            sequences = [_SpatialFilterSequence(seq) for seq in sequences]

        if args.intensity_percentile:

            intensities_path = os.path.join(tseries_dir, 'intensities.pkl')
            try:
                with open(intensities_path, 'rb') as fp:
                    intensities = pkl.load(fp)

            except IOError:
                with File(h5_path, 'r') as f:
                    intensities = [np.nanmean(frame) for frame in f['imaging']]

                with open(intensities_path, 'wb') as fw:
                    pkl.dump(intensities, fw)

            threshold = np.percentile(intensities, args.intensity_percentile)
            high_idx = list(np.where(intensities > threshold)[0])

            primary_sequences = [_SplicedSequence(seq, high_idx) for seq in sequences]

            bright_dir = os.path.join(tseries_dir,
                                      os.path.basename(tseries_dir) +
                                      '_{}' + '.sima').format(int(args.intensity_percentile))

            try:

                ds = translation.correct(
                    sima.ImagingDataset(primary_sequences, None), bright_dir,
                    channel_names=ch_names, correction_channels=ch_idx,
                    trim_criterion=args.trim_criterion)

            except IndexError:
                print 'File is too short (this may be a single image): {}, skipping'.format(h5_path)
                continue

            ds = strip_sequence(ds, _SpatialFilterSequence)

            time_averages = ds.time_averages

            shutil.copy(os.path.join(bright_dir, 'time_averages.pkl'),
                        os.path.join(os.path.dirname(bright_dir), 'bright_time_avg.pkl'))

            ds.export_averages([os.path.join(tseries_dir, 'bright_time_avg_' + n + '.tif')
                                for n in ds.channel_names])

            # Define new MC model with newly-computed template
            translation = sima.motion.PlaneTranslation2D(
                max_displacement=max_displacement,
                max_levels=args.max_levels,
                n_processes=args.n_processes,
                template=time_averages,
                template_weight=args.template_weight)

            shutil.rmtree(bright_dir)

            # Now seed motion correction with this time average as a template...

        dataset = translation.correct(
            sima.ImagingDataset(sequences, None), save_dir,
            channel_names=ch_names, correction_channels=ch_idx,
            trim_criterion=args.trim_criterion)

        if args.intensity_percentile:
            for n in dataset.channel_names:
                ta_name = 'bright_time_avg_' + n + '.tif'
                shutil.move(os.path.join(tseries_dir, ta_name), os.path.join(save_dir, ta_name))

        # Remove Spatial Filter Sequence
        if args.spatial_filter:
            dataset = strip_sequence(dataset, _SpatialFilterSequence)

        if args.align_planes:
            align_planes.main([save_dir, '-d', str(np.max([max_x, max_y]))])

        else:

            dataset.time_averages

        dataset.export_averages([os.path.join(save_dir, 'time_avg_' + n + '.tif')
                                 for n in dataset.channel_names])

        print 'Done Correcting {}'.format(save_dir)

if __name__ == '__main__':
    main(sys.argv[1:])
