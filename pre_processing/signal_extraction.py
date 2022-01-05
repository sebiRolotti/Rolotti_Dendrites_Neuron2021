"""Losonczy Lab signal extraction script"""

import os
import warnings
import argparse
import sys
from datetime import datetime

import matplotlib as mpl
mpl.use('pdf')

from sima import ImagingDataset

from lab_repo.classes.dbclasses import dbExperiment

from merge_roi_segments import merge_roi_segments

EXCLUDED_ROI_LABELS = ('auto', 'ignore', 'test', 'junk', 'bad')


def mtime(path):

    modtime_in_seconds = os.path.getmtime(path)
    modtime_as_datetime = datetime.fromtimestamp(modtime_in_seconds)
    modtime_string = datetime.strftime(
        modtime_as_datetime, '%Y-%m-%d-%Hh%Mm%Ss')

    return modtime_string


def labels_to_extract(
        dataset, signal_channel='Ch2', overwrite=False, label=None):
    """Return a list of the roi labels to be extracted"""
    labels = []
    for roi_label in dataset.ROIs:
        if (label is not None and roi_label == label) \
                or (label is None
                    and not any(
                        [bad in roi_label for bad in EXCLUDED_ROI_LABELS])
                    and not roi_label.startswith('_')
                    and len(dataset.ROIs[roi_label])):
            if overwrite or roi_label not in dataset.signals(signal_channel):
                labels.append(roi_label)
            else:
                roi_time = dataset.ROIs[roi_label].timestamp
                signals_time = dataset.signals(
                    signal_channel)[roi_label]['timestamp']
                dataset_time = mtime(
                    os.path.join(dataset.savedir, 'dataset.pkl'))
                sequences_time = mtime(
                    os.path.join(dataset.savedir, 'sequences.pkl'))
                if any(t > signals_time for t in
                       (roi_time, dataset_time, sequences_time)):
                    labels.append(roi_label)
    return labels


def datasets_to_extract(search_directory):
    for directory, folders, files in os.walk(search_directory):
        if directory.endswith('.sima'):
            try:
                dataset = ImagingDataset.load(directory)
            except IOError:
                continue
            else:
                yield dataset


def extract_dataset(dataset, signal_channel='Ch2', demix_channel='Ch1',
                    overwrite=False, include_overlap=False, label=None,
                    n_processes=1):
    try:
        demix_channel = dataset._resolve_channel(demix_channel)
    except ValueError:
        demix_channel = None
    for roi_label in labels_to_extract(dataset, signal_channel, overwrite,
                                       label):
        print("Extracting label '{}' from {}".format(roi_label,
                                                     dataset.savedir))
        dataset.extract(rois=dataset.ROIs[roi_label],
                        label=roi_label,
                        signal_channel=signal_channel,
                        demix_channel=demix_channel,
                        remove_overlap=not include_overlap,
                        n_processes=n_processes)

        print('Extraction complete: {}'.format(dataset.savedir))


def main(argv):
    """Find all imaging data within the directory and extract signals."""

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "-d", "--demix_channel", action="store", type=str,
        help="Name of the channel to demix from the signal channel")
    argParser.add_argument(
        "-i", "--include_overlap", action="store_true",
        help="Include pixels that overlap between ROIs")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="Re-extract datasets with pre-existing signals files")
    argParser.add_argument(
        "-s", "--signal_channel", action="store", type=str, default="Ch2",
        help="Name of the signal channel to extract, defaults to \'Ch2\'")
    argParser.add_argument(
        "-l", "--label", action="store", type=str,
        help="Label to extract, defaults to all valid labels")
    argParser.add_argument(
        "-p", "--processes", action="store", type=int, default=1,
        help="Number of processes to pool the extraction across."
        + " Don't go too crazy.")
    argParser.add_argument(
        "directory", action="store", type=str, default=os.curdir,
        help="Locate all datasets below this folder. If integer, use \
              tSeriesDirectory for corresponding trial id")

    args = argParser.parse_args(argv)

    warnings.simplefilter("ignore", RuntimeWarning)

    directory = args.directory
    try:
        int(directory)
    except ValueError:
        pass
    else:
        directory = dbExperiment(int(directory)).tSeriesDirectory

    if args.label in ['merged', 'merge', 'ROIs']:
        merge_roi_segments(directory)

        for dataset in datasets_to_extract(directory):
            extract_dataset(
                dataset, signal_channel=args.signal_channel,
                demix_channel=args.demix_channel, overwrite=args.overwrite,
                include_overlap=args.include_overlap, label='mergedmerged',
                n_processes=args.processes)

    else:

        for dataset in datasets_to_extract(directory):
            extract_dataset(
                dataset, signal_channel=args.signal_channel,
                demix_channel=args.demix_channel, overwrite=args.overwrite,
                include_overlap=args.include_overlap, label=args.label,
                n_processes=args.processes)

if __name__ == '__main__':
    main(sys.argv[1:])
