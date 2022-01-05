import sima
import numpy as np

import argparse
import os

from lab_repo.misc.misc import locate


def merge_roi_segments(directory):

    directory = os.path.abspath(directory)
    dirs = [os.path.split(d)[0] for d in locate('rois.pkl', directory)]

    for filepath in dirs:

        ds = sima.ImagingDataset.load(filepath)

        try:
            rois = ds.ROIs['merged']
        except KeyError:
            continue

        print 'Merging {}'.format(filepath)

        im_shape = rois[0].im_shape

        new_label = 'mergedmerged'

        rois_by_label = {}
        tags_by_label = {}

        for roi in rois:

            # Strip segment labels
            name = roi.label.split('_segment')[0]
            verts = roi.coords

            if name in rois_by_label:
                # If two ROIs have the same name they'll have the same
                # tags and id, so no need to check here
                rois_by_label[name]['polygons'].extend(verts)

                tags_by_label[name]['d'].extend([float(t.strip('d_')) for t in roi.tags if t.startswith('d_')])
                tags_by_label[name]['l'].extend([float(t.strip('l_')) for t in roi.tags if t.startswith('l_')])

            else:
                rois_by_label[name] = {}
                # Take apical/basal label and order, these will be the same across rois being merged
                rois_by_label[name]['tags'] = set([t for t in roi.tags if not t.startswith('d_') and not t.startswith('l_')])
                rois_by_label[name]['id'] = roi.id
                rois_by_label[name]['label'] = name
                rois_by_label[name]['polygons'] = verts

                if len(rois_by_label[name]['tags']) != len(roi.tags):
                    tags_by_label[name] = {}
                    tags_by_label[name]['d'] = [float(t.strip('d_')) for t in roi.tags if t.startswith('d_')]
                    tags_by_label[name]['l'] = [float(t.strip('l_')) for t in roi.tags if t.startswith('l_')]

        for label in tags_by_label.keys():
            # Distance is just closest distance to soma over all rois being merged
            d = np.min(tags_by_label[label]['d'])
            # Length is distance from closest tip to tip of farthest roi, plus length of farthest roi
            farthest_point = np.max([dr + lr for dr, lr in zip(tags_by_label[label]['d'], tags_by_label[label]['l'])])
            l = farthest_point - d

            rois_by_label[label]['tags'].add('d_' + str(d))
            rois_by_label[label]['tags'].add('l_' + str(l))

        ROIs = sima.ROI.ROIList([])
        for label in rois_by_label:
            ROIs.append(sima.ROI.ROI(polygons=rois_by_label[label]['polygons'],
                                     im_shape=im_shape,
                                     tags=rois_by_label[label]['tags'],
                                     id=rois_by_label[label]['label'],  # Set ID to be label
                                     label=rois_by_label[label]['label']))

        try:
            # Every roi should either have dendrite tags or be a cell body
            assert(all([(len(r.tags) == 4) or ('_' not in r.label) for r in ROIs]))
        except AssertionError as e:
            print [(r.label, r.tags) for r in ROIs if len(r.tags) != 4]
            raise e

        try:
            assert(all([r.label.lower().startswith('cell') for r in ROIs]))
        except AssertionError as e:
            print [r.label for r in ROIs if not r.label.lower().startswith('cell')]
            raise e

        ds.add_ROIs(ROIs, label=new_label)


if __name__ == '__main__':

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-d", "--directory", action="store", type=str, default='',
        help="Process the t-series folders contained in 'directory'")
    args = argParser.parse_args()

    merge_roi_segments(args.directory)
