"""Calculate the planewise correspondence between a Z-stack and T-series.

Use this transform to generate T-series rois from Z-traces.
"""
import os
from os.path import join
import argparse
import glob

import warnings


def custom_formatwarning(msg, category, *a):
    """Ignore everything except the message and warning type."""
    return str(category.__name__) + ': ' + str(msg) + '\n'

warnings.formatwarning = custom_formatwarning

import cPickle
import zipfile

import numpy as np
from scipy.ndimage import binary_dilation
import skimage.measure
from sympy import Plane, Point3D

from sima.misc.imagej import read_roi
from sima.ROI import ROI, ROIList, mask2poly
from sima import ImagingDataset

from lab_repo.misc.registration_helpers import get_plane_spacing, get_z_buffer, \
    directory_pairs_to_process, volume_pattern_pairs_to_process
from lab_repo.misc.misc import get_element_size_um, get_prairieview_version
from parse_swc import parse_swc

# Segments overlap by one pixel at branch points.
# When drawing ROIs, ignore branch points as well as any points within
# BRANCH_BUFFER pixels of the branch point
BRANCH_BUFFER = 4

# Number of microns above and below T plane through Z space
# that segments should be included and projected
PLANE_BUFFER = 6

# When forming the ROIs, perform recursive binary dilations on the
# linear trajectory through pixel space.
N_DILATIONS = 3


def points_needed_for_method(method):
    """Return the number of points per plane needed for each method."""
    if method == 'linear':
        return 3
    elif method == 'affine':
        return 4


def read_point_roi_zip(filename):
    """Read an ImageJ single point ROI zip set and parse each ROI individually.

    Parameters
    ----------
    filename : string
        Path to the ImageJ ROis zip file

    Returns
    -------
    roi_list : list
        List of point coordinates

    """
    roi_list = []
    with zipfile.ZipFile(filename) as zf:
        for name in zf.namelist():
            roi = read_roi(zf.open(name))
            if roi is None:
                continue
            roi_list.append(roi)
        return np.array(roi_list)


def reg_points(pair, method='affine', Z_buffer=None,
               Z_plane_spacing=None, T_plane_spacing=None):
    """Read in two ImageJ point sets and calculates the tranform between them.

    Parameters
    ----------
    pair : list
        Pair of T and Z series paths
    method : str
        Method used to calculate correspondence between anchor segments
        'affine' or 'linear'
    Z_buffer : float
        Distance in microns top Z plane is above top T plane
    Z_plane_spacing : float
        Distance in microns between adjacent Z planes
    T_plane_spacing : float
        Distance in microns between adjacent T planes

    Returns
    -------
    transforms : list
        np.array transformation matrices A: Z->T
    planes : list
        sympy Plane objects representing T-planes embedded in Z-space

    """
    T_path = join(pair[0], 'RoiSet.zip')
    Z_path = pair[1]

    T_xml_dirname = os.path.dirname(os.path.dirname(T_path))
    T_xml_path = join(T_xml_dirname, os.path.basename(T_xml_dirname) + '.xml')

    Z_xml_dirname = os.path.dirname(Z_path)
    Z_xml_path = join(Z_xml_dirname, os.path.basename(Z_xml_dirname) + '.xml')

    if not Z_buffer:
        Z_buffer = get_z_buffer(T_xml_path, Z_xml_path)
        print 'Z-Buffer: {}'.format(str(Z_buffer))
    if not T_plane_spacing:
        T_plane_spacing = get_plane_spacing(T_xml_path)
        print 'T-Plane Spacing: {}'.format(str(T_plane_spacing))
    if not Z_plane_spacing:
        Z_plane_spacing = get_plane_spacing(Z_xml_path)
        print 'Z-Plane Spacing: {}'.format(str(Z_plane_spacing))

    pv_version = get_prairieview_version(T_xml_path)

    [T_y_spacing, T_x_spacing] = get_element_size_um(T_xml_path,
                                                     pv_version)[-2:]
    [Z_y_spacing, Z_x_spacing] = get_element_size_um(Z_xml_path,
                                                     pv_version)[-2:]

    # Coordinates should be found in TZ pairs
    coords = read_point_roi_zip(T_path)

    # Figure out whether T or Z was drawn first
    # by leveraging fact that z stack z-coordinate is always higher

    Z_ind = np.where(coords[:2, 2] == np.max(coords[:2, 2]))[0][0]
    T_ind = 1 - Z_ind

    T_coords = coords[T_ind::2, :]
    Z_coords = coords[Z_ind::2, :]

    n_points = T_coords.shape[0]
    n_points_per_plane = points_needed_for_method(method)

    assert(Z_coords.shape[0] == n_points)
    assert(n_points % n_points_per_plane == 0)

    # Transform coordinates into real space
    T_coords[:, 0] = T_coords[:, 0] * T_x_spacing
    T_coords[:, 1] = T_coords[:, 1] * T_y_spacing
    T_coords[:, 2] = T_coords[:, 2] * T_plane_spacing + Z_buffer

    Z_coords[:, 0] = Z_coords[:, 0] * Z_x_spacing
    Z_coords[:, 1] = Z_coords[:, 1] * Z_y_spacing
    Z_coords[:, 2] = Z_coords[:, 2] * Z_plane_spacing

    transforms = []
    planes = []

    for plane_start in xrange(0, n_points, n_points_per_plane):

        plane_stop = plane_start + n_points_per_plane
        T = T_coords[plane_start:plane_stop, :].T
        Z = Z_coords[plane_start:plane_stop, :].T

        if method == 'linear':
            # T = A*Z
            transform = np.dot(T, np.linalg.inv(Z))

            z_points = [Point3D(coord) for coord in Z.T]
            z_plane = Plane(*z_points)

        elif method == 'affine':
            Tpad = np.vstack([T, np.ones((1, n_points_per_plane))])
            Zpad = np.vstack([Z, np.ones((1, n_points_per_plane))])

            try:

                transform = np.dot(Tpad, np.linalg.inv(Zpad))

                # Best fit plane conains centroid and has normal vector
                # Equal to smallest left eigenvector of point cloud
                centroid = np.mean(Z, axis=1, keepdims=True)
                Z_centered = Z - centroid
                U, _, _ = np.linalg.svd(Z_centered)

                z_plane = Plane(Point3D(centroid.squeeze()), normal_vector=U[:, -1])

            except np.linalg.LinAlgError:

                # If points were picked badly for a plane there may be no
                # way to invert Z - note that this is a convenient way to
                # dismiss a badly collected plane during point dropping.
                transform = None
                z_plane = None
                print 'Undefined transformation for plane ' + str(len(planes))

        transforms.append(transform)
        planes.append(z_plane)

    return transforms, planes


def generate_rois(pair, transforms, planes, method='affine',
                  Z_plane_spacing=None, T_plane_spacing=None):
    """Create sima.ROIs from traces and precomputed Z->T transforms.

    Parameters
    ----------
    pair : list
        Pair of T and Z series paths
    transforms : list
        np.array transformation matrices A: Z->T
        List should be either len 1 or len num_planes
    planes : list
        sympy Plane objects representing T-planes embedded in Z-space
        Must be same length as transforms
    method : str
        Method used to calculate correspondence between anchor segments
        'affine' or 'linear'
    Z_plane_spacing : float
        Distance in microns between adjacent Z planes
    T_plane_spacing : float
        Distance in microns between adjacent T planes

    Returns
    -------
    rois : sima.ROIList

    """
    [t_vol_path, z_vol_path] = pair

    if not os.path.exists(os.path.join(z_vol_path, 'traces')):
        return

    cell_ids = os.listdir(os.path.join(z_vol_path, 'traces'))

    t_vol = os.path.join(t_vol_path, 'time_averages.pkl')
    with open(t_vol, 'rb') as f:
        t_vol = cPickle.load(f)[..., -1]

    n_planes = t_vol.shape[0]

    T_xml_dirname = os.path.dirname(t_vol_path)
    T_xml_path = join(T_xml_dirname, os.path.basename(T_xml_dirname) + '.xml')

    Z_xml_dirname = os.path.dirname(z_vol_path)
    Z_xml_path = join(Z_xml_dirname, os.path.basename(Z_xml_dirname) + '.xml')
    pv_version = get_prairieview_version(Z_xml_path)
    [Z_y_spacing, Z_x_spacing] = get_element_size_um(Z_xml_path,
                                                     pv_version)[-2:]

    if not T_plane_spacing:
        T_plane_spacing = get_plane_spacing(T_xml_path)
    if not Z_plane_spacing:
        Z_plane_spacing = get_plane_spacing(Z_xml_path)

    px_to_micron = np.array([Z_x_spacing, Z_y_spacing, Z_plane_spacing])

    # If only a single plane/transform was supplied, reuse transform
    # and generate other planes by translating through Z by T-step
    assert(len(transforms) == len(planes))

    if len(transforms) == 1:
        transforms = transforms * n_planes

        for i in xrange(1, n_planes):
            tmp_pt = np.array(planes[i - 1].p1).astype(float)
            # Translate z-coordinate
            tmp_pt[2] = tmp_pt[2] + T_plane_spacing
            planes.append(Plane(Point3D(tmp_pt), planes[0].normal_vector))

    rois = []
    roi_original_pts = {}

    for cell_id in cell_ids:
        # One .swc file per primary dendrite tree
        swc_files = glob.glob(
            os.path.join(z_vol_path, 'traces', cell_id, '*.swc'))

        for swc_file in swc_files:

            # NOTE: naming convention, all dends are assumed basal
            # unless apical is present in the swc-file name
            swc_base = os.path.basename(swc_file)
            if 'apical' in swc_base:
                apical_or_basal = 'apical'
                primary_branch_id = len(swc_files)
            else:
                apical_or_basal = 'basal'
                # Assume format standard_name_bID.swc
                primary_branch_id = swc_base.rsplit('b', 1)[1].rstrip('.swc')

            branches = parse_swc(swc_file, str(primary_branch_id))

            for branch in branches:

                # Length of dendrite up to each point
                real_coords = branch.coords * px_to_micron
                cum_dist = np.cumsum(np.linalg.norm(
                                     real_coords[1:] - real_coords[:-1],
                                     axis=1)).tolist()
                cum_dist.insert(0, 0)

                # Exclude pixels near the branch points (within 2 pixels)
                crop_idx = slice(1 + BRANCH_BUFFER, -1 * (1 + BRANCH_BUFFER))
                cropped_branch_coords = real_coords[crop_idx, :]
                cum_dist = cum_dist[crop_idx]

                if not len(cropped_branch_coords):
                    warnings.warn(
                        'Branch {} too short to transform'.format(branch.id))
                    continue

                mask = np.zeros(t_vol.shape)
                # Keep track of distance for every branch coordinate mapped
                # to plane. Needs to be a list in case multiple coordinates
                # are mapped to the same point
                dist_mask = [[[[] for x in range(t_vol.shape[2])]
                             for y in range(t_vol.shape[1])]
                             for z in range(t_vol.shape[0])]

                point_correspondence = []

                # Project traces in Z space onto nearby planes, then
                # Transform these points to T space with precomputed affine

                for coord, cdist in zip(cropped_branch_coords, cum_dist):
                    z_point = Point3D(coord.T)

                    for plane_idx, (z_plane, transform) in enumerate(
                            zip(planes, transforms)):

                        if transform is None:
                            continue

                        elif z_plane.distance(z_point) < PLANE_BUFFER:

                            z_proj = np.array(z_plane.projection(
                                              z_point)).astype(float)

                            # Pad point for affine representation
                            if method == 'affine':
                                z_proj = np.hstack([z_proj, 1.])

                            # Transform to T-space and remove padding
                            t_point = np.dot(transform, z_proj)[:3]

                            t_point = np.divide(t_point,
                                                px_to_micron).astype(int)

                            # Swap x and y for sima compliance
                            try:
                                mask[plane_idx, t_point[1], t_point[0]] = 1
                                dist_mask[plane_idx][t_point[1]][t_point[0]].append(cdist)

                                t_point = tuple([t_point[0], t_point[1],
                                                 plane_idx])
                                point_correspondence.append([t_point,
                                                             np.divide(coord, px_to_micron)])

                            except IndexError:
                                pass

                            # Points only get projected to one plane
                            break

                dist_mask = np.array(dist_mask)

                for plane_idx in xrange(mask.shape[0]):

                    for dilation_idx in range(N_DILATIONS):
                        mask[plane_idx] = binary_dilation(mask[plane_idx])

                    # Each connected components gets its own roi and dist
                    connected_components = skimage.measure.label(mask[plane_idx])
                    for cc_idx in xrange(1, np.max(connected_components) + 1):

                        cc = np.zeros(t_vol.shape[1:]).astype(int)
                        cc[np.where(connected_components == cc_idx)] = 1

                        roi_mask = np.zeros(t_vol.shape)
                        roi_mask[plane_idx] = cc

                        included_dists = dist_mask[plane_idx, cc.astype(bool)]
                        included_dists = [d for px in included_dists.flatten()
                                          for d in px]

                        included_pts = []
                        for pair in point_correspondence:
                            tpt, zpt = pair

                            # If t_point was included in this roi,
                            # save out the corresponding point on original tree
                            # in z-space

                            if roi_mask[tpt[2]][tpt[1], tpt[0]] == 1:

                                included_pts.append(zpt)

                        cum_len = np.mean(included_dists)

                        polys = mask2poly(roi_mask)

                        label = cell_id.replace('_', '') + '_' + branch.id + \
                            '_segment_' + str(plane_idx) + '_' + str(cc_idx)
                        tags = ['o_{}'.format(str(branch.order)),
                                'l_{:1.4f}'.format(cum_len),
                                'd_{:1.4f}'.format(branch.distance_to_soma()),
                                apical_or_basal]

                        rois.append(ROI(
                            polygons=polys, label=label, tags=tags,
                            im_shape=t_vol.shape))

                        roi_original_pts[label] = included_pts

    imset = ImagingDataset.load(t_vol_path)
    imset.add_ROIs(ROIList(rois), label='auto')

    original_pt_file = os.path.join(t_vol_path, 'original_pts.pkl')
    with open(original_pt_file, 'wb') as fpw:
        cPickle.dump(roi_original_pts, fpw)

    return rois


if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-d", "--directory", action="store", type=str, default='',
        help="Process the t-series folders contained in 'directory'")
    argParser.add_argument(
        "-v", "--volume", action="store", type=str,
        help="Path to sima folder of structural volume")
    argParser.add_argument(
        "-f", "--functional_pattern", action="store", type=str,
        help="Pattern for T series sima paths to match used in glob")
    argParser.add_argument(
        "-b", "--z_buffer", action="store", type=float, default=None,
        help="Distance in microns top Z plane is above top T plane."
              "Distance is inferred from xmls if not given.")
    argParser.add_argument(
        "-z", "--z_step", action="store", type=float, default=None,
        help="Distance in microns between Z planes."
              "Distance is inferred from xml if not given.")
    argParser.add_argument(
        "-t", "--t_step", action="store", type=float, default=None,
        help="Distance in microns between T planes."
              "Distance is inferred from xml if not given.")
    argParser.add_argument(
        "-m", "--method", action="store", type=str, default="affine",
        help="Method for calculating point cloud registration transform")
    args = argParser.parse_args()

    if args.directory:
        directory = os.path.abspath(args.directory)
        pairs_to_process = directory_pairs_to_process(directory)

    elif args.volume and args.functional_pattern:

        pairs_to_process = volume_pattern_pairs_to_process(
            args.volume, args.functional_pattern)

    else:
        raise ValueError(
            'Pass in either a directory or a volume and functional pattern.')

    for pair in pairs_to_process:

        print pair[0]

        try:
            transforms, planes = reg_points(pair, method=args.method,
                                            Z_buffer=args.z_buffer,
                                            Z_plane_spacing=args.z_step,
                                            T_plane_spacing=args.t_step)
        except IOError as e:
            print e
            continue
        else:

            generate_rois(pair, transforms, planes, method=args.method,
                          Z_plane_spacing=args.z_step, T_plane_spacing=args.t_step)
