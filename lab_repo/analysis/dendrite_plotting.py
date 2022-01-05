import matplotlib.pyplot as plt

from shapely.geometry import Polygon, LineString
from shapely.geometry.point import Point

from descartes import PolygonPatch

import numpy as np

import os
import glob

import cPickle as pkl

from lab_repo.misc.registration_helpers import parse_swc
from lab_repo.misc.misc import get_element_size_um, get_prairieview_version


def get_microns(z_dir, plane_spacing):

    z_xml_dirname = os.path.dirname(z_dir)
    z_xml = os.path.join(z_xml_dirname,
                         os.path.basename(z_xml_dirname) + '.xml')

    pv_version = get_prairieview_version(z_xml)
    [y_spacing, x_spacing] = get_element_size_um(z_xml, pv_version)[-2:]

    px_to_micron = np.array([x_spacing, y_spacing, plane_spacing])

    return px_to_micron


def build_len_dict(swc_files, px_to_micron):

    len_dict = {}

    for swc_file in swc_files:

        # NOTE: naming convention, all dends are assumed basal
        # unless apical is present in the swc-file name
        if 'apical' in swc_file:
            primary_branch_id = str(len(swc_files))
        else:
            # Assume format standard_name_bID.swc
            primary_branch_id = os.path.basename(swc_file).replace('b', '.').split('.')[1]

        branches = parse_swc(swc_file, primary_branch_id)

        for branch in branches:

            # Length of branch
            real_coords = branch.coords * px_to_micron
            length = np.sum(np.linalg.norm(
                            real_coords[1:] - real_coords[:-1],
                            axis=1)).tolist()

            len_dict[branch.id] = length

    return len_dict


def num_branches(cell_id, label_list):
    return len([x for x in label_list
                if x == cell_id or x.startswith(cell_id + '_')])


def num_children(label, label_list):
    return len([x for x in label_list
                if x.startswith(label + '_') and not x == label])


def get_parent(label):
    return label[:-2]


def get_children(label, label_list):
    return [x for x in label_list
            if x.startswith(label) and len(x) == len(label) + 2]


def num_generations(label, label_list):

    farthest_node = max([x for x in label_list
                         if x.startswith(label)], key=len)

    current_generation = label.count('_')
    farthest_generation = farthest_node.count('_')

    return farthest_generation - current_generation


def get_alpha(label, label_list):
    return num_generations(label, label_list) + 1


def get_proportions(labels, label_list):
    alphas = [get_alpha(x, label_list) for x in labels]
    N = float(sum(alphas))
    return [x / N for x in alphas]


def get_extents(current_gen, label_list, range_dict, center_dict):

    parent = get_parent(current_gen[0])
    if parent:
        parent_arc = range_dict[parent]
        parent_arclength = parent_arc[1] - parent_arc[0]
        proportions = get_proportions(current_gen, label_list)
    else:
        # If these are the primary branches, do a bit of setup
        # First move last branch to beginning (i.e. start with 'apical')
        current_gen.insert(0, current_gen.pop())
        proportions = get_proportions(current_gen, label_list)
        # First arc should start 1/2 of proportion of full circle
        # away from straight down
        parent_arc = [1.5 * np.pi + proportions[0] * np.pi] * 2
        parent_arclength = 2 * np.pi

    arc_stop = parent_arc[0]
    for child, prop in zip(current_gen, proportions):
        arc_start = arc_stop
        arc_stop = arc_start + prop * parent_arclength
        range_dict[child] = [arc_start, arc_stop]

        next_gen = get_children(child, label_list)
        if next_gen:
            # If there are children, determine their ranges
            get_extents(next_gen, label_list, range_dict, center_dict)

            # Center of current node is betwen centers of furthest children
            center_dict[child] = np.mean([center_dict[next_gen[0]],
                                          center_dict[next_gen[-1]]])
        else:
            # We're at a leaf, center is just in the middle of the range
            center_dict[child] = np.mean(range_dict[child])


def _prepare_schematic(cell_id, z_dir, px_to_micron=None, plane_spacing=2.,
                       spacing='num_generations'):

    if not px_to_micron:
        px_to_micron = get_microns(z_dir, plane_spacing)

    swc_files = glob.glob(
        os.path.join(z_dir, 'traces', cell_id, '*.swc'))

    # Get length and id of every branch segment
    len_dict = {}
    len_dict = build_len_dict(swc_files, px_to_micron)

    # Get range and centroid subtended by each segment
    # In format CW-centroid-CCW in radians
    branch_ids = sorted(len_dict.keys())
    range_dict = {}
    center_dict = {}

    n_primaries = len(swc_files)
    primary_ids = [str(i) for i in xrange(1, n_primaries + 1)]

    get_extents(primary_ids, branch_ids, range_dict, center_dict)

    # For ease of book-keeping, generate a dict with the starting
    # radial distance for each segment
    start_dict = {}
    for i in xrange(1, n_primaries + 1):
        start_dict[str(i)] = 0

    for branch_id in branch_ids:
        if branch_id not in start_dict:

            parent = branch_id[:-2]

            start_dict[branch_id] = start_dict[parent] + len_dict[parent]

    return len_dict, range_dict, center_dict, start_dict


def _plot_prepped_schematic(len_dict, range_dict, center_dict,
                            start_dict, ax=None, color_dict=None):

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        show_plot = True
    else:
        show_plot = False

    # Now we have all the info we need to plot

    branch_ids = sorted(len_dict.keys())
    rmax = int(np.ceil(np.max([len_dict[b] + start_dict[b]
                               for b in branch_ids])))
    # We'll draw a soma that is 10% the size of rmax
    soma_r = int(rmax * 0.1)
    circle_theta = np.arange(0, 2 * np.pi, 0.1)
    circle_r = [soma_r] * len(circle_theta)
    ax.plot(circle_theta, circle_r, '0.25', lw=1.5)

    # Move all starts out by this amount
    start_dict = {k: v + soma_r for k, v in start_dict.items()}

    for branch_id in branch_ids:
        # Draw radial segment
        try:
            color = color_dict[branch_id]
        except (TypeError, KeyError):
            # ToDo: Option to make un-colored branches dashed
            color = '0.5'

        ax.plot([center_dict[branch_id]] * 2, [start_dict[branch_id],
                                               start_dict[branch_id] +
                                               len_dict[branch_id]],
                color=color, lw=1.5)
        # Draw segment connecting to parent
        parent = branch_id[:-2]
        if parent in range_dict:
            ax.plot(np.linspace(center_dict[branch_id],
                                center_dict[parent], 100),
                    np.ones(100) * start_dict[branch_id],
                    color=color, lw=1)

    rmax = rmax + soma_r
    rmax += int(np.ceil(0.04 * rmax))

    ax.set_thetagrids([])
    ax.set_rmax(rmax)
    ax.set_rticks(range(soma_r, rmax, 50))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if show_plot:
        plt.show()
    else:
        return ax


# High level function that should be called:
def plot_schematic(cell_id, z_dir, ax=None,
                   px_to_micron=None, plane_spacing=2.,
                   colors=None, spacing='num_generations'):

    len_dict, range_dict, center_dict, start_dict = \
        _prepare_schematic(cell_id, z_dir, px_to_micron,
                           plane_spacing, spacing=spacing)

    ax = _plot_prepped_schematic(len_dict, range_dict, center_dict, start_dict,
                                 ax, color_dict=colors)

    return ax
