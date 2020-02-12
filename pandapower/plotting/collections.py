# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import inspect
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection, Collection
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, Rectangle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
from pandas import isnull
from pandapower.plotting.patch_makers import load_patches, node_patches, gen_patches,\
    sgen_patches, ext_grid_patches, trafo_patches
from pandapower.plotting.plotting_toolbox import _rotate_dim2, coords_from_node_geodata, \
    position_on_busbar, get_index_array

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CustomTextPath(TextPath):
    """
    Create a path from the text. This class provides functionality for deepcopy, which is not
    implemented for TextPath.
    """

    def __init__(self, xy, s, size=None, prop=None, _interpolation_steps=1, usetex=False):
        """
        Create a path from the text. No support for TeX yet. Note that
        it simply is a path, not an artist. You need to use the
        PathPatch (or other artists) to draw this path onto the
        canvas.

        xy : position of the text.
        s : text
        size : font size
        prop : font property
        """
        if prop is None:
            prop = FontProperties()
        TextPath.__init__(self, xy, s, size=size, prop=prop,
                          _interpolation_steps=_interpolation_steps, usetex=usetex)
        self.s = s
        self.usetex = usetex
        self.prop = prop

    def __deepcopy__(self, memo=None):
        """
        Returns a deepcopy of the `CustomTextPath`, which is not implemented for TextPath
        """
        return self.__class__(copy.deepcopy(self._xy), copy.deepcopy(self.s), size=self.get_size(),
                              prop=copy.deepcopy(self.prop),
                              _interpolation_steps=self._interpolation_steps, usetex=self.usetex)


def create_annotation_collection(texts, coords, size, prop=None, **kwargs):
    """
    Creates PatchCollection of Texts shown at the given coordinates

    Input:
        **texts** (iterable of strings) - The texts to be
        **coords** (iterable of tuples) - The pandapower network
        **size** (int) - The pandapower network

    OPTIONAL:
        **prop** - FontProperties being passed to the TextPatches
        **kwargs** - Any other keyword-arguments will be passed to the PatchCollection.
    """
    tp = []
    # we convert TextPaths to PathPatches to create a PatchCollection
    if hasattr(size, "__iter__"):
        for i, t in enumerate(texts):
            tp.append(PathPatch(CustomTextPath(coords[i], t, size=size[i], prop=prop)))
    else:
        for t, c in zip(texts, coords):
            tp.append(PathPatch(CustomTextPath(c, t, size=size, prop=prop)))

    return PatchCollection(tp, **kwargs)


def add_cmap_to_collection(collection, cmap, norm, z, cbar_title, plot_colormap=True, clim=None):
    """
    Adds a colormap to the given collection.

    :param collection: collection for which to add colormap
    :type collection: matplotlib.collections.collection
    :param cmap: colormap which to use
    :type cmap: any colormap from matplotlib.colors
    :param norm: any norm which to use to translate values into colors
    :type norm: any norm from matplotlib.colors
    :param z: the array which to use in order to create the colors for the given collection
    :type z: iterable
    :param cbar_title: title of the colorbar
    :type cbar_title: str
    :param plot_colormap: flag whether the colormap is actually drawn (if False, is excluded in\
        :func:`add_single_collection`)
    :type plot_colormap: bool, default True
    :param clim: color limit of the collection
    :type clim: list(float), default None
    :return: collection - the given collection with added colormap (no copy!)
    """
    collection.set_cmap(cmap)
    collection.set_norm(norm)
    collection.set_array(np.ma.masked_invalid(z))
    collection.has_colormap = plot_colormap
    collection.cbar_title = cbar_title
    if clim is not None:
        collection.set_clim(clim)
    return collection


def _create_node_collection(nodes, coords, size=5, patch_type="circle", color=None, picker=False,
                            infos=None, **kwargs):
    """
    Creates a collection with patches for the given nodes. Can be used generically for different \
    types of nodes (bus in pandapower network, but also other nodes, e.g. in a networkx graph).

    :param nodes: indices of the nodes to plot
    :type nodes: iterable
    :param coords: list of node coordinates (shape (2, N))
    :type coords: iterable
    :param size: size of the patches (handed over to patch creation function)
    :type size: float
    :param patch_type: type of patches that chall be created for the nodes - can be one of\
        - "circle" for a circle\
        - "rect" for a rectangle\
        - "poly<n>" for a polygon with n edges
    :type patch_type: str, default "circle"
    :param color: colors or color of the node patches
    :type color: iterable, float
    :param picker: picker argument passed to the patch collection
    :type picker: bool, default False
    :param infos: list of infos belonging to each of the patches (can be displayed when hovering \
        over the elements)
    :type infos: list, default None
    :param kwargs: keyword arguments are passed to the patch maker and patch collection
    :type kwargs:
    :return: pc - patch collection for the nodes
    """
    if len(coords) == 0:
        return None

    infos = list(infos) if infos is not None else []
    patches = node_patches(coords, size, patch_type, color, **kwargs)
    pc = PatchCollection(patches, match_original=True, picker=picker)
    pc.node_indices = np.array(nodes)

    pc.patch_type = patch_type
    pc.size = size
    if 'orientation' in kwargs:
        pc.orientation = kwargs['orientation']
    if "zorder" in kwargs:
        pc.set_zorder(kwargs["zorder"])
    pc.info = infos
    pc.patch_type = patch_type
    pc.size = size
    if "zorder" in kwargs:
        pc.set_zorder(kwargs["zorder"])
    if 'orientation' in kwargs:
        pc.orientation = kwargs['orientation']

    return pc


def _create_line2d_collection(coords, indices, infos=None, picker=False, **kwargs):
    """
    Generic function to create a LineCollection from coordinates.

    :param coords: list of line coordinates (list should look like this: \
        `[[(x11, y11), (x12, y12), (x13, y13), ...], [(x21, y21), (x22, x23), ...], ...]`)
    :type coords: list or np.array
    :param indices: list of node indices
    :type indices: list or np.array
    :param infos: list of infos belonging to each of the lines (can be displayed when hovering \
        over them)
    :type infos: list, default None
    :param picker: picker argument passed to the line collection
    :type picker: bool, default False
    :param kwargs: keyword arguments are passed to the line collection
    :type kwargs:
    :return: lc - line collection for the given coordinates
    """
    # This would be done anyways by matplotlib - doing it explicitly makes it a) clear and
    # b) prevents unexpected behavior when observing colors being "none"
    lc = LineCollection(coords, picker=picker, **kwargs)
    lc.indices = np.array(indices)
    lc.info = infos if infos is not None else list()
    return lc


def _create_node_element_collection(node_coords, patch_maker, size=1., infos=None,
                                    repeat_infos=(1, 1), orientation=np.pi, picker=False,
                                    patch_facecolor="w", patch_edgecolor="k", line_color="k",
                                    **kwargs):
    """
    Creates matplotlib collections of node elements. All node element collections usually consist of
    one patch collection representing the element itself and a small line collection that connects
    the element to the respective node.

    :param node_coords: the coordinates (x, y) of the nodes with shape (N, 2)
    :type node_coords: iterable
    :param patch_maker: a function to generate the patches of which the collections consist (cf. \
        the patch_maker module)
    :type patch_maker: function
    :param size: patch size
    :type size: float, default 1
    :param infos: list of infos belonging to each of the elements (can be displayed when hovering \
        over them)
    :type infos: iterable, default None
    :param repeat_infos: determines how many times the info shall be repeated to match the number \
        of patches (first element) and lines (second element) returned by the patch maker
    :type repeat_infos: tuple (length 2), default (1, 1)
    :param orientation: orientation of load collection. pi is directed downwards, increasing values\
        lead to clockwise direction changes.
    :type orientation: float, default np.pi
    :param picker: picker argument passed to the line collection
    :type picker: bool, default False
    :param patch_facecolor: color of the patch face (content)
    :type patch_facecolor: matplotlib color, "w"
    :param patch_edgecolor: color of the patch edges
    :type patch_edgecolor: matplotlib color, "k"
    :param line_color: color of the connecting lines
    :type line_color: matplotlib color, "k"
    :param kwargs: key word arguments are passed to the patch function
    :type kwargs:
    :return: Return values:\
        - patch_coll - patch collection representing the element\
        - line_coll - connecting line collection

    """
    angles = orientation if hasattr(orientation, '__iter__') else [orientation] * len(node_coords)
    assert len(node_coords) == len(angles), \
        "The length of coordinates does not match the length of the orientation angles!"
    if infos is None:
        infos_pc = []
        infos_lc = []
    else:
        infos_pc = list(np.repeat(infos, repeat_infos[0]))
        infos_lc = list(np.repeat(infos, repeat_infos[1]))

    lines, polys, popped_keywords = patch_maker(
        node_coords, size, angles, patch_facecolor=patch_facecolor, patch_edgecolor=patch_edgecolor,
        **kwargs)
    for kw in set(popped_keywords) & set(kwargs.keys()):
        kwargs.pop(kw)
    patch_coll = PatchCollection(polys, match_original=True, picker=picker, **kwargs)
    line_coll = LineCollection(lines, color=line_color, picker=picker, **kwargs)
    patch_coll.info = infos_pc
    line_coll.info = infos_lc
    return patch_coll, line_coll


def _create_complex_branch_collection(coords, patch_maker, size=1, infos=None, repeat_infos=(2, 2),
                                      picker=False, patch_facecolor="w", patch_edgecolor="k",
                                      line_color="k", linewidths=2., **kwargs):
    """
    Creates a matplotlib line collection and a matplotlib patch collection representing a branch\
    element that cannot be represented by just a line.

    :param coords: list of connecting node coordinates (usually should be \
        `[((x11, y11), (x12, y12)), ((x21, y21), (x22, y22)), ...]`)
    :type coords: (N, (2, 2)) shaped iterable
    :param patch_maker: a function to generate the patches of which the collections consist (cf. \
        the patch_maker module)
    :type patch_maker: function
    :param size: patch size
    :type size: float, default 1
    :param infos: list of infos belonging to each of the branches (can be displayed when hovering \
        over them)
    :type infos: iterable, default None
    :param repeat_infos: determines how many times the info shall be repeated to match the number \
        of patches (first element) and lines (second element) returned by the patch maker
    :type repeat_infos: tuple (length 2), default (1, 1)
    :param picker: picker argument passed to the line collection
    :type picker: bool, default False
    :param patch_facecolor: color or colors of the patch face (content)
    :type patch_facecolor: matplotlib color or iterable, "w"
    :param patch_edgecolor: color or colors of the patch edges
    :type patch_edgecolor: matplotlib color or iterable, "k"
    :param line_color: color or colors of the connecting lines
    :type line_color: matplotlib color or iterable, "k"
    :param linewidths: linewidths of the connecting lines and the patch edges
    :type linewidths: float, default 2.
    :param kwargs: key word arguments are passed to the patch maker and the patch and line \
        collections
    :type kwargs:
    :return: Return values:\
        - patch_coll - patch collection representing the branch element\
        - line_coll - line collection connecting the patches with the nodes
    """
    if infos is None:
        infos_pc = []
        infos_lc = []
    else:
        infos_pc = list(np.repeat(infos, repeat_infos[0]))
        infos_lc = list(np.repeat(infos, repeat_infos[1]))

    lines, patches, popped_keywords = patch_maker(coords, size, patch_facecolor=patch_facecolor,
                                                  patch_edgecolor=patch_edgecolor, **kwargs)
    for kw in set(popped_keywords) & set(kwargs.keys()):
        kwargs.pop(kw)
    patch_coll = PatchCollection(patches, match_original=True, picker=picker, **kwargs)
    line_coll = LineCollection(lines, color=line_color, picker=picker, linewidths=linewidths,
                               **kwargs)
    patch_coll.info = infos_pc
    line_coll.info = infos_lc
    return patch_coll, line_coll


def create_bus_collection(net, buses=None, size=5, patch_type="circle", color=None, z=None,
                          cmap=None, norm=None, infofunc=None, picker=False, bus_geodata=None,
                          cbar_title="Bus Voltage [pu]", **kwargs):
    """
    Creates a matplotlib patch collection of pandapower buses.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **buses** (list, None) - The buses for which the collections are created.
            If None, all buses in the network are considered.

        **size** (int, 5) - patch size

        **patch_type** (str, "circle") - patch type, can be

                - "circle" for a circle
                - "rect" for a rectangle
                - "poly<n>" for a polygon with n edges

        **infofunc** (function, None) - infofunction for the patch element

        **color** (list or color, None) - color or list of colors for every element

        **z** (array, None) - array of bus voltage magnitudes for colormap. Used in case of given
            cmap. If None net.res_bus.vm_pu is used.

        **cmap** (ListedColormap, None) - colormap for the patch colors

        **norm** (matplotlib norm object, None) - matplotlib norm object

        **picker** (bool, False) - picker argument passed to the patch collection

        **bus_geodata** (DataFrame, None) - coordinates to use for plotting
            If None, net["bus_geodata"] is used

        **cbar_title** (str, "Bus Voltage [pu]") - colormap bar title in case of given cmap

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **pc** - patch collection
    """
    buses = get_index_array(buses, net.bus.index)
    if len(buses) == 0:
        return None
    if bus_geodata is None:
        bus_geodata = net["bus_geodata"]

    coords = list(zip(bus_geodata.loc[buses, "x"].values, bus_geodata.loc[buses, "y"].values))

    infos = [infofunc(bus) for bus in buses] if infofunc is not None else []

    pc = _create_node_collection(buses, coords, size, patch_type, color, picker, infos, **kwargs)

    if cmap is not None:
        add_cmap_to_collection(pc, cmap, norm, z, cbar_title)

    return pc


def create_line_collection(net, lines=None, line_geodata=None, bus_geodata=None,
                           use_bus_geodata=False, infofunc=None, cmap=None, norm=None, picker=False,
                           z=None, cbar_title="Line Loading [%]", clim=None, plot_colormap=True,
                           **kwargs):
    """
    Creates a matplotlib line collection of pandapower lines.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **lines** (list, None) - The lines for which the collections are created. If None, all lines
            in the network are considered.

        **line_geodata** (DataFrame, None) - coordinates to use for plotting. If None,
            net["line_geodata"] is used

        **bus_geodata** (DataFrame, None) - coordinates to use for plotting
            If None, net["bus_geodata"] is used

        **use_bus_geodata** (bool, False) - Defines whether bus or line geodata are used.

         **infofunc** (function, None) - infofunction for the patch element

        **cmap** - colormap for the patch colors

        **norm** (matplotlib norm object, None) - matplotlib norm object

        **picker** (bool, False) - picker argument passed to the line collection

        **z** (array, None) - array of line loading magnitudes for colormap. Used in case of given
            cmap. If None net.res_line.loading_percent is used.

        **cbar_title** (str, "Line Loading [%]") - colormap bar title in case of given cmap

        **clim** (tuple of floats, None) - setting the norm limits for image scaling

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **lc** - line collection
    """
    if use_bus_geodata is False and line_geodata is None and net.line_geodata.empty:
        # if bus geodata is available, but no line geodata
        logger.warning("use_bus_geodata is automatically set to True, since net.line_geodata is "
                       "empty.")
        use_bus_geodata = True

    lines = get_index_array(lines, net.line.index)
    if len(lines) == 0:
        return None

    if use_bus_geodata:
        coords, lines_with_geo = coords_from_node_geodata(
            lines, net.line.from_bus.loc[lines].values, net.line.to_bus.loc[lines].values,
            bus_geodata if bus_geodata is not None else net["bus_geodata"], "line")
    else:
        line_geodata = line_geodata if line_geodata is not None else net.line_geodata
        lines_with_geo = lines[np.isin(lines, line_geodata.index.values)]
        coords = list(line_geodata.loc[lines_with_geo, 'coords'])
        lines_without_geo = set(lines) - set(lines_with_geo)
        if lines_without_geo:
            logger.warning("Could not plot lines %s. %s geodata is missing for those lines!"
                           % (lines_without_geo, "Bus" if use_bus_geodata else "Line"))

    if len(lines_with_geo) == 0:
        return None

    infos = [infofunc(line) for line in lines_with_geo] if infofunc else []

    lc = _create_line2d_collection(coords, lines_with_geo, infos=infos, picker=picker, **kwargs)

    if cmap is not None:
        if z is None:
            z = net.res_line.loading_percent.loc[lines_with_geo]
        add_cmap_to_collection(lc, cmap, norm, z, cbar_title, plot_colormap, clim)

    return lc


def create_trafo_connection_collection(net, trafos=None, bus_geodata=None, infofunc=None,
                                       cmap=None, clim=None, norm=None, z=None,
                                       cbar_title="Transformer Loading", picker=False, **kwargs):
    """
    Creates a matplotlib line collection of pandapower transformers.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The transformers for which the collections are created.
            If None, all transformers in the network are considered.

        **bus_geodata** (DataFrame, None) - coordinates to use for plotting
            If None, net["bus_geodata"] is used

        **infofunc** (function, None) - infofunction for the patch element

        **cmap** - colormap for the patch colors

        **clim** (tuple of floats, None) - setting the norm limits for image scaling

        **norm** (matplotlib norm object, None) - matplotlib norm object

        **z** (array, None) - array of line loading magnitudes for colormap. Used in case of given
            cmap. If None net.res_line.loading_percent is used.

        **cbar_title** (str, "Line Loading [%]") - colormap bar title in case of given cmap

        **picker** (bool, False) - picker argument passed to the line collection

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **lc** - line collection
    """
    trafos = get_index_array(trafos, net.trafo.index)
    trafo_table = net.trafo.loc[trafos]

    if bus_geodata is None:
        bus_geodata = net["bus_geodata"]

    hv_geo = list(zip(bus_geodata.loc[trafo_table["hv_bus"], "x"].values,
                      bus_geodata.loc[trafo_table["hv_bus"], "y"].values))
    lv_geo = list(zip(bus_geodata.loc[trafo_table["lv_bus"], "x"].values,
                      bus_geodata.loc[trafo_table["lv_bus"], "y"].values))
    tg = list(zip(hv_geo, lv_geo))

    info = [infofunc(tr) for tr in trafos] if infofunc is not None else []

    lc = _create_line2d_collection(tg, trafos, info, picker=picker, **kwargs)

    if cmap is not None:
        if z is None:
            z = net.res_trafo.loading_percent.loc[trafos]
        add_cmap_to_collection(lc, cmap, norm, z, cbar_title, True, clim)

    return lc


def create_trafo3w_connection_collection(net, trafos=None, bus_geodata=None, infofunc=None,
                                         **kwargs):
    """
    Creates a matplotlib line collection of pandapower 3W-transformers.
    This function can be used to create line collections for voltage fall diagrams.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The 3W-transformers for which the collections are created.
            If None, all 3W-transformers in the network are considered.

        **bus_geodata** (DataFrame, None) - coordinates to use for plotting
            If None, net["bus_geodata"] is used

         **infofunc** (function, None) - infofunction for the patch element

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **lc** - line collection
    """
    trafos = get_index_array(trafos, net.trafo3w.index)
    trafo_table = net.trafo3w.loc[trafos]

    if bus_geodata is None:
        bus_geodata = net["bus_geodata"]

    hv_geo, mv_geo, lv_geo = (list(zip(*(bus_geodata.loc[trafo_table[column], var].values
                                         for var in ['x', 'y'])))
                              for column in ['hv_bus', 'mv_bus', 'lv_bus'])

    # create 3 connection lines, each of 2 points, for every trafo3w
    tg = [x for c in [list(combinations(y, 2)) for y in zip(hv_geo, mv_geo, lv_geo)] for x in c]

    # 3 times infofunc for every trafo
    info = [infofunc(x) if infofunc is not None else []
            for tr in [(t, t, t) for t in trafos]
            for x in tr]

    lc = LineCollection(tg, **kwargs)
    lc.info = info

    return lc


def create_trafo_collection(net, trafos=None, picker=False, size=None, infofunc=None, cmap=None,
                            norm=None, z=None, clim=None, cbar_title="Transformer Loading",
                            plot_colormap=True, bus_geodata=None, **kwargs):
    """
    Creates a matplotlib line collection of pandapower transformers.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The transformers for which the collections are created.
            If None, all transformers in the network are considered.

        **picker** (bool, False) - picker argument passed to the patch collection

        **size** (int, None) - size of transformer symbol circles. Should be >0 and
            < 0.35*bus_distance

         **infofunc** (function, None) - infofunction for the patch element

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **lc** - line collection

        **pc** - patch collection
    """
    trafos = get_index_array(trafos, net.trafo.index)
    trafo_table = net.trafo.loc[trafos]

    coords, trafos_with_geo = coords_from_node_geodata(
        trafos, trafo_table.hv_bus.values, trafo_table.lv_bus.values,
        bus_geodata if bus_geodata is not None else net["bus_geodata"], "trafo")

    if len(trafos_with_geo) == 0:
        return None

    colors = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    if cmap is not None:
        if z is None:
            z = net.res_trafo.loading_percent
        colors = [cmap(norm(z.at[idx])) for idx in trafos_with_geo]

    infos = [infofunc(i) for i in range(len(trafos_with_geo))] if infofunc is not None else []

    lc, pc = _create_complex_branch_collection(
        coords, trafo_patches, size, infos, patch_facecolor="none", patch_edgecolor=colors,
        line_color=colors, picker=picker, linewidths=linewidths, **kwargs)

    if cmap is not None:
        z_duplicated = np.repeat(z.values, 2)
        add_cmap_to_collection(lc, cmap, norm, z_duplicated, cbar_title, plot_colormap, clim)
    return lc, pc


# noinspection PyArgumentList
def create_trafo3w_collection(net, trafo3ws=None, picker=False, infofunc=None, cmap=None, norm=None,
                              z=None, clim=None, cbar_title="3W-Transformer Loading",
                              plot_colormap=True, **kwargs):
    """
    Creates a matplotlib line collection of pandapower transformers.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafo3ws** (list, None) - The three winding transformers for which the collections are
            created. If None, all three winding transformers in the network are considered.

        **picker** (bool, False) - picker argument passed to the patch collection

         **infofunc** (function, None) - infofunction for the patch element

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **lc** - line collection

        **pc** - patch collection
    """
    trafo3ws = get_index_array(trafo3ws, net.trafo3w.index)
    trafo3w_table = net.trafo3w.loc[trafo3ws]
    lines = []
    circles = []
    infos = []
    color = kwargs.pop("color", "k")
    linewidth = kwargs.pop("linewidths", 2.)
    if cmap is not None and z is None:
        z = net.res_trafo3w.loading_percent
    for i, idx in enumerate(trafo3w_table.index):
        # get bus geodata
        p1 = net.bus_geodata[["x", "y"]].loc[net.trafo3w.at[idx, "hv_bus"]].values
        p2 = net.bus_geodata[["x", "y"]].loc[net.trafo3w.at[idx, "mv_bus"]].values
        p3 = net.bus_geodata[["x", "y"]].loc[net.trafo3w.at[idx, "lv_bus"]].values
        if np.all(p1 == p2) and np.all(p1 == p3):
            continue
        p = np.array([p1, p2, p3])
        # determine center of buses and minimum distance center-buses
        center = sum(p) / 3
        d = np.linalg.norm(p - center, axis=1)
        r = d.min() / 3
        # determine closest bus to center and vector from center to circle midpoint in closest
        # direction
        closest = d.argmin()
        to_closest = (p[closest] - center) / d[closest] * 2 * r / 3
        # determine vectors from center to circle midpoint
        order = list(range(closest, 3)) + list(range(closest))
        cm = np.empty((3, 2))
        cm[order.pop(0)] = to_closest
        ang = 2 * np.pi / 3  # 120 degree
        cm[order.pop(0)] = _rotate_dim2(to_closest, ang)
        cm[order.pop(0)] = _rotate_dim2(to_closest, -ang)
        # determine midpoints of circles
        m = center + cm
        # determine endpoints of circles
        e = (center - p) * (1 - 5 * r / 3 / d).reshape(3, 1) + p
        # save circle and line collection data
        ec = color if cmap is None else cmap(norm(z.at[idx]))
        for j in range(3):
            circles.append(Circle(m[j], r, fc=(1, 0, 0, 0), ec=ec))
            lines.append([p[j], e[j]])

        if infofunc is not None:
            infos.append(infofunc(i))
            infos.append(infofunc(i))
    if len(circles) == 0:
        return None, None
    lc = LineCollection(lines, color=color, picker=picker, linewidths=linewidth, **kwargs)
    lc.info = infos
    pc = PatchCollection(circles, match_original=True, picker=picker, linewidth=linewidth, **kwargs)
    pc.info = infos
    if cmap is not None:
        z_duplicated = np.repeat(z.values, 3)
        lc.set_cmap(cmap)
        lc.set_norm(norm)
        if clim is not None:
            lc.set_clim(clim)
        lc.set_array(np.ma.masked_invalid(z_duplicated))
        lc.has_colormap = plot_colormap
        lc.cbar_title = cbar_title
    return lc, pc


def create_busbar_collection(net, buses=None, infofunc=None, cmap=None, norm=None, picker=False,
                             z=None, cbar_title="Bus Voltage [p.u.]", clim=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapower buses plotted as busbars

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **buses** (list, None) - The buses for which the collections are created. If None, all buses
            which have the entry coords in bus_geodata are considered.

        **line_geodata** (DataFrame, None) - coordinates to use for plotting. If None,
            net["line_geodata"] is used

        **infofunc** (function, None) - infofunction for the line element

        **cmap** - colormap for the line colors

        **norm** (matplotlib norm object, None) - matplotlib norm object

        **picker** (bool, False) - picker argument passed to the patch collection

        **z** (array, None) - array of line loading magnitudes for colormap. Used in case of given
            cmap. If None net.res_line.loading_percent is used.

        **cbar_title** (str, "Line Loading [%]") - colormap bar title in case of given cmap

        **clim** (tuple of floats, None) - setting the norm limits for image scaling

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **bbc** - busbar collection
    """

    if buses is None:
        buses = net.bus_geodata.loc[~isnull(net.bus_geodata.coords)].index

    if cmap is not None:
        # determine color of busbar by vm_pu
        if z is None and net is not None:
            z = net.res_bus.vm_pu.loc[buses]
        else:
            logger.warning("z is None and no net is provided")

    # the busbar is just a line collection with coords from net.bus_geodata
    return create_line_collection(net, lines=buses, line_geodata=net.bus_geodata, bus_geodata=None,
                                  norm=norm, cmap=cmap, infofunc=infofunc, picker=picker, z=z,
                                  cbar_title=cbar_title, clim=clim, **kwargs)


def create_load_collection(net, loads=None, size=1., infofunc=None, orientation=np.pi, picker=False,
                           **kwargs):
    """
    Creates a matplotlib patch collection of pandapower loads.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **loads** (list of ints, None) - the loads to include in the collection

        **size** (float, 1) - patch size

        **infofunc** (function, None) - infofunction for the patch element

        **orientation** (float, np.pi) - orientation of load collection. pi is directed downwards,
            increasing values lead to clockwise direction changes.

        **picker** (bool, False) - picker argument passed to the patch collectionent

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **load_pc** - patch collection

        **load_lc** - line collection
    """
    loads = get_index_array(loads, net.load.index)
    infos = [infofunc(i) for i in range(len(loads))] if infofunc is not None else []
    node_coords = net.bus_geodata.loc[net.load.loc[loads, "bus"].values, ["x", "y"]].values
    load_pc, load_lc = _create_node_element_collection(
        node_coords, load_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, **kwargs)
    return load_pc, load_lc


def create_gen_collection(net, gens=None, size=1., infofunc=None, orientation=np.pi, picker=False,
                          **kwargs):
    """
    Creates a matplotlib patch collection of pandapower gens.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **gens** (list of ints, None) - the generators to include in the collection

        **size** (float, 1) - patch size

        **infofunc** (function, None) - infofunction for the patch element

        **orientation** (float or list of floats, np.pi) - orientation of gen collection. pi is\
            directed downwards, increasing values lead to clockwise direction changes.

        **picker** (bool, False) - picker argument passed to the patch collectionent

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **gen_pc** - patch collection

        **gen_lc** - line collection
    """
    gens = get_index_array(gens, net.gen.index)
    infos = [infofunc(i) for i in range(len(gens))] if infofunc is not None else []
    node_coords = net.bus_geodata.loc[:, ["x", "y"]].values[net.gen.loc[gens, "bus"].values]
    gen_pc, gen_lc = _create_node_element_collection(
        node_coords, gen_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, **kwargs)
    return gen_pc, gen_lc


def create_sgen_collection(net, sgens=None, size=1., infofunc=None, orientation=np.pi, picker=False,
                           **kwargs):
    """
    Creates a matplotlib patch collection of pandapower sgen.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **sgens** (list of ints, None) - the static generators to include in the collection

        **size** (float, 1) - patch size

        **infofunc** (function, None) - infofunction for the patch elem

        **picker** (bool, False) - picker argument passed to the patch collectionent

        **orientation** (float, np.pi) - orientation of static generator collection. pi is directed\
            downwards, increasing values lead to clockwise direction changes.

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **sgen_pc** - patch collection

        **sgen_lc** - line collection
    """
    gens = get_index_array(sgens, net.sgen.index)
    infos = [infofunc(i) for i in range(len(sgens))] if infofunc is not None else []
    node_coords = net.bus_geodata.loc[net.sgen.loc[sgens, "bus"].values, ["x", "y"]].values
    sgen_pc, sgen_lc = _create_node_element_collection(
        node_coords, sgen_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, **kwargs)
    return sgen_pc, sgen_lc


def create_ext_grid_collection(net, size=1., infofunc=None, orientation=0, picker=False,
                               ext_grids=None, ext_grid_buses=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapower ext_grid. Parameters
    ext_grids, ext_grid_buses can be used to specify, which ext_grids the collection should be
    created for.

    Input:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **size** (float, 1) - patch size

        **infofunc** (function, None) - infofunction for the patch element

        **orientation** (float, 0) - orientation of load collection. 0 is directed upwards,
            increasing values lead to clockwise direction changes.

        **picker** (bool, False) - picker argument passed to the patch collection

        **ext_grid_buses** (np.ndarray, None) - buses to be used as ext_grid locations

        **kwargs - key word arguments are passed to the patch function

    OUTPUT:
        **ext_grid1** - patch collection

        **ext_grid2** - patch collection
    """
    ext_grids = get_index_array(ext_grids, net.ext_grid.index)
    if ext_grid_buses is None:
        ext_grid_buses = net.ext_grid.bus.loc[ext_grids].values
    else:
        assert len(ext_grids) == len(ext_grid_buses), \
            "Length mismatch between chosen ext_grids and ext_grid_buses."
    infos = [infofunc(ext_grid_idx) for ext_grid_idx in ext_grids] if infofunc is not None else []

    node_coords = net.bus_geodata.loc[ext_grid_buses, ["x", "y"]].values

    ext_grid_pc, ext_grid_lc = _create_node_element_collection(
        node_coords, ext_grid_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, hatch='XXX', **kwargs)

    return ext_grid_pc, ext_grid_lc


def create_line_switch_collection(net, size=1, distance_to_bus=3, use_line_geodata=False, **kwargs):
    """
    Creates a matplotlib patch collection of pandapower line-bus switches.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:

        **size** (float, 1) - Size of the switch patches

        **distance_to_bus** (float, 3) - Distance of the switch patch from the bus patch

        **use_line_geodata** (bool, False) - If True, line coordinates are used to identify the
                                             switch position

        **kwargs - Key word arguments are passed to the patch function

    OUTPUT:
        **switches** - patch collection
    """
    lbs_switches = net.switch.index[net.switch.et == "l"]

    color = kwargs.pop("color", "k")

    switch_patches = []
    for switch in lbs_switches:
        sb = net.switch.bus.loc[switch]
        line = net.line.loc[net.switch.element.loc[switch]]
        fb = line.from_bus
        tb = line.to_bus

        line_buses = {fb, tb}
        target_bus = list(line_buses - {sb})[0]

        if sb not in net.bus_geodata.index or target_bus not in net.bus_geodata.index:
            logger.warning("Bus coordinates for switch %s not found, skipped switch!" % switch)
            continue

        # switch bus and target coordinates
        pos_sb = net.bus_geodata.loc[sb, ["x", "y"]].values
        pos_tb = np.zeros(2)

        use_bus_geodata = False

        if use_line_geodata:
            if line.name in net.line_geodata.index:
                line_coords = net.line_geodata.coords.loc[line.name]
                # check, which end of the line is nearer to the switch bus
                intersection = position_on_busbar(net, target_bus, busbar_coords=line_coords)
                if intersection is not None:
                    pos_sb = intersection
                if len(line_coords) >= 2:
                    if abs(line_coords[0][0] - pos_sb[0]) < 0.01 and \
                            abs(line_coords[0][1] - pos_sb[1]) < 0.01:
                        pos_tb = np.array([line_coords[1][0], line_coords[1][1]])
                    else:
                        pos_tb = np.array([line_coords[-2][0], line_coords[-2][1]])
                else:
                    use_bus_geodata = True
            else:
                use_bus_geodata = True

        if not use_line_geodata or use_bus_geodata:
            pos_tb = net.bus_geodata.loc[target_bus, ["x", "y"]]

        # position of switch symbol
        vec = pos_tb - pos_sb
        mag = np.linalg.norm(vec)
        pos_sw = pos_sb + vec / mag * distance_to_bus

        # rotation of switch symbol
        angle = np.arctan2(vec[1], vec[0])
        rotation = Affine2D().rotate_around(pos_sw[0], pos_sw[1], angle)

        # color switch by state
        col = color if net.switch.closed.loc[switch] else "white"

        # create switch patch (switch size is respected to center the switch on the line)
        patch = Rectangle((pos_sw[0] - size / 2, pos_sw[1] - size / 2), size, size, facecolor=col,
                          edgecolor=color)
        # apply rotation
        patch.set_transform(rotation)

        switch_patches.append(patch)

    switches = PatchCollection(switch_patches, match_original=True, **kwargs)
    return switches


def create_bus_bus_switch_collection(net, size=1., helper_line_style=':', helper_line_size=1.,
                                     helper_line_color="gray", **kwargs):
    """
    Creates a matplotlib patch collection of pandapower bus-bus switches. Switches are plotted in
    the center between two buses with a "helper" line (dashed and thin) being drawn between the
    buses as well.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:

        **size** (float, 1.0) - Size of the switch patches

        **helper_line_style** (string, ':') - Line style of the "helper" line being plotted between\
            two buses connected by a bus-bus switch

        **helper_line_size** (float, 1.0) - Line width of the "helper" line being plotted between \
            two buses connected by a bus-bus switch

        **helper_line_color** (string, "gray") - Line color of the "helper" line being plotted \
            between two buses connected by a bus-bus switch

        **kwargs - Key word arguments are passed to the patch function

    OUTPUT:
        **switches**, **helper_lines** - tuple of patch collections
    """
    lbs_switches = net.switch.index[net.switch.et == "b"]
    color = kwargs.pop("color", "k")
    switch_patches = []
    line_patches = []
    for switch in lbs_switches:
        switch_bus = net.switch.bus.loc[switch]
        target_bus = net.switch.element.loc[switch]
        if switch_bus not in net.bus_geodata.index or target_bus not in net.bus_geodata.index:
            logger.warning("Bus coordinates for switch %s not found, skipped switch!" % switch)
            continue
        # switch bus and target coordinates
        pos_sb = net.bus_geodata.loc[switch_bus, ["x", "y"]].values.astype(np.float64)
        pos_tb = net.bus_geodata.loc[target_bus, ["x", "y"]].values.astype(np.float64)
        # position of switch symbol
        vec = pos_tb - pos_sb
        pos_sw = pos_sb + vec * 0.5 if not np.allclose(pos_sb, pos_tb) else pos_tb
        # rotation of switch symbol
        angle = np.arctan2(vec[1], vec[0])
        rotation = Affine2D().rotate_around(pos_sw[0], pos_sw[1], angle)
        # color switch by state
        col = color if net.switch.closed.loc[switch] else "white"
        # create switch patch (switch size is respected to center the switch on the line)
        patch = Rectangle((pos_sw[0] - size / 2, pos_sw[1] - size / 2), size, size, facecolor=col,
                          edgecolor=color)
        # apply rotation
        patch.set_transform(rotation)
        # add to collection lists
        switch_patches.append(patch)
        line_patches.append([pos_sb.tolist(), pos_tb.tolist()])
    # create collections and return
    switches = PatchCollection(switch_patches, match_original=True, **kwargs)
    helper_lines = LineCollection(line_patches, linestyles=helper_line_style,
                                  linewidths=helper_line_size, colors=helper_line_color)
    return switches, helper_lines


def draw_collections(collections, figsize=(10, 8), ax=None, plot_colorbars=True, set_aspect=True,
                     axes_visible=(False, False), copy_collections=True, draw=True):
    """
    Draws matplotlib collections which can be created with the create collection functions.

    Input:
        **collections** (list) - iterable of collection objects, may include tuples of collections

    OPTIONAL:
        **figsize** (tuple, (10,8)) - figsize of the matplotlib figure

        **ax** (axis, None) - matplotlib axis object to plot into, new axis is created if None

        **plot_colorbars** (bool, True) - defines whether colorbars should be plotted

        **set_aspect** (bool, True) - defines whether 'equal' and 'datalim' aspects of axis scaling\
            should be set.

        **axes_visible** (tuple, (False, False)) - defines visibility of (xaxis, yaxis)

    OUTPUT:
        **ax** - matplotlib axes
    """

    if ax is None:
        plt.figure(facecolor="white", figsize=figsize)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.05,
                            wspace=0.02, hspace=0.04)
    ax = ax or plt.gca()

    add_collections_to_axes(ax, collections, plot_colorbars=plot_colorbars,
                            copy_collections=copy_collections)

    try:
        ax.set_facecolor("white")
    except:
        ax.set_axis_bgcolor("white")
    ax.xaxis.set_visible(axes_visible[0])
    ax.yaxis.set_visible(axes_visible[1])
    if not any(axes_visible):
        # removes bounding box of the plot also
        ax.axis("off")
    if set_aspect:
        ax.set_aspect('equal', 'datalim')
    ax.autoscale_view(True, True, True)
    ax.margins(.02)
    if draw:
        plt.draw()
    return ax


def add_single_collection(c, ax, plot_colorbars, copy_collections):
    if copy_collections:
        c = copy.copy(c)
    ax.add_collection(c)
    if plot_colorbars and hasattr(c, "has_colormap") and c.has_colormap:
        extend = c.extend if hasattr(c, "extend") else "neither"
        cbar_load = plt.colorbar(c, extend=extend, ax=ax)
        if hasattr(c, "cbar_title"):
            cbar_load.ax.set_ylabel(c.cbar_title)


def add_collections_to_axes(ax, collections, plot_colorbars=True, copy_collections=True):
    for i, c in enumerate(collections):
        if Collection in inspect.getmro(c.__class__):
            # if Collection is in one of the base classes of c
            add_single_collection(c, ax, plot_colorbars, copy_collections)
        elif isinstance(c, tuple) or isinstance(c, list):
            # if c is a tuple or a list of collections
            add_collections_to_axes(ax, c, plot_colorbars, copy_collections)
        else:
            logger.warning("{} in collections is of unknown type. Skipping".format(i))


if __name__ == "__main__":
    # if 0:
    #     import pandapower as pp
    #
    #     ntw = pp.create_empty_network()
    #     b1 = pp.create_bus(ntw, 10, geodata=(5, 10))
    #     b2 = pp.create_bus(ntw, 0.4, geodata=(5, 15))
    #     b3 = pp.create_bus(ntw, 0.4, geodata=(0, 22))
    #     b4 = pp.create_bus(ntw, 0.4, geodata=(8, 20))
    #     pp.create_gen(ntw, b1, p_mw=0.1)
    #     pp.create_load(ntw, b3, p_mw=0.1)
    #     pp.create_ext_grid(ntw, b4)
    #
    #     pp.create_line(ntw, b2, b3, 2.0, std_type="NAYY 4x50 SE")
    #     pp.create_line(ntw, b2, b4, 2.0, std_type="NAYY 4x50 SE")
    #     pp.create_transformer(ntw, b1, b2, std_type="0.63 MVA 10/0.4 kV")
    #     pp.create_transformer(ntw, b3, b4, std_type="0.63 MVA 10/0.4 kV")
    #
    #     bus_col = create_bus_collection(ntw, size=0.2, color="k")
    #     line_col = create_line_collection(ntw, use_line_geodata=False, color="k", linewidth=3.)
    #     lt, bt = create_trafo_collection(ntw, size=2, linewidth=3.)
    #     load_col1, load_col2 = create_load_collection(ntw, linewidth=2.,
    #                                                   infofunc=lambda x: ("load", x))
    #     gen1, gen2 = create_gen_collection(ntw, linewidth=2.,
    #                                        infofunc=lambda x: ("gen", x))
    #     eg1, eg2 = create_ext_grid_collection(ntw, size=2.,
    #                                           infofunc=lambda x: ("ext_grid", x))
    #
    #     draw_collections([bus_col, line_col, load_col1, load_col2, gen1, gen2, lt, bt, eg1, eg2])
    # else:
    #     pass
    pass
