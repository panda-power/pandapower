# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap, Normalize, \
    LogNorm
import numpy as np


def cmap_discrete(cmap_list):
    """
    Can be used to create a discrete colormap.

    INPUT:
        - cmap_list (list) - list of tuples, where each tuple represents one range. Each tuple has
                             the form of ((from, to), color).

    OUTPUT:
        - cmap - matplotlib colormap

        - norm - matplotlib norm object

    EXAMPLE:
        >>> from pandapower.plotting import cmap_discrete, create_line_collection, draw_collections
        >>> from pandapower.networks import mv_oberrhein
        >>> net = mv_oberrhein("generation")
        >>> cmap_list = [((0, 10), "green"), ((10, 30), "yellow"), ((30, 100), "red")]
        >>> cmap, norm = cmap_discrete(cmap_list)
        >>> lc = create_line_collection(net, cmap=cmap, norm=norm)
        >>> draw_collections([lc])
    """
    cmap_colors = []
    boundaries = []
    last_upper = None
    for (lower, upper), color in cmap_list:
        if last_upper is not None and lower != last_upper:
            raise ValueError("Ranges for colormap must be continuous")
        cmap_colors.append(color)
        boundaries.append(lower)
        last_upper = upper
    boundaries.append(upper)
    cmap = ListedColormap(cmap_colors)
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def cmap_continuous(cmap_list):
    """
    Can be used to create a continuous colormap.

    INPUT:
        - cmap_list (list) - list of tuples, where each tuple represents one color. Each tuple has
                             the form of (center, color). The colorbar is a linear segmentation of
                             the colors between the centers.

    OUTPUT:
        - cmap - matplotlib colormap

        - norm - matplotlib norm object

    EXAMPLE:
        >>> from pandapower.plotting import cmap_continuous, create_bus_collection, draw_collections
        >>> from pandapower.networks import mv_oberrhein
        >>> net = mv_oberrhein("generation")
        >>> cmap_list = [(0.97, "blue"), (1.0, "green"), (1.03, "red")]
        >>> cmap, norm = cmap_continuous(cmap_list)
        >>> bc = create_bus_collection(net, size=70, cmap=cmap, norm=norm)
        >>> draw_collections([bc])
    """
    min_loading = cmap_list[0][0]
    max_loading = cmap_list[-1][0]
    cmap_colors = [((loading-min_loading)/(max_loading - min_loading), color) for
                 (loading, color) in cmap_list]
    cmap = LinearSegmentedColormap.from_list('name', cmap_colors)
    norm = Normalize(min_loading, max_loading)
    return cmap, norm


def cmap_logarithmic(min_value, max_value, colors):
    """
    Can be used to create a logarithmic colormap. The colormap itself has a linear segmentation of
    the given colors. The values however will be matched to the colors based on a logarithmic
    normalization (c.f. matplotlib.colors.LogNorm for more information on how the logarithmic
    normalization works).\nPlease note: There are numerous ways of how a logarithmic scale might
    be created, the intermediate values on the scale are created automatically based on the minimum
    and maximum given values in analogy to the LogNorm. Also, the logarithmic colormap can only be
    used with at least 3 colors and increasing values which all have to be above 0.

    INPUT:
        - min_value (float) - the minimum value of the colorbar

        - max_value (float) - the maximum value for the colorbar

        - colors (list) - list of colors to be used for the colormap

    OUTPUT:
        - cmap - matplotlib colormap

        - norm - matplotlib norm object

    EXAMPLE:
        >>> from pandapower.plotting import cmap_logarithmic, create_bus_collection, draw_collections
        >>> from pandapower.networks import mv_oberrhein
        >>> net = mv_oberrhein("generation")
        >>> min_value, max_value = 1.0, 1.03
        >>> colors = ["blue", "green", "red"]
        >>> cmap, norm = cmap_logarithmic(min_value, max_value, colors)
        >>> bc = create_bus_collection(net, size=70, cmap=cmap, norm=norm)
        >>> draw_collections([bc])
    """
    num_values = len(colors)
    if num_values < 2:
        raise UserWarning("Cannot create a logarithmic colormap less than 2 colors.")
    if min_value <= 0:
        raise UserWarning("The minimum value must be above 0.")
    if max_value <= min_value:
        raise UserWarning("The upper bound must be larger than the lower bound.")
    values = np.arange(num_values + 1)
    diff = (max_value - min_value) / (num_values - 1)
    values = (np.log(min_value + values * diff) - np.log(min_value)) \
             / (np.log(max_value) - np.log(min_value))
    cmap = LinearSegmentedColormap.from_list("name", list(zip(values, colors)))
    norm = LogNorm(min_value, max_value)
    return cmap, norm
