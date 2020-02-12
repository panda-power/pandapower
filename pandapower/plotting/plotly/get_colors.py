# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mplc

import numpy as np

try:
    import seaborn
except ImportError:
    pass


def get_plotly_color(color_string):
    try:
        converted = _to_plotly_color(mplc.to_rgba(color_string))
        return converted
    except ValueError:
        return color_string


def get_plotly_color_palette(n):
    if 'seaborn' in sys.modules:
        return _to_plotly_palette(seaborn.color_palette("hls", n))
    else:
        hsv = plt.get_cmap('hsv')
        return _to_plotly_palette(hsv(np.linspace(0, 1.0, n)))


def _to_plotly_palette(scl, transparence=None):
    """
    converts a rgb color palette in format (0-1,0-1,0-1) to a plotly color palette
    'rgb(0-255,0-255,0-255)'
    """
    _out = []
    for color in scl:
        plotly_col = [255 * _c for _c in mplc.to_rgba(color)]
        if transparence:
            assert 0. <= transparence <= 1.0
            plotly_col[3] = transparence
            plotly_col = "rgba({:.0f}, {:.0f}, {:.0f}, {:.4f})".format(*plotly_col)
        else:
            plotly_col = "rgb({:.0f}, {:.0f}, {:.0f})".format(*plotly_col[:3])
        _out.append(plotly_col)
    return _out


def _to_plotly_color(scl, transparence=None):
    """
    converts a rgb color in format (0-1,0-1,0-1) to a plotly color 'rgb(0-255,0-255,0-255)'
    """
    plotly_col = [255 * _c for _c in mplc.to_rgba(scl)] if len(scl) == 3 else [255 * _c for _c in
                                                                               mplc.to_rgb(scl)]
    if transparence is not None:
        assert 0. <= transparence <= 1.0
        plotly_col[3] = transparence
        return "rgba({:.0f}, {:.0f}, {:.0f}, {:.4f})".format(*plotly_col)
    else:
        return "rgb({:.0f}, {:.0f}, {:.0f})".format(*plotly_col[:3])


def get_plotly_cmap(values, cmap_name='jet', cmin=None, cmax=None):
    cmap = cm.get_cmap(cmap_name)
    if cmin is None:
        cmin = values.min()
    if cmax is None:
        cmax = values.max()
    norm = mplc.Normalize(vmin=cmin, vmax=cmax)
    bus_fill_colors_rgba = cmap(norm(values).data)[:, 0:3] * 255.
    return ['rgb({0},{1},{2})'.format(r, g, b) for r, g, b in bus_fill_colors_rgba]
