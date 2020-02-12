# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd

from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting.plotly.traces import create_bus_trace, create_line_trace, \
    create_trafo_trace, draw_traces, version_check
from pandapower.plotting.plotly.mapbox_plot import *

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


def get_hoverinfo(net, element, precision=3, sub_index=None):
    hover_index = net[element].index
    if element == "bus":
        load_str, sgen_str = [], []
        for ln in [net.load.loc[net.load.bus == b, "p_mw"].sum() for b in net.bus.index]:
            load_str.append("Load: {:.3f} MW<br />".format(ln) if ln != 0. else "")
        for s in [net.sgen.loc[net.sgen.bus == b, "p_mw"].sum() for b in net.bus.index]:
            sgen_str.append("Static generation: {:.3f} MW<br />".format(s) if s != 0. else "")
        hoverinfo = (
                "Index: " + net.bus.index.astype(str) + '<br />' +
                "Name: " + net.bus['name'].astype(str) + '<br />' +
                'V_n: ' + net.bus['vn_kv'].round(precision).astype(str) + ' kV' + '<br />' + load_str + sgen_str)\
            .tolist()
    elif element == "line":
        hoverinfo = (
                "Index: " + net.line.index.astype(str) + '<br />' +
                "Name: " + net.line['name'].astype(str) + '<br />' +
                'Length: ' + net.line['length_km'].round(precision).astype(str) + ' km' + '<br />' +
                'R: ' + (net.line['length_km'] * net.line['r_ohm_per_km']).round(precision).astype(str)
                + ' Ohm' + '<br />'
                + 'X: ' + (net.line['length_km'] * net.line['x_ohm_per_km']).round(precision).astype(str)
                + ' Ohm' + '<br />').tolist()
    elif element == "trafo":
        hoverinfo = (
                "Index: " + net.trafo.index.astype(str) + '<br />' +
                "Name: " + net.trafo['name'].astype(str) + '<br />' +
                'V_n HV: ' + net.trafo['vn_hv_kv'].round(precision).astype(str) + ' kV' + '<br />' +
                'V_n LV: ' + net.trafo['vn_lv_kv'].round(precision).astype(str) + ' kV' + '<br />' +
                'Tap pos.: ' + net.trafo['tap_pos'].astype(str) + '<br />').tolist()
    elif element == "ext_grid":
        hoverinfo = (
                "Index: " + net.ext_grid.index.astype(str) + '<br />' +
                "Name: " + net.ext_grid['name'].astype(str) + '<br />' +
                'V_m: ' + net.ext_grid['vm_pu'].round(precision).astype(str) + ' p.u.' + '<br />' +
                'V_a: ' + net.ext_grid['va_degree'].round(precision).astype(str) + ' °' + '<br />').tolist()
        hover_index = net.ext_grid.bus.tolist()
    else:
        return None
    hoverinfo = pd.Series(index=hover_index, data=hoverinfo)
    if sub_index is not None:
        hoverinfo = hoverinfo.loc[list(sub_index)]
    return hoverinfo


def simple_plotly(net, respect_switches=True, use_line_geodata=None, on_map=False,
                  projection=None, map_style='basic', figsize=1, aspectratio='auto', line_width=1,
                  bus_size=10, ext_grid_size=20.0, bus_color="blue", line_color='grey',
                  trafo_color='green', ext_grid_color="yellow"):
    """
    Plots a pandapower network as simple as possible in plotly.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
            plotted as an example

    OPTIONAL:
        **respect_switches** (bool, True) - Respect switches when artificial geodata is created

        *use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata of the lines (True)
            or on net.bus_geodata of the connected buses (False)

        **on_map** (bool, False) - enables using mapbox plot in plotly.
            If provided geodata are not real geo-coordinates in lon/lat form, on_map will be set to False.

        **projection** (String, None) - defines a projection from which network geo-data will be transformed to
            lat-long. For each projection a string can be found at http://spatialreference.org/ref/epsg/


        **map_style** (str, 'basic') - enables using mapbox plot in plotly

            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the network geodata;
            any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **line_width** (float, 1.0) - width of lines

        **bus_size** (float, 10.0) -  size of buses to plot.

        **ext_grid_size** (float, 20.0) - size of ext_grids to plot.

            See bus sizes for details. Note: ext_grids are plotted as rectangles

        **bus_color** (String, "blue") - Bus Color. Init as first value of color palette.

        **line_color** (String, 'grey') - Line Color. Init is grey

        **trafo_color** (String, 'green') - Trafo Color. Init is green

        **ext_grid_color** (String, 'yellow') - External Grid Color. Init is yellow

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object
    """
    version_check()
    # create geocoord if none are available
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x", "y"])
    if len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time...")
        create_generic_coordinates(net, respect_switches=respect_switches)
        if on_map:
            logger.warning("Map plots not available with artificial coordinates and will be disabled!")
            on_map = False

    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
        geo_data_to_latlong(net, projection=projection)

    # ----- Buses ------
    # initializating bus trace
    hoverinfo = get_hoverinfo(net, element="bus")
    bus_trace = create_bus_trace(net, net.bus.index, size=bus_size, color=bus_color, infofunc=hoverinfo)

    # ----- Lines ------
    # if bus geodata is available, but no line geodata
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning("No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False

    hoverinfo = get_hoverinfo(net, element="line")
    line_traces = create_line_trace(net, net.line.index, respect_switches=respect_switches,
                                    color=line_color, width=line_width,
                                    use_line_geodata=use_line_geodata, infofunc=hoverinfo)

    # ----- Trafos ------
    hoverinfo = get_hoverinfo(net, element="trafo")
    trafo_trace = create_trafo_trace(net, color=trafo_color, width=line_width * 5, infofunc=hoverinfo,
                                     use_line_geodata=use_line_geodata)

    # ----- Ext grid ------
    # get external grid from create_bus_trace
    marker_type = 'circle' if on_map else 'square'  # workaround because doesn't appear on mapbox if square
    hoverinfo = get_hoverinfo(net, element="ext_grid")
    ext_grid_trace = create_bus_trace(net, buses=net.ext_grid.bus,
                                      color=ext_grid_color, size=ext_grid_size,
                                      patch_type=marker_type, trace_name='external_grid', infofunc=hoverinfo)

    return draw_traces(line_traces + trafo_trace + ext_grid_trace + bus_trace,
                       aspectratio=aspectratio, figsize=figsize, on_map=on_map, map_style=map_style)
