# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd

from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting.plotly.traces import create_bus_trace, create_line_trace, \
    create_trafo_trace, draw_traces, version_check, _create_node_trace, _create_branch_trace
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

    elif element == "trafo3w":
        hoverinfo = (
                "Index: " + net.trafo3w.index.astype(str) + '<br />' +
                "Name: " + net.trafo3w['name'].astype(str) + '<br />' +
                'V_n HV: ' + net.trafo3w['vn_hv_kv'].round(precision).astype(str) + ' kV' + '<br />' +
                'V_n MV: ' + net.trafo3w['vn_mv_kv'].round(precision).astype(str) + ' kV' + '<br />' +
                'V_n LV: ' + net.trafo3w['vn_lv_kv'].round(precision).astype(str) + ' kV' + '<br />' +
                'Tap pos.: ' + net.trafo3w['tap_pos'].astype(str) + '<br />').tolist()



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
                  trafo_color='green',trafo3w_color='green', ext_grid_color="yellow", filename='temp-plot.html', auto_open=True):
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

        **trafo3w_color** (String, 'green') - Trafo 3W Color. Init is blue

        **ext_grid_color** (String, 'yellow') - External Grid Color. Init is yellow

        **auto_open** (bool, True) - automatically open plot in browser

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object
    """
    node_element = "bus"
    branch_element = "line"
    trans_element = "trafo"
    trans3w_element = "trafo3w"
    separator_element = "switch"
    return _simple_plotly_generic(net, respect_switches, use_line_geodata, on_map, projection,
                                  map_style, figsize, aspectratio, line_width, bus_size,
                                  ext_grid_size, bus_color, line_color, trafo_color,trafo3w_color, ext_grid_color,
                                  node_element, branch_element, trans_element, trans3w_element, separator_element,
                                  create_line_trace, create_bus_trace, get_hoverinfo, filename,
                                  auto_open=auto_open)


def _simple_plotly_generic(net, respect_separators, use_branch_geodata, on_map, projection, map_style,
                           figsize, aspectratio, branch_width, node_size, ext_grid_size, node_color,
                           branch_color, trafo_color,trafo3w_color, ext_grid_color, node_element, branch_element,
                           trans_element, trans3w_element, separator_element, branch_trace_func, node_trace_func,
                           hoverinfo_func, filename='temp-plot.html', auto_open=True):
    version_check()
    # create geocoord if none are available
    branch_geodata = branch_element + "_geodata"
    node_geodata = node_element + "_geodata"

    if branch_geodata not in net:
        net[branch_geodata] = pd.DataFrame(columns=['coords'])
    if node_geodata not in net:
        net[node_geodata] = pd.DataFrame(columns=["x", "y"])
    if len(net[node_geodata]) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time...")
        create_generic_coordinates(net, respect_switches=respect_separators)
        if on_map:
            logger.warning(
                "Map plots not available with artificial coordinates and will be disabled!")
            on_map = False
    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
        geo_data_to_latlong(net, projection=projection)
    # ----- Nodes (Buses) ------
    # initializating node trace
    hoverinfo = hoverinfo_func(net, element=node_element)
    node_trace = node_trace_func(net, net[node_element].index, size=node_size, color=node_color,
                                    infofunc=hoverinfo)
    # ----- branches (Lines) ------
    # if node geodata is available, but no branch geodata
    if use_branch_geodata is None:
        use_branch_geodata = False if len(net[branch_geodata]) == 0 else True
    elif use_branch_geodata and len(net[branch_geodata]) == 0:
        logger.warning(
            "No or insufficient line geodata available --> only bus geodata will be used.")
        use_branch_geodata = False
    hoverinfo = hoverinfo_func(net, element=branch_element)
    branch_traces = branch_trace_func(net, net[branch_element].index, use_branch_geodata,
                                      respect_separators,
                                      color=branch_color, width=branch_width,
                                      infofunc=hoverinfo)
    # ----- Trafos ------
    if 'trafo' in net:
        hoverinfo = hoverinfo_func(net, element=trans_element)
        trans_trace = create_trafo_trace(net, color=trafo_color, width=branch_width * 5,
                                         infofunc=hoverinfo,
                                         use_line_geodata=use_branch_geodata)
    # ----- 3W Trafos ------
    if 'trafo3w' in net:
        hoverinfo = hoverinfo_func(net, element=trans3w_element)
        trans_trace = create_trafo_trace(net, color=trafo3w_color, trafotype='3W', width=branch_width * 5, trace_name='3w_trafos',
                                         infofunc=hoverinfo,
                                         use_line_geodata=use_branch_geodata)
    else:
        trans_trace = []
    # ----- Ext grid ------
    # get external grid from _create_node_trace
    marker_type = 'circle' if on_map else 'square'  # workaround because doesn't appear on mapbox if square
    hoverinfo = hoverinfo_func(net, element="ext_grid")
    ext_grid_trace = _create_node_trace(net, nodes=net.ext_grid[node_element], size=ext_grid_size,
                                        patch_type=marker_type, color=ext_grid_color,
                                        infofunc=hoverinfo, trace_name='external_grid',
                                        node_element=node_element, branch_element=branch_element)
    return draw_traces(branch_traces + trans_trace + ext_grid_trace + node_trace,
                       aspectratio=aspectratio, figsize=figsize, on_map=on_map,
                       map_style=map_style, filename=filename, auto_open=auto_open)
