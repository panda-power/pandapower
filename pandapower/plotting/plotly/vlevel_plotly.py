# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd

from pandapower.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting.plotly.traces import create_bus_trace, create_line_trace, \
    create_trafo_trace, draw_traces, version_check
from pandapower.plotting.plotly.get_colors import get_plotly_color_palette
from pandapower.plotting.plotly.mapbox_plot import geo_data_to_latlong
from pandapower.topology import create_nxgraph, connected_components

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


def vlevel_plotly(net, respect_switches=True, use_line_geodata=None, colors_dict=None, on_map=False,
                  projection=None, map_style='basic', figsize=1, aspectratio='auto', line_width=2,
                  bus_size=10):
    """
    Plots a pandapower network in plotly
    using lines/buses colors according to the voltage level they belong to.
    If no geodata is available, artificial geodata is generated. For advanced plotting see the
    tutorial

    INPUT:
        **net** - The pandapower format network. If none is provided, mv_oberrhein() will be
        plotted as an example

    OPTIONAL:
        **respect_switches** (bool, True) - Respect switches when artificial geodata is created

        **use_line_geodata** (bool, True) - defines if lines patches are based on net.line_geodata
            of the lines (True) or on net.bus_geodata of the connected buses (False)

        *colors_dict** (dict, None) - dictionary for customization of colors for each voltage level
            in the form: voltage : color

        **on_map** (bool, False) - enables using mapbox plot in plotly If provided geodata are not
            real geo-coordinates in lon/lat form, on_map will be set to False.

        **projection** (String, None) - defines a projection from which network geo-data will be
            transformed to lat-long. For each projection a string can be found at
            http://spatialreference.org/ref/epsg/

        **map_style** (str, 'basic') - enables using mapbox plot in plotly

            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the
            network geodata any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **line_width** (float, 1.0) - width of lines

        **bus_size** (float, 10.0) -  size of buses to plot.

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object

    """
    version_check()
    # create geocoord if none are available
    if 'line_geodata' not in net:
        net.line_geodata = pd.DataFrame(columns=['coords'])
    if 'bus_geodata' not in net:
        net.bus_geodata = pd.DataFrame(columns=["x", "y"])
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=respect_switches)
        if on_map:
            logger.warning(
                "Map plots not available with artificial coordinates and will be disabled!")
            on_map = False

    # check if geodata are real geographycal lat/lon coordinates using geopy
    if on_map and projection is not None:
        geo_data_to_latlong(net, projection=projection)

    # if bus geodata is available, but no line geodata
    if use_line_geodata is None:
        use_line_geodata = False if len(net.line_geodata) == 0 else True
    elif use_line_geodata and len(net.line_geodata) == 0:
        logger.warning(
            "No or insufficient line geodata available --> only bus geodata will be used.")
        use_line_geodata = False

    # getting connected componenets without consideration of trafos
    graph = create_nxgraph(net, include_trafos=False)
    vlev_buses = connected_components(graph)
    # getting unique sets of buses for each voltage level
    vlev_bus_dict = {}
    for vl_buses in vlev_buses:
        if net.bus.loc[vl_buses, 'vn_kv'].unique().shape[0] > 1:
            logger.warning('buses from the same voltage level does not have the same vn_kv !?')
        vn_kv = net.bus.loc[vl_buses, 'vn_kv'].unique()[0]
        if vlev_bus_dict.get(vn_kv):
            vlev_bus_dict[vn_kv].update(vl_buses)
        else:
            vlev_bus_dict[vn_kv] = vl_buses

    # create a default colormap for voltage levels
    nvlevs = len(vlev_bus_dict)
    colors = get_plotly_color_palette(nvlevs)
    colors_dict = colors_dict or dict(zip(vlev_bus_dict.keys(), colors))

    # creating traces for buses and lines for each voltage level
    bus_traces = []
    line_traces = []
    for vn_kv, buses_vl in vlev_bus_dict.items():

        vlev_color = colors_dict[vn_kv]
        bus_trace_vlev = create_bus_trace(
            net, buses=buses_vl, size=bus_size, legendgroup=str(vn_kv), color=vlev_color,
            trace_name='buses {0} kV'.format(vn_kv))
        if bus_trace_vlev is not None:
            bus_traces += bus_trace_vlev

        vlev_lines = net.line[net.line.from_bus.isin(buses_vl) &
                              net.line.to_bus.isin(buses_vl)].index.tolist()
        line_trace_vlev = create_line_trace(
            net, lines=vlev_lines, use_line_geodata=use_line_geodata,
            respect_switches=respect_switches, legendgroup=str(vn_kv), color=vlev_color,
            width=line_width, trace_name='lines {0} kV'.format(vn_kv))
        if line_trace_vlev is not None:
            line_traces += line_trace_vlev

    trafo_traces = create_trafo_trace(net, color='gray', width=line_width * 2)

    return draw_traces(line_traces + trafo_traces + bus_traces, showlegend=True,
                       aspectratio=aspectratio, on_map=on_map, map_style=map_style, figsize=figsize)


if __name__ == '__main__':
    from pandapower.plotting.plotly import simple_plotly
    from pandapower.networks import mv_oberrhein
    from pandapower import runpp
    net = mv_oberrhein()
    vlevel_plotly(net)
    runpp(net)
    line_width = 2
    bus_size = 10
    use_line_geodata = None
    graph = create_nxgraph(net, include_trafos=False)
    vlev_buses = connected_components(graph)
    respect_switches = True

    # getting unique sets of buses for each voltage level
    vlev_bus_dict = {}
    for vl_buses in vlev_buses:
        if net.bus.loc[vl_buses, 'vn_kv'].unique().shape[0] > 1:
            logger.warning('buses from the same voltage level does not have the same vn_kv !?')
        vn_kv = net.bus.loc[vl_buses, 'vn_kv'].unique()[0]
        if vlev_bus_dict.get(vn_kv):
            vlev_bus_dict[vn_kv].update(vl_buses)
        else:
            vlev_bus_dict[vn_kv] = vl_buses

    # create a default colormap for voltage levels
    nvlevs = len(vlev_bus_dict)
    colors = get_plotly_color_palette(nvlevs)
    colors_dict = dict(zip(vlev_bus_dict.keys(), colors))

    # creating traces for buses and lines for each voltage level
    bus_traces = []
    line_traces = []
    for vn_kv, buses_vl in vlev_bus_dict.items():

        vlev_color = colors_dict[vn_kv]
        bus_trace_vlev = create_bus_trace(
            net, buses=buses_vl, size=bus_size, legendgroup=str(vn_kv),
            color=vlev_color, trace_name='buses {0} kV'.format(vn_kv))
        if bus_trace_vlev is not None:
            bus_traces += bus_trace_vlev

        vlev_lines = net.line[net.line.from_bus.isin(buses_vl) &
                              net.line.to_bus.isin(buses_vl)].index.tolist()
        print(vlev_lines)
        line_trace_vlev = create_line_trace(
            net, lines=vlev_lines, use_line_geodata=use_line_geodata,
            respect_switches=respect_switches, legendgroup=str(vn_kv), color="r",
            width=line_width, trace_name='lines {0} kV'.format(vn_kv))
        if line_trace_vlev is not None:
            line_traces += line_trace_vlev
