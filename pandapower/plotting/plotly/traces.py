# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import math

import numpy as np
import pandas as pd
from packaging import version
from collections.abc import Iterable

from pandapower.plotting.plotly.get_colors import get_plotly_color, get_plotly_cmap
from pandapower.plotting.plotly.mapbox_plot import _on_map_test, _get_mapbox_token, \
    MapboxTokenMissing

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

try:
    from plotly import __version__ as plotly_version
    from plotly.graph_objs.scatter.marker import ColorBar
    from plotly.graph_objs import Figure, Layout
    from plotly.graph_objs.layout import XAxis, YAxis
    from plotly.graph_objs.scatter import Line, Marker
    from plotly.graph_objs.scattermapbox import Line as scmLine
    from plotly.graph_objs.scattermapbox import Marker as scmMarker
except ImportError:
    logger.info("Failed to import plotly - interactive plotting will not be available")


def version_check():
    if "plotly_version" not in locals() and "plotly_version" not in globals():
        raise UserWarning("You are trying to use plotly, which is not installed.\r\n"
                          "Please upgrade your python-plotly installation, "
                          "e.g., via pip install --upgrade plotly")
    if version.parse(plotly_version) < version.parse("3.1.1"):
        raise UserWarning("Your plotly version {} is no longer supported.\r\n"
                          "Please upgrade your python-plotly installation, "
                          "e.g., via pip install --upgrade plotly".format(__version__))


def _in_ipynb():
    """
    an auxiliary function which checks if plot is called from a jupyter-notebook or not
    """
    import __main__ as main
    return not hasattr(main, '__file__')


def sum_line_length(pts):
    pt_diff = lambda p: (p[0][0] - p[1][0], p[0][1] - p[1][1])
    diffs = map(pt_diff, zip(pts[:-1], pts[1:]))
    line_length = sum(math.hypot(d1, d2) for d1, d2 in diffs)
    return line_length


def get_line_neutral(coord):
    if len(coord) == 1:
        return coord[0]
    half_length = sum_line_length(coord) / 2.0
    length = 0.0
    ind = 0
    while length < half_length:
        ind += 1
        length = sum_line_length(coord[:ind])

    start_coord = coord[ind - 2]
    end_coord = coord[ind - 1]
    mid = [(a1 + a2) / 2.0 for a1, a2 in zip(start_coord, end_coord)]

    return mid


def create_edge_center_trace(line_trace, size=1, patch_type="circle", color="white", infofunc=None,
                             trace_name='edge_center', use_line_geodata=False):
    """
    Creates a plotly trace of pandapower buses.

    INPUT:
        **line traces** (from pandapowerNet) - The already generated line traces with center geodata

    OPTIONAL:

        **size** (int, 5) - patch size

        **patch_type** (str, "circle") - patch type, can be

                - "circle" for a circle
                - "square" for a rectangle
                - "diamond" for a diamond
                - much more pathc types at https://plot.ly/python/reference/#scatter-marker

        **infofunc** (pd.Series, None) - hoverinfo for each trace element. Indices should correspond
            to the pandapower element indices

        **trace_name** (String, "buses") - name of the trace which will appear in the legend

        **color** (String, "blue") - color of buses in the trace

    """
    # color = get_plotly_color(color)

    center_trace = dict(type='scatter', text=[], mode='markers', hoverinfo='text', name=trace_name,
                        marker=dict(color=color, size=size, symbol=patch_type))

    if not use_line_geodata:
        center_trace['x'], center_trace['y'] = (line_trace[0]["x"][1::4], line_trace[0]["y"][1::4])
    else:
        x, y = [], []
        for trace in line_trace:
            coord = list(zip(trace["x"], trace["y"]))
            mid_coord = get_line_neutral(coord)
            x.append(mid_coord[0])
            y.append(mid_coord[1])

        center_trace['x'], center_trace['y'] = (x, y)

    center_trace['text'] = infofunc

    return center_trace


def create_bus_trace(net, buses=None, size=5, patch_type="circle", color="blue", infofunc=None,
                     trace_name='buses', legendgroup=None, cmap=None, cmap_vals=None,
                     cbar_title=None, cmin=None, cmax=None, cpos=1.0, colormap_column="vm_pu"):
    """
    Creates a plotly trace of pandapower buses.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **buses** (list, None) - The buses for which the collections are created.
        If None, all buses in the network are considered.

        **size** (int, 5) - patch size

        **patch_type** (str, "circle") - patch type, can be

                - "circle" for a circle
                - "square" for a rectangle
                - "diamond" for a diamond
                - much more pathc types at https://plot.ly/python/reference/#scatter-marker

        **infofunc** (pd.Series, None) - hoverinfo for bus elements. Indices should correspond to
            the pandapower element indices

        **trace_name** (String, "buses") - name of the trace which will appear in the legend

        **color** (String, "blue") - color of buses in the trace

        **cmap** (String, None) - name of a colormap which exists within plotly
            (Greys, YlGnBu, Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow,
            Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis) alternatively a custom
            discrete colormap can be used

        **cmap_vals** (list, None) - values used for coloring using colormap

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum

        **cpos** (float, 1.1) - position of the colorbar

        **colormap_column** (str, "vm_pu") - set color of bus according to this variable

    """
    color = get_plotly_color(color)

    bus_trace = dict(type='scatter', text=[], mode='markers', hoverinfo='text', name=trace_name,
                     marker=dict(color=color, size=size, symbol=patch_type))

    buses = net.bus.index.tolist() if buses is None else list(buses)
    bus_plot_index = [b for b in buses if b in list(set(buses) & set(net.bus_geodata.index))]

    bus_trace['x'], bus_trace['y'] = (net.bus_geodata.loc[bus_plot_index, 'x'].tolist(),
                                      net.bus_geodata.loc[bus_plot_index, 'y'].tolist())

    if not isinstance(infofunc, pd.Series) and isinstance(infofunc, Iterable) and \
            len(infofunc) == len(buses):
        infofunc = pd.Series(index=buses, data=infofunc)

    bus_trace['text'] = net.bus.loc[bus_plot_index, 'name'] if infofunc is None else \
        infofunc.loc[buses]

    if legendgroup:
        bus_trace['legendgroup'] = legendgroup

    # if color map is set
    if cmap is not None:
        # TODO introduce discrete colormaps (see contour plots in plotly)
        # if cmap_vals are not given

        cmap = 'Jet' if cmap is True else cmap

        if cmap_vals is not None:
            cmap_vals = cmap_vals
        else:
            if net.res_line.shape[0] == 0:
                logger.error("There are no power flow results for buses voltage magnitudes which are default for bus "
                             "colormap coloring..."
                             "set cmap_vals input argument if you want colormap according to some specific values...")
            cmap_vals = net.res_bus.loc[bus_plot_index, colormap_column].values

        cmap_vals = net.res_bus.loc[bus_plot_index, colormap_column] if cmap_vals is None else cmap_vals

        cmin = cmin if cmin else cmap_vals.min()
        cmax = cmax if cmax else cmap_vals.max()

        bus_trace['marker'] = Marker(size=size,
                                     color=cmap_vals, cmin=cmin, cmax=cmax,
                                     colorscale=cmap,
                                     colorbar=ColorBar(thickness=10,
                                                       x=cpos),
                                     symbol=patch_type
                                     )

        if cbar_title:
            bus_trace['marker']['colorbar']['title'] = cbar_title

        bus_trace['marker']['colorbar']['title']['side'] = 'right'

    return [bus_trace]


def _get_line_geodata_plotly(net, lines, use_line_geodata):
    xs = []
    ys = []
    if use_line_geodata:
        for line_ind, _ in lines.iterrows():
            line_coords = net.line_geodata.loc[line_ind, 'coords']
            linex, liney = list(zip(*line_coords))
            xs += linex
            xs += [None]
            ys += liney
            ys += [None]
    else:
        # getting x and y values from bus_geodata for from and to side of each line

        from_bus = net.bus_geodata.loc[lines.from_bus, 'x'].tolist()
        to_bus = net.bus_geodata.loc[lines.to_bus, 'x'].tolist()
        # center point added because of the hovertool
        center = (np.array(from_bus) + np.array(to_bus)) / 2
        none_list = [None] * len(from_bus)
        xs = np.array([from_bus, center, to_bus, none_list]).T.flatten().tolist()

        from_bus = net.bus_geodata.loc[lines.from_bus, 'y'].tolist()
        to_bus = net.bus_geodata.loc[lines.to_bus, 'y'].tolist()
        # center point added because of the hovertool
        center = (np.array(from_bus) + np.array(to_bus)) / 2
        none_list = [None] * len(from_bus)
        ys = np.array([from_bus, center, to_bus, none_list]).T.flatten().tolist()

    # [:-1] is because the trace will not appear on maps if None is at the end
    return xs[:-1], ys[:-1]


def create_line_trace(net, lines=None, use_line_geodata=True, respect_switches=False, width=1.0,
                      color='grey', infofunc=None, trace_name='lines', legendgroup=None,
                      cmap=None, cbar_title=None, show_colorbar=True, cmap_vals=None, cmin=None,
                      cmax=None, cpos=1.1):
    """
    Creates a plotly trace of pandapower lines.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **lines** (list, None) - The lines for which the collections are created.
        If None, all lines in the network are considered.

        **width** (int, 1) - line width

        **respect_switches** (bool, False) - flag for consideration of disconnected lines

        **infofunc** (pd.Series, None) - hoverinfo for line elements. Indices should correspond to
            the pandapower element indices

        **trace_name** (String, "lines") - name of the trace which will appear in the legend

        **color** (String, "grey") - color of lines in the trace

        **legendgroup** (String, None) - defines groups of layers that will be displayed in a legend
        e.g. groups according to voltage level (as used in `vlevel_plotly`)

        **cmap** (String, None) - name of a colormap which exists within plotly if set to True default `Jet`
        colormap is used, alternative colormaps : Greys, YlGnBu, Greens, YlOrRd,
        Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot, Blackbody, Earth, Electric, Viridis

        **cmap_vals** (list, None) - values used for coloring using colormap

        **show_colorbar** (bool, False) - flag for showing or not corresponding colorbar

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum

        **cpos** (float, 1.1) - position of the colorbar

        """

    color = get_plotly_color(color)

    # defining lines to be plot
    lines = net.line.index.tolist() if lines is None else list(lines)
    if len(lines) == 0:
        return []

    if infofunc is not None:
        if not isinstance(infofunc, pd.Series) and isinstance(infofunc, Iterable) and \
                len(infofunc) == len(lines):
            infofunc = pd.Series(index=lines, data=infofunc)
        if len(infofunc) != len(lines) and len(infofunc) != len(net.line):
            raise UserWarning("Different amount of hover info than lines to plot")
        assert isinstance(infofunc, pd.Series), \
            "infofunc should be a pandas series with the net.line.index to the infofunc contents"

    no_go_lines = set()
    if respect_switches:
        no_go_lines = set(lines) & set(net.switch.element[(net.switch.et == "l") &
                                                          (net.switch.closed == 0)])

    lines_to_plot = net.line.loc[set(net.line.index) & (set(lines) - no_go_lines)]
    no_go_lines_to_plot = None
    use_line_geodata = use_line_geodata if net.line_geodata.shape[0] > 0 else False

    if use_line_geodata:
        lines_to_plot = lines_to_plot.loc[set(lines_to_plot.index) & set(net.line_geodata.index)]
    else:
        lines_with_geodata = lines_to_plot.from_bus.isin(net.bus_geodata.index) & \
                             lines_to_plot.to_bus.isin(net.bus_geodata.index)
        lines_to_plot = lines_to_plot.loc[lines_with_geodata]

    cmap_lines = None
    if cmap is not None:
        # workaround: if colormap plot is used, each line need to be separate scatter object because
        # plotly still doesn't support appropriately colormap for line objects
        # TODO correct this when plotly solves existing github issue about Line colorbar

        cmap = 'jet' if cmap is True else cmap

        if cmap_vals is not None:
            if not isinstance(cmap_vals, np.ndarray):
                cmap_vals = np.asarray(cmap_vals)
        else:
            if net.res_line.shape[0] == 0:
                logger.error("There are no power flow results for lines which are default for line colormap coloring..."
                             "set cmap_vals input argument if you want colormap according to some specific values...")
            cmap_vals = net.res_line.loc[lines_to_plot.index, 'loading_percent'].values

        cmap_lines = get_plotly_cmap(cmap_vals, cmap_name=cmap, cmin=cmin, cmax=cmax)
        if len(cmap_lines) == len(net.line):
            # some lines are not plotted although cmap_value were provided for all lines
            line_idx_map = dict(zip(net.line.loc[lines].index.tolist(), range(len(lines))))
            cmap_lines = [cmap_lines[line_idx_map[idx]] for idx in lines_to_plot.index]
        else:
            assert len(cmap_lines) == len(lines_to_plot), \
                "Different amounts of cmap values and lines to plot were supplied"

    line_traces = []
    for col_i, (idx, line) in enumerate(lines_to_plot.iterrows()):
        line_color = color
        line_info = line['name']
        if cmap is not None:
            try:
                line_color = cmap_lines[col_i]
                line_info = line['name'] if infofunc is None else infofunc.loc[idx]
            except IndexError:
                logger.warning("No color and info for line {:d} (name: {}) available".format(
                    idx, line['name']))

        line_trace = dict(type='scatter', text=[], hoverinfo='text', mode='lines', name=trace_name,
                          line=Line(width=width, color=color))

        line_trace['x'], line_trace['y'] = _get_line_geodata_plotly(net, lines_to_plot.loc[idx:idx],
                                                                    use_line_geodata)

        line_trace['line']['color'] = line_color

        line_trace['text'] = line_info

        line_traces.append(line_trace)

    if show_colorbar and cmap is not None:

        cmin = cmin if cmin else cmap_vals.min()
        cmax = cmax if cmax else cmap_vals.max()
        try:
            # TODO for custom colormaps
            cbar_cmap_name = 'Jet' if cmap == 'jet' else cmap
            # workaround to get colorbar for lines (an unvisible node is added)
            # get x and y of first line.from_bus:
            x = [net.bus_geodata.x[net.line.from_bus[net.line.index[0]]]]
            y = [net.bus_geodata.y[net.line.from_bus[net.line.index[0]]]]
            lines_cbar = dict(type='scatter', x=x, y=y, mode='markers',
                              marker=Marker(size=0, cmin=cmin, cmax=cmax,
                                            color='rgb(255,255,255)',
                                            opacity=0,
                                            colorscale=cbar_cmap_name,
                                            colorbar=ColorBar(thickness=10,
                                                              x=cpos),
                                            ))
            if cbar_title:
                lines_cbar['marker']['colorbar']['title'] = cbar_title

            lines_cbar['marker']['colorbar']['title']['side'] = 'right'

            line_traces.append(lines_cbar)
        except:
            pass

    if len(no_go_lines) > 0:
        no_go_lines_to_plot = net.line.loc[no_go_lines]
        for idx, line in no_go_lines_to_plot.iterrows():
            line_color = color
            line_trace = dict(type='scatter',
                              text=[], hoverinfo='text', mode='lines', name='disconnected lines',
                              line=Line(width=width / 2, color='grey', dash='dot'))

            line_trace['x'], line_trace['y'] = _get_line_geodata_plotly(
                net, no_go_lines_to_plot.loc[idx:idx], use_line_geodata)

            line_trace['line']['color'] = line_color
            try:
                line_trace['text'] = infofunc.loc[idx]
            except (KeyError, IndexError, AttributeError):
                line_trace["text"] = line['name']

            line_traces.append(line_trace)

            if legendgroup:
                line_trace['legendgroup'] = legendgroup

    # sort infofunc so that it is the correct order lines_to_plot + no_go_lines_to_plot
    if infofunc is not None:
        if not isinstance(infofunc, pd.Series) and isinstance(infofunc, Iterable) and \
                len(infofunc) == len(net.line):
            infofunc = pd.Series(index=net.line.index, data=infofunc)
        assert isinstance(infofunc, pd.Series), \
            "infofunc should be a pandas series with the net.line.index to the infofunc contents"
        sorted_idx = lines_to_plot.index.tolist()
        if no_go_lines_to_plot is not None:
            sorted_idx += no_go_lines_to_plot.index.tolist()
        infofunc = infofunc.loc[sorted_idx]

    center_trace = create_edge_center_trace(line_traces, color=color, infofunc=infofunc,
                                            use_line_geodata=use_line_geodata)
    line_traces.append(center_trace)
    return line_traces


def create_trafo_trace(net, trafos=None, color='green', width=5, infofunc=None, cmap=None,
                       trace_name='trafos', cmin=None, cmax=None, cmap_vals=None,
                       use_line_geodata=None):
    """
    Creates a plotly trace of pandapower trafos.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTIONAL:
        **trafos** (list, None) - The trafos for which the collections are created.
        If None, all trafos in the network are considered.

        **width** (int, 5) - line width

        **infofunc** (pd.Series, None) - hoverinfo for trafo elements. Indices should correspond
            to the pandapower element indices

        **trace_name** (String, "lines") - name of the trace which will appear in the legend

        **color** (String, "green") - color of lines in the trace

        **cmap** (bool, False) - name of a colormap which exists within plotly (Greys, YlGnBu,
            Greens, YlOrRd, Bluered, RdBu, Reds, Blues, Picnic, Rainbow, Portland, Jet, Hot,
            Blackbody, Earth, Electric, Viridis)

        **cmap_vals** (list, None) - values used for coloring using colormap

        **cbar_title** (String, None) - title for the colorbar

        **cmin** (float, None) - colorbar range minimum

        **cmax** (float, None) - colorbar range maximum


    """
    color = get_plotly_color(color)

    # defining lines to be plot
    trafos = net.trafo.index.tolist() if trafos is None else list(trafos)
    if len(trafos) == 0:
        return []

    trafo_buses_with_geodata = net.trafo.hv_bus.isin(net.bus_geodata.index) & \
        net.trafo.lv_bus.isin(net.bus_geodata.index)

    trafos_mask = net.trafo.index.isin(trafos)
    trafos_to_plot = net.trafo[trafo_buses_with_geodata & trafos_mask]

    if infofunc is not None:
        if not isinstance(infofunc, pd.Series) and isinstance(infofunc, Iterable) and \
                len(infofunc) == len(trafos):
            infofunc = pd.Series(index=trafos, data=infofunc)
        assert isinstance(infofunc, pd.Series), \
            "infofunc should be a pandas series with the net.trafo.index to the infofunc contents"
        infofunc = infofunc.loc[trafos_to_plot.index]

    cmap_colors = []
    if cmap is not None:
        cmap = 'jet' if cmap is None else cmap

        cmin = 0 if cmin is None else cmin
        cmax = 100 if cmin is None else cmax

        if cmap_vals is not None:
            cmap_vals = cmap_vals
        else:
            if net.res_trafo.shape[0] == 0:
                logger.error("There are no power flow results for lines which are default for line colormap coloring..."
                             "set cmap_vals input argument if you want colormap according to some specific values...")
            cmap_vals = net.res_trafo.loc[trafos_to_plot.index, 'loading_percent'].values

        cmap_colors = get_plotly_cmap(cmap_vals, cmap_name=cmap, cmin=cmin, cmax=cmax)

    trafo_traces = []
    for col_i, (idx, trafo) in enumerate(trafos_to_plot.iterrows()):
        if cmap is not None:
            color = cmap_colors[col_i]

        trafo_trace = dict(type='scatter', text=[], line=Line(width=width, color=color),
                           hoverinfo='text', mode='lines', name=trace_name)

        trafo_trace['text'] = trafo['name'] if infofunc is None else infofunc.loc[idx]

        from_bus = net.bus_geodata.loc[trafo.hv_bus, 'x']
        to_bus = net.bus_geodata.loc[trafo.lv_bus, 'x']
        trafo_trace['x'] = [from_bus, (from_bus + to_bus) / 2, to_bus]

        from_bus = net.bus_geodata.loc[trafo.hv_bus, 'y']
        to_bus = net.bus_geodata.loc[trafo.lv_bus, 'y']
        trafo_trace['y'] = [from_bus, (from_bus + to_bus) / 2, to_bus]

        trafo_traces.append(trafo_trace)

    center_trace = create_edge_center_trace(trafo_traces, color=color, infofunc=infofunc,
                                            use_line_geodata=use_line_geodata)
    trafo_traces.append(center_trace)
    return trafo_traces


def draw_traces(traces, on_map=False, map_style='basic', showlegend=True, figsize=1,
                aspectratio='auto', filename="temp-plot.html"):
    """
    plots all the traces (which can be created using :func:`create_bus_trace`, :func:`create_line_trace`,
    :func:`create_trafo_trace`)
    to PLOTLY (see https://plot.ly/python/)

    INPUT:
        **traces** - list of dicts which correspond to plotly traces
        generated using: `create_bus_trace`, `create_line_trace`, `create_trafo_trace`

    OPTIONAL:
        **on_map** (bool, False) - enables using mapbox plot in plotly

        **map_style** (str, 'basic') - enables using mapbox plot in plotly

            - 'streets'
            - 'bright'
            - 'light'
            - 'dark'
            - 'satellite'

        **showlegend** (bool, 'True') - enables legend display

        **figsize** (float, 1) - aspectratio is multiplied by it in order to get final image size

        **aspectratio** (tuple, 'auto') - when 'auto' it preserves original aspect ratio of the
            network geodata any custom aspectration can be given as a tuple, e.g. (1.2, 1)

        **filename** (str, "temp-plot.html") - plots to a html file called filename

    OUTPUT:
        **figure** (graph_objs._figure.Figure) figure object

    """

    if on_map:
        try:
            on_map = _on_map_test(traces[0]['x'][0], traces[0]['y'][0])
        except:
            logger.warning("Test if geo-data are in lat/long cannot be performed using geopy -> "
                           "eventual plot errors are possible.")

        if on_map is False:
            logger.warning("Existing geodata are not real lat/lon geographical coordinates. -> "
                           "plot on maps is not possible.\n"
                           "Use geo_data_to_latlong(net, projection) to transform geodata from specific projection.")

    if on_map:
        # change traces for mapbox
        # change trace_type to scattermapbox and rename x to lat and y to lon
        for trace in traces:
            trace['lat'] = trace.pop('x')
            trace['lon'] = trace.pop('y')
            trace['type'] = 'scattermapbox'
            if "line" in trace and isinstance(trace["line"], Line):
                # scattermapboxplot lines do not support dash for some reason, make it a red line instead
                if "dash" in trace["line"]._props:
                    _prps = dict(trace["line"]._props)
                    _prps.pop("dash", None)
                    _prps["color"] = "red"
                    trace["line"] = scmLine(_prps)
                else:
                    trace["line"] = scmLine(dict(trace["line"]._props))
            elif "marker" in trace and isinstance(trace["marker"], Marker):
                trace["marker"] = scmMarker(trace["marker"]._props)

    # setting Figure object
    fig = Figure(data=traces,  # edge_trace
                 layout=Layout(
                     titlefont=dict(size=16),
                     showlegend=showlegend,
                     autosize=True if aspectratio is 'auto' else False,
                     hovermode='closest',
                     margin=dict(b=5, l=5, r=5, t=5),
                     # annotations=[dict(
                     #     text="",
                     #     showarrow=False,
                     #     xref="paper", yref="paper",
                     #     x=0.005, y=-0.002)],
                     xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                     yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False),
                     # legend=dict(x=0, y=1.0)
                 ), )

    # check if geodata are real geographycal lat/lon coordinates using geopy

    if on_map:
        try:
            mapbox_access_token = _get_mapbox_token()
        except Exception:
            logger.exception('mapbox token required for map plots. '
                             'Get Mapbox token by signing in to https://www.mapbox.com/.\n'
                             'After getting a token, set it to pandapower using:\n'
                             'pandapower.plotting.plotly.mapbox_plot.set_mapbox_token(\'<token>\')')
            raise MapboxTokenMissing

        fig['layout']['mapbox'] = dict(accesstoken=mapbox_access_token,
                                       bearing=0,
                                       center=dict(lat=pd.Series(traces[0]['lat']).dropna().mean(),
                                                   lon=pd.Series(traces[0]['lon']).dropna().mean()),
                                       style=map_style,
                                       pitch=0,
                                       zoom=11)

    # default aspectratio: if on_map use auto, else use 'original'
    aspectratio = 'original' if not on_map and aspectratio is 'auto' else aspectratio

    if aspectratio is not 'auto':
        if aspectratio is 'original':
            # TODO improve this workaround for getting original aspectratio
            xs = []
            ys = []
            for trace in traces:
                xs += trace['x']
                ys += trace['y']
            x_dropna = pd.Series(xs).dropna()
            y_dropna = pd.Series(ys).dropna()
            xrange = x_dropna.max() - x_dropna.min()
            yrange = y_dropna.max() - y_dropna.min()
            ratio = xrange / yrange
            if ratio < 1:
                aspectratio = (ratio, 1.)
            else:
                aspectratio = (1., 1 / ratio)

        aspectratio = np.array(aspectratio) / max(aspectratio)
        fig['layout']['width'], fig['layout']['height'] = ([ar * figsize * 700 for ar in aspectratio])

    # check if called from ipynb or not in order to consider appropriate plot function
    if _in_ipynb():
        from plotly.offline import init_notebook_mode, iplot as plot
        init_notebook_mode()
    else:
        from plotly.offline import plot as plot

    plot(fig, filename=filename)

    return fig
