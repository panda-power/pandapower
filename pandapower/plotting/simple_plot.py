# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import matplotlib.pyplot as plt

from pandapower.plotting.plotting_toolbox import get_collection_sizes
from pandapower.plotting.collections import create_bus_collection, create_line_collection, \
    create_trafo_collection, create_trafo3w_collection, \
    create_line_switch_collection, draw_collections, create_bus_bus_switch_collection, create_sgen_collection, \
    create_load_collection
from pandapower.plotting.generic_geodata import create_generic_coordinates

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def simple_plot(net, respect_switches=False, line_width=1.0, bus_size=1.0, ext_grid_size=1.0,
                trafo_size=1.0, plot_loads=False, plot_sgens=False, load_size=1.0, sgen_size=1.0,
                switch_size=2.0, switch_distance=1.0, plot_line_switches=False, scale_size=True,
                bus_color="b", line_color='grey', trafo_color='k', ext_grid_color='y',
                switch_color='k', library="igraph", show_plot=True, ax=None):
    """
    Plots a pandapower network as simple as possible. If no geodata is available, artificial
    geodata is generated. For advanced plotting see the tutorial

    INPUT:
        **net** - The pandapower format network.

    OPTIONAL:
        **respect_switches** (bool, False) - Respect switches if artificial geodata is created.

                                            .. note::
                                                This Flag is ignored if plot_line_switches is True

        **line_width** (float, 1.0) - width of lines

        **bus_size** (float, 1.0) - Relative size of buses to plot.

            The value bus_size is multiplied with mean_distance_between_buses, which equals the
            distance between
            the max geoocord and the min divided by 200.
            mean_distance_between_buses = sum((net['bus_geodata'].max()
                                          - net['bus_geodata'].min()) / 200)

        **ext_grid_size** (float, 1.0) - Relative size of ext_grids to plot.

            See bus sizes for details. Note: ext_grids are plottet as rectangles

        **trafo_size** (float, 1.0) - Relative size of trafos to plot.

        **plot_loads** (bool, False) - Flag to decide whether load symbols should be drawn.

        **plot_sgens** (bool, False) - Flag to decide whether sgen symbols should be drawn.

        **load_size** (float, 1.0) - Relative size of loads to plot.

        **sgen_size** (float, 1.0) - Relative size of sgens to plot.

        **switch_size** (float, 2.0) - Relative size of switches to plot. See bus size for details

        **switch_distance** (float, 1.0) - Relative distance of the switch to its corresponding \
                                           bus. See bus size for details

        **plot_line_switches** (bool, False) - Flag if line switches are plotted

        **scale_size** (bool, True) - Flag if bus_size, ext_grid_size, bus_size- and distance \
                                      will be scaled with respect to grid mean distances

        **bus_color** (String, colors[0]) - Bus Color. Init as first value of color palette.
        Usually colors[0] = "b".

        **line_color** (String, 'grey') - Line Color. Init is grey

        **trafo_color** (String, 'k') - Trafo Color. Init is black

        **ext_grid_color** (String, 'y') - External Grid Color. Init is yellow

        **switch_color** (String, 'k') - Switch Color. Init is black

        **library** (String, "igraph") - library name to create generic coordinates (case of
            missing geodata). "igraph" to use igraph package or "networkx" to use networkx package.

        **show_plot** (bool, True) - Shows plot at the end of plotting

		**ax** (object, None) - matplotlib axis to plot to

    OUTPUT:
        **ax** - axes of figure
    """
    # don't hide lines if switches are plotted
    if plot_line_switches:
        respect_switches = False

    # create geocoord if none are available
    if len(net.line_geodata) == 0 and len(net.bus_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, respect_switches=respect_switches, library=library)

    if scale_size:
        # if scale_size -> calc size from distance between min and max geocoord
        sizes = get_collection_sizes(net, bus_size, ext_grid_size, switch_size,
                                     switch_distance, load_size, sgen_size, trafo_size)
        bus_size = sizes["bus"]
        ext_grid_size = sizes["ext_grid"]
        trafo_size = sizes["trafo"]
        sgen_size = sizes["sgen"]
        load_size = sizes["load"]
        switch_size = sizes["switch"]
        switch_distance = sizes["switch_distance"]

    # create bus collections to plot
    bc = create_bus_collection(net, net.bus.index, size=bus_size, color=bus_color, zorder=10)

    # if bus geodata is available, but no line geodata
    use_bus_geodata = len(net.line_geodata) == 0
    in_service_lines = net.line[net.line.in_service].index
    nogolines = set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)]) \
        if respect_switches else set()
    plot_lines = in_service_lines.difference(nogolines)

    # create line collections
    lc = create_line_collection(net, plot_lines, color=line_color, linewidths=line_width,
                                use_bus_geodata=use_bus_geodata)
    collections = [bc, lc]

    # create ext_grid collections
    eg_buses_with_geo_coordinates = set(net.ext_grid.bus.values) & set(net.bus_geodata.index)
    if len(eg_buses_with_geo_coordinates) > 0:
        sc = create_bus_collection(net, eg_buses_with_geo_coordinates, patch_type="rect",
                                   size=ext_grid_size, color=ext_grid_color, zorder=11)
        collections.append(sc)

    # create trafo collection if trafo is available
    trafo_buses_with_geo_coordinates = [t for t, trafo in net.trafo.iterrows()
                                        if trafo.hv_bus in net.bus_geodata.index and
                                        trafo.lv_bus in net.bus_geodata.index]
    if len(trafo_buses_with_geo_coordinates) > 0:
        tc = create_trafo_collection(net, trafo_buses_with_geo_coordinates,
                                     color=trafo_color, size=trafo_size)
        collections.append(tc)

    # create trafo3w collection if trafo3w is available
    trafo3w_buses_with_geo_coordinates = [
        t for t, trafo3w in net.trafo3w.iterrows() if trafo3w.hv_bus in net.bus_geodata.index and
                                                      trafo3w.mv_bus in net.bus_geodata.index and trafo3w.lv_bus in net.bus_geodata.index]
    if len(trafo3w_buses_with_geo_coordinates) > 0:
        tc = create_trafo3w_collection(net, trafo3w_buses_with_geo_coordinates,
                                       color=trafo_color)
        collections.append(tc)

    if plot_line_switches and len(net.switch):
        sc = create_line_switch_collection(
            net, size=switch_size, distance_to_bus=switch_distance,
            use_line_geodata=not use_bus_geodata, zorder=12, color=switch_color)
        collections.append(sc)

    if plot_sgens and len(net.sgen):
        sgc = create_sgen_collection(net, size=sgen_size)
        collections.append(sgc)
    if plot_loads and len(net.load):
        lc = create_load_collection(net, size=load_size)
        collections.append(lc)

    if len(net.switch):
        bsc = create_bus_bus_switch_collection(net, size=switch_size)
        collections.append(bsc)

    ax = draw_collections(collections, ax=ax)
    if show_plot:
        plt.show()
    return ax


if __name__ == "__main__":
    import pandapower.networks as nw

    net = nw.case145()
    #    net = nw.create_cigre_network_mv()
    #    net = nw.mv_oberrhein()
    simple_plot(net, bus_size=0.4)
