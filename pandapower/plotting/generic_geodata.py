# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy

import networkx as nx
import pandas as pd

import pandapower.topology as top

try:
    import igraph
    IGRAPH_INSTALLED = True
except ImportError:
    IGRAPH_INSTALLED = False

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def build_igraph_from_pp(net, respect_switches=False):
    """
    This function uses the igraph library to create an igraph graph for a given pandapower network.
    Lines, transformers and switches are respected.
    Performance vs. networkx: https://graph-tool.skewed.de/performance

    :param net: pandapower network
    :type net: pandapowerNet
    :param respect_switches: if True, exclude edges for open switches (also lines that are \
        connected via line switches)
    :type respect_switches: bool, default False

    :Example:
        graph, meshed, roots = build_igraph_from_pp(net)
    """
    try:
        import igraph as ig
    except (DeprecationWarning, ImportError):
        raise ImportError("Please install python-igraph")
    g = ig.Graph(directed=True)
    g.add_vertices(net.bus.shape[0])
    # g.vs["label"] = [s.encode('unicode-escape') for s in net.bus.name.tolist()]
    g.vs["label"] = net.bus.index.tolist()
    pp_bus_mapping = dict(list(zip(net.bus.index, list(range(net.bus.index.shape[0])))))

    # add lines
    nogolines = set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)]) \
        if respect_switches else set()
    for lix in (ix for ix in net.line.index if ix not in nogolines):
        fb, tb = net.line.at[lix, "from_bus"], net.line.at[lix, "to_bus"]
        g.add_edge(pp_bus_mapping[fb], pp_bus_mapping[tb], weight=net.line.at[lix, "length_km"])

    # add trafos
    for _, trafo in net.trafo.iterrows():
        g.add_edge(pp_bus_mapping[trafo.hv_bus], pp_bus_mapping[trafo.lv_bus], weight=0.01)

    for _, trafo3w in net.trafo3w.iterrows():
        g.add_edge(pp_bus_mapping[trafo3w.hv_bus], pp_bus_mapping[trafo3w.lv_bus], weight=0.01)
        g.add_edge(pp_bus_mapping[trafo3w.hv_bus], pp_bus_mapping[trafo3w.mv_bus], weight=0.01)

    # add switches
    bs = net.switch[(net.switch.et == "b") & (net.switch.closed == 1)] if respect_switches else \
        net.switch[(net.switch.et == "b")]
    for fb, tb in zip(bs.bus, bs.element):
        g.add_edge(pp_bus_mapping[fb], pp_bus_mapping[tb], weight=0.001)

    meshed = False
    for i in range(1, net.bus.shape[0]):
        if len(g.get_all_shortest_paths(0, i, mode="ALL")) > 1:
            meshed = True
            break

    roots = [pp_bus_mapping[s] for s in net.ext_grid.bus.values]
    return g, meshed, roots  # g, (not g.is_dag())


def coords_from_igraph(graph, roots, meshed=False, calculate_meshed=False):
    """
    Create a list of generic coordinates from an igraph graph layout.

    :param graph: The igraph graph on which the coordinates shall be based
    :type graph: igraph.Graph
    :param roots: The root buses of the graph
    :type roots: iterable
    :param meshed: determines if the graph has any meshes
    :type meshed: bool, default False
    :param calculate_meshed: determines whether to calculate the meshed status
    :type calculate_meshed: bool, default False
    :return: coords - list of coordinates from the graph layout
    """
    if calculate_meshed:
        meshed = False
        for i in range(1, len(graph.vs)):
            if len(graph.get_all_shortest_paths(0, i, mode="ALL")) > 1:
                meshed = True
                break
    if meshed is True:
        layout = graph.layout("kk")
    else:
        graph.to_undirected(mode="each", combine_edges="first")
        layout = graph.layout("rt", root=roots)
    return list(zip(*layout.coords))


def coords_from_nxgraph(mg=None):
    """
    Create a list of generic coordinates from a networkx graph layout.

    :param mg: The networkx graph on which the coordinates shall be based
    :type mg: networkx.Graph
    :return: coords - list of coordinates from the graph layout
    """
    # workaround for bug in agraph
    for u, v in mg.edges(data=False, keys=False):
        if 'key' in mg[int(u)][int(v)]:
            del mg[int(u)][int(v)]['key']
        if 'key' in mg[int(u)][int(v)][0]:
            del mg[int(u)][int(v)][0]['key']
    # ToDo: Insert fallback layout for nxgraph
    return list(zip(*(list(nx.drawing.nx_agraph.graphviz_layout(mg, prog='neato').values()))))


def create_generic_coordinates(net, mg=None, library="igraph", respect_switches=False):
    """
    This function will add arbitrary geo-coordinates for all buses based on an analysis of branches
    and rings. It will remove out of service buses/lines from the net. The coordinates will be
    created either by igraph or by using networkx library.

    :param net: pandapower network
    :type net: pandapowerNet
    :param mg: Existing networkx multigraph, if available. Convenience to save computation time.
    :type mg: networkx.Graph
    :param library: "igraph" to use igraph package or "networkx" to use networkx package
    :type library: str
    :return: net - pandapower network with added geo coordinates for the buses

    :Example:
        net = create_generic_coordinates(net)
    """

    if "bus_geodata" in net and net.bus_geodata.shape[0]:
        logger.warning("Please delete all geodata. This function cannot be used with pre-existing"
                       " geodata.")
        return
    if "bus_geodata" not in net or net.bus_geodata is None:
        net.bus_geodata = pd.DataFrame(columns=["x", "y"])

    gnet = copy.deepcopy(net)
    gnet.bus = gnet.bus[gnet.bus.in_service == True]

    if library == "igraph":
        if not IGRAPH_INSTALLED:
            raise UserWarning("The library igraph is selected for plotting, but not installed "
                              "correctly.")
        graph, meshed, roots = build_igraph_from_pp(gnet, respect_switches)
        coords = coords_from_igraph(graph, roots, meshed)
    elif library == "networkx":
        if mg is None:
            nxg = top.create_nxgraph(gnet, respect_switches=respect_switches)
        else:
            nxg = copy.deepcopy(mg)
        coords = coords_from_nxgraph(nxg)
    else:
        raise ValueError("Unknown library %s - chose 'igraph' or 'networkx'" % library)

    net.bus_geodata.x = coords[1]
    net.bus_geodata.y = coords[0]
    net.bus_geodata.index = gnet.bus.index
    return net


def fuse_geodata(net):
    mg = top.create_nxgraph(net, include_lines=False, include_impedances=False,
                            respect_switches=False)
    geocoords = set(net.bus_geodata.index)
    for area in top.connected_components(mg):
        if len(area & geocoords) > 1:
            geo = net.bus_geodata.loc[area & geocoords].values[0]
            for bus in area:
                net.bus_geodata.loc[bus] = geo
