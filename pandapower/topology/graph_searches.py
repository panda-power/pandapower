# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import networkx as nx
import pandas as pd
from collections import deque
from itertools import combinations

from pandapower.topology.create_graph import create_nxgraph


def connected_component(mg, bus, notravbuses=[]):
    """
    Finds all buses in a NetworkX graph that are connected to a certain bus.

    INPUT:
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.

        **bus** (integer) - Index of the bus at which the search for connected components originates


    OPTIONAL:
     **notravbuses** (list/set) - indices of notravbuses: lines connected to these buses are
                                     not being considered in the graph

    OUTPUT:
        **cc** (generator) - Returns a generator that yields all buses connected to the input bus

    EXAMPLE:
         import pandapower.topology as top

         mg = top.create_nx_graph(net)

         cc = top.connected_component(mg, 5)

    """
    yield bus
    visited = {bus}
    stack = deque([iter(mg[bus])])
    while stack:
        for child in stack.pop():
            if child not in visited:
                yield child
                visited.add(child)
                if child not in notravbuses:
                    stack.append(iter(mg[child]))


def connected_components(mg, notravbuses=set()):
    """
     Clusters all buses in a NetworkX graph that are connected to each other.

     INPUT:
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.


     OPTIONAL:
     **notravbuses** (set) - Indices of notravbuses: lines connected to these buses are
                                       not being considered in the graph

     OUTPUT:
        **cc** (generator) - Returns a generator that yields all clusters of buses connected
                             to each other.

     EXAMPLE:
         import pandapower.topology as top

         mg = top.create_nx_graph(net)

         cc = top.connected_components(net, 5)

    """

    nodes = set(mg.nodes()) - notravbuses
    while nodes:
        cc = set(connected_component(mg, nodes.pop(), notravbuses=notravbuses))
        yield cc
        nodes -= cc
    # the above does not work if two notravbuses are directly connected
    if len(notravbuses) > 0:
        for f, t in mg.edges(notravbuses):
            if f in notravbuses and t in notravbuses:
                yield set([f, t])


def calc_distance_to_bus(net, bus, respect_switches=True, nogobuses=None,
                         notravbuses=None, weight='weight'):
    """
        Calculates the shortest distance between a source bus and all buses connected to it.

     INPUT:
        **net** (pandapowerNet) - Variable that contains a pandapower network.

        **bus** (integer) - Index of the source bus.


     OPTIONAL:
        **respect_switches** (boolean, True) - True: open line switches are being considered
                                                     (no edge between nodes)
                                               False: open line switches are being ignored

        **nogobuses** (integer/list, None) - nogobuses are not being considered

        **notravbuses** (integer/list, None) - lines connected to these buses are not being
                                              considered
        **weight** (string, None) – Edge data key corresponding to the edge weight

     OUTPUT:
        **dist** - Returns a pandas series with containing all distances to the source bus
                   in km. If weight=None dist is the topological distance (int).

     EXAMPLE:
         import pandapower.topology as top

         dist = top.calc_distance_to_bus(net, 5)

    """
    g = create_nxgraph(net, respect_switches=respect_switches,
                       nogobuses=nogobuses, notravbuses=notravbuses)
    return pd.Series(nx.single_source_dijkstra_path_length(g, bus, weight=weight))


def unsupplied_buses(net, mg=None, slacks=None, respect_switches=True):
    """
     Finds buses, that are not connected to an external grid.

     INPUT:
        **net** (pandapowerNet) - variable that contains a pandapower network

     OPTIONAL:
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.

        **in_service_only** (boolean, False) - Defines whether only in service buses should be
            included in unsupplied_buses.

        **slacks** (set, None) - buses which are considered as root / slack buses. If None, all
            existing slack buses are considered.

        **respect_switches** (boolean, True) - Fixes how to consider switches - only in case of no
            given mg.

     OUTPUT:
        **ub** (set) - unsupplied buses

     EXAMPLE:
         import pandapower.topology as top

         top.unsupplied_buses(net)
    """

    mg = mg or create_nxgraph(net, respect_switches=respect_switches)
    if slacks is None:
        slacks = set(net.ext_grid[net.ext_grid.in_service].bus.values) | set(
            net.gen[net.gen.in_service & net.gen.slack].bus.values)
    not_supplied = set()
    for cc in nx.connected_components(mg):
        if not set(cc) & slacks:
            not_supplied.update(set(cc))

    return not_supplied


def find_basic_graph_characteristics(g, roots, characteristics):
    """
    Determines basic characteristics of the given graph like connected buses, stubs, bridges,
    and articulation points.

    .. note::

        This is the base function for find_graph_characteristics. Please use the latter
        function instead!
    """
    connected = 'connected' in characteristics
    stub_buses = 'stub_buses' in characteristics
    bridges = {'bridges', 'required_bridges'} & set(characteristics)
    articulation_points = {'articulation_points', 'notn1_areas'} & set(characteristics)
    notn1_starts = 'notn1_areas' in characteristics

    char_dict = {'connected': set(), 'stub_buses': set(), 'bridges': set(),
                 'articulation_points': set(), 'notn1_starts': set()}

    discovery = {root: 0 for root in roots}  # "time" of first discovery of node during search
    low = {root: 0 for root in roots}
    visited = set(roots)
    path = []
    stack = [(root, root, iter(g[root])) for root in roots]
    while stack:
        grandparent, parent, children = stack[-1]
        try:
            child = next(children)
            if stub_buses:
                if child not in visited:
                    path.append(child)  # keep track of movement through the graph
            if grandparent == child:
                continue
            if child in visited:
                if discovery[child] <= discovery[parent]:  # back edge
                    low[parent] = min(low[parent], discovery[child])
            else:
                low[child] = discovery[child] = len(discovery)
                visited.add(child)
                stack.append((parent, child, iter(g[child])))
        except StopIteration:
            back = stack.pop()
            path.append(back[0])
            if low[parent] >= discovery[grandparent]:
                # Articulation points and start of not n-1 safe buses
                if grandparent not in roots:
                    if articulation_points:
                        char_dict['articulation_points'].add(grandparent)
                    if notn1_starts:
                        char_dict['notn1_starts'].add(parent)
                if low[parent] > discovery[grandparent]:
                    # Bridges
                    if bridges:
                        char_dict['bridges'].add((grandparent, parent))

                    # Stub buses
                    if stub_buses:
                        stub = path.pop()
                        if stub != grandparent:
                            char_dict['stub_buses'].add(stub)
                        while path and path[-1] != grandparent and path[-1] not in roots:
                            stub = path.pop()
                            char_dict['stub_buses'].add(stub)
            low[grandparent] = min(low[parent], low[grandparent])

    if connected:
        char_dict['connected'] = visited
    return char_dict


def find_graph_characteristics(g, roots, characteristics):
    """
    Finds and returns different characteristics of the given graph which can be specified.

    INPUT:
        **g** (NetworkX graph) - Graph of the network

        **roots** (list) - Root buses of the graphsearch

        **characteristics** (list) - List of characteristics this function determines and returns

        .. note::

            Possible characteristics:

            - 'connected' - All buses which have a connection to at least one of the root buses
            - 'articulation_points' - Buses which lead to disconnected areas if they get removed
            - 'bridges' - Edges which lead to disconnected areas if they get removed
            - 'stub_buses' - Buses which arent't connected if one specific edge gets removed
            - 'required_bridges' - Bridges which are strictly needed to connect a specific bus
            - 'notn1_areas' - Areas which aren't connected if one specific bus gets removed

    OUTPUT:

        **char_dict** (dict) - dictionary which contains the wanted characteristics

        ======================= ================================================================
        key                     dict value
        ======================= ================================================================
        'connected'             set of all connected buses
        'articulation_points'   set of all articulation points
        'bridges'               set of tuples which represent start and end bus of each bridge
        'stub_buses'            set of all buses which lie on a stub
        'required_bridges'      dict of all buses which are connected via at least one bridge.
                                The dict values contain a set of bridges which are needed to
                                connect the key buses
        'notn1_areas'           dict of not n-1 safe areas. The dict values contain a set of
                                not n-1 safe buses which aren't connected if the key bus gets
                                removed
        ======================= ================================================================

    EXAMPLE::

        import topology as top
        g = top.create_nxgraph(net, respect_switches=False)
        char_dict = top.find_graph_characteristics(g, roots=[0, 3], characteristics=['connected', 'stub_buses'])
    """
    char_dict = find_basic_graph_characteristics(g, roots, characteristics)

    required_bridges = 'required_bridges' in characteristics
    notn1_areas = 'notn1_areas' in characteristics

    if not required_bridges and not notn1_areas:
        return {key: char_dict[key] for key in characteristics}

    char_dict.update({'required_bridges': dict(), 'notn1_areas': dict()})

    visited = set(roots)
    visited_bridges = []
    notn1_area_start = None
    curr_notn1_area = []

    stack = [(root, root, iter(g[root])) for root in roots]
    while stack:
        grandparent, parent, children = stack[-1]
        try:
            child = next(children)
            if child == grandparent:
                continue
            if child not in visited:
                visited.add(child)
                stack.append((parent, child, iter(g[child])))
                if required_bridges and ((parent, child) in char_dict['bridges'] or
                                         (child, parent) in char_dict['bridges']):
                    visited_bridges.append((parent, child))

                if notn1_areas:
                    if child in char_dict['notn1_starts'] and not notn1_area_start:
                        notn1_area_start = parent
                    if notn1_area_start:
                        curr_notn1_area.append(child)

        except StopIteration:
            stack.pop()
            if required_bridges:
                if len(visited_bridges) > 0:
                    char_dict['required_bridges'][parent] = visited_bridges[:]
                if ((parent, grandparent) in char_dict['bridges'] or
                    (grandparent, parent) in char_dict['bridges']):
                    visited_bridges.pop()

            if notn1_areas and grandparent == notn1_area_start:
                if grandparent in char_dict["notn1_areas"]:
                    char_dict["notn1_areas"][grandparent].update(set(curr_notn1_area[:]))
                else:
                    char_dict["notn1_areas"][grandparent] = set(curr_notn1_area[:])
                del curr_notn1_area[:]
                notn1_area_start = None

    return {key: char_dict[key] for key in characteristics}


def get_2connected_buses(g, roots):
    """
    Get all buses which have at least two connections to the roots

    INPUT:
        **g** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network

        **roots** - Roots of the graphsearch
    """
    char_dict = find_graph_characteristics(g, roots, characteristics=['connected', 'stub_buses'])
    connected, stub_buses = char_dict['connected'], char_dict['stub_buses']
    two_connected = connected - stub_buses
    return connected, two_connected


def determine_stubs(net, roots=None, mg=None, respect_switches=False):
    """
     Finds stubs in a network. Open switches are being ignored. Results are being written in a new
     column in the bus table ("on_stub") and line table ("is_stub") as True/False value.


     INPUT:
        **net** (pandapowerNet) - Variable that contains a pandapower network.

     OPTIONAL:
        **roots** (integer/list, None) - indices of buses that should be excluded (by default, the
                                         ext_grid buses will be set as roots)

     EXAMPLE:
         import pandapower.topology as top

         top.determine_stubs(net, roots = [0, 1])


    """
    if mg is None:
        mg = create_nxgraph(net, respect_switches=respect_switches)
    # remove buses with degree lower 2 until none left
    if roots is None:
        roots = set(net.ext_grid.bus)
    #    mg.add_edges_from((a, b) for a, b in zip(list(roots)[:-1], list(roots)[1:]))
    #    while True:
    #        dgo = {g for g, d in list(mg.degree().items()) if d < 2} #- roots
    #        if not dgo:
    #            break
    #        mg.remove_nodes_from(dgo)
    #    n1_buses = mg.nodes()
    _, n1_buses = get_2connected_buses(mg, roots)
    net.bus["on_stub"] = True
    net.bus.loc[n1_buses, "on_stub"] = False
    net.line["is_stub"] = ~((net.line.from_bus.isin(n1_buses)) & (net.line.to_bus.isin(n1_buses)))
    stubs = set(net.bus.index) - set(n1_buses)
    return stubs


def lines_on_path(mg, path):
    """
     Finds all lines that connect a given path of buses.

     INPUT:
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.

        **path** (list) - List of connected buses.

     OUTPUT:
        **lines** (list) - Returns a list of all lines on the path.

     EXAMPLE:
         import topology as top

         mg = top.create_nxgraph(net)
         lines = top.lines_on_path(mg, [4, 5, 6])

     """

    return elements_on_path(mg, path, "line")


def elements_on_path(mg, path, element="line"):
    """
     Finds all elements that connect a given path of buses.

     INPUT:
        **mg** (NetworkX graph) - NetworkX Graph or MultiGraph that represents a pandapower network.

        **path** (list) - List of connected buses.

        **element** (string, "l") - element type

        **multi** (boolean, True) - True: Applied on a NetworkX MultiGraph
                                    False: Applied on a NetworkX Graph

     OUTPUT:
        **elements** (list) - Returns a list of all lines on the path.

     EXAMPLE:
         import topology as top

         mg = top.create_nxgraph(net)
         elements = top.elements_on_path(mg, [4, 5, 6])

     """
    if element not in ["line", "switch", "trafo", "trafo3w"]:
        raise ValueError("Invalid element type %s"%element)
    if isinstance(mg, nx.MultiGraph):
        return [edge[1] for b1, b2 in zip(path, path[1:]) for edge in mg.get_edge_data(b1, b2).keys()
                if edge[0]==element]
    else:
        return [mg.get_edge_data(b1, b2)["key"][1] for b1, b2 in zip(path, path[1:])
                if mg.get_edge_data(b1, b2)["key"][0]==element]


def get_end_points_of_continuously_connected_lines(net, lines):
    mg = nx.MultiGraph()
    line_buses = net.line.loc[lines, ["from_bus", "to_bus"]].values
    mg.add_edges_from(line_buses)
    switch_buses = net.switch[["bus", "element"]].values[net.switch.et.values=="b"]
    mg.add_edges_from(switch_buses)

    all_buses = set(line_buses.flatten())
    longest_path = []
    for b1, b2 in combinations(all_buses, 2):
        try:
            path = nx.shortest_path(mg, b1, b2)
        except nx.NetworkXNoPath:
            raise UserWarning("Lines not continuously connected")
        if len(path) > len(longest_path):
            longest_path = path
    if all_buses - set(longest_path):
        raise UserWarning("Lines have branching points")
    return longest_path[0], longest_path[-1]
