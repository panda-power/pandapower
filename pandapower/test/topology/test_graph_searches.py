# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandapower as pp
import pytest
import pandapower.topology as top
import pandapower.networks as nw
import networkx as nx


@pytest.fixture
def feeder_network():
    net = pp.create_empty_network()
    current_bus = pp.create_bus(net, vn_kv=20.)
    pp.create_ext_grid(net, current_bus)
    for length in [12, 6, 8]:
        new_bus = pp.create_bus(net, vn_kv=20.)
        pp.create_line(net, current_bus, new_bus, length_km=length,
                       std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
        current_bus = new_bus
    pp.create_line(net, current_bus, 0, length_km=5, std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    return net


@pytest.fixture
def meshed_network():
    net = pp.create_empty_network("7bus_system")

    # ext grid
    b = []
    b.append(pp.create_bus(net, vn_kv=380., name="exi", geodata=(0, 0)))
    pp.create_ext_grid(net, b[0], name="exi")

    # create 110kV buses
    for i in range(1, 7):
        b.append(pp.create_bus(net, vn_kv=110., name="bus" + str(i), geodata=(0, 0)))
    # connect buses b1 to b6 with overhead lines
    for i in range(1, 6):
        l = pp.create_line(net, b[i], b[i + 1], length_km=10. * i * 2.,
                           std_type="149-AL1/24-ST1A 110.0", name="line" + str(i), index=i + 2)
        pp.create_switch(net, b[i], l, et="l", name="bl_switch_" + str(i), index=i + 3)

    # create trafo
    pp.create_transformer(net, hv_bus=b[0], lv_bus=b[1], std_type="160 MVA 380/110 kV", name="trafo")

    # create some more lines between b6-b1 and b1-b4
    pp.create_line(net, b[1], b[4], length_km=100., std_type="149-AL1/24-ST1A 110.0", name="line6")
    pp.create_line(net, b[6], b[1], length_km=100., std_type="149-AL1/24-ST1A 110.0", name="line7")
    return net

@pytest.fixture
def mixed_network():
    net = pp.create_empty_network()
    pp.create_buses(net, nr_buses=5, vn_kv=20.)
    connections = [(0, 1), (1, 2), (2, 3), (2, 4)]
    for fb, tb in connections:
        pp.create_line(net, fb, tb, length_km=1, std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    for b in [1, 4, 3]:
        pp.create_ext_grid(net, b)
    return net

def test_connected_components(feeder_network):
    net = feeder_network
    mg = top.create_nxgraph(net)
    cc = top.connected_components(mg)
    assert list(cc) == [{0, 1, 2, 3}]
    cc_notrav = top.connected_components(mg, notravbuses={0,2})
    assert list(cc_notrav) == [{0, 1, 2}, {0, 2, 3}]

def test_determine_stubs(feeder_network):
    net = feeder_network
    sec_bus = pp.create_bus(net, vn_kv=20.)
    sec_line = pp.create_line(net, 3, sec_bus, length_km=3, std_type="NA2XS2Y 1x185 RM/25 12/20 kV")
    top.determine_stubs(net)
    assert not np.any(net.bus.on_stub.loc[set(net.bus.index) - {sec_bus}].values)
    assert not np.any(net.line.is_stub.loc[set(net.line.index) - {sec_line}].values)
    assert net.bus.on_stub.at[sec_bus]
    assert net.line.is_stub.at[sec_line]

def test_determine_stubs_meshed(meshed_network):
    net = meshed_network
    # root == LV side of trafo at ext_grid. Then ext_grid bus itself (0) == stub
    stubs = top.determine_stubs(net, roots=[1])
    assert len(stubs) == 1
    assert stubs.pop() == 0

def test_determine_stubs_mixed(mixed_network):
    net = mixed_network
    stubs = top.determine_stubs(net, roots=[1, 4, 3])
    assert stubs == {0}
    stubs = top.determine_stubs(net, roots=[4, 3, 1])
    assert stubs == {0}

def test_distance(feeder_network):
    net = feeder_network
    dist = top.calc_distance_to_bus(net, 0)
    assert np.allclose(dist.sort_index().values, [0, 12, 13, 5])

    dist = top.calc_distance_to_bus(net, 0, notravbuses={3})
    assert np.allclose(dist.sort_index().values, [0, 12, 18, 5])

    pp.create_switch(net, bus=3, element=2, et="l", closed=False)
    dist = top.calc_distance_to_bus(net, 0)
    assert np.allclose(dist.sort_index().values, [0, 12, 18, 5])
    
    dist = top.calc_distance_to_bus(net, 0, weight=None)
    assert np.allclose(dist.sort_index().values, [0, 1, 2, 1])

def test_unsupplied_buses_with_in_service():
    # IS ext_grid --- open switch --- OOS bus --- open switch --- IS bus
    net = pp.create_empty_network()

    slack_bus = pp.create_bus(net, 0.4)
    pp.create_ext_grid(net, slack_bus)

    bus0 = pp.create_bus(net, 0.4, in_service=False)
    pp.create_switch(net, slack_bus, bus0, 'b', False)

    bus1 = pp.create_bus(net, 0.4, in_service=True)
    pp.create_switch(net, bus0, bus1, 'b', False)

    ub = top.unsupplied_buses(net)
    assert ub == {2}



    # OOS ext_grid --- closed switch --- IS bus
    net = pp.create_empty_network()

    bus_sl = pp.create_bus(net, 0.4)
    pp.create_ext_grid(net, bus_sl, in_service=False)

    bus0 = pp.create_bus(net, 0.4, in_service=True)
    pp.create_switch(net, bus_sl, bus0, 'b', True)

    ub = top.unsupplied_buses(net)
    assert ub == {0, 1}


def test_unsupplied_buses_with_switches():
    net = pp.create_empty_network()
    pp.create_buses(net, 8, 20)
    pp.create_buses(net, 5, 0.4)
    pp.create_ext_grid(net, 0)
    pp.create_line(net, 0, 1, 1.2, "NA2XS2Y 1x185 RM/25 12/20 kV")
    pp.create_switch(net, 0, 0, "l", closed=True)
    pp.create_switch(net, 1, 0, "l", closed=False)
    pp.create_line(net, 0, 2, 1.2, "NA2XS2Y 1x185 RM/25 12/20 kV")
    pp.create_switch(net, 0, 1, "l", closed=False)
    pp.create_switch(net, 2, 1, "l", closed=True)
    pp.create_line(net, 0, 3, 1.2, "NA2XS2Y 1x185 RM/25 12/20 kV")
    pp.create_switch(net, 0, 2, "l", closed=False)
    pp.create_switch(net, 3, 2, "l", closed=False)
    pp.create_line(net, 0, 4, 1.2, "NA2XS2Y 1x185 RM/25 12/20 kV")
    pp.create_switch(net, 0, 3, "l", closed=True)
    pp.create_switch(net, 4, 3, "l", closed=True)
    pp.create_line(net, 0, 5, 1.2, "NA2XS2Y 1x185 RM/25 12/20 kV")

    pp.create_switch(net, 0, 6, "b", closed=True)
    pp.create_switch(net, 0, 7, "b", closed=False)

    pp.create_transformer(net, 0, 8, "0.63 MVA 20/0.4 kV")
    pp.create_switch(net, 0, 0, "t", closed=True)
    pp.create_switch(net, 8, 0, "t", closed=False)
    pp.create_transformer(net, 0, 9, "0.63 MVA 20/0.4 kV")
    pp.create_switch(net, 0, 1, "t", closed=False)
    pp.create_switch(net, 9, 1, "t", closed=True)
    pp.create_transformer(net, 0, 10, "0.63 MVA 20/0.4 kV")
    pp.create_switch(net, 0, 2, "t", closed=False)
    pp.create_switch(net, 10, 2, "t", closed=False)
    pp.create_transformer(net, 0, 11, "0.63 MVA 20/0.4 kV")
    pp.create_switch(net, 0, 3, "t", closed=True)
    pp.create_switch(net, 11, 3, "t", closed=True)
    pp.create_transformer(net, 0, 12, "0.63 MVA 20/0.4 kV")

    pp.create_buses(net, 2, 20)
    pp.create_impedance(net, 0, 13, 1, 1, 10)
    pp.create_impedance(net, 0, 14, 1, 1, 10, in_service=False)

    ub = top.unsupplied_buses(net)
    assert ub == {1, 2, 3, 7, 8, 9, 10, 14}
    ub = top.unsupplied_buses(net, respect_switches=False)
    assert ub == {14}


def test_graph_characteristics(feeder_network):
    # adapt network
    net = feeder_network
    bus0 = pp.create_bus(net, vn_kv=20.0)
    bus1 = pp.create_bus(net, vn_kv=20.0)
    bus2 = pp.create_bus(net, vn_kv=20.0)
    bus3 = pp.create_bus(net, vn_kv=20.0)
    bus4 = pp.create_bus(net, vn_kv=20.0)
    bus5 = pp.create_bus(net, vn_kv=20.0)
    bus6 = pp.create_bus(net, vn_kv=20.0)
    bus7 = pp.create_bus(net, vn_kv=20.0)
    bus8 = pp.create_bus(net, vn_kv=20.0)
    bus9 = pp.create_bus(net, vn_kv=20.0)
    new_connections = [(3, bus0), (bus0, bus1), (bus0, bus2), (1, bus3), (2, bus4), (bus3, bus4),
                       (bus4, bus5), (bus4, bus6), (bus5, bus6), (2, bus7), (bus7, bus8),
                       (bus8, bus9), (bus9, bus7)]
    for fb, tb in new_connections:
        pp.create_line(net, fb, tb, length_km=1.0, std_type="NA2XS2Y 1x185 RM/25 12/20 kV")

    # get characteristics
    mg = top.create_nxgraph(net, respect_switches=False)
    characteristics = ["bridges", "articulation_points", "connected", "stub_buses",
                       "required_bridges", "notn1_areas"]
    char_dict = top.find_graph_characteristics(mg, net.ext_grid.bus, characteristics)
    bridges = char_dict["bridges"]
    articulation_points = char_dict["articulation_points"]
    connected = char_dict["connected"]
    stub_buses = char_dict["stub_buses"]
    required_bridges = char_dict["required_bridges"]
    notn1_areas = char_dict["notn1_areas"]
    assert bridges == {(3, 4), (4, 5), (4, 6), (2, 11)}
    assert articulation_points == {8, 3, 4, 2, 11}
    assert connected == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
    assert stub_buses == {4, 5, 6, 11, 12, 13}
    assert required_bridges == {4: [(3, 4)], 5: [(3, 4), (4, 5)], 6: [(3, 4), (4, 6)], 11: [(2, 11)],
                                12: [(2, 11)], 13: [(2, 11)]}
    assert notn1_areas == {8: {9, 10}, 3: {4, 5, 6}, 2: {11, 12, 13}}


def test_elements_on_path():
    net = nw.example_simple()
    for multi in [True, False]:
        mg = top.create_nxgraph(net, multi=multi)
        path = nx.shortest_path(mg, 0, 6)
        assert top.elements_on_path(mg, path, "line") == [0, 3]
        assert top.lines_on_path(mg, path) == [0, 3]
        assert top.elements_on_path(mg, path, "trafo") == [0]
        assert top.elements_on_path(mg, path, "trafo3w") == []
        assert top.elements_on_path(mg, path, "switch") == [0, 1]
        with pytest.raises(ValueError) as exception_info:
            top.elements_on_path(mg, path, element="sgen")
        assert str(exception_info.value) == "Invalid element type sgen"


def test_end_points_of_continuously_connected_lines():
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=20.)
    b1 = pp.create_bus(net, vn_kv=20.)
    b2 = pp.create_bus(net, vn_kv=20.)
    b3 = pp.create_bus(net, vn_kv=20.)
    b4 = pp.create_bus(net, vn_kv=20.)
    b5 = pp.create_bus(net, vn_kv=20.)
    b5 = pp.create_bus(net, vn_kv=20.)
    b5 = pp.create_bus(net, vn_kv=20.)
    b6 = pp.create_bus(net, vn_kv=20.)
    b7 = pp.create_bus(net, vn_kv=20.)

    l1 = pp.create_line(net, from_bus=b0, to_bus=b1, length_km=2., std_type="34-AL1/6-ST1A 20.0")
    l2 = pp.create_line(net, from_bus=b1, to_bus=b2, length_km=2., std_type="34-AL1/6-ST1A 20.0")
    pp.create_switch(net, bus=b2, element=b3, et="b")
    pp.create_switch(net, bus=b3, element=b4, et="b")
    pp.create_switch(net, bus=b4, element=b5, et="b")
    l3 = pp.create_line(net, from_bus=b5, to_bus=b6, length_km=2., std_type="34-AL1/6-ST1A 20.0")
    l4 = pp.create_line(net, from_bus=b6, to_bus=b7, length_km=2., std_type="34-AL1/6-ST1A 20.0")

    f, t = top.get_end_points_of_continuously_connected_lines(net, lines=[l2, l1])
    assert {f, t} == {b0, b2}

    f, t = top.get_end_points_of_continuously_connected_lines(net, lines=[l2, l1, l3])
    assert {f, t} == {b0, b6}

    f, t = top.get_end_points_of_continuously_connected_lines(net, lines=[l3])
    assert {f, t} == {b5, b6}

    with pytest.raises(UserWarning) as exception_info:
        top.get_end_points_of_continuously_connected_lines(net, lines=[l1, l2, l4])
    assert str(exception_info.value) == "Lines not continuously connected"

    with pytest.raises(UserWarning) as exception_info:
        top.get_end_points_of_continuously_connected_lines(net, lines=[l1, l4])
    assert str(exception_info.value) == "Lines not continuously connected"

    b8 = pp.create_bus(net, vn_kv=20.)
    l5 = pp.create_line(net, 8, b8, length_km=1., std_type="34-AL1/6-ST1A 20.0")
    with pytest.raises(UserWarning) as exception_info:
        top.get_end_points_of_continuously_connected_lines(net, lines=[l1, l2, l3, l4, l5])
    assert str(exception_info.value) == "Lines have branching points"




if __name__ == '__main__':
    pytest.main([__file__])
