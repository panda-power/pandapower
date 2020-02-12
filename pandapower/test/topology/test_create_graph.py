# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from itertools import combinations

import numpy as np
import pytest

import pandapower as pp
from pandapower.pypower.idx_brch import BR_R, BR_X
from pandapower.test.loadflow.result_test_network_generator import add_test_trafo3w, \
    add_test_trafo, add_test_line, \
    add_test_impedance, \
    add_test_bus_bus_switch
from pandapower.test.loadflow.test_scenarios import network_with_trafo3ws
from pandapower.topology import create_nxgraph
from pandapower.topology.create_graph import graph_tool_available


libraries = ["networkx"]
if graph_tool_available:
    libraries.append("graph_tool")


def test_line():
    net = pp.create_empty_network()
    add_test_line(net)
    line, open_loop_line, oos_line = net.line.index
    f, t = net.line.from_bus.at[line], net.line.to_bus.at[line]

    # check that oos lines are neglected and switches are respected
    for library in libraries:
        mg = create_nxgraph(net, library=library)
        assert set(mg.get_edge_data(f, t).keys()) == {("line", line)}
        assert set(mg.nodes()) == set(net.bus.index)

        # check respect_switches
        mg = create_nxgraph(net, respect_switches=False, library=library)
        assert set(mg.get_edge_data(f, t).keys()) == {("line", line), ("line", open_loop_line)}
        assert set(mg.nodes()) == set(net.bus.index)

        # check not including lines
        mg = create_nxgraph(net, include_lines=False, library=library)
        assert mg.get_edge_data(f, t) is None
        assert set(mg.nodes()) == set(net.bus.index)

        # check edge attributes
        mg = create_nxgraph(net, calc_branch_impedances=True, library=library)
        line_tab = net.line.loc[line]
        par = mg.get_edge_data(f, t, key=("line", line))
        r = line_tab.length_km / line_tab.parallel * line_tab.r_ohm_per_km
        x = line_tab.length_km / line_tab.parallel * line_tab.x_ohm_per_km
        z = np.sqrt(r ** 2 + x ** 2)
        assert np.isclose(par["r_ohm"], r)
        assert np.isclose(par["x_ohm"], x)
        assert np.isclose(par["z_ohm"], z)
        assert np.isclose(par["weight"], line_tab.length_km)
        assert par["path"] == 1

        mg = create_nxgraph(net, calc_branch_impedances=True, branch_impedance_unit="pu", library=library)
        line_tab = net.line.loc[line]
        par = mg.get_edge_data(f, t, key=("line", line))
        pp.runpp(net)
        f, t = net._pd2ppc_lookups["branch"]["line"]
        assert np.isclose(par["r_pu"], net._ppc["branch"][f, BR_R])
        assert np.isclose(par["x_pu"], net._ppc["branch"][f, BR_X])


def test_trafo():
    net = pp.create_empty_network()
    add_test_trafo(net)

    for library in libraries:
        trafo, open_loop_trafo, oos_trafo = net.trafo.index
        f, t = net.trafo.hv_bus.at[trafo], net.trafo.lv_bus.at[trafo]
        # check that oos trafos are neglected and switches are respected
        mg = create_nxgraph(net, library=library)
        assert set(mg.get_edge_data(f, t).keys()) == {("trafo", trafo)}
        assert set(mg.nodes()) == set(net.bus.index)

        # check respect_switches
        mg = create_nxgraph(net, respect_switches=False, library=library)
        assert set(mg.get_edge_data(f, t).keys()) == {("trafo", trafo), ("trafo", open_loop_trafo)}
        assert set(mg.nodes()) == set(net.bus.index)

        # check not including trafos
        mg = create_nxgraph(net, include_trafos=False, library=library)
        assert mg.get_edge_data(f, t) is None
        assert set(mg.nodes()) == set(net.bus.index)

        # check edge attributes
        net.trafo.vn_hv_kv = 20
        net.trafo.vn_lv_kv = 0.4
        net.trafo.pfe_kw = 0
        net.trafo.i0_percent = 0
        mg = create_nxgraph(net, calc_branch_impedances=True, library=library)
        trafo_tab = net.trafo.loc[trafo]
        par = mg.get_edge_data(f, t, key=("trafo", trafo))
        base_Z = (trafo_tab.sn_mva) / (trafo_tab.vn_hv_kv ** 2)
        r = (trafo_tab.vkr_percent / 100) / base_Z / trafo_tab.parallel
        z = (trafo_tab.vk_percent / 100) / base_Z / trafo_tab.parallel
        assert np.isclose(par["r_ohm"], r)
        assert np.isclose(par["z_ohm"], z)
        assert par["weight"] == 0
        assert par["path"] == 1

        mg = create_nxgraph(net, calc_branch_impedances=True, branch_impedance_unit="pu", library=library)
        par = mg.get_edge_data(f, t, key=("trafo", trafo))
        pp.runpp(net)
        f, t = net._pd2ppc_lookups["branch"]["trafo"]
        assert np.isclose(par["r_pu"], net._ppc["branch"][f, BR_R])
        assert np.isclose(par["x_pu"], net._ppc["branch"][f, BR_X])


def test_trafo3w():
    for library in libraries:
        net = pp.create_empty_network()
        add_test_trafo3w(net)

        t1, t2 = net.trafo3w.index
        trafo3 = net.trafo3w.iloc[t1]
        hv, mv, lv = trafo3.hv_bus, trafo3.mv_bus, trafo3.lv_bus
        mg = create_nxgraph(net, library=library)
        for f, t in combinations([hv, mv, lv], 2):
            assert set(mg.get_edge_data(f, t).keys()) == {("trafo3w", t1)}
            assert set(mg.nodes()) == set(net.bus.index)

        net.trafo3w.in_service.at[t2] = True
        mg = create_nxgraph(net, library=library)
        for f, t in combinations([hv, mv, lv], 2):
            assert set(mg.get_edge_data(f, t).keys()) == {("trafo3w", t1), ("trafo3w", t2)}
            assert set(mg.nodes()) == set(net.bus.index)

        for sb in [hv, mv, lv]:
            sw = pp.create_switch(net, bus=sb, element=t1, et="t3", closed=False)
            mg = create_nxgraph(net, library=library)
            for f, t in combinations([hv, mv, lv], 2):
                if sb == f or t == sb:
                    assert set(mg.get_edge_data(f, t).keys()) == {("trafo3w", t2)}
                else:
                    assert set(mg.get_edge_data(f, t).keys()) == {("trafo3w", t1), ("trafo3w", t2)}
                assert set(mg.nodes()) == set(net.bus.index)
            net.switch.closed.at[sw] = True


def test_trafo3w_impedances(network_with_trafo3ws):
    net, t3, hv, mv, lv = network_with_trafo3ws
    net.trafo3w.vn_hv_kv = 20
    net.trafo3w.vn_mv_kv = 0.6
    net.trafo3w.vn_lv_kv = 0.4
    t3 = net.trafo3w.index[0]
    for library in libraries:
        mg = create_nxgraph(net, calc_branch_impedances=True, library=library)
        trafo3 = net.trafo3w.loc[t3]
        hv, mv, lv = trafo3.hv_bus, trafo3.mv_bus, trafo3.lv_bus
        base_Z_hv = min(trafo3.sn_hv_mva, trafo3.sn_mv_mva) / (trafo3.vn_hv_kv ** 2)
        base_Z_mv = min(trafo3.sn_mv_mva, trafo3.sn_lv_mva) / (trafo3.vn_hv_kv ** 2)
        base_Z_lv = min(trafo3.sn_hv_mva, trafo3.sn_lv_mva) / (trafo3.vn_hv_kv ** 2)

        par = mg.get_edge_data(hv, mv, key=("trafo3w", t3))
        assert np.isclose(par["r_ohm"], (trafo3.vkr_hv_percent / 100) / base_Z_hv)
        assert np.isclose(par["z_ohm"], (trafo3.vk_hv_percent / 100) / base_Z_hv)
        assert par["weight"] == 0
        assert par["path"] == 1

        par = mg.get_edge_data(mv, lv, key=("trafo3w", t3))
        assert np.isclose(par["r_ohm"], (trafo3.vkr_mv_percent / 100) / base_Z_mv)
        assert np.isclose(par["z_ohm"], (trafo3.vk_mv_percent / 100) / base_Z_mv)
        assert par["weight"] == 0
        assert par["path"] == 1

        par = mg.get_edge_data(hv, lv, key=("trafo3w", t3))
        assert np.isclose(par["r_ohm"], (trafo3.vkr_lv_percent / 100) / base_Z_lv)
        assert np.isclose(par["z_ohm"], (trafo3.vk_lv_percent / 100) / base_Z_lv)
        assert par["weight"] == 0
        assert par["path"] == 1


def test_impedance():
    net = pp.create_empty_network()
    add_test_impedance(net)

    for library in libraries:
        impedance, oos_impedance = net.impedance.index
        f, t = net.impedance.from_bus.at[impedance], net.impedance.to_bus.at[impedance]
        # check that oos impedances are neglected and switches are respected
        mg = create_nxgraph(net, library=library)
        assert set(mg.get_edge_data(f, t).keys()) == {("impedance", impedance)}
        assert set(mg.nodes()) == set(net.bus.index)

        # check not including impedances
        mg = create_nxgraph(net, include_impedances=False, library=library)
        assert mg.get_edge_data(f, t) is None
        assert set(mg.nodes()) == set(net.bus.index)

        # check edge attributes
        mg = create_nxgraph(net, calc_branch_impedances=True, branch_impedance_unit="pu", library=library)
        pp.runpp(net)

        par = mg.get_edge_data(f, t, key=("impedance", impedance))
        assert np.isclose(par["weight"], 0)
        assert par["path"] == 1
        f, t = net._pd2ppc_lookups["branch"]["impedance"]
        assert np.isclose(par["r_pu"], net._ppc["branch"][f, BR_R])
        assert np.isclose(par["x_pu"], net._ppc["branch"][f, BR_X])


def test_bus_bus_switches():
    net = pp.create_empty_network()
    add_test_bus_bus_switch(net)

    for library in libraries:
        s = net.switch.index[0]
        f, t = net.switch.bus.iloc[0], net.switch.element.iloc[0]

        net.switch.closed.at[s] = True
        mg = create_nxgraph(net, library=library)
        assert set(mg.get_edge_data(f, t)) == {("switch", s)}
        assert set(mg.nodes()) == set(net.bus.index)

        net.switch.closed.at[s] = False
        mg = create_nxgraph(net, library=library)
        assert mg.get_edge_data(f, t) is None
        assert set(mg.nodes()) == set(net.bus.index)

        mg = create_nxgraph(net, respect_switches=False, library=library)
        assert set(mg.get_edge_data(f, t)) == {("switch", s)}
        assert set(mg.nodes()) == set(net.bus.index)

        mg = create_nxgraph(net, respect_switches=False, calc_branch_impedances=True, library=library)
        # TODO check R/X/Z values
        par = mg.get_edge_data(f, t, key=("switch", s))
        assert np.isclose(par["r_ohm"], 0)
        assert np.isclose(par["z_ohm"], 0)
        assert np.isclose(par["weight"], 0)
        assert par["path"] == 1


def test_nogo():
    net = pp.create_empty_network()
    add_test_line(net)
    mg = create_nxgraph(net)
    assert set(mg.nodes()) == set(net.bus.index)
    mg = create_nxgraph(net, nogobuses=[0])
    assert set(mg.nodes()) == set(net.bus.index) - {0}


def test_branch_impedance_unit():
    net = pp.create_empty_network()
    with pytest.raises(ValueError) as exception_info:
        mg = create_nxgraph(net, branch_impedance_unit="p.u.")
    assert str(exception_info.value) == "branch impedance unit can be either 'ohm' or 'pu'"

@pytest.mark.xfail(reason="This test fails, since graph_tool bus indices must be a range(0, n_buses). "
                          "If a bus is removed, graph-tool is not working.")
def test_nogo_graph_tool():
    net = pp.create_empty_network()
    add_test_line(net)
    mg = create_nxgraph(net, library="graph_tool")
    assert set(mg.nodes()) == set(net.bus.index)
    mg = create_nxgraph(net, nogobuses=[0], library="graph_tool")
    assert set(mg.nodes()) == set(net.bus.index) - {0}


if __name__ == '__main__':
    pytest.main(__file__)
