# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import pytest

from numpy import isclose
import pandapower as pp
import pandapower.shortcircuit as sc
from pandapower.test.shortcircuit.test_meshing_detection import meshed_grid

@pytest.fixture
def radial_grid():
    net = pp.create_empty_network(sn_mva=2.)
    b0 = pp.create_bus(net, 220)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b0, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    pp.create_transformer(net, b0, b1, "100 MVA 220/110 kV")
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20.)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
    return net

def test_radial_network(radial_grid):   
    net = radial_grid
    sc_bus = 3
    sc.calc_sc(net)
    ik = net.res_bus_sc.ikss_ka.at[sc_bus]
    sc.calc_single_sc(net, bus=sc_bus)
    assert isclose(net.res_bus_sc.vm_pu.at[sc_bus], 0)
    assert isclose(net.res_line_sc.i_ka.at[1], ik)
    assert isclose(net.res_line_sc.i_ka.at[0], ik)
    assert isclose(net.res_trafo_sc.i_lv_ka.at[0], ik)
    trafo_ratio = net.trafo.vn_lv_kv.values / net.trafo.vn_hv_kv.values
    assert isclose(net.res_trafo_sc.i_hv_ka.at[0], ik*trafo_ratio)
    
    sc_bus = 2
    sc.calc_sc(net)
    ik = net.res_bus_sc.ikss_ka.at[sc_bus]
    sc.calc_single_sc(net, bus=sc_bus)
    assert isclose(net.res_bus_sc.vm_pu.at[sc_bus], 0)
    assert isclose(net.res_line_sc.i_ka.at[1], 0)
    assert isclose(net.res_line_sc.i_ka.at[0], ik)
    assert isclose(net.res_trafo_sc.i_lv_ka.at[0], ik)
    trafo_ratio = net.trafo.vn_lv_kv.values / net.trafo.vn_hv_kv.values
    assert isclose(net.res_trafo_sc.i_hv_ka.at[0], ik*trafo_ratio)


def test_meshed_network(meshed_grid):
    net = meshed_grid
    sc.calc_sc(net)
    sc_bus = 5
    ik = net.res_bus_sc.ikss_ka.at[sc_bus]
    
    sc.calc_single_sc(net, bus=sc_bus)
    assert isclose(net.res_bus_sc.vm_pu.at[sc_bus], 0)
    line_flow_into_sc = net.res_line_sc.i_ka[(net.line.to_bus==sc_bus) | (net.line.from_bus==sc_bus)].sum()
    assert isclose(line_flow_into_sc, ik, atol=2e-3)


if __name__ == '__main__': 
    pytest.main([__file__])