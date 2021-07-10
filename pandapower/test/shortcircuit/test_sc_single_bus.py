 # -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import pytest

import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc
from pandapower.test.shortcircuit.test_meshing_detection import meshed_grid

# @pytest.fixture
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

# @pytest.fixture
def three_bus_big_sgen_example():
     net = pp.create_empty_network()
     b1 = pp.create_bus(net, 110)
     b2 = pp.create_bus(net, 110)
     b3 = pp.create_bus(net, 110)

     pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
     pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20.)
     pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
     net.line["endtemp_degree"] = 80

     pp.create_sgen(net, b2, sn_mva=200., p_mw=0, k=1.2)
     return net

def test_radial_network():   
    net = radial_grid()
    sc_bus = 3
    sc.calc_sc(net)
    ik = net.res_bus_sc.ikss_ka.at[sc_bus]
    sc.calc_sc(net, bus=sc_bus, branch_results=True)

    assert np.isclose(net.res_line_sc.ikss_ka.at[1], ik)
    assert np.isclose(net.res_line_sc.ikss_ka.at[0], ik)
    assert np.isclose(net.res_trafo_sc.ikss_lv_ka.at[0], ik)
    trafo_ratio = net.trafo.vn_lv_kv.values / net.trafo.vn_hv_kv.values
    assert np.isclose(net.res_trafo_sc.ikss_hv_ka.at[0], ik*trafo_ratio)
    
    for inv_y in (False, True):
        sc_bus = 2
        sc.calc_sc(net, inverse_y=inv_y, branch_results=True)
        ik = net.res_bus_sc.ikss_ka.at[sc_bus]
        sc.calc_sc(net, bus=sc_bus, branch_results=True)

        assert np.isclose(net.res_line_sc.ikss_ka.at[1], 0)
        assert np.isclose(net.res_line_sc.ikss_ka.at[0], ik)
        assert np.isclose(net.res_trafo_sc.ikss_lv_ka.at[0], ik)
        trafo_ratio = net.trafo.vn_lv_kv.values / net.trafo.vn_hv_kv.values
        assert np.isclose(net.res_trafo_sc.ikss_hv_ka.at[0], ik*trafo_ratio)



def test_meshed_network(meshed_grid):
    net = meshed_grid
    sc.calc_sc(net)
    sc_bus = 5
    ik = net.res_bus_sc.ikss_ka.at[sc_bus]
    
    for inv_y in (False, True):
        sc.calc_sc(net, bus=sc_bus, inverse_y=inv_y, branch_results=True)

        line_flow_into_sc = net.res_line_sc.ikss_ka[(net.line.to_bus==sc_bus) | (net.line.from_bus==sc_bus)].sum()
        assert np.isclose(line_flow_into_sc, ik, atol=2e-3)


def test_big_gen_network_calc_sc():
    net = three_bus_big_sgen_example()
       
    for inv_y in (False, True):
        sc_bus = 0
        sc.calc_sc(net, bus=sc_bus,
                   branch_results=True, inverse_y=inv_y)
    
        assert np.isclose(net.res_line_sc.ikss_ka.at[0], 1.25967331, atol=1e-3)
        assert np.isclose(net.res_line_sc.ikss_ka.at[1], 0., atol=1e-3)
    
        sc_bus = 2
        sc.calc_sc(net, bus=sc_bus,
                   branch_results=True, inverse_y=inv_y)
        assert np.isclose(net.res_line_sc.ikss_ka.at[0], 0.46221808, atol=1e-3)
        assert np.isclose(net.res_line_sc.ikss_ka.at[1], 1.72233192, atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__])
     