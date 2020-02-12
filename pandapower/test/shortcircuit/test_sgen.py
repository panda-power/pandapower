# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc

@pytest.fixture
def wind_park_example():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110., index=1)
    b2 = pp.create_bus(net, vn_kv=110., index=2)
    b3 = pp.create_bus(net, vn_kv=110., index=3)
    b4 = pp.create_bus(net, vn_kv=110., index=4)
    pp.create_ext_grid(net, b1, s_sc_max_mva=20*110*np.sqrt(3), rx_max=0.1)

    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=100, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b3, length_km=50, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3, length_km=50, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b3, to_bus=b4, length_km=25, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)

    pp.create_sgen(net, b2, p_mw=0.1e3, sn_mva=100)
    pp.create_sgen(net, b3, p_mw=0.050e3, sn_mva=50)
    pp.create_sgen(net, b4, p_mw=0.050e3, sn_mva=50)
    net.sgen["k"] = 1.2
    return net

@pytest.fixture
def three_bus_example():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20.)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
    net.line["endtemp_degree"] = 80

    pp.create_sgen(net, b2, sn_mva=2, p_mw=0, k=1.2)
    return net

def test_max_branch_results(three_bus_example):
    net = three_bus_example
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, np.array([0.53746061, 0.50852707, 0.4988896]))
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([ 0.49593034,  0.4988896 ]))
    assert np.allclose(net.res_line_sc.ip_ka.values, np.array([ 0.92787443,  0.9251165 ]))
    assert np.allclose(net.res_line_sc.ith_ka.values, np.array([ 0.49811957,  0.50106881]))

def test_min_branch_results(three_bus_example):
    net = three_bus_example
    sc.calc_sc(net, case="min", ip=True, ith=True, branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, np.array([ 0.43248784,  0.41156533,  0.40431286]))
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([ 0.39171613,  0.40431286]))
    assert np.allclose(net.res_line_sc.ip_ka.values, np.array([ 0.72795118,  0.74576565]))
    assert np.allclose(net.res_line_sc.ith_ka.values, np.array([ 0.39340278,  0.40605375]))

def test_wind_park(wind_park_example):
    net = wind_park_example
    sc.calc_sc(net, ip=True)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[2], 3.9034, rtol=1e-4)
    assert np.isclose(net.res_bus_sc.ip_ka.at[2], 7.3746, rtol=1e-4)

if __name__ == '__main__':
    pytest.main(["test_sgen.py"])



