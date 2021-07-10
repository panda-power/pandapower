# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.shortcircuit as sc


@pytest.fixture
def net_transformer():
    net = pp.create_empty_network(sn_mva=2)
    b1a = pp.create_bus(net, vn_kv=10.)
    b1b = pp.create_bus(net, vn_kv=10.)
    b2 = pp.create_bus(net, vn_kv=.4)
    pp.create_bus(net, vn_kv=0.4, in_service=False) #add out of service bus to test oos indexing
    pp.create_ext_grid(net, b1a, s_sc_max_mva=100., s_sc_min_mva=40., rx_min=0.1, rx_max=0.1)
    pp.create_switch(net, b1a, b1b, et="b")
    pp.create_transformer_from_parameters(net, b1b, b2, vn_hv_kv=11., vn_lv_kv=0.42, vk_percent=6.,
                                          vkr_percent=0.5, pfe_kw=14, shift_degree=0.0,
                                          tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2, tap_pos=2,
                                          tap_step_percent=2.5, parallel=2, sn_mva=0.4, i0_percent=0.5)
    pp.create_shunt(net, b2, q_mvar=0.050, p_mw=0.0500) #adding a shunt shouldn't change the result
    return net

def test_max_10_trafo(net_transformer):
    net = net_transformer
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent= 10.)
    assert (abs(net.res_bus_sc.ikss_ka.at[0] - 5.77350301940194) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[1] - 5.77350301940194) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[2] - 16.992258758) <1e-5)

    assert (abs(net.res_bus_sc.ip_ka.at[0] - 14.25605) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[1] - 14.25605) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[2] - 42.739927153) <1e-5)

    assert (abs(net.res_bus_sc.ith_ka.at[0] - 5.8711913689) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[1] - 5.8711913689) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[2] - 17.328354145) <1e-5)

def test_max_6_trafo(net_transformer):
    net = net_transformer
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent = 6.)
    assert (abs(net.res_bus_sc.ikss_ka.at[0] - 5.77350301940194) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[1] - 5.77350301940194) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[2] - 16.905912296) <1e-5)

    assert (abs(net.res_bus_sc.ip_ka.at[0] - 14.256046241) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[1] - 14.256046241) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[2] - 42.518706441) <1e-5)

    assert (abs(net.res_bus_sc.ith_ka.at[0] - 5.8711913689) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[1] - 5.8711913689) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[2] - 17.240013111) <1e-5)

def test_min_10_trafo(net_transformer):
    net = net_transformer
    sc.calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent= 10.)
    assert (abs(net.res_bus_sc.ikss_ka.at[0] - 2.309401) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[1] - 2.309401) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[2] - 12.912468695) <1e-5)

    assert (abs(net.res_bus_sc.ip_ka.at[0] - 5.702418) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[1] - 5.702418) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[2] - 32.405489528) <1e-5)

    assert (abs(net.res_bus_sc.ith_ka.at[0] - 2.348476) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[1] - 2.348476) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[2] - 13.162790807) <1e-5)

def test_min_6_trafo(net_transformer):
    net = net_transformer
    sc.calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent = 6.)
    assert (abs(net.res_bus_sc.ikss_ka.at[0] - 2.309401) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[1] - 2.309401) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[2] - 13.39058012) <1e-5)

    assert (abs(net.res_bus_sc.ip_ka.at[0] - 5.702418) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[1] - 5.702418) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[2] - 33.599801499) <1e-5)

    assert (abs(net.res_bus_sc.ith_ka.at[0] - 2.348476) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[1] - 2.348476) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[2] - 13.649789214) <1e-5)

def test_min_10_trafo_2ph(net_transformer):
    net = net_transformer
    sc.calc_sc(net, fault="2ph", case='min', ip=True, ith=True, lv_tol_percent = 10.)
    assert (abs(net.res_bus_sc.ikss_ka.at[0] - 2.0000000702) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[2] - 11.182525915) <1e-5)

    assert (abs(net.res_bus_sc.ip_ka.at[0] - 4.9384391739) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[2] - 28.063977154) <1e-5)

#    assert (abs(net.res_bus_sc.ith_ka.at[0] - 2.0000000702) <1e-5)
#    assert (abs(net.res_bus_sc.ith_ka.at[2] - 11.182525915) <1e-5)

if __name__ == '__main__':
    pytest.main(["test_transformer.py"])
