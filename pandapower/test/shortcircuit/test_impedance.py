# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc


@pytest.fixture
def impedance_net():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 220)
    b2 = pp.create_bus(net, 30)
    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=40., rx_min=0.1, rx_max=0.1)
    pp.create_ext_grid(net, b2, s_sc_max_mva=100., s_sc_min_mva=40., rx_min=0.1, rx_max=0.1)
    pp.create_impedance(net, b1, b2, rft_pu=0.01, xft_pu=0.02, rtf_pu=0.05, xtf_pu=0.01, sn_mva=1)
    return net

def test_impedance_max(impedance_net):
    net = impedance_net
    sc.calc_sc(net, case="max", ip=True, ith=True, kappa_method="B")
    assert np.allclose(net.res_bus_sc.ikss_ka.values, [0.38042409891, 2.0550304761])
    assert np.allclose(net.res_bus_sc.ip_ka.values, [0.88029252774, 4.3947194836])
    assert np.allclose(net.res_bus_sc.ith_ka.values, [0.38460749284, 2.0703298078])

def test_impedance_min(impedance_net):
    net = impedance_net
    sc.calc_sc(net, case="min", ip=True, ith=True, kappa_method="B")
    assert np.allclose(net.res_bus_sc.ikss_ka.values, [0.19991981619, 0.86978512768])
    assert np.allclose(net.res_bus_sc.ip_ka.values, [0.50118698745, 1.755888097])
    assert np.allclose(net.res_bus_sc.ith_ka.values, [0.20375890703, 0.87488745362])

if __name__ == '__main__':
    pytest.main(['test_impedance.py'])

