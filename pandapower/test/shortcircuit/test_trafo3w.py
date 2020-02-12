# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc


@pytest.fixture
def trafo3w_net():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 220)
    b2 = pp.create_bus(net, 30)
    b3 = pp.create_bus(net, 10)
    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=40., rx_min=0.1, rx_max=0.1)
    pp.create_load(net, b2, 10000, 2000)
    pp.create_load(net, b3, 10000, 4000)
    pp.create_transformer3w_from_parameters(net, hv_bus=b1, mv_bus=b2, lv_bus=b3, vn_hv_kv=222,
                                            vn_mv_kv=33, vn_lv_kv=11., sn_hv_mva=50,
                                            sn_mv_mva=30, sn_lv_mva=20, vk_hv_percent=11,
                                            vkr_hv_percent=1., vk_mv_percent=11,
                                            vkr_mv_percent=1., vk_lv_percent=11.,
                                            vkr_lv_percent=1., pfe_kw=10, i0_percent=0.2)
    return net

def test_trafo3w_max(trafo3w_net):
    net = trafo3w_net
    sc.calc_sc(net, case="max", lv_tol_percent=6., ip=True, ith=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, [0.26243195543, 1.2151357496, 3.2407820253])
    assert np.allclose(net.res_bus_sc.ip_ka.values, [0.64800210157, 3.0086118915, 8.0313060686])
    assert np.allclose(net.res_bus_sc.ith_ka.values, [0.26687233494, 1.2361480166, 3.2972358704])

def test_trafo3w_min(trafo3w_net):
    net = trafo3w_net
    sc.calc_sc(net, case="min", lv_tol_percent=6., ip=True, ith=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, [0.1049727799, 0.56507157823, 1.5934473235])
    assert np.allclose(net.res_bus_sc.ip_ka.values, [0.25920083485, 1.3972274925, 3.9422963436])
    assert np.allclose(net.res_bus_sc.ith_ka.values, [0.10674893166, 0.57473904595, 1.6208335668])

if __name__ == '__main__':
    pytest.main(['test_trafo3w.py'])
