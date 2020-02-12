# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc


@pytest.fixture
def ring_network():
    net = pp.create_empty_network(sn_mva=2.)
    b0 = pp.create_bus(net, 220)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b0, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    pp.create_transformer(net, b0, b1, "100 MVA 220/110 kV")
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20.)
    l2 = pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
    pp.create_line(net, b3, b1, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=10.)
    pp.create_switch(net, b3, l2, closed=False, et="l")
    return net


def test_branch_results_open_ring(ring_network):
    net = ring_network
    sc.calc_sc(net, branch_results=True)
    assert np.allclose(net.res_trafo_sc.ikss_lv_ka.values, [0.47705988])
    assert np.allclose(net.res_line_sc.ikss_ka.values, [0.45294928, 0.44514686, 0.47125418])

def test_branch_results_closed_ring(ring_network):
    net = ring_network
    net.switch.closed = True
    sc.calc_sc(net, branch_results=True)

    assert np.allclose(net.res_trafo_sc.ikss_lv_ka.values, [0.47705988])
    assert np.allclose(net.res_line_sc.ikss_ka.values, [0.17559325, 0.29778739, 0.40286545])

def test_kappa_methods(ring_network):
    net = ring_network
    net.switch.closed = True
    sc.calc_sc(net, kappa_method="B", ip=True)
    assert np.allclose(net.res_bus_sc.ip_ka.values,
                       [0.48810547956, 0.91192962511, 1.0264898716, 1.0360554521])
    sc.calc_sc(net, kappa_method="C", ip=True, topology="auto")
    assert np.allclose(net.res_bus_sc.ip_ka.values,
                       [0.48810547956, 0.91192962511, 0.89331396461, 0.90103415924])

if __name__ == '__main__':
#    net = ring_network()
#    test_branch_results_open_ring(net)
    pytest.main(["test_ring.py"])

