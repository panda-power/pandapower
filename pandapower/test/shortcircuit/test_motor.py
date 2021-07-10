# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc

@pytest.fixture
def motor_net():
    net = pp.create_empty_network(sn_mva=10)
    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)
    b3 = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, b1, s_sc_max_mva=10., rx_max=0.1, s_sc_min_mva=8.,
                       rx_min=0.1)
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=1.,
                                   r_ohm_per_km=0.32, c_nf_per_km=0,
                                   x_ohm_per_km=0.07, max_i_ka=1,
                                   endtemp_degree=80)
    pp.create_motor(net, b2, pn_mech_mw=0.5, lrc_pu=7., vn_kv=0.45, rx=0.4,
                    efficiency_n_percent=95, cos_phi_n=0.9, cos_phi=0.9)
    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3, length_km=2.,
                                   r_ohm_per_km=0.32, c_nf_per_km=0,
                                   x_ohm_per_km=0.07, max_i_ka=1,
                                   endtemp_degree=80)
    return net

def test_motor_min(motor_net):
    net = motor_net
    sc.calc_sc(net, case="min")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:3],
                       [11.547005315, 0.53709235574, 0.18070949061], rtol=1e-4)

def test_motor_max(motor_net):
    net = motor_net
    sc.calc_sc(net, case="max")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:3],
                       [14.743809197, 5.626994278, 0.370730612], rtol=1e-4)

def test_motor_max_branch(motor_net):
    net = motor_net
    net.motor.in_service = False
    sc.calc_sc(net, case="max")
    ikss_without_motor = net.res_bus_sc.ikss_ka.copy()

    net.motor.in_service = True
    sc.calc_sc(net, case="max", branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:3],
                       [14.743809197, 5.626994278, 0.370730612], rtol=1e-4)
    # The maximum current through the first branch is the short-circuit current
    # at the second bus without the motor contribution, which does not flow
    # through the line
    assert np.isclose(net.res_line_sc.ikss_ka.at[0], ikss_without_motor.at[1])
    # The maximum current through the second branch is the short-circuit current
    # at the third bus
    assert np.isclose(net.res_line_sc.ikss_ka.at[1], net.res_bus_sc.ikss_ka.at[2])

def test_motor_min_branch(motor_net):
    net = motor_net
    net.motor.in_service = False
    sc.calc_sc(net, case="min", branch_results=True)
    ikss_without_motor = net.res_line_sc.ikss_ka.values

    net.motor.in_service = True
    sc.calc_sc(net, case="min", branch_results=True)
    assert np.allclose(ikss_without_motor, net.res_line_sc.ikss_ka.values)

def test_large_motor(motor_net):
    net = motor_net
    net.motor.pn_mech_mw = 10
    sc.calc_sc(net, case="max")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:3],
                       [14.695869025, 103.16722971, 0.38693418116], rtol=1e-4)

if __name__ == '__main__':
    pytest.main([__file__])