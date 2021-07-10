# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower as pp
import pytest
from pandapower import control
from pandapower.control import Characteristic


def test_continuous_p():
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=110.)
    lv = pp.create_bus(net, vn_kv=20)
    t = pp.create_transformer(net, hv, lv, std_type='40 MVA 110/20 kV')
    pp.create_ext_grid(net, hv)
    eps = 0.0005

    c = control.ContinuousTapControl(net, t, 1., tol=eps)

    tc = control.USetTapControl(net, 0, characteristic=Characteristic([10, 20], [0.95, 1.05]), tol=eps)

    # create 20kW load
    lid = pp.create_load(net, lv, 20)
    gid = pp.create_sgen(net, lv, 0)
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.05) < eps

    # power sums up to 15kW
    net.sgen.p_mw.at[gid] = 5
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at 1.0 pu
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.) < eps

    # generation now cancels load
    net.sgen.p_mw.at[gid] = 10
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 0.95) < eps

    # testing limits
    # power flowing back
    net.sgen.p_mw.at[gid] = 30
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit and not drop even lower
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 0.95) < eps

    # excessive load
    net.sgen.p_mw.at[gid] = 0
    net.load.p_mw.at[lid] = 30
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit and not to go beyond
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.05) < eps


def test_continuous_i():
    net = pp.create_empty_network()
    hv = pp.create_bus(net, vn_kv=110.)
    lv = pp.create_bus(net, vn_kv=20)
    t = pp.create_transformer(net, hv, lv, std_type='40 MVA 110/20 kV')
    pp.create_ext_grid(net, hv)
    eps = 0.0005

    c = control.ContinuousTapControl(net, t, 1., tol=eps)

    # a different characteristic for i_lv_ka rather than for p_lv_mw
    tc = control.USetTapControl(net, 0, variable='i_lv_ka', characteristic=Characteristic([0.315, 0.55], [0.95, 1.05]), tol=eps)

    # create 20kW load
    lid = pp.create_load(net, lv, 20)
    gid = pp.create_sgen(net, lv, 0)
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.05) < eps

    # power sums up to 15kW
    net.sgen.p_mw.at[gid] = 5
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at 1.0 pu
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.) < eps

    # generation now cancels load
    net.sgen.p_mw.at[gid] = 10
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 0.95) < eps

    # testing limits
    # power flowing back
    net.sgen.p_mw.at[gid] = 30
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at lower voltage limit and not drop even lower
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 0.95) < eps

    # excessive load
    net.sgen.p_mw.at[gid] = 0
    net.load.p_mw.at[lid] = 30
    pp.runpp(net, run_control=True)

    # we expect the tap to converge at upper voltage limit and not to go beyond
    assert abs(net.res_bus.vm_pu.at[c.controlled_bus] - 1.05) < eps

if __name__ == '__main__':
    pytest.main(['-s', __file__])