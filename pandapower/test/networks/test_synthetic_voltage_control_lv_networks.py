# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.networks as pn


def test_rural_1():
    net = pn.create_synthetic_voltage_control_lv_network('rural_1')
    assert abs(net.line.length_km.sum() - 1.616) < 1e-6
    assert abs(net.load.p_mw.sum() - 77e-3) < 1e-6
    assert len(net.bus.index) == 26
    assert len(net.line.index) == 24
    assert len(net.trafo.index) == 1
    pp.runpp(net)
    assert net.converged


def test_rural_2():
    net = pn.create_synthetic_voltage_control_lv_network('rural_2')
    assert abs(net.line.length_km.sum() - 0.567) < 1e-6
    assert abs(net.load.p_mw.sum() - 64.5e-3) < 1e-6
    assert len(net.bus.index) == 18
    assert len(net.line.index) == 16
    assert len(net.trafo.index) == 1
    pp.runpp(net)
    assert net.converged


def test_village_1():
    net = pn.create_synthetic_voltage_control_lv_network('village_1')
    assert abs(net.line.length_km.sum() - 2.6) < 1e-6
    assert abs(net.load.p_mw.sum() - 262.1e-3) < 1e-6
    assert len(net.bus.index) == 80
    assert len(net.line.index) == 78
    assert len(net.trafo.index) == 1
    pp.runpp(net)
    assert net.converged


def test_village_2():
    net = pn.create_synthetic_voltage_control_lv_network('village_2')
    assert abs(net.line.length_km.sum() - 1.832) < 1e-6
    assert abs(net.load.p_mw.sum() - 183.6e-3) < 1e-6
    assert len(net.bus.index) == 74
    assert len(net.line.index) == 72
    assert len(net.trafo.index) == 1
    pp.runpp(net)
    assert net.converged


def test_suburb_1():
    net = pn.create_synthetic_voltage_control_lv_network('suburb_1')
    assert abs(net.line.length_km.sum() - 4.897) < 1e-6
    assert abs(net.load.p_mw.sum() - 578.3e-3) < 1e-6
    assert len(net.bus.index) == 204
    assert len(net.line.index) == 202
    assert len(net.trafo.index) == 1
    pp.runpp(net)
    assert net.converged


if __name__ == '__main__':
    pytest.main(['-x', "test_synthetic_voltage_control_lv_networks.py"])
