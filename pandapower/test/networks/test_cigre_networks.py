# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
import pytest

import pandapower as pp
import pandapower.networks as pn


def test_cigre_hv():
    net = pn.create_cigre_network_hv()  # length_km_6a_6b=0.1
    pp.runpp(net)

    all_vn_kv = pd.Series([22, 220, 380])
    assert net.bus.vn_kv.isin(all_vn_kv).all()
    all_length_km = pd.Series([100, 300, 600, 0.1])
    assert net.line.length_km.isin(all_length_km).all()
    assert len(net.bus) == 13
    assert len(net.line) == 9
    assert len(net.gen) == 3
    assert len(net.sgen) == 0
    assert len(net.shunt) == 3
    assert len(net.trafo) == 6
    assert len(net.load) == 5
    assert len(net.ext_grid) == 1
    assert net.converged

    net = pn.create_cigre_network_hv(length_km_6a_6b=80)
    assert net.line.length_km[8] == 80


def test_cigre_mv():
    net = pn.create_cigre_network_mv()  # with_der=False
    pp.runpp(net)

    all_vn_kv = pd.Series([110, 20])
    assert net.bus.vn_kv.isin(all_vn_kv).all()
    assert len(net.bus) == 15
    assert len(net.line) == 15
    assert len(net.gen) == 0
    assert len(net.sgen) == 0
    assert len(net.shunt) == 0
    assert len(net.trafo) == 2
    assert len(net.load) == 18
    assert len(net.ext_grid) == 1
    assert len(net.switch) == 8
    assert net.converged

    net = pn.create_cigre_network_mv(with_der="pv_wind")
    pp.runpp(net)

    all_vn_kv = pd.Series([110, 20])
    assert net.bus.vn_kv.isin(all_vn_kv).all()
    assert len(net.bus) == 15
    assert len(net.line) == 15
    assert len(net.gen) == 0
    assert len(net.sgen) == 9
    assert len(net.shunt) == 0
    assert len(net.trafo) == 2
    assert len(net.load) == 18
    assert len(net.ext_grid) == 1
    assert len(net.switch) == 8
    assert net.converged

    net = pn.create_cigre_network_mv(with_der="all")
    pp.runpp(net)

    all_vn_kv = pd.Series([110, 20])
    assert net.bus.vn_kv.isin(all_vn_kv).all()
    assert len(net.bus) == 15
    assert len(net.line) == 15
    assert len(net.gen) == 0
    assert len(net.sgen) == 13
    assert len(net.storage) == 2
    assert len(net.shunt) == 0
    assert len(net.trafo) == 2
    assert len(net.load) == 18
    assert len(net.ext_grid) == 1
    assert len(net.switch) == 8
    assert net.converged


def test_cigre_lv():
    net = pn.create_cigre_network_lv()
    pp.runpp(net)

    all_vn_kv = pd.Series([20, 0.4])
    assert net.bus.vn_kv.isin(all_vn_kv).all()
    assert len(net.bus) == 44
    assert len(net.line) == 37
    assert len(net.gen) == 0
    assert len(net.sgen) == 0
    assert len(net.shunt) == 0
    assert len(net.trafo) == 3
    assert len(net.load) == 15
    assert len(net.ext_grid) == 1
    assert len(net.switch) == 3
    assert net.converged

if __name__ == '__main__':
    pytest.main(['-x', "test_cigre_networks.py"])
