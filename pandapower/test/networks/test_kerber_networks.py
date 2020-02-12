# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import random as rd

import pytest

import pandapower as pp
import pandapower.networks as pn
from pandapower.networks.kerber_networks import _create_empty_network_with_transformer, \
    _add_lines_and_loads, _add_lines_with_branched_loads


def test_create_empty_network_with_transformer():
    # BUILD:
    trafotype = "0.16 MVA 10/0.4 kV"  # example trafo type
    v_os = 10
    v_us = 0.4
    # OPERATE:
    net = _create_empty_network_with_transformer(trafotype, V_OS=v_os, V_US=v_us)[0]
    # CHECK:
    assert len(net.bus.index) == 2
    assert len(net.trafo.index) == 1
    assert len(net.ext_grid.index) == 1
    assert net.bus.vn_kv.loc[0] == v_os
    assert net.bus.vn_kv.loc[1] == v_us


def test_add_lines_and_loads():
    # BUILD:
    pd_net = pp.create_empty_network()
    busnr1 = pp.create_bus(pd_net, name="startbus", vn_kv=.4)
    n_lines_add = int(10. * rd.random() + 1)
    l_per_line = 0.10 * rd.random()
    # OPERATE:
    _add_lines_and_loads(pd_net, n_lines=n_lines_add, startbusnr=busnr1,
                         length_per_line=l_per_line, p_load_mw=2,
                         q_load_mvar=1, branchnr=2)

    assert len(pd_net.bus.index) == n_lines_add + 1
    assert len(pd_net.line.index) == n_lines_add
    assert len(pd_net.load.index) == n_lines_add
    assert abs(pd_net.line.length_km.sum() - n_lines_add * l_per_line) < 0.0000001


def test_add_lines_with_branched_loads():
    # BUILD:
    pd_net = pp.create_empty_network()
    busnr1 = pp.create_bus(pd_net, name="startbus", vn_kv=.4)
    n_lines_add = int(10. * rd.random() + 1)
    l_per_line = 0.10 * rd.random()
    l_branchout_line = 0.022
    # OPERATE:
    _add_lines_with_branched_loads(pd_net, n_lines_add, startbus=busnr1,
                                   length_per_line=l_per_line,
                                   p_load_mw=2., q_load_mvar=0,
                                   length_branchout_line_1=l_branchout_line,
                                   prob_branchout_line_1=0.5, branchnr=1)

    assert len(pd_net.bus.index) == 2 * n_lines_add + 1
    assert len(pd_net.line.index) == 2 * n_lines_add
    assert len(pd_net.load.index) == n_lines_add
    assert abs(pd_net.line.length_km.sum() - n_lines_add * (l_per_line + l_branchout_line)) < 0.0000001


@pytest.mark.slow
def test_kerber_landnetz_freileitung_1():
    pd_net = pn.create_kerber_landnetz_freileitung_1(p_load_mw=.002, q_load_mvar=0.001)
    assert abs(pd_net.line.length_km.sum() - 0.273) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - 0.026) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.013) < 0.00000001
    assert len(pd_net.bus.index) == 15
    assert len(pd_net.line.index) == 13
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


@pytest.mark.slow
def test_kerber_landnetz_freileitung_2():
    pd_net = pn.create_kerber_landnetz_freileitung_2(p_load_mw=.002, q_load_mvar=.001)
    assert abs(pd_net.line.length_km.sum() - 0.390) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - 0.016) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.008) < 0.00000001
    assert len(pd_net.bus.index) == 10
    assert len(pd_net.line.index) == 8
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


@pytest.mark.slow
def test_create_kerber_landnetz_kabel_1():
    pd_net = pn.create_kerber_landnetz_kabel_1(p_load_mw=.002, q_load_mvar=.001)
    assert abs(pd_net.line.length_km.sum() - 1.046) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - 0.016) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.008) < 0.00000001
    assert len(pd_net.bus.index) == 18
    assert len(pd_net.line.index) == 16
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


@pytest.mark.slow
def test_create_kerber_landnetz_kabel_2():
    pd_net = pn.create_kerber_landnetz_kabel_2(p_load_mw=.002, q_load_mvar=.001)
    assert abs(pd_net.line.length_km.sum() - 1.343) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - 0.028) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.014) < 0.00000001
    assert len(pd_net.bus.index) == 30
    assert len(pd_net.line.index) == 28
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


@pytest.mark.slow
def test_create_kerber_dorfnetz():
    pd_net = pn.create_kerber_dorfnetz(p_load_mw=.002, q_load_mvar=.001)
    assert abs(pd_net.line.length_km.sum() - 3.412) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - .114) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.057) < 0.00000001
    assert len(pd_net.bus.index) == 116
    assert len(pd_net.line.index) == 114
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


@pytest.mark.slow
def test_create_kerber_vorstadtnetz_kabel_1():
    pd_net = pn.create_kerber_vorstadtnetz_kabel_1(p_load_mw=.002, q_load_mvar=0.001)
    assert abs(pd_net.line.length_km.sum() - 4.476) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - .292) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - .146) < 0.00000001
    assert len(pd_net.bus.index) == 294
    assert len(pd_net.line.index) == 292
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


@pytest.mark.slow
def test_create_kerber_vorstadtnetz_kabel_2():
    pd_net = pn.create_kerber_vorstadtnetz_kabel_2(p_load_mw=.002, q_load_mvar=.001)
    assert abs(pd_net.line.length_km.sum() - 4.689) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - .288) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - .144) < 0.00000001
    assert len(pd_net.bus.index) == 290
    assert len(pd_net.line.index) == 288
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


if __name__ == '__main__':
    pytest.main(['-x', __file__])
#    test_create_kerber_vorstadtnetz_kabel_2()
