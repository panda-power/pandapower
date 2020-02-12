# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.networks as pn


def test_kb_extrem_landnetz_freileitung():
    pd_net = pn.kb_extrem_landnetz_freileitung(p_load_mw=0.002, q_load_mvar=0.001)
    assert abs(pd_net.line.length_km.sum() - 0.312) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - 0.052) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.026) < 0.00000001
    assert len(pd_net.bus.index) == 28
    assert len(pd_net.line.index) == 26
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


def test_kb_extrem_landnetz_kabel():
    pd_net = pn.kb_extrem_landnetz_kabel(p_load_mw=0.002, q_load_mvar=0.001)
    assert abs(pd_net.line.length_km.sum() - 1.339) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - 0.052) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.026) < 0.00000001
    assert len(pd_net.bus.index) == 54
    assert len(pd_net.line.index) == 52
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


def test_kb_extrem_landnetz_freileitung_trafo():
    pd_net = pn.kb_extrem_landnetz_freileitung_trafo(p_load_mw=0.002, q_load_mvar=0.001)
    assert abs(pd_net.line.length_km.sum() - 0.348) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - 0.054) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.027) < 0.00000001
    assert len(pd_net.bus.index) == 29
    assert len(pd_net.line.index) == 27
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


def test_kb_extrem_landnetz_kabel_trafo():
    pd_net = pn.kb_extrem_landnetz_kabel_trafo(p_load_mw=0.002, q_load_mvar=.001)
    assert abs(pd_net.line.length_km.sum() - 1.435) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - 0.054) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.027) < 0.00000001
    assert len(pd_net.bus.index) == 56
    assert len(pd_net.line.index) == 54
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


def test_kb_extrem_dorfnetz():
    pd_net = pn.kb_extrem_dorfnetz(p_load_mw=.002, q_load_mvar=.001)
    assert abs(pd_net.line.length_km.sum() - 3.088) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - .116) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.058) < 0.00000001
    assert len(pd_net.bus.index) == 118
    assert len(pd_net.line.index) == 116
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


def test_kb_extrem_dorfnetz_trafo():
    pd_net = pn.kb_extrem_dorfnetz_trafo(p_load_mw=0.002, q_load_mvar=0.001)
    assert abs(pd_net.line.length_km.sum() - 6.094) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - .234) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - .117) < 0.00000001
    assert len(pd_net.bus.index) == 236
    assert len(pd_net.line.index) == 234
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


def test_kb_extrem_vorstadtnetz_1():
    pd_net = pn.kb_extrem_vorstadtnetz_1(p_load_mw=.002, q_load_mvar=0.001)
    assert abs(pd_net.line.length_km.sum() - 3.296) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - 0.290) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - 0.145) < 0.00000001
    assert len(pd_net.bus.index) == 292
    assert len(pd_net.line.index) == 290
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


def test_kb_extrem_vorstadtnetz_2():
    pd_net = pn.kb_extrem_vorstadtnetz_2(p_load_mw=.002, q_load_mvar=0.001)
    assert abs(pd_net.line.length_km.sum() - 4.019) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - 0.290) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - .145) < 0.00000001
    assert len(pd_net.bus.index) == 292
    assert len(pd_net.line.index) == 290
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


def test_kb_extrem_vorstadtnetz_trafo_1():
    pd_net = pn.kb_extrem_vorstadtnetz_trafo_1(p_load_mw=.002, q_load_mvar=0.001)
    assert abs(pd_net.line.length_km.sum() - 5.256) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - .382) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - .191) < 0.00000001
    assert len(pd_net.bus.index) == 384
    assert len(pd_net.line.index) == 382
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged


def test_kb_extrem_vorstadtnetz_trafo_2():
    pd_net = pn.kb_extrem_vorstadtnetz_trafo_2(p_load_mw=.002, q_load_mvar=0.001)
    assert abs(pd_net.line.length_km.sum() - 5.329) < 0.00000001
    assert abs(pd_net.load.p_mw.sum() - .384) < 0.00000001
    assert abs(pd_net.load.q_mvar.sum() - .192) < 0.00000001
    assert len(pd_net.bus.index) == 386
    assert len(pd_net.line.index) == 384
    assert len(pd_net.trafo.index) == 1
    pp.runpp(pd_net)
    assert pd_net.converged

if __name__ == '__main__':
    pytest.main(['-x', "test_kerber_extreme_networks.py"])