# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.networks as pn


def test_panda_four_load_branch():
    pd_net = pn.panda_four_load_branch()
    assert len(pd_net.bus) == 6
    assert len(pd_net.ext_grid) == 1
    assert len(pd_net.trafo) == 1
    assert len(pd_net.line) == 4
    assert len(pd_net.load) == 4
    pp.runpp(pd_net)
    assert pd_net.converged


def test_four_loads_with_branches_out():
    pd_net = pn.four_loads_with_branches_out()
    assert len(pd_net.bus) == 10
    assert len(pd_net.ext_grid) == 1
    assert len(pd_net.trafo) == 1
    assert len(pd_net.line) == 8
    assert len(pd_net.load) == 4
    pp.runpp(pd_net)
    assert pd_net.converged


def test_simple_four_bus_system():
    net = pn.simple_four_bus_system()
    assert len(net.bus) == 4
    assert len(net.ext_grid) == 1
    assert len(net.trafo) == 1
    assert len(net.line) == 2
    assert len(net.sgen) == 2
    assert len(net.load) == 2
    pp.runpp(net)
    assert net.converged


def test_simple_mv_open_ring_net():
    net = pn.simple_mv_open_ring_net()
    assert len(net.bus) == 7
    assert len(net.ext_grid) == 1
    assert len(net.trafo) == 1
    assert len(net.line) == 6
    assert len(net.load) == 5
    pp.runpp(net)
    assert net.converged


if __name__ == '__main__':
    pytest.main(['-x', "test_simple_pandapower_test_networks.py"])
