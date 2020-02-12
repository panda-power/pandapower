# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.networks as pn


def _ppc_element_test(net, n_bus=None, n_branch=None, n_gen=None, gencost=False):
    pp.runpp(net, trafo_model="pi")
    if n_bus:
        assert len(net.bus) == n_bus
    if n_branch:
        assert len(net.line) + len(net.trafo) == n_branch
    if n_gen:
        assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    if gencost:
        if isinstance(gencost, bool):
            assert net.poly_cost.shape[0] == n_gen
        else:
            assert net.poly_cost.shape[0] == gencost
    assert net.converged


def test_case4gs():
    net = pn.case4gs()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 4, 4, 2, False)


def test_case5():
    net = pn.case5()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 5, 6, 5, False)


def test_case6ww():
    net = pn.case6ww()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 6, 11, 3, True)


def test_case9():
    net = pn.case9()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 9, 9, 3, True)


def test_case14():
    net = pn.case14()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 14, 20, 5, True)


def test_case24_ieee_rts():
    net = pn.case24_ieee_rts()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 24)


def test_case30():
    net = pn.case30()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 30, 41, 6, True)


def test_case_ieee30():
    net = pn.case_ieee30()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 30, 41, 6, False)


def test_case33bw():
    net = pn.case33bw()
    pp.runpp(net, trafo_mode="pi")
    assert net.converged
    _ppc_element_test(net, 33, 37, 1, True)


def test_case39():
    net = pn.case39()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 39, 46, 10, True)


def test_case57():
    net = pn.case57()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 57, 80, 7, True)


def test_case89pegase():
    net = pn.case89pegase()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 89, 210, 12+6, 12)


def test_case118():
    net = pn.case118()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 118, 186, 54, True)


def test_case145():
    net = pn.case145()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 145, 453, 50+9, 50)


def test_case_illinois200():
    net = pn.case_illinois200()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 200, 245, 49, False)


def test_case300():
    net = pn.case300()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 300, 411, 69+8, 69)


def test_case1354pegase():
    net = pn.case1354pegase()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 1354, 1991, 260+52, 260)


def test_case1888rte():
    net = pn.case1888rte()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 1888, 2531, 354, 297)


def test_case1888rte_changed_slack():
    ref_bus_idx = 1233
    net = pn.case1888rte(ref_bus_idx=ref_bus_idx)
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 1888, 2531, 354, 297)
    assert net.ext_grid.bus.at[0] == ref_bus_idx

    ref_bus_idx = [1233, 1854]
    net = pn.case1888rte(ref_bus_idx=ref_bus_idx)
    pp.runpp(net, trafo_model='pi')
    assert list(net.ext_grid.bus.sort_values()) == ref_bus_idx
    assert net.converged


def test_case2848rte():
    net = pn.case2848rte()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 2848, 3776, 670, 547)


def test_case2869pegase():
    net = pn.case2869pegase()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 2869, 4582, 690, 510)


def test_case3120sp():
    net = pn.case3120sp()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 3120, 3693, 505, True)


def test_case6470rte():
    net = pn.case6470rte()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 6470, 9005, 1578, 1330)


def test_case6495rte():
    net = pn.case6495rte()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 6495, 9019, 1650, 1372)


def test_case6515rte():
    net = pn.case6515rte()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 6515, 9037, 1651, 1388)


def test_case9241pegase():
    net = pn.case9241pegase()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 9241, 16049, 1879, 1445)


def test_GBreducednetwork():
    net = pn.GBreducednetwork()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 29, 99, 66, True)


def test_GBnetwork():
    net = pn.GBnetwork()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 2224, 3207, 394+37, 394)


def test_iceland():
    net = pn.iceland()
    pp.runpp(net)
    assert net.converged
    _ppc_element_test(net, 189, 206, 35, True)


if __name__ == '__main__':
    pytest.main(["test_power_system_test_cases.py", "-xs"])
