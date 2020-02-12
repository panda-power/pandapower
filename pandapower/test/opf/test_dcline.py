# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
from numpy import array, allclose, isclose

import pandapower as pp
from pandapower.test.consistency_checks import consistency_checks

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


@pytest.fixture
def dcline_net():
    net = pp.create_empty_network()

    b5 = pp.create_bus(net, 380)
    b3 = pp.create_bus(net, 380)
    b2 = pp.create_bus(net, 380)
    b4 = pp.create_bus(net, 380)
    b1 = pp.create_bus(net, 380)

    pp.create_ext_grid(net, b1, 1.02, min_p_mw=0., max_p_mw=1e9)
    pp.create_line(net, b1, b2, 30, "490-AL1/64-ST1A 380.0")
    pp.create_dcline(net, name="dc line", from_bus=b2, to_bus=b3, p_mw=200, loss_percent=1.0,
                     loss_mw=0.5, vm_from_pu=1.01, vm_to_pu=1.012, max_p_mw=1000,
                     in_service=True, index=4)
    pp.create_line(net, b3, b4, 20, "490-AL1/64-ST1A 380.0")

    pp.create_load(net, bus=b4, p_mw=800, controllable=False)
    pp.create_line(net, b4, b5, 20, "490-AL1/64-ST1A 380.0")
    pp.create_ext_grid(net, b5, 1.02, min_p_mw=0., max_p_mw=1e9)

    return net


def get_delta_try_except(net):
    for delta in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]:
        try:
            pp.runopp(net, delta=delta)
            return delta
        except pp.OPFNotConverged:
            continue
    return 1e-10


@pytest.mark.xfail(reason="numerical issue with OPF convergence. The failure seems to depend on the"
                          " python version. Should be reworked.")
def test_dispatch1(dcline_net):
    net = dcline_net
    pp.create_pwl_cost(net, 0, "ext_grid", [[-1e12, 1e9, 100]])
    pp.create_pwl_cost(net, 1, "ext_grid", [[-1e12, 1e9, 80]])
    net.bus["max_vm_pu"] = 2
    net.bus["min_vm_pu"] = 0  # needs to be constrained more than default
    net.line["max_loading_percent"] = 1000  # does not converge if unconstrained
    pp.runopp(net, delta=1e-8)
    consistency_checks(net)
    rel_loss_expect = (net.res_dcline.pl_mw - net.dcline.loss_mw) / \
                      (net.res_dcline.p_from_mw - net.res_dcline.pl_mw) * 100
    assert allclose(rel_loss_expect.values, net.dcline.loss_percent.values, rtol=1e-2)

    assert allclose(net.res_ext_grid.p_mw.values, [0.5, 805], atol=0.1)
    assert allclose(net.res_ext_grid.q_mvar.values, [-7.78755773243, 0.62830727889], atol=1e-3)

    assert allclose(net.res_dcline.p_from_mw.values, [0.500754071], atol=1e-3)
    assert allclose(net.res_dcline.q_from_mvar.values, [7.78745600524])

    assert allclose(net.res_dcline.p_to_mw.values, array([-5.48553789e-05]))
    assert allclose(net.res_dcline.q_to_mvar.values, array([-.62712636707]))


@pytest.mark.xfail(reason="numerical issue with OPF convergence. If vm_pu delta is != 0. at "
                          "ext_grid -> fail. See build_gen() in line 111 + 112")
def test_dcline_dispatch2(dcline_net):
    net = dcline_net
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=80)
    pp.create_poly_cost(net, 1, "ext_grid", cp1_eur_per_mw=100)
    #    pp.create_poly_cost(net, 0, "ext_grid", array([.08, 0]))
    #    pp.create_poly_cost(net, 1, "ext_grid", array([.1, 0]))

    net.bus["max_vm_pu"] = 2
    net.bus["min_vm_pu"] = 0  # needs to be constrained more than default
    net.line["max_loading_percent"] = 1000  # does not converge if unconstrained

    # pp.runopp(net, delta=get_delta_try_except(net))
    pp.runopp(net)
    consistency_checks(net, rtol=1e-3)
    rel_loss_expect = (net.res_dcline.pl_mw - net.dcline.loss_mw) / \
                      (net.res_dcline.p_from_mw - net.res_dcline.pl_mw) * 100
    assert allclose(rel_loss_expect.values, net.dcline.loss_percent.values)

    p_eg_expect = array([8.21525358e+02, 5.43498903e-05])
    q_eg_expect = array([-7787.55852923e-3, -21048.59213887e-3])
    assert allclose(net.res_ext_grid.p_mw.values, p_eg_expect)
    assert allclose(net.res_ext_grid.q_mvar.values, q_eg_expect)

    p_from_expect = array([813573.88366999e-3])
    q_from_expect = array([-26446.0473644e-3])

    assert allclose(net.res_dcline.p_from_mw.values, p_from_expect)
    assert allclose(net.res_dcline.q_from_mvar.values, q_from_expect)

    p_to_expect = array([-805023.64719801e-3])
    q_to_expect = array([-21736.31196315e-3])

    assert allclose(net.res_dcline.p_to_mw.values, p_to_expect)
    assert allclose(net.res_dcline.q_to_mvar.values, q_to_expect)


@pytest.mark.xfail(reason="numerical issue with OPF convergence. If vm_pu delta is != 0. at "
                          "ext_grid -> fail. See build_gen() in line 111 + 112")
def test_dcline_dispatch3(dcline_net):
    net = dcline_net
    pp.create_poly_cost(net, 4, "dcline", cp1_eur_per_mw=1.5)
    net.bus["max_vm_pu"] = 1.03  # needs to be constrained more than default
    net.line["max_loading_percent"] = 1000  # does not converge if unconstrained
    # pp.runopp(net, delta=get_delta_try_except(net))
    pp.runopp(net)
    consistency_checks(net, rtol=1e-1)

    # dc line is not dispatched because of the assigned costs
    assert isclose(net.res_dcline.at[4, "p_to_mw"], 0, atol=1e-2)
    assert all(net.res_ext_grid.p_mw.values > 0)

    # costs for ext_grid at the end of the DC line get double the costs of DC line transfer
    pp.create_poly_cost(net, 1, "ext_grid", cp1_eur_per_mw=2000)
    pp.runopp(net)
    # pp.runopp(net, delta=get_delta_try_except(net))

    # now the total power is supplied through the DC line
    assert (net.res_dcline.at[4, "p_to_mw"]) < 1e3
    assert net.res_ext_grid.p_mw.at[1] < 1
    assert isclose(net.res_cost, net.res_dcline.at[4, "p_from_mw"] * 1.5)


if __name__ == "__main__":
    pytest.main([__file__])
