# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


def test_cost_piecewise_linear_gen():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.05, max_p_mw=0.15, max_q_mvar=0.05,
                  min_q_mvar=-0.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "gen", [[0, 75, 1.5], [75, 150, 1.5]])

    pp.runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values * 1.5, atol=1e-3)


def test_cost_piecewise_linear_eg():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10)
    pp.create_ext_grid(net, 0, min_p_mw=0, max_p_mw=0.050)
    pp.create_gen(net, 1, p_mw=0.01, min_p_mw=0, max_p_mw=0.050, controllable=True)
    # pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "ext_grid", [[0, 50, -10]])
    # run OPF
    pp.runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, -10*net.res_ext_grid.p_mw.values)


def test_get_costs():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.05, max_p_mw=0.15, max_q_mvar=0.05,
                  min_q_mvar=-0.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "gen", [[0, 150, 2]])
    # run OPF
    pp.runopp(net)

    assert net["OPF_converged"]
    assert net.res_gen.p_mw.values[0] - net.gen.min_p_mw.values[0] < 1e-2
    assert net.res_cost == 2 * net.res_gen.p_mw.values
    # check and assert result


def test_cost_piecewise_linear_sgen():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_sgen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.05, max_p_mw=0.15, max_q_mvar=0.05,
                  min_q_mvar=-0.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "sgen", [[0, 150, 2]])
    # run OPF
    pp.runopp(net)

    assert net["OPF_converged"]
    assert net.res_sgen.p_mw.values[0] - net.sgen.min_p_mw.values[0] < 1e-2
    assert net.res_cost == 2 * net.res_sgen.p_mw.values


def test_cost_piecewise_linear_load():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_load(net, 1, p_mw=0.1, controllable=True, max_p_mw=0.15, min_p_mw=0.050, max_q_mvar=0,
                   min_q_mvar=0)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "load", [[0, 75, 1.5], [75, 150, 1.5]])

    pp.runopp(net)

    assert net["OPF_converged"]
    assert abs(net.res_cost - net.res_load.p_mw.values * 1.5) < 1e-3

def test_cost_piecewise_linear_sgen_uneven_slopes():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_sgen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.05, max_p_mw=0.15, max_q_mvar=0.05,
                   min_q_mvar=-0.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "sgen", [[0, 75, 1.5], [75, 150, 1.5]])
    # run OPF
    pp.runopp(net)

    assert net["OPF_converged"]
    assert net.res_cost - net.res_sgen.p_mw.values * 1.5 < 1e-3


def test_cost_piecewise_linear_load_uneven_slopes():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_load(net, 1, p_mw=0.050)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)


    pp.create_pwl_cost(net, 0, "ext_grid", [(0, 0.075, 1), (0.075, 150, 2)])

    pp.runopp(net)
    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_ext_grid.p_mw.values[0])

    net.load.p_mw = 0.1
    pp.runopp(net)
    assert np.isclose(net.res_cost, (0.075 + 2*(net.res_ext_grid.p_mw.values[0] - 0.075)), rtol=1e-2)

def test_cost_piecewise_linear_sgen_very_unsteady_slopes():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """

    # boundaries:
    vm_max = 1.5
    vm_min = 0.5

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_sgen(net, 1, p_mw=0.10, controllable=True, min_p_mw=0, max_p_mw=1.50,
                   max_q_mvar=0.05, min_q_mvar=-0.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_pwl_cost(net, 0, "sgen", [[0, 0.75, -1], [0.75, 1500, 2]])
    # run OPF
    pp.runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_sgen.p_mw.values[0], .75, rtol=1e-2)
    assert np.isclose(net.res_sgen.p_mw.values[0], -net.res_cost, rtol=1e-2)


if __name__ == "__main__":
    pytest.main(["-xs"])