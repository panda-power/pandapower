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


def test_cost_mixed():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=-0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=.05,
                  min_q_mvar=-.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False, max_q_mvar=.05, max_p_mw=0.1, min_p_mw=0.0050,
                   min_q_mvar=-.05)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    # testing some combinations
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)
    pp.runopp(net)
    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values[0])

    net.poly_cost.cp1_eur_per_mw.at[0] = 0
    net.poly_cost.cp2_eur_per_mw2.at[0] = 1
    pp.runopp(net)
    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values**2)

    net.poly_cost.cp0_eur.at[0] = 1
    pp.runopp(net)
    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values**2 + 1)

    net.load.controllable.at[0] = True
    pp.runopp(net)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values ** 2 + 1)

    net.load.controllable.at[0] = False
    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)
    pp.create_pwl_cost(net, 0, "ext_grid", [[-1000, 0, -2000], [0, 1000, 2000]], power_type="p")

    net.poly_cost.cp1_eur_per_mw.at[0] = 1000
    net.poly_cost.cp2_eur_per_mw2.at[0] = 0
    pp.runopp(net)
    assert np.isclose(net.res_ext_grid.p_mw.values[0], 0, atol=1e-4)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values[0]*1000, atol=1e-3)


def test_mixed_p_q_pol():
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=.05,
                  min_q_mvar=-.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False, max_q_mvar=.05, max_p_mw=0.1, min_p_mw=0.0050,
                   min_q_mvar=-.05)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    # testing some combinations
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1, cq1_eur_per_mvar=1)
    pp.runopp(net)
    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, (net.res_gen.p_mw.values + net.res_gen.q_mvar.values))


def test_mixed_p_q_pwl():
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=.05,
                  min_q_mvar=-.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False, max_q_mvar=.05, max_p_mw=0.1, min_p_mw=0.005,
                   min_q_mvar=-.05)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    # testing some combinations
    pp.create_pwl_cost(net, 0, "gen", [[-150, 150, 1]])
    pp.create_pwl_cost(net, 0, "gen", [[-150, 150, 1]], power_type="q")
    pp.runopp(net)
    assert net["OPF_converged"]
    assert np.allclose(net.res_cost, net.res_gen.p_mw.values + net.res_gen.q_mvar.values)


if __name__ == "__main__":
    # vm_max = 1.05
    # vm_min = 0.95
    #
    # # create net
    # net = pp.create_empty_network()
    # pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    # pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    # pp.create_gen(net, 1, p_mw=-0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=.05,
    #               min_q_mvar=-.05)
    # pp.create_ext_grid(net, 0)
    # pp.create_load(net, 1, p_mw=0.02, controllable=False, max_q_mvar=.05, max_p_mw=0.1, min_p_mw=0.0050,
    #                min_q_mvar=-.05)
    # pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
    #                                c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
    #                                max_loading_percent=100 * 690)
    #
    # # testing some combinations
    # pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)
    # pp.runopp(net)
    # assert net["OPF_converged"]
    # assert net.res_cost == net.res_gen.p_mw.values[0]
    #
    # net.poly_cost.cp1_eur_per_mw.at[0] = 0
    # net.poly_cost.cp2_eur_per_mw2.at[0] = 1
    # pp.runopp(net)
    # assert net["OPF_converged"]
    # assert np.isclose(net.res_cost, net.res_gen.p_mw.values**2)
    #
    # net.poly_cost.cp0_eur.at[0] = 1
    # pp.runopp(net)
    # assert net["OPF_converged"]
    # assert np.isclose(net.res_cost, net.res_gen.p_mw.values**2 + 1)
    #
    # net.load.controllable.at[0] = True
    # pp.runopp(net)
    # assert np.isclose(net.res_cost, net.res_gen.p_mw.values ** 2 + 1)
    #
    # net.load.controllable.at[0] = False
    # net.pwl_cost.drop(net.pwl_cost.index, inplace=True)
    # pp.create_pwl_cost(net, 0, "ext_grid", [[-1000, 0, -2000], [0, 1000, 2000]], power_type="p")
    #
    # net.poly_cost.cp1_eur_per_mw.at[0] = 1000
    # net.poly_cost.cp2_eur_per_mw2.at[0] = 0
    # pp.runopp(net)
    # assert np.isclose(net.res_ext_grid.p_mw.values[0], 0, atol=1e-4)
    # assert np.isclose(net.res_cost, net.res_gen.p_mw.values[0]*1000, atol=1e-3)

   pytest.main([__file__, "-xs"])
