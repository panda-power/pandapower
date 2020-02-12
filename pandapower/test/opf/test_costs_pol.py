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


def test_cost_pol_gen():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.05,
                  min_q_mvar=-0.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)

    pp.runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values)

    net.poly_cost.cp1_eur_per_mw.at[0] = 0
    net.poly_cost.cp2_eur_per_mw2.at[0] = 1
    # run OPF
    pp.runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values**2)


def test_cost_pol_all_elements():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.05,
                  min_q_mvar=-0.05)
    pp.create_sgen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.05,
                   min_q_mvar=-0.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)
    pp.create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=1)
    # run OPF
    pp.runopp(net)

    assert net["OPF_converged"]
    assert abs(net.res_cost - (net.res_gen.p_mw.values + net.res_sgen.p_mw.values)) < 1e-2

    net.poly_cost.cp1_eur_per_mw.at[0] = 0
    net.poly_cost.cp2_eur_per_mw2.at[0] = 1

    pp.runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_gen.p_mw.values**2 + net.res_sgen.p_mw.values)

def test_cost_pol_q():
    """ Testing a very simple network for the resulting cost value
    constraints with OPF """
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_sgen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.05,
                   min_q_mvar=-0.05)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)

    pp.create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=0, cq1_eur_per_mvar=-1)
    # run OPF
    pp.runopp(net)

    assert net["OPF_converged"]
    assert abs(net.res_cost + (net.res_sgen.q_mvar.values)) < 1e-2

    net.poly_cost.cq1_eur_per_mvar.at[0] = 0
    net.poly_cost.cq2_eur_per_mvar2.at[0] = 1
#    net.poly_cost.c.at[0] = np.array([[1, 0, 0]])
    # run OPF
    pp.runopp(net)

    assert net["OPF_converged"]
    assert np.isclose(net.res_cost, net.res_sgen.q_mvar.values**2)



if __name__ == "__main__":
       pytest.main(["test_costs_pol.py", "-xs"])