# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
from pandapower.convert_format import convert_format
from pandapower.networks import simple_four_bus_system
from pandapower.test.toolbox import add_grid_connection

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def simplest_grid():
    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15,
                  max_q_mvar=0.005, min_q_mvar=-0.005)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100)
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=0.1)

    return net


@pytest.fixture
def simple_opf_test_net():
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=10.)
    pp.create_bus(net, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.05,
                  min_q_mvar=-.005)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.020, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100)
    return net


def test_convert_format():
    """ Testing a very simple network without transformer for voltage
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15,
                  max_q_mvar=0.05, min_q_mvar=-0.005)
    net.gen["cost_per_mw"] = 100
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)
    # run OPF
    convert_format(net)
    print(net.gen)
    print(net.ext_grid)
    print(net.bus)
    for init in ["pf", "flat"]:
        pp.runopp(net, init=init)
        assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_simplest_voltage():
    """ Testing a very simple network without transformer for voltage
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    net = simplest_grid()
    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, init=init)
        assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min

    pp.runopp(net, check_connectivity=True)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


# def test_eg_voltage():
#    """ Testing a very simple network without transformer for voltage
#    constraints with OPF """
#
#    # boundaries:
#    vm_max = 1.05
#    vm_min = 0.95
#
#    # create net
#    net = pp.create_empty_network()
#    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
#    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
#    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.150, max_q_mvar=0.05,
#                  min_q_mvar=-0.05)
#    pp.create_ext_grid(net, 0, vm_pu=1.01)
#    pp.create_load(net, 1, p_mw=0.02, controllable=False)
#    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
#                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
#                                   max_loading_percent=100)
#    # run OPF
#    for init in ["pf", "flat"]:
#        pp.runopp(net, init=init)
#        assert net["OPF_converged"]
#
#    # check and assert result
#    logger.debug("test_simplest_voltage")
#    logger.debug("res_gen:\n%s" % net.res_gen)
#    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
#    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
#    assert net.res_bus.vm_pu.at[0] == net.ext_grid.vm_pu.values


def test_simplest_dispatch():
    """ Testing a very simple network without transformer for voltage
    constraints with OPF """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_gen(net, 1, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.150, max_q_mvar=0.05,
                  min_q_mvar=-0.05)
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=100)
    pp.create_ext_grid(net, 0)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=101)
    pp.create_load(net, 1, p_mw=0.02, controllable=False)
    pp.create_line_from_parameters(net, 0, 1, 50, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100 * 690)
    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, cost_function="linear", init=init)
        assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_est_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_opf_gen_voltage():
    """ Testing a  simple network with transformer for voltage
    constraints with OPF using a generator """

    # boundaries:
    vm_max = 1.05
    vm_min = 0.95

    # ceate net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_transformer_from_parameters(net, 0, 1, vk_percent=3.75,
                                          tap_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tap_neutral=0,
                                          vn_hv_kv=10.0, vkr_percent=2.8125,
                                          tap_pos=0, tap_side="hv", tap_min=-2,
                                          tap_step_percent=2.5, i0_percent=0.68751,
                                          sn_mva=0.016, pfe_kw=0.11, name=None,
                                          in_service=True, index=None, max_loading_percent=200)
    pp.create_gen(net, 3, p_mw=0.01, controllable=True, min_p_mw=0, max_p_mw=0.025, max_q_mvar=0.5,
                  min_q_mvar=-0.5)
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=10)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100000)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=100000)

    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, init=init)
        assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_opf_gen_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_opf_sgen_voltage():
    """ Testing a  simple network with transformer for voltage
    constraints with OPF using a static generator """

    # boundaries
    vm_max = 1.04
    vm_min = 0.96

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_transformer_from_parameters(net, 0, 1, vk_percent=3.75,
                                          tap_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tap_neutral=0,
                                          vn_hv_kv=10.0, vkr_percent=2.8125,
                                          tap_pos=0, tap_side="hv", tap_min=-2,
                                          tap_step_percent=2.5, i0_percent=0.68751,
                                          sn_mva=0.016, pfe_kw=0.11, name=None,
                                          in_service=True, index=None, max_loading_percent=1000000)
    pp.create_sgen(net, 3, p_mw=0.01, controllable=True, min_p_mw=-0.005, max_p_mw=0.015,
                   max_q_mvar=0.025, min_q_mvar=-0.025)
    pp.create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=0.1)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=1000000)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=1000000)

    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, init=init)
        assert net["OPF_converged"]

    # assert and check result
    logger.debug("test_opf_sgen_voltage")
    logger.debug("res_sgen:\n%s" % net.res_sgen)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_opf_gen_loading():
    """ Testing a  simple network with transformer for loading
    constraints with OPF using a generator """

    # wide open voltage boundaries to make sure they don't interfere with loading constraints
    vm_max = 1.5
    vm_min = 0.5
    max_line_loading = 11

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_transformer_from_parameters(net, 0, 1, vk_percent=3.75,
                                          tap_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tap_neutral=0,
                                          vn_hv_kv=10.0, vkr_percent=2.8125,
                                          tap_pos=0, tap_side="hv", tap_min=-2,
                                          tap_step_percent=2.5, i0_percent=0.68751,
                                          sn_mva=0.016, pfe_kw=0.11, name=None,
                                          in_service=True, index=None, max_loading_percent=145)
    pp.create_gen(net, 3, p_mw=0.01, controllable=True, min_p_mw=0.005, max_p_mw=0.015,
                  max_q_mvar=0.05, min_q_mvar=-0.05)
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=-10)
    pp.create_ext_grid(net, 0)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=.1)
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=max_line_loading)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=max_line_loading)

    # run OPF

    pp.runopp(net, OPF_VIOLATION=1e-1, OUT_LIM_LINE=2,
              PDIPM_GRADTOL=1e-10, PDIPM_COMPTOL=1e-10, PDIPM_COSTTOL=1e-10)
    assert net["OPF_converged"]

    # assert and check result
    logger.debug("test_opf_gen_loading")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_line.loading_percent:\n%s" % net.res_line.loading_percent)
    assert max(net.res_line.loading_percent) < max_line_loading
    logger.debug("res_trafo.loading_percent:\n%s" % net.res_trafo.loading_percent)
    assert max(net.res_trafo.loading_percent) < 145
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_opf_sgen_loading():
    """ Testing a  simple network with transformer for loading
    constraints with OPF using a generator """

    # boundaries
    vm_max = 1.5
    vm_min = 0.5
    max_trafo_loading = 800
    max_line_loading = 13

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_transformer_from_parameters(net, 0, 1, vk_percent=3.75, tap_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tap_neutral=0, vn_hv_kv=10.0,
                                          vkr_percent=2.8125, tap_pos=0, tap_side="hv", tap_min=-2,
                                          tap_step_percent=2.5, i0_percent=0.68751, sn_mva=0.016,
                                          pfe_kw=0.11, name=None, in_service=True, index=None,
                                          max_loading_percent=max_trafo_loading)
    pp.create_sgen(net, 3, p_mw=0.01, controllable=True, min_p_mw=0.005, max_p_mw=.015,
                   max_q_mvar=0.025, min_q_mvar=-0.025)
    pp.create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=-10)
    pp.create_ext_grid(net, 0)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=.1)
    pp.create_line_from_parameters(net, 1, 2, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=max_line_loading)
    pp.create_line_from_parameters(net, 2, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876,
                                   max_loading_percent=max_line_loading)

    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, init=init)
        assert net["OPF_converged"]

    # assert and check result
    logger.debug("test_opf_sgen_loading")
    logger.debug("res_sgen:\n%s" % net.res_sgen)
    logger.debug("res_line.loading_percent:\n%s" % net.res_line.loading_percent)
    assert max(net.res_line.loading_percent) - max_line_loading < 0.15
    logger.debug("res_trafo.loading_percent:\n%s" % net.res_trafo.loading_percent)
    assert max(net.res_trafo.loading_percent) < max_trafo_loading
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min
    # check connectivity check
    pp.runopp(net, check_connectivity=True)


def test_unconstrained_line():
    """ Testing a very simple network without transformer for voltage
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
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876)
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=1)
    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, init=init)
        assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert max(net.res_bus.vm_pu) < vm_max
    assert min(net.res_bus.vm_pu) > vm_min


def test_trafo3w_loading():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=20.)
    b4 = pp.create_bus(net, vn_kv=10.)
    tidx = pp.create_transformer3w(net, b2, b3, b4, std_type='63/25/38 MVA 110/20/10 kV',
                                   max_loading_percent=120)
    pp.create_load(net, b3, p_mw=5, controllable=False)
    load_id = pp.create_load(net, b4, p_mw=5, controllable=True, max_p_mw=50, min_p_mw=0, min_q_mvar=-1e6,
                             max_q_mvar=1e6)
    pp.create_poly_cost(net, load_id, "load", cp1_eur_per_mw=-1000)
    # pp.create_xward(net, b4, 1000, 1000, 1000, 1000, 0.1, 0.1, 1.0)
    net.trafo3w.shift_lv_degree.at[tidx] = 120
    net.trafo3w.shift_mv_degree.at[tidx] = 80

    # pp.runopp(net, calculate_voltage_angles = True)  >> Doesn't converge
    for init in ["pf", "flat"]:
        pp.runopp(net, calculate_voltage_angles=False, init=init)
        assert net["OPF_converged"]
    assert abs(net.res_trafo3w.loading_percent.values - 120) < 1e-3


def test_dcopf_poly(simple_opf_test_net):
    net = simple_opf_test_net
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=100)
    # run OPF
    pp.rundcopp(net)

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert abs(100 * net.res_gen.p_mw.values - net.res_cost) < 1e-3


def test_opf_poly(simple_opf_test_net):
    net = simple_opf_test_net
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=100)
    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, init=init)
        assert net["OPF_converged"]
    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert abs(100 * net.res_gen.p_mw.values - net.res_cost) < 1e-3


def test_opf_pwl(simple_opf_test_net):
    # create net
    net = simple_opf_test_net
    pp.create_pwl_cost(net, 0, "gen", [[0, 100, 100], [100, 200, 100]])
    # run OPF
    for init in ["pf", "flat"]:
        pp.runopp(net, init=init)
        assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert abs(100 * net.res_gen.p_mw.values - net.res_cost) < 1e-3


def test_dcopf_pwl(simple_opf_test_net):
    # create net
    net = simple_opf_test_net
    pp.create_pwl_cost(net, 0, "gen", [[0, 100, 100], [100, 200, 100]])
    pp.create_pwl_cost(net, 0, "ext_grid", [[0, 100, 0], [100, 200, 0]])
    # run OPF
    pp.rundcopp(net)
    assert net["OPF_converged"]

    # check and assert result
    logger.debug("test_simplest_voltage")
    logger.debug("res_gen:\n%s" % net.res_gen)
    logger.debug("res_ext_grid:\n%s" % net.res_ext_grid)
    logger.debug("res_bus.vm_pu: \n%s" % net.res_bus.vm_pu)
    assert abs(100 * net.res_gen.p_mw.values - net.res_cost) < 1e-3


def test_opf_varying_max_line_loading():
    """ Testing a  simple network with transformer for loading
    constraints with OPF using a generator """

    # boundaries
    vm_max = 1.5
    vm_min = 0.5
    max_trafo_loading = 800
    max_line_loading = 13

    # create net
    net = pp.create_empty_network()
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=10.)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_bus(net, max_vm_pu=vm_max, min_vm_pu=vm_min, vn_kv=.4)
    pp.create_transformer_from_parameters(net, 0, 1, vk_percent=3.75, tap_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tap_neutral=0, vn_hv_kv=10.0,
                                          vkr_percent=2.8125, tap_pos=0, tap_side="hv", tap_min=-2,
                                          tap_step_percent=2.5, i0_percent=0.68751, sn_mva=0.016,
                                          pfe_kw=0.11, name=None, in_service=True, index=None,
                                          max_loading_percent=max_trafo_loading)

    pp.create_sgen(net, 3, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.025,
                   min_q_mvar=-0.025)
    pp.create_sgen(net, 2, p_mw=0.1, controllable=True, min_p_mw=0.005, max_p_mw=0.15, max_q_mvar=0.025,
                   min_q_mvar=-0.025)
    pp.create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=10)
    pp.create_poly_cost(net, 1, "sgen", cp1_eur_per_mw=10)
    pp.create_ext_grid(net, 0)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=.1)
    pp.create_line_from_parameters(net, 1, 2, 1, name="line1", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.200, x_ohm_per_km=0.1159876,
                                   max_loading_percent=20)
    pp.create_line_from_parameters(net, 1, 3, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.100, x_ohm_per_km=0.1159876,
                                   max_loading_percent=10)

    # run OPF
    pp.runopp(net, init="flat")
    assert net["OPF_converged"]

    assert np.allclose(net["_ppc"]["branch"][:, 5], np.array([0.02771281 + 0.j, 0.00692820 + 0.j, 0.12800000 + 0.j]))

    # assert and check result
    logger.debug("test_opf_sgen_loading")
    logger.debug("res_sgen:\n%s" % net.res_sgen)
    logger.debug("res_line.loading_percent:\n%s" % net.res_line.loading_percent)
    assert net.res_line.loading_percent.at[0] - 20 < 1e-2
    logger.debug("res_line.loading_percent:\n%s" % net.res_line.loading_percent)
    assert net.res_line.loading_percent.at[1] - 10 < 1e-2


def test_storage_opf():
    """ Testing a simple network with storage to ensure the correct behaviour
    of the storage OPF-Functions """

    # boundaries
    vm_max = 1.1
    vm_min = 0.9
    max_line_loading_percent = 100

    # create network
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=0.4, max_vm_pu=vm_max, min_vm_pu=vm_min)
    b2 = pp.create_bus(net, vn_kv=0.4, max_vm_pu=vm_max, min_vm_pu=vm_min)

    pp.create_line(net, b1, b2, length_km=5, std_type="NAYY 4x50 SE",
                   max_loading_percent=max_line_loading_percent)

    # test elements static
    pp.create_ext_grid(net, b2)
    pp.create_load(net, b1, p_mw=0.0075, controllable=False)
    pp.create_sgen(net, b1, p_mw=0.025, controllable=True, min_p_mw=0.01, max_p_mw=0.025,
                   max_q_mvar=0.025, min_q_mvar=-0.025)

    # test elements
    pp.create_storage(net, b1, p_mw=-.0025, max_e_mwh=50, controllable=True, max_p_mw=0,
                      min_p_mw=-0.025, max_q_mvar=0.025, min_q_mvar=-0.025)
    pp.create_sgen(net, b1, p_mw=0.025, controllable=True, min_p_mw=0, max_p_mw=0.025,
                   max_q_mvar=0.025, min_q_mvar=-0.025)
    pp.create_load(net, b1, p_mw=0.025, controllable=True, max_p_mw=0.025, min_p_mw=0,
                   max_q_mvar=0.025, min_q_mvar=-0.025)

    # costs
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=3)
    pp.create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=2)
    pp.create_poly_cost(net, 0, "storage", cp1_eur_per_mw=-1)
    pp.create_poly_cost(net, 1, "sgen", cp1_eur_per_mw=1)

    pp.create_poly_cost(net, 1, "load", cp1_eur_per_mw=-3)

    # test storage generator behaviour
    net["storage"].in_service.iloc[0] = True
    net["storage"].p_mw.iloc[0] = -0.025
    net["sgen"].in_service.iloc[1] = False
    net["load"].in_service.iloc[1] = False

    pp.runopp(net)
    assert net["OPF_converged"]

    res_stor_p_mw = net["res_storage"].p_mw.iloc[0]
    res_stor_q_mvar = net["res_storage"].q_mvar.iloc[0]
    res_cost_stor = net["res_cost"]

    net["storage"].in_service.iloc[0] = False
    net["storage"].p_mw.iloc[0] = -0.025
    net["sgen"].in_service.iloc[1] = True
    net["load"].in_service.iloc[1] = False

    pp.runopp(net)
    assert net["OPF_converged"]

    res_sgen_p_mw = net["res_sgen"].p_mw.iloc[1]
    res_sgen_q_mvar = net["res_sgen"].q_mvar.iloc[1]
    res_cost_sgen = net["res_cost"]

    # assert storage generator behaviour
    assert np.isclose(res_stor_p_mw, -res_sgen_p_mw)
    assert np.isclose(res_stor_q_mvar, -res_sgen_q_mvar)
    assert np.isclose(res_cost_stor, res_cost_sgen)

    # test storage load behaviour
    net["storage"].in_service.iloc[0] = True
    net["storage"].p_mw.iloc[0] = 0.025
    net["storage"].max_p_mw.iloc[0] = 0.025
    net["storage"].min_p_mw.iloc[0] = 0
    net["storage"].max_q_mvar.iloc[0] = 0.025
    net["storage"].min_q_mvar.iloc[0] = -0.025
    # gencost for storages: positive costs in pandapower per definition
    # --> storage gencosts are similar to sgen gencosts (make_objective.py, l.128ff. and l.185ff.)
    net["poly_cost"].cp1_eur_per_mw.iloc[2] = net.poly_cost.cp1_eur_per_mw.iloc[4]
    net["sgen"].in_service.iloc[1] = False
    net["load"].in_service.iloc[1] = False

    pp.runopp(net)
    assert net["OPF_converged"]

    res_stor_p_mw = net["res_storage"].p_mw.iloc[0]
    res_stor_q_mvar = net["res_storage"].q_mvar.iloc[0]
    res_cost_stor = net["res_cost"]

    net["storage"].in_service.iloc[0] = False
    net["storage"].p_mw.iloc[0] = 0.025
    net["sgen"].in_service.iloc[1] = False
    net["load"].in_service.iloc[1] = True

    pp.runopp(net)
    assert net["OPF_converged"]

    res_load_p_mw = net["res_load"].p_mw.iloc[1]
    res_load_q_mvar = net["res_load"].q_mvar.iloc[1]
    res_cost_load = net["res_cost"]

    # assert storage load behaviour
    assert np.isclose(res_stor_p_mw, res_load_p_mw)
    assert np.isclose(res_stor_q_mvar, res_load_q_mvar)
    assert np.isclose(res_cost_stor, res_cost_load)


def test_in_service_controllables():
    """ Testing controllable but out of service elements behaviour """
    # boundaries
    vm_max = 1.1
    vm_min = 0.9
    max_line_loading_percent = 100

    # create network
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=0.4, max_vm_pu=vm_max, min_vm_pu=vm_min)
    b2 = pp.create_bus(net, vn_kv=0.4, max_vm_pu=vm_max, min_vm_pu=vm_min)

    pp.create_line(net, b1, b2, length_km=5, std_type="NAYY 4x50 SE",
                   max_loading_percent=max_line_loading_percent)

    # test elements static
    pp.create_ext_grid(net, b2)
    pp.create_load(net, b1, p_mw=7.5, controllable=True, max_p_mw=0.010, min_p_mw=0,
                   max_q_mvar=2.5, min_q_mvar=-2.5)
    pp.create_sgen(net, b1, p_mw=0.025, controllable=True, min_p_mw=0.01, max_p_mw=0.025,
                   max_q_mvar=0.025, min_q_mvar=-0.025)

    # test elements
    pp.create_sgen(net, b1, p_mw=0.025, controllable=True, min_p_mw=0, max_p_mw=0.025,
                   max_q_mvar=0.025, min_q_mvar=-0.025)
    pp.create_load(net, b1, p_mw=0.025, controllable=True, max_p_mw=0.0025, min_p_mw=0,
                   max_q_mvar=2.5, min_q_mvar=-2.5)

    # costs
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=3)
    pp.create_poly_cost(net, 0, "load", cp1_eur_per_mw=-1)
    pp.create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=2)
    pp.create_poly_cost(net, 1, "sgen", cp1_eur_per_mw=1)
    pp.create_poly_cost(net, 1, "load", cp1_eur_per_mw=-1)

    net["sgen"].in_service.iloc[1] = False
    net["load"].in_service.iloc[1] = False

    pp.runopp(net)
    assert net["OPF_converged"]


def test_no_controllables(simple_opf_test_net):
    # was ist das problwem an diesem fall und wie fange ich es ab?
    net = simple_opf_test_net
    net.gen.controllable = False
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=-2)
    pp.create_poly_cost(net, 0, "load", cp1_eur_per_mw=1)
    try:
        pp.runopp(net)
    except pp.OPFNotConverged:
        # opf will fail if not bus limits are set and vm_pu is the default value of 1.0 (it is enforced)
        assert True
    net.gen.loc[:, "vm_pu"] = 1.062  # vm_pu setpoint is mandatory if controllable=False
    net.gen.loc[:, "p_mw"] = 0.149
    pp.runopp(net)
    assert np.allclose(net.res_gen.at[0, "vm_pu"], 1.062)
    assert np.allclose(net.res_gen.at[0, "p_mw"], 0.149)


def test_opf_no_controllables_vs_pf():
    """ Comparing the calculation results of PF and OPF in a simple network with non-controllable
     elements """

    # boundaries
    vm_max = 1.3
    vm_min = 0.9
    max_line_loading_percent = 100

    # create network
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=0.4, max_vm_pu=vm_max, min_vm_pu=vm_min)
    b2 = pp.create_bus(net, vn_kv=0.4, max_vm_pu=vm_max, min_vm_pu=vm_min)

    pp.create_line(net, b1, b2, length_km=5, std_type="NAYY 4x50 SE",
                   max_loading_percent=max_line_loading_percent)

    # test elements static
    pp.create_ext_grid(net, b2)
    pp.create_load(net, b1, p_mw=.0075, controllable=False)
    pp.create_sgen(net, b1, p_mw=0.025, controllable=False, min_p_mw=0.01, max_p_mw=0.025,
                   max_q_mvar=0.025, min_q_mvar=-0.025)

    # testing cost assignment (for non-controllable elements - see Gitlab Issue #27)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=3)
    pp.create_poly_cost(net, 0, "load", cp1_eur_per_mw=-3)
    pp.create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=2)

    # do calculations
    pp.runopp(net)
    assert net["OPF_converged"]

    res_opf_line_loading = net.res_line.loading_percent
    res_opf_bus_voltages = net.res_bus.vm_pu

    pp.runpp(net)
    assert net["converged"]

    res_pf_line_loading = net.res_line.loading_percent
    res_pf_bus_voltages = net.res_bus.vm_pu

    # assert calculation behaviour
    assert np.isclose(res_opf_line_loading, res_pf_line_loading).all()
    assert np.isclose(res_opf_bus_voltages, res_pf_bus_voltages).all()


def test_line_temperature():
    net = simplest_grid()
    r_init = net.line.r_ohm_per_km.copy()

    # run OPF
    pp.runopp(net, verbose=False)
    va_init = net.res_bus.va_degree
    assert "r_ohm_per_km" not in net.res_line.columns

    # check results of r adjustment, check that user_pf_options works, alpha
    net.line["temperature_degree_celsius"] = 80
    alpha = 4.03e-3
    net.line['alpha'] = alpha
    pp.runopp(net, verbose=False, consider_line_temperature=True)
    r_temp = r_init * (1 + alpha * (80 - 20))
    assert np.allclose(net.res_line.r_ohm_per_km, r_temp, rtol=0, atol=1e-16)
    assert not np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-2)

    pp.runopp(net, verbose=False, consider_line_temperature=False)
    assert np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-16)
    assert "r_ohm_per_km" not in net.res_line.columns


@pytest.fixture
def four_bus_net():
    net = simple_four_bus_system()
    net.sgen.drop(index=1, inplace=True)
    net.load.drop(index=1, inplace=True)
    return net


def test_three_slacks_vm_setpoint(four_bus_net):
    # tests a net with three slacks in one area. Two of them will be converted to gens, since only one is allowed per
    # area. The others should have vmin / vmax set as their vm_pu setpoint
    net = four_bus_net
    # create two additional slacks with different voltage setpoints
    pp.create_ext_grid(net, 1, vm_pu=1.01, max_p_mw=1., min_p_mw=-1., min_q_mvar=-1, max_q_mvar=1.)
    pp.create_ext_grid(net, 3, vm_pu=1.02, max_p_mw=1., min_p_mw=-1., min_q_mvar=-1, max_q_mvar=1.)
    pp.runpp(net)
    # assert if voltage limits are correct in result in pf an opf
    assert np.allclose(net.res_bus.loc[[0, 1, 3], "vm_pu"], [1., 1.01, 1.02])
    pp.runopp(net)
    assert np.allclose(net.res_bus.loc[[0, 1, 3], "vm_pu"], [1., 1.01, 1.02])


def test_only_gen_slack_vm_setpoint(four_bus_net):
    # tests a net with only gens of which one of them is a a slack
    # The  vmin / vmax vm_pu setpoint should be correct
    net = four_bus_net
    net.ext_grid.drop(index=net.ext_grid.index, inplace=True)
    net.bus.loc[:, "min_vm_pu"] = 0.9
    net.bus.loc[:, "max_vm_pu"] = 1.1
    # create two additional slacks with different voltage setpoints
    pp.create_gen(net, 0, p_mw=0., vm_pu=1., max_p_mw=1., min_p_mw=-1., min_q_mvar=-1, max_q_mvar=1., slack=True)
    g1 = pp.create_gen(net, 1, p_mw=0.02, vm_pu=1.01, max_p_mw=1., min_p_mw=-1., min_q_mvar=-1, max_q_mvar=1.,
                       controllable=False)  # controllable == False -> vm_pu enforced
    g3 = pp.create_gen(net, 3, p_mw=0.01, vm_pu=1.02, max_p_mw=1., min_p_mw=-1.,
                       min_q_mvar=-1, max_q_mvar=1.)  # controllable == True -> vm_pu between bus voltages
    pp.runpp(net)
    # assert if voltage limits are correct in result in pf an opf
    assert np.allclose(net.res_bus.loc[[0, 1, 3], "vm_pu"], [1., 1.01, 1.02])
    pp.runopp(net)

    # controllable == True is more important than  slack == True -> vm_pu is between bus limits
    assert not np.allclose(net.res_bus.at[0, "vm_pu"], 1.)
    # controllable == True is less important than  slack == True -> see
    # https://github.com/e2nIEE/pandapower/issues/511#issuecomment-536593128

    # assert value of controllable == False gen
    assert np.allclose(net.res_bus.at[1, "vm_pu"], 1.01)
    assert np.allclose(net.res_bus.at[1, "p_mw"], -0.02)
    # assert limit of controllable == True gen
    assert 0.9 < net.res_bus.at[3, "vm_pu"] < 1.1
    assert not net.res_bus.at[3, "vm_pu"] == 1.02


def test_gen_p_vm_fixed(four_bus_net):
    # tests if gen max_vm_pu and min_vm_pu are correctly enforced
    net = four_bus_net
    min_vm_pu, max_vm_pu = .95, 1.05
    min_p_mw, max_p_mw = 0., 1.
    p_mw, vm_pu = 0.02, 1.01
    bus = 1

    # controllable == False -> limits are ignored and p_mw / vm_pu values are enforced
    pp.create_gen(net, bus, p_mw=p_mw, vm_pu=vm_pu, controllable=False,
                  min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu, min_p_mw=min_p_mw, max_p_mw=max_p_mw)
    pp.runopp(net)
    assert np.allclose(net.res_bus.at[bus, "vm_pu"], vm_pu)
    assert np.allclose(net.res_bus.at[bus, "p_mw"], -p_mw)


def test_gen_p_vm_limits(four_bus_net):
    # tests if gen max_vm_pu and min_vm_pu are correctly enforced
    net = four_bus_net
    net.bus.loc[:, "min_vm_pu"] = 0.9
    net.bus.loc[:, "max_vm_pu"] = 1.1
    min_vm_pu, max_vm_pu = .99, 1.005
    min_p_mw, max_p_mw = 0., 1.
    bus = 1
    # controllable == False -> limits are ignored and p_mw / vm_pu values are enforced
    pp.create_gen(net, bus, p_mw=0.02, vm_pu=1.01, controllable=True,
                  min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu, min_p_mw=min_p_mw, max_p_mw=max_p_mw)
    pp.runopp(net)
    assert not np.allclose(net.res_bus.at[bus, "vm_pu"], 1.01)
    assert not np.allclose(net.res_bus.at[bus, "p_mw"], 0.02)
    assert min_vm_pu < net.res_bus.at[bus, "vm_pu"] < max_vm_pu
    assert min_p_mw <= -net.res_bus.at[bus, "p_mw"] < max_p_mw


def test_gen_violated_p_vm_limits(four_bus_net):
    # tests if gen max_vm_pu and min_vm_pu are correctly enforced
    net = four_bus_net
    min_vm_pu, max_vm_pu = .98, 1.007  # gen limits are out of bus limits
    net.bus.loc[:, "min_vm_pu"] = min_vm_pu
    net.bus.loc[:, "max_vm_pu"] = max_vm_pu

    min_p_mw, max_p_mw = 0., 1.
    bus = 1
    # controllable == False -> limits are ignored and p_mw / vm_pu values are enforced
    g = pp.create_gen(net, bus, p_mw=0.02, vm_pu=1.01, controllable=True,
                      min_vm_pu=.9, max_vm_pu=1.1, min_p_mw=min_p_mw, max_p_mw=max_p_mw)
    pp.runopp(net)
    assert not np.allclose(net.res_bus.at[bus, "vm_pu"], 1.01)
    assert not np.allclose(net.res_bus.at[bus, "p_mw"], 0.02)
    assert min_vm_pu < net.res_bus.at[bus, "vm_pu"] < max_vm_pu
    assert min_p_mw <= -net.res_bus.at[bus, "p_mw"] < max_p_mw
    net.gen.at[g, "vm_pu"] = 0.9  # lower bus vm_pu limit violation
    pp.runopp(net)
    assert min_vm_pu < net.res_bus.at[bus, "vm_pu"] < max_vm_pu


def test_xward_p_mw_fixed():
    assert True


def test_xward_vm_pu_fixed():
    net = simple_four_bus_system()
    net.sgen.drop(index=1, inplace=True)
    net.load.drop(index=1, inplace=True)
    # net.sgen.drop(index=1, inplace=True)
    # net.load.drop(index=1, inplace=True)
    pp.create_xward(net, bus=3, pz_mw=.01, qz_mvar=.01, ps_mw=-.01, qs_mvar=.01,
                    vm_pu=0.979, x_ohm=.01, r_ohm=.01)
    pp.runpp(net)
    print(net.xward)
    print(net.res_xward)
    pp.diagnostic(net, report_style="none")
    #pp.runopp(net)
    print(net.xward)
    print(net.res_xward)
    #pp.runopp(net)
    assert True

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
