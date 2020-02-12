# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import os

import numpy as np
import pandas as pd
import pytest

import pandapower as pp
from pandapower.auxiliary import _check_connectivity, _add_ppc_options
from pandapower.networks import create_cigre_network_mv, four_loads_with_branches_out, \
    example_simple, simple_four_bus_system, example_multivoltage
from pandapower.pd2ppc import _pd2ppc
from pandapower.pf.create_jacobian import _create_J_without_numba
from pandapower.pf.run_newton_raphson_pf import _get_pf_variables_from_ppci
from pandapower.powerflow import LoadflowNotConverged
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.result_test_network_generator import add_test_xward, add_test_trafo3w, \
    add_test_line, add_test_oos_bus_with_is_element, result_test_network_generator, add_test_trafo
from pandapower.test.toolbox import add_grid_connection, create_test_line, assert_net_equal
from pandapower.toolbox import nets_equal


def test_minimal_net():
    # tests corner-case when the grid only has 1 bus and an ext-grid
    net = pp.create_empty_network()
    b = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b)
    runpp_with_consistency_checks(net)

    pp.create_load(net, b, p_mw=0.1)
    runpp_with_consistency_checks(net)

    b2 = pp.create_bus(net, 110)
    pp.create_switch(net, b, b2, 'b')
    pp.create_sgen(net, b2, p_mw=0.2)
    runpp_with_consistency_checks(net)


def test_set_user_pf_options():
    net = example_simple()
    pp.runpp(net)

    old_options = net._options.copy()
    test_options = {key: i for i, key in enumerate(old_options.keys())}

    pp.set_user_pf_options(net, hello='bye', **test_options)
    test_options.update({'hello': 'bye'})

    assert net.user_pf_options == test_options

    # remove what is in user_pf_options and add hello=world
    pp.set_user_pf_options(net, overwrite=True, hello='world')
    assert net.user_pf_options == {'hello': 'world'}

    # check if 'hello' is added to net._options, but other options are untouched
    pp.runpp(net)
    assert 'hello' in net._options.keys() and net._options['hello'] == 'world'
    net._options.pop('hello')
    assert net._options == old_options

    # check if user_pf_options can be deleted and net._options is as it was before
    pp.set_user_pf_options(net, overwrite=True, hello='world')
    pp.set_user_pf_options(net, overwrite=True)
    assert net.user_pf_options == {}
    pp.runpp(net)
    assert 'hello' not in net._options.keys()

    # see if user arguments overrule user_pf_options, but other user_pf_options still have the
    # priority
    pp.set_user_pf_options(net, tolerance_mva=1e-6, max_iteration=20)
    pp.runpp(net, tolerance_mva=1e-2)
    assert net.user_pf_options['tolerance_mva'] == 1e-6
    assert net._options['tolerance_mva'] == 1e-2
    assert net._options['max_iteration'] == 20


def test_kwargs_with_user_options():
    net = example_simple()
    pp.runpp(net)
    assert net._options["trafo3w_losses"] == "hv"
    pp.set_user_pf_options(net, trafo3w_losses="lv")
    pp.runpp(net)
    assert net._options["trafo3w_losses"] == "lv"


@pytest.mark.xfail(reason="Until now there was no way found to dynamically identify "
                          "the arguments passed to runpp, so if the user options are "
                          "overwritten with the default values, this is not recognized.")
def test_overwrite_default_args_with_user_options():
    net = example_simple()
    pp.runpp(net)
    assert net._options["check_connectivity"] is True
    pp.set_user_pf_options(net, check_connectivity=False)
    pp.runpp(net)
    assert net._options["check_connectivity"] is False
    pp.runpp(net, check_connectivity=True)
    assert net._options["check_connectivity"] is True


def test_runpp_init():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b3 = pp.create_bus(net, vn_kv=0.4)
    tidx = pp.create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.25 MVA 20/0.4 kV")
    net.trafo.shift_degree.at[tidx] = 70
    pp.runpp(net)
    va = net.res_bus.va_degree.at[4]
    pp.runpp(net, calculate_voltage_angles=True, init_va_degree="dc")
    assert np.allclose(va - net.trafo.shift_degree.at[tidx], net.res_bus.va_degree.at[4])
    pp.runpp(net, calculate_voltage_angles=True, init_va_degree="results")
    assert np.allclose(va - net.trafo.shift_degree.at[tidx], net.res_bus.va_degree.at[4])


def test_runpp_init_auxiliary_buses():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=20.)
    b4 = pp.create_bus(net, vn_kv=10.)
    tidx = pp.create_transformer3w(net, b2, b3, b4, std_type='63/25/38 MVA 110/20/10 kV')
    pp.create_load(net, b3, p_mw=5)
    pp.create_load(net, b4, p_mw=5)
    pp.create_xward(net, b4, ps_mw=1, qs_mvar=1, pz_mw=1, qz_mvar=1, r_ohm=0.1, x_ohm=0.1,
                    vm_pu=1.0)
    net.trafo3w.shift_lv_degree.at[tidx] = 120
    net.trafo3w.shift_mv_degree.at[tidx] = 80
    pp.runpp(net)
    va = net.res_bus.va_degree.at[b2]
    pp.runpp(net, calculate_voltage_angles=True, init_va_degree="dc")
    assert np.allclose(va - net.trafo3w.shift_mv_degree.at[tidx], net.res_bus.va_degree.at[b3],
                       atol=2)
    assert np.allclose(va - net.trafo3w.shift_lv_degree.at[tidx], net.res_bus.va_degree.at[b4],
                       atol=2)
    pp.runpp(net, calculate_voltage_angles=True, init_va_degree="results")
    assert np.allclose(va - net.trafo3w.shift_mv_degree.at[tidx], net.res_bus.va_degree.at[b3],
                       atol=2)
    assert np.allclose(va - net.trafo3w.shift_lv_degree.at[tidx], net.res_bus.va_degree.at[b4],
                       atol=2)


def test_result_iter():
    for net in result_test_network_generator():
        try:
            runpp_with_consistency_checks(net, enforce_q_lims=True)
        except (AssertionError):
            raise UserWarning("Consistency Error after adding %s" % net.last_added_case)
        except(LoadflowNotConverged):
            raise UserWarning("Power flow did not converge after adding %s" % net.last_added_case)


@pytest.fixture
def bus_bus_net():
    net = pp.create_empty_network()
    add_grid_connection(net)
    for _u in range(4):
        pp.create_bus(net, vn_kv=.4)
    pp.create_load(net, 5, p_mw=0.01)
    pp.create_switch(net, 3, 6, et="b")
    pp.create_switch(net, 4, 5, et="b")
    pp.create_switch(net, 6, 5, et="b")
    pp.create_switch(net, 0, 7, et="b")
    create_test_line(net, 4, 7)
    pp.create_load(net, 4, p_mw=0.01)
    return net


def test_bus_bus_switches(bus_bus_net):
    net = bus_bus_net
    pp.runpp(net)
    assert net.res_bus.vm_pu.at[3] == net.res_bus.vm_pu.at[4] == net.res_bus.vm_pu.at[5] == \
           net.res_bus.vm_pu.at[6]
    assert net.res_bus.vm_pu.at[0] == net.res_bus.vm_pu.at[7]

    net.bus.in_service.at[5] = False
    pp.runpp(net)
    assert net.res_bus.vm_pu.at[3] == net.res_bus.vm_pu.at[6]
    assert net.res_bus.vm_pu.at[0] == net.res_bus.vm_pu.at[7]
    assert pd.isnull(net.res_bus.vm_pu.at[5])
    assert net.res_bus.vm_pu.at[6] != net.res_bus.vm_pu.at[4]


def test_bus_bus_switches_merges_two_gens(bus_bus_net):
    "buses should not be fused if two gens are connected"
    net = bus_bus_net
    net.bus.in_service.at[5] = False
    pp.create_gen(net, 6, 10)
    pp.create_gen(net, 4, 10)
    net.bus.in_service.at[5] = True
    pp.runpp(net)
    assert net.converged


def test_bus_bus_switches_throws_exception_for_two_gen_with_diff_vm(bus_bus_net):
    "buses should not be fused if two gens are connected"
    net = bus_bus_net
    pp.create_gen(net, 6, 10, 1.)
    pp.create_gen(net, 4, 10, 1.1)
    with pytest.raises(UserWarning):
        pp.runpp(net)


@pytest.fixture
def z_switch_net():
    net = pp.create_empty_network()
    for i in range(3):
        pp.create_bus(net, vn_kv=.4)
        pp.create_load(net, i, p_mw=0.1)
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.1 / np.sqrt(2),
                                   x_ohm_per_km=0.1 / np.sqrt(2),
                                   c_nf_per_km=0, max_i_ka=.2)
    pp.create_switch(net, 0, 2, et="b", z_ohm=0.1)
    return net


@pytest.mark.parametrize("numba", [True, False])
def test_z_switch(z_switch_net, numba):
    net = z_switch_net
    pp.runpp(net, numba=numba, switch_rx_ratio=1)
    assert net.res_bus.vm_pu.at[1] == net.res_bus.vm_pu.at[2]

    net_zero_z_switch = copy.deepcopy(net)
    net_zero_z_switch.switch.z_ohm = 0
    pp.runpp(net_zero_z_switch, numba=numba, switch_rx_ratio=1)
    assert net_zero_z_switch.res_bus.vm_pu.at[0] == net_zero_z_switch.res_bus.vm_pu.at[2]


@pytest.fixture
def z_switch_net_4bus_parallel():
    net = pp.create_empty_network()
    for i in range(4):
        pp.create_bus(net, vn_kv=.4)
        pp.create_load(net, i, p_mw=0.1)
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.1 / np.sqrt(2),
                                   x_ohm_per_km=0.1 / np.sqrt(2),
                                   c_nf_per_km=0, max_i_ka=.2)
    pp.create_line_from_parameters(net, 1, 3, 1, r_ohm_per_km=0.1 / np.sqrt(2),
                                   x_ohm_per_km=0.1 / np.sqrt(2),
                                   c_nf_per_km=0, max_i_ka=.2)
    pp.create_switch(net, 0, 2, et="b", z_ohm=0.1)
    pp.create_switch(net, 0, 2, et="b", z_ohm=0)
    return net


@pytest.fixture
def z_switch_net_4bus():
    net = pp.create_empty_network()
    for i in range(4):
        pp.create_bus(net, vn_kv=.4)
        pp.create_load(net, i, p_mw=0.01)
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.1 / np.sqrt(2),
                                   x_ohm_per_km=0.1 / np.sqrt(2),
                                   c_nf_per_km=0, max_i_ka=.2)
    pp.create_switch(net, 1, 2, et="b", z_ohm=0.1)
    pp.create_switch(net, 2, 3, et="b", z_ohm=0)
    return net


@pytest.mark.parametrize("numba", [True, False])
def test_switch_fuse_z_ohm_0(z_switch_net_4bus_parallel, z_switch_net_4bus, numba):
    net = z_switch_net_4bus_parallel
    pp.runpp(net, numba=numba)
    assert net.res_bus.vm_pu[0] == net.res_bus.vm_pu[2]
    assert net.res_switch.i_ka[0] == 0

    net = z_switch_net_4bus
    pp.runpp(net, numba=numba)
    assert net.res_bus.vm_pu[2] == net.res_bus.vm_pu[3]
    assert net.res_bus.vm_pu[1] != net.res_bus.vm_pu[2]


@pytest.mark.parametrize("numba", [True, False])
def test_switch_z_ohm_different(z_switch_net_4bus_parallel, z_switch_net_4bus, numba):
    net = z_switch_net_4bus_parallel
    net.switch.at[1, 'z_ohm'] = 0.2
    pp.runpp(net, numba=numba)
    assert net.res_bus.vm_pu[0] != net.res_bus.vm_pu[2]
    assert np.all(net.res_switch.i_ka > 0)

    net = z_switch_net_4bus
    net.switch.at[1, 'z_ohm'] = 0.2
    pp.runpp(net, numba=numba)
    assert net.res_bus.vm_pu[2] != net.res_bus.vm_pu[3]
    assert net.res_bus.vm_pu[1] != net.res_bus.vm_pu[2]


def test_two_open_switches():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b3 = pp.create_bus(net, vn_kv=20.)
    l2 = create_test_line(net, b2, b3)
    create_test_line(net, b3, b1)
    pp.create_switch(net, b2, l2, et="l", closed=False)
    pp.create_switch(net, b3, l2, et="l", closed=False)
    pp.runpp(net)
    assert np.isnan(net.res_line.i_ka.at[l2]) or net.res_line.i_ka.at[l2] == 0


def test_oos_bus():
    net = pp.create_empty_network()
    add_test_oos_bus_with_is_element(net)
    assert runpp_with_consistency_checks(net)

    #    test for pq-node result
    pp.create_shunt(net, 6, q_mvar=0.8)
    assert runpp_with_consistency_checks(net)

    #   1test for pv-node result
    pp.create_gen(net, 4, p_mw=0.5)
    assert runpp_with_consistency_checks(net)


def get_isolated(net):
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=False,
                     trafo_model="t", check_connectivity=False,
                     mode="pf", switch_rx_ratio=2, init_vm_pu="flat",
                     init_va_degree="flat",
                     enforce_q_lims=False, recycle=None)

    ppc, ppci = _pd2ppc(net)
    return _check_connectivity(ppc)


def test_connectivity_check_island_without_pv_bus():
    # Network with islands without pv bus -> all buses in island should be set out of service
    net = create_cigre_network_mv(with_der=False)
    iso_buses, iso_p, iso_q = get_isolated(net)
    assert len(iso_buses) == 0
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    isolated_bus1 = pp.create_bus(net, vn_kv=20., name="isolated Bus1")
    isolated_bus2 = pp.create_bus(net, vn_kv=20., name="isolated Bus2")
    pp.create_line(net, isolated_bus2, isolated_bus1, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                   name="IsolatedLine")
    iso_buses, iso_p, iso_q = get_isolated(net)
    assert len(iso_buses) == 2
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    pp.create_load(net, isolated_bus1, p_mw=0.2, q_mvar=0.02)
    pp.create_sgen(net, isolated_bus2, p_mw=0.15, q_mvar=0.01)

    # with pytest.warns(UserWarning):
    iso_buses, iso_p, iso_q = get_isolated(net)
    assert len(iso_buses) == 2
    assert np.isclose(iso_p, 350)
    assert np.isclose(iso_q, 30)
    # with pytest.warns(UserWarning):
    runpp_with_consistency_checks(net, check_connectivity=True)


def test_connectivity_check_island_with_one_pv_bus():
    # Network with islands with one PV bus -> PV bus should be converted to the reference bus
    net = create_cigre_network_mv(with_der=False)
    iso_buses, iso_p, iso_q = get_isolated(net)
    assert len(iso_buses) == 0
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    isolated_bus1 = pp.create_bus(net, vn_kv=20., name="isolated Bus1")
    isolated_bus2 = pp.create_bus(net, vn_kv=20., name="isolated Bus2")
    isolated_gen = pp.create_bus(net, vn_kv=20., name="isolated Gen")
    isolated_pv_bus = pp.create_gen(net, isolated_gen, p_mw=0.35, vm_pu=1.0, name="isolated PV bus")
    pp.create_line(net, isolated_bus2, isolated_bus1, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="IsolatedLine")
    pp.create_line(net, isolated_gen, isolated_bus1, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="IsolatedLineToGen")
    # with pytest.warns(UserWarning):
    iso_buses, iso_p, iso_q = get_isolated(net)

    # assert len(iso_buses) == 0
    # assert np.isclose(iso_p, 0)
    # assert np.isclose(iso_q, 0)
    #
    # pp.create_load(net, isolated_bus1, p_mw=0.200., q_mvar=0.020)
    # pp.create_sgen(net, isolated_bus2, p_mw=0.0150., q_mvar=-0.010)
    #
    # iso_buses, iso_p, iso_q = get_isolated(net)
    # assert len(iso_buses) == 0
    # assert np.isclose(iso_p, 0)
    # assert np.isclose(iso_q, 0)

    # with pytest.warns(UserWarning):
    runpp_with_consistency_checks(net, check_connectivity=True)


def test_connectivity_check_island_with_multiple_pv_buses():
    # Network with islands an multiple PV buses in the island -> Error should be thrown since it
    # would be random to choose just some PV bus as the reference bus
    net = create_cigre_network_mv(with_der=False)
    iso_buses, iso_p, iso_q = get_isolated(net)
    assert len(iso_buses) == 0
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    isolated_bus1 = pp.create_bus(net, vn_kv=20., name="isolated Bus1")
    isolated_bus2 = pp.create_bus(net, vn_kv=20., name="isolated Bus2")
    isolated_pv_bus1 = pp.create_bus(net, vn_kv=20., name="isolated PV bus1")
    isolated_pv_bus2 = pp.create_bus(net, vn_kv=20., name="isolated PV bus2")
    pp.create_gen(net, isolated_pv_bus1, p_mw=0.3, vm_pu=1.0, name="isolated PV bus1")
    pp.create_gen(net, isolated_pv_bus2, p_mw=0.05, vm_pu=1.0, name="isolated PV bus2")

    pp.create_line(net, isolated_pv_bus1, isolated_bus1, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                   name="IsolatedLineToGen1")
    pp.create_line(net, isolated_pv_bus2, isolated_bus2, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                   name="IsolatedLineToGen2")
    pp.create_line(net, isolated_bus2, isolated_bus1, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                   name="IsolatedLine")
    # ToDo with pytest.warns(UserWarning):
    iso_buses, iso_p, iso_q = get_isolated(net)


def test_isolated_in_service_bus_at_oos_line():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b = pp.create_bus(net, vn_kv=135)
    l = pp.create_line(net, b2, b, 0.1, std_type="NAYY 4x150 SE")
    net.line.loc[l, "in_service"] = False
    assert runpp_with_consistency_checks(net, init="flat")


def test_isolated_in_service_line():
    # ToDo: Fix this
    net = pp.create_empty_network()
    _, b2, l1 = add_grid_connection(net)
    b = pp.create_bus(net, vn_kv=20.)
    pp.create_line(net, b2, b, 0.1, std_type="NAYY 4x150 SE")
    net.line.loc[l1, "in_service"] = False
    assert runpp_with_consistency_checks(net, init="flat")


def test_makeYbus():
    # tests if makeYbus fails for nets where every bus is connected to each other
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)

    # number of buses to create
    n_bus = 20
    bus_list = []
    # generate buses and connect them
    for _ in range(n_bus):
        bus_list.append(pp.create_bus(net, vn_kv=20.))

    # connect the first bus to slack node
    create_test_line(net, bus_list[0], b2)
    # iterate over every bus and add connection to every other bus
    for bus_1 in bus_list:
        for bus_2 in bus_list:
            # add no connection to itself
            if bus_1 == bus_2:
                continue
            create_test_line(net, bus_1, bus_2)

    assert runpp_with_consistency_checks(net)


def test_test_sn_mva():
    test_net_gen1 = result_test_network_generator(sn_mva=1)
    test_net_gen2 = result_test_network_generator(sn_mva=2)
    for net1, net2 in zip(test_net_gen1, test_net_gen2):
        pp.runpp(net1)
        pp.runpp(net2)
        try:
            assert_net_equal(net1, net2)
        except:
            raise UserWarning("Result difference due to sn_mva after adding %s" %
                              net1.last_added_case)


def test_bsfw_algorithm():
    net = example_simple()

    pp.runpp(net)
    vm_nr = copy.copy(net.res_bus.vm_pu)
    va_nr = copy.copy(net.res_bus.va_degree)

    pp.runpp(net, algorithm='bfsw')
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree

    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


@pytest.mark.xfail(reason="unknown")
def test_bsfw_algorithm_multi_net():
    net = example_simple()
    add_grid_connection(net, vn_kv=110., zone="second")

    pp.runpp(net)
    vm_nr = copy.copy(net.res_bus.vm_pu)
    va_nr = copy.copy(net.res_bus.va_degree)

    pp.runpp(net, algorithm='bfsw')
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree

    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


def test_bsfw_algorithm_with_trafo_shift_and_voltage_angles():
    net = example_simple()
    net["trafo"].loc[:, "shift_degree"] = 180.

    pp.runpp(net, calculate_voltage_angles=True)
    vm_nr = net.res_bus.vm_pu
    va_nr = net.res_bus.va_degree

    pp.runpp(net, algorithm='bfsw', calculate_voltage_angles=True)
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree
    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


def test_bsfw_algorithm_with_enforce_q_lims():
    net = example_simple()
    net.ext_grid["max_q_mvar"] = [0.1]
    net.ext_grid["min_q_mvar"] = [-0.1]
    net.gen["max_q_mvar"] = [5.]
    net.gen["min_q_mvar"] = [4.]

    pp.runpp(net, enforce_q_lims=True)
    vm_nr = net.res_bus.vm_pu
    va_nr = net.res_bus.va_degree

    pp.runpp(net, algorithm='bfsw', enforce_q_lims=True)
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree
    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


def test_bsfw_algorithm_with_branch_loops():
    net = example_simple()
    pp.create_line(net, 0, 6, length_km=2.5,
                   std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line meshed")
    net.switch.loc[:, "closed"] = True

    pp.runpp(net)
    vm_nr = net.res_bus.vm_pu
    va_nr = net.res_bus.va_degree

    pp.runpp(net, algorithm='bfsw')
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree
    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


@pytest.mark.slow
def test_pypower_algorithms_iter():
    alg_to_test = ['fdbx', 'fdxb', 'gs']
    for alg in alg_to_test:
        for net in result_test_network_generator(skip_test_impedance=True):
            try:
                runpp_with_consistency_checks(net, enforce_q_lims=True, algorithm=alg)
                runpp_with_consistency_checks(net, enforce_q_lims=False, algorithm=alg)
            except (AssertionError):
                raise UserWarning("Consistency Error after adding %s" % net.last_added_case)
            except(LoadflowNotConverged):
                raise UserWarning("Power flow did not converge after adding %s" %
                                  net.last_added_case)


@pytest.mark.xfail
def test_zip_loads_gridcal():
    # Tests newton power flow considering zip loads against GridCal's pf result

    # Results used for benchmarking are obtained using GridCal with the following code:
    # from GridCal.grid.CalculationEngine import *
    #
    # np.set_printoptions(precision=4)
    # grid = MultiCircuit()
    #
    # # Add buses
    # bus1 = Bus('Bus 1', vnom=20)
    #
    # bus1.controlled_generators.append(ControlledGenerator('Slack Generator', voltage_module=1.0))
    # grid.add_bus(bus1)
    #
    # bus2 = Bus('Bus 2', vnom=20)
    # bus2.loads.append(Load('load 2',
    #                        power=0.2 * complex(40, 20),
    #                        impedance=1 / (0.40 * (40. - 20.j)),
    #                        current=np.conj(0.40 * (40. + 20.j)) / (20 * np.sqrt(3)),
    #                        ))
    # grid.add_bus(bus2)
    #
    # bus3 = Bus('Bus 3', vnom=20)
    # bus3.loads.append(Load('load 3', power=complex(25, 15)))
    # grid.add_bus(bus3)
    #
    # bus4 = Bus('Bus 4', vnom=20)
    # bus4.loads.append(Load('load 4', power=complex(40, 20)))
    # grid.add_bus(bus4)
    #
    # bus5 = Bus('Bus 5', vnom=20)
    # bus5.loads.append(Load('load 5', power=complex(50, 20)))
    # grid.add_bus(bus5)
    #
    # # add branches (Lines in this case)
    # grid.add_branch(Branch(bus1, bus2, 'line 1-2', r=0.05, x=0.11, b=0.02))
    #
    # grid.add_branch(Branch(bus1, bus3, 'line 1-3', r=0.05, x=0.11, b=0.02))
    #
    # grid.add_branch(Branch(bus1, bus5, 'line 1-5', r=0.03, x=0.08, b=0.02))
    #
    # grid.add_branch(Branch(bus2, bus3, 'line 2-3', r=0.04, x=0.09, b=0.02))
    #
    # grid.add_branch(Branch(bus2, bus5, 'line 2-5', r=0.04, x=0.09, b=0.02))
    #
    # grid.add_branch(Branch(bus3, bus4, 'line 3-4', r=0.06, x=0.13, b=0.03))
    #
    # grid.add_branch(Branch(bus4, bus5, 'line 4-5', r=0.04, x=0.09, b=0.02))
    #
    # grid.compile()
    #
    # print('Ybus:\n', grid.circuits[0].power_flow_input.Ybus.todense())
    #
    # options = PowerFlowOptions(SolverType.NR, robust=False)
    # power_flow = PowerFlow(grid, options)
    # power_flow.run()
    #
    # print('\n\n', grid.name)
    # print('\t|V|:', abs(grid.power_flow_results.voltage))
    # print('\tVang:', np.rad2deg(np.angle(grid.power_flow_results.voltage)))

    vm_pu_gridcal = np.array([1., 0.9566486349, 0.9555640318, 0.9340468428, 0.9540542172])
    va_degree_gridcal = np.array([0., -2.3717973886, -2.345654238, -3.6303651197, -2.6713716569])

    Ybus_gridcal = np.array(
        [[10.9589041096 - 25.9973972603j, -3.4246575342 + 7.5342465753j,
          -3.4246575342 + 7.5342465753j,
          0.0000000000 + 0.j, -4.1095890411 + 10.9589041096j],
         [-3.4246575342 + 7.5342465753j, 11.8320802147 - 26.1409476063j,
          -4.1237113402 + 9.2783505155j,
          0.0000000000 + 0.j, -4.1237113402 + 9.2783505155j],
         [-3.4246575342 + 7.5342465753j, -4.1237113402 + 9.2783505155j,
          10.4751981427 - 23.1190605054j,
          -2.9268292683 + 6.3414634146j, 0.0000000000 + 0.j],
         [0.0000000000 + 0.j, 0.0000000000 + 0.j, -2.9268292683 + 6.3414634146j,
          7.0505406085 - 15.5948139301j,
          -4.1237113402 + 9.2783505155j],
         [-4.1095890411 + 10.9589041096j, -4.1237113402 + 9.2783505155j, 0.0000000000 + 0.j,
          -4.1237113402 + 9.2783505155j, 12.3570117215 - 29.4856051405j]])

    losses_gridcal = 4.69773448916 - 2.710430515j

    abs_path = os.path.join(pp.pp_dir, 'networks', 'power_system_test_case_jsons',
                            'case5_demo_gridcal.json')
    net = pp.from_json(abs_path)

    pp.runpp(net, voltage_depend_loads=True, recycle=None)

    # Test Ybus matrix
    Ybus_pp = net["_ppc"]['internal']['Ybus'].todense()
    bus_ord = net["_pd2ppc_lookups"]["bus"]
    Ybus_pp = Ybus_pp[bus_ord, :][:, bus_ord]

    assert np.allclose(Ybus_pp, Ybus_gridcal)

    # Test Results
    assert np.allclose(net.res_bus.vm_pu, vm_pu_gridcal)
    assert np.allclose(net.res_bus.va_degree, va_degree_gridcal)

    # Test losses
    losses_pp = net.res_bus.p_mw.sum() + 1.j * net.res_bus.q_mvar.sum()
    assert np.isclose(losses_gridcal, - losses_pp / 1.e3)

    # Test bfsw algorithm
    pp.runpp(net, voltage_depend_loads=True, algorithm='bfsw')
    assert np.allclose(net.res_bus.vm_pu, vm_pu_gridcal)
    assert np.allclose(net.res_bus.va_degree, va_degree_gridcal)


def test_zip_loads_consistency():
    net = four_loads_with_branches_out()
    net.load['const_i_percent'] = 40
    net.load['const_z_percent'] = 40
    assert runpp_with_consistency_checks(net)


def test_zip_loads_pf_algorithms():
    net = four_loads_with_branches_out()
    net.load['const_i_percent'] = 40
    net.load['const_z_percent'] = 40

    alg_to_test = ['bfsw']
    for alg in alg_to_test:
        pp.runpp(net, algorithm='nr')
        vm_nr = net.res_bus.vm_pu
        va_nr = net.res_bus.va_degree

        pp.runpp(net, algorithm=alg)
        vm_alg = net.res_bus.vm_pu
        va_alg = net.res_bus.va_degree

        assert np.allclose(vm_nr, vm_alg, rtol=1e-6)
        assert np.allclose(va_nr.values, va_alg.values, rtol=1e-4)


def test_zip_loads_with_voltage_angles():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=1.)
    b2 = pp.create_bus(net, vn_kv=1.)
    pp.create_ext_grid(net, b1)
    pp.create_line_from_parameters(net, b1, b2, length_km=1, r_ohm_per_km=0.3,
                                   x_ohm_per_km=0.3, c_nf_per_km=10, max_i_ka=1)
    pp.create_load(net, b2, p_mw=0.002, const_z_percent=0, const_i_percent=100)

    pp.set_user_pf_options(net, calculate_voltage_angles=True, init='dc')

    pp.runpp(net)

    res_load = net.res_load.copy()
    net.ext_grid.va_degree = 100

    pp.runpp(net)

    assert np.allclose(net.res_load.values, res_load.values)


def test_xward_buses():
    """
    Issue: xward elements create dummy buses for the load flow, that are cleaned up afterwards.
    However, if the load flow does not converge, those buses end up staying in the net and don't get
    removed. This can potentially lead to thousands of dummy buses in net.
    """
    net = pp.create_empty_network()
    bus_sl = pp.create_bus(net, 110, name='ExtGrid')
    pp.create_ext_grid(net, bus_sl, vm_pu=1)
    bus_x = pp.create_bus(net, 110, name='XWARD')
    pp.create_xward(net, bus_x, 0, 0, 0, 0, 0, 10, 1.1)
    iid = pp.create_impedance(net, bus_sl, bus_x, 0.2, 0.2, 1e3)

    bus_num1 = len(net.bus)

    pp.runpp(net)

    bus_num2 = len(net.bus)

    assert bus_num1 == bus_num2

    # now - make sure that the loadflow doesn't converge:
    net.impedance.at[iid, 'rft_pu'] = 1
    pp.create_load(net, bus_x, 1e6, 0)
    with pytest.raises(LoadflowNotConverged):
        # here the load flow doesn't converge and there is an extra bus in net
        pp.runpp(net)

    bus_num3 = len(net.bus)
    assert bus_num3 == bus_num1


def test_pvpq_lookup():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=0.4, index=4)
    b2 = pp.create_bus(net, vn_kv=0.4, index=2)
    b3 = pp.create_bus(net, vn_kv=0.4, index=3)

    pp.create_gen(net, b1, p_mw=0.01, vm_pu=0.4)
    pp.create_load(net, b2, p_mw=0.01)
    pp.create_ext_grid(net, b3)

    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=0.5, std_type="NAYY 4x120 SE")
    pp.create_line(net, from_bus=b1, to_bus=b3, length_km=0.5, std_type="NAYY 4x120 SE")
    net_numba = copy.deepcopy(net)
    pp.runpp(net_numba, numba=True)
    pp.runpp(net, numba=False)

    assert nets_equal(net, net_numba)


def test_get_internal():
    net = example_simple()
    # for Newton raphson
    pp.runpp(net)
    J_intern = net._ppc["internal"]["J"]

    ppc = net._ppc
    V_mag = ppc["bus"][:, 7][:-2]
    V_ang = ppc["bus"][:, 8][:-2]
    V = V_mag * np.exp(1j * V_ang / 180 * np.pi)

    # Get stored Ybus in ppc
    Ybus = ppc["internal"]["Ybus"]

    _, ppci = _pd2ppc(net)
    baseMVA, bus, gen, branch, ref, pv, pq, _, _, V0, _ = _get_pf_variables_from_ppci(ppci)

    pvpq = np.r_[pv, pq]

    J = _create_J_without_numba(Ybus, V, pvpq, pq)

    assert sum(sum(abs(abs(J.toarray()) - abs(J_intern.toarray())))) < 0.05
    # get J for all other algorithms


def test_storage_pf():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)

    pp.create_line(net, b1, b2, length_km=5, std_type="NAYY 4x50 SE")

    pp.create_ext_grid(net, b2)
    pp.create_load(net, b1, p_mw=0.010)
    pp.create_sgen(net, b1, p_mw=0.010)

    # test generator behaviour
    pp.create_storage(net, b1, p_mw=-0.010, max_e_mwh=0.010)
    pp.create_sgen(net, b1, p_mw=0.010, in_service=False)

    res_gen_beh = runpp_with_consistency_checks(net)
    res_ll_stor = net["res_line"].loading_percent.iloc[0]

    net["storage"].in_service.iloc[0] = False
    net["sgen"].in_service.iloc[1] = True

    runpp_with_consistency_checks(net)
    res_ll_sgen = net["res_line"].loading_percent.iloc[0]

    assert np.isclose(res_ll_stor, res_ll_sgen)

    # test load behaviour
    pp.create_load(net, b1, p_mw=0.01, in_service=False)
    net["storage"].in_service.iloc[0] = True
    net["storage"].p_mw.iloc[0] = 0.01
    net["sgen"].in_service.iloc[1] = False

    res_load_beh = runpp_with_consistency_checks(net)
    res_ll_stor = net["res_line"].loading_percent.iloc[0]

    net["storage"].in_service.iloc[0] = False
    net["load"].in_service.iloc[1] = True

    runpp_with_consistency_checks(net)
    res_ll_load = net["res_line"].loading_percent.iloc[0]

    assert np.isclose(res_ll_stor, res_ll_load)

    assert res_gen_beh and res_load_beh


def test_add_element_and_init_results():
    net = simple_four_bus_system()
    pp.runpp(net, init="flat")
    pp.create_bus(net, vn_kv=20.)
    pp.create_line(net, from_bus=2, to_bus=3, length_km=1, name="new line" + str(1),
                   std_type="NAYY 4x150 SE")
    pp.runpp(net, init="results")


def test_pp_initialization():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)

    pp.create_ext_grid(net, b1, vm_pu=0.7)
    pp.create_line(net, b1, b2, 0.5, std_type="NAYY 4x50 SE", index=4)
    pp.create_load(net, b2, p_mw=0.01)

    pp.runpp(net, init_va_degree="flat", init_vm_pu=1.02)
    assert net._ppc["iterations"] == 5

    pp.runpp(net, init_va_degree="dc", init_vm_pu=0.8)
    assert net._ppc["iterations"] == 4

    pp.runpp(net, init_va_degree="flat", init_vm_pu=np.array([0.75, 0.7]))
    assert net._ppc["iterations"] == 3

    pp.runpp(net, init_va_degree="dc", init_vm_pu=[0.75, 0.7])
    assert net._ppc["iterations"] == 3

    pp.runpp(net, init_va_degree="flat", init_vm_pu="auto")
    assert net._ppc["iterations"] == 3

    pp.runpp(net, init_va_degree="dc")
    assert net._ppc["iterations"] == 3


def test_equal_indices_res():
    # tests if res_bus indices of are the same as the ones in bus.
    # If this is not the case and you init from results, the PF will fail
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=10., index=3)
    b2 = pp.create_bus(net, vn_kv=0.4, index=1)
    b3 = pp.create_bus(net, vn_kv=0.4, index=2)

    pp.create_ext_grid(net, b1)
    pp.create_transformer(net, b1, b2, std_type="0.63 MVA 20/0.4 kV")
    pp.create_line(net, b2, b3, 0.5, std_type="NAYY 4x50 SE", index=4)
    pp.create_load(net, b3, p_mw=0.010)
    pp.runpp(net)
    net["bus"] = net["bus"].sort_index()
    try:
        pp.runpp(net, init_vm_pu="results", init_va_degree="results")
        assert True
    except LoadflowNotConverged:
        assert False


def test_ext_grid_and_gen_at_one_bus():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)
    pp.create_ext_grid(net, b1, vm_pu=1.01)
    pp.create_line(net, b1, b2, 1., std_type="305-AL1/39-ST1A 110.0")
    pp.create_load(net, bus=b2, p_mw=3.5, q_mvar=1)

    runpp_with_consistency_checks(net)
    q = net.res_ext_grid.q_mvar.sum()

    ##create two gens at the slack bus
    g1 = pp.create_gen(net, b1, vm_pu=1.01, p_mw=1)
    g2 = pp.create_gen(net, b1, vm_pu=1.01, p_mw=1)
    runpp_with_consistency_checks(net)

    # all the reactive power previously provided by the ext_grid is now provided by the generators
    assert np.isclose(net.res_ext_grid.q_mvar.values, 0)
    assert np.isclose(net.res_gen.q_mvar.sum(), q)
    # since no Q-limits were set, reactive power is distributed equally to both generators
    assert np.isclose(net.res_gen.q_mvar.at[g1], net.res_gen.q_mvar.at[g2])

    # set reactive power limits at the generators
    net.gen["max_q_mvar"] = [0.1, 0.01]
    net.gen["min_q_mvar"] = [-0.1, -0.01]
    runpp_with_consistency_checks(net)
    # g1 now has 10 times the reactive power of g2 in accordance with the different Q ranges
    assert np.isclose(net.res_gen.q_mvar.at[g1], net.res_gen.q_mvar.at[g2] * 10)
    # all the reactive power is still provided by the generators, because Q-lims are not enforced
    assert np.allclose(net.res_ext_grid.q_mvar.values, [0])
    assert np.isclose(net.res_gen.q_mvar.sum(), q)

    # now enforce Q-lims
    runpp_with_consistency_checks(net, enforce_q_lims=True)
    # both generators are at there lower limit with regard to the reactive power
    assert np.allclose(net.res_gen.q_mvar.values, net.gen.max_q_mvar.values)
    # the total reactive power remains unchanged, but the rest of the power is now provided by the ext_grid
    assert np.isclose(net.res_gen.q_mvar.sum() + net.res_ext_grid.q_mvar.sum(), q)

    # second ext_grid at the slack bus
    pp.create_ext_grid(net, b1, vm_pu=1.01)
    runpp_with_consistency_checks(net, enforce_q_lims=False)
    # gens still have the correct active power
    assert np.allclose(net.gen.p_mw.values, net.res_gen.p_mw.values)
    # slack active power is evenly distributed to both ext_grids
    assert np.isclose(net.res_ext_grid.p_mw.values[0], net.res_ext_grid.p_mw.values[1])

    # q limits at the ext_grids are not enforced
    net.ext_grid["max_q_mvar"] = [0.1, 0.01]
    net.ext_grid["min_q_mvar"] = [-0.1, -0.01]
    runpp_with_consistency_checks(net, enforce_q_lims=True)
    assert net.res_ext_grid.q_mvar.values[0] > net.ext_grid.max_q_mvar.values[0]
    assert np.allclose(net.res_gen.q_mvar.values, net.gen.max_q_mvar.values)


def two_ext_grids_at_one_bus():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110, index=3)
    b2 = pp.create_bus(net, vn_kv=110, index=5)
    pp.create_ext_grid(net, b1, vm_pu=1.01, index=2)
    pp.create_line(net, b1, b2, 1., std_type="305-AL1/39-ST1A 110.0")
    pp.create_load(net, bus=b2, p_mw=3.5, q_mvar=1)
    pp.create_gen(net, b1, vm_pu=1.01, p_mw=1)
    runpp_with_consistency_checks(net)
    assert net.converged

    # connect second ext_grid to b1 with different angle but out of service
    eg2 = pp.create_ext_grid(net, b1, vm_pu=1.01, va_degree=20, index=5, in_service=False)
    runpp_with_consistency_checks(net)  # power flow still converges since eg2 is out of service
    assert net.converged

    # error is raised after eg2 is set in service
    net.ext_grid.in_service.at[eg2] = True
    with pytest.raises(UserWarning):
        pp.runpp(net)

    #  error is also raised when eg2 is connected to first ext_grid through bus-bus switch
    b3 = pp.create_bus(net, vn_kv=110)
    pp.create_switch(net, b1, b3, et="b")
    net.ext_grid.bus.at[eg2] = b3
    with pytest.raises(UserWarning):
        pp.runpp(net)

    # no error is raised when voltage angles are not calculated
    runpp_with_consistency_checks(net, calculate_voltage_angles=False)
    assert net.converged

    # same angle but different voltage magnitude also raises an error
    net.ext_grid.vm_pu.at[eg2] = 1.02
    net.ext_grid.va_degree.at[eg2] = 0
    with pytest.raises(UserWarning):
        pp.runpp(net)


def test_dc_with_ext_grid_at_one_bus():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110)
    b2 = pp.create_bus(net, vn_kv=110)

    pp.create_ext_grid(net, b1, vm_pu=1.01)
    pp.create_ext_grid(net, b2, vm_pu=1.01)

    pp.create_dcline(net, from_bus=b1, to_bus=b2, p_mw=10,
                     loss_percent=0, loss_mw=0, vm_from_pu=1.01, vm_to_pu=1.01)

    pp.create_sgen(net, b1, p_mw=10)
    pp.create_load(net, b2, p_mw=10)

    runpp_with_consistency_checks(net)
    assert np.allclose(net.res_ext_grid.p_mw.values, [0, 0])


def test_init_results_without_results():
    # should switch to "auto" mode and not fail
    net = example_multivoltage()
    pp.reset_results(net)
    pp.runpp(net, init="results")
    assert net.converged
    pp.reset_results(net)
    pp.runpp(net, init_vm_pu="results")
    assert net.converged
    pp.reset_results(net)
    pp.runpp(net, init_va_degree="results")
    assert net.converged
    pp.reset_results(net)
    pp.runpp(net, init_va_degree="results", init_vm_pu="results")
    assert net.converged


def test_init_results():
    net = pp.create_empty_network()
    add_test_line(net)  # line network with switch at to bus
    assert_init_results(net)
    net.switch.at[0, "bus"] = 0  # switch at from bus
    assert_init_results(net)

    add_test_trafo(net)  # trafo network with switch at lv bus
    assert_init_results(net)
    net.switch.at[0, "bus"] = 7  # switch at hv bus
    assert_init_results(net)

    add_test_xward(net)  # xward with internal node
    assert_init_results(net)
    add_test_trafo3w(net)  # trafo3w with internal node
    assert_init_results(net)
    t3idx = net.trafo3w.index[0]
    t3_switch = pp.create_switch(net, bus=net.trafo3w.hv_bus.at[t3idx],
                                 element=t3idx, et="t3", closed=False)  # trafo3w switch at hv side
    assert_init_results(net)
    net.switch.bus.at[t3_switch] = net.trafo3w.mv_bus.at[t3idx]  # trafo3w switch at mv side
    assert_init_results(net)
    net.switch.bus.at[t3_switch] = net.trafo3w.lv_bus.at[t3idx]  # trafo3w switch at lv side
    assert_init_results(net)


def assert_init_results(net):
    pp.runpp(net, init="auto")
    assert net._ppc["iterations"] > 0
    pp.runpp(net, init="results")
    assert net._ppc["iterations"] == 0


def test_wye_delta():
    from pandapower.pypower.idx_brch import BR_R, BR_X, BR_B
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=110)
    pp.create_buses(net, nr_buses=4, vn_kv=20)
    trafo = pp.create_transformer(net, hv_bus=0, lv_bus=1, std_type='25 MVA 110/20 kV')
    pp.create_line(net, 1, 2, length_km=2.0, std_type="NAYY 4x50 SE")
    pp.create_line(net, 2, 3, length_km=6.0, std_type="NAYY 4x50 SE")
    pp.create_line(net, 3, 4, length_km=10.0, std_type="NAYY 4x50 SE")
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 4, p_mw=0.1)
    pp.create_sgen(net, 2, p_mw=4.)
    pp.create_sgen(net, 3, p_mw=4.)

    pp.runpp(net, trafo_model="pi")
    f, t = net._pd2ppc_lookups["branch"]["trafo"]
    assert np.isclose(net.res_trafo.p_hv_mw.at[trafo], -7.560996, rtol=1e-7)
    assert np.allclose(net._ppc["branch"][f:t, [BR_R, BR_X, BR_B]].flatten(),
                       np.array([0.0001640 + 0.j, 0.0047972 + 0.j, -0.0105000 - 0.014j]),
                       rtol=1e-7)

    pp.runpp(net, trafo_model="t")
    assert np.allclose(net._ppc["branch"][f:t, [BR_R, BR_X, BR_B]].flatten(),
                       np.array([0.00016392 + 0.j, 0.00479726 + 0.j, -0.01050009 - 0.01399964j]))
    assert np.isclose(net.res_trafo.p_hv_mw.at[trafo], -7.561001, rtol=1e-7)


def test_line_temperature():
    net = four_loads_with_branches_out()
    r_init = net.line.r_ohm_per_km.values.copy()

    # r_ohm_per_km is not in line results by default
    pp.runpp(net)
    v_init = net.res_bus.vm_pu.values.copy()
    va_init = net.res_bus.va_degree.values.copy()
    assert "r_ohm_per_km" not in net.res_line.columns

    # no temperature adjustment performed if not explicitly set in options/arguments to runpp
    net.line["temperature_degree_celsius"] = 20
    pp.runpp(net)
    assert "r_ohm_per_km" not in net.res_line.columns
    assert np.allclose(net.res_bus.vm_pu, v_init, rtol=0, atol=1e-16)
    assert np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-16)

    # argument in runpp is considered
    pp.runpp(net, consider_line_temperature=True)
    assert "r_ohm_per_km" in net.res_line.columns
    assert np.allclose(net.res_line.r_ohm_per_km, r_init, rtol=0, atol=1e-16)
    assert np.allclose(net.res_bus.vm_pu, v_init, rtol=0, atol=1e-16)
    assert np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-16)

    # check results of r adjustment, check that user_pf_options works, alpha is 4e-3 by default
    t = np.arange(0, 80, 10)
    net.line.temperature_degree_celsius = t
    pp.set_user_pf_options(net, consider_line_temperature=True)
    pp.runpp(net)
    alpha = 4e-3
    r_temp = r_init * (1 + alpha * (t - 20))
    assert np.allclose(net.res_line.r_ohm_per_km, r_temp, rtol=0, atol=1e-16)
    assert not np.allclose(net.res_bus.vm_pu, v_init, rtol=0, atol=1e-4)
    assert not np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-2)

    # check reults with user-defined alpha
    alpha = np.arange(3e-3, 5e-3, 2.5e-4)
    net.line['alpha'] = alpha
    pp.runpp(net)
    r_temp = r_init * (1 + alpha * (t - 20))
    assert np.allclose(net.res_line.r_ohm_per_km, r_temp, rtol=0, atol=1e-16)
    assert not np.allclose(net.res_bus.vm_pu, v_init, rtol=0, atol=1e-4)
    assert not np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-2)

    # not anymore in net if not considered
    pp.set_user_pf_options(net, overwrite=True)
    pp.runpp(net)
    assert np.allclose(net.res_bus.vm_pu, v_init, rtol=0, atol=1e-16)
    assert np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-16)
    assert "r_ohm_per_km" not in net.res_line.columns


def test_results_for_line_temperature():
    net = pp.create_empty_network()
    pp.create_bus(net, 0.4)
    pp.create_buses(net, 2, 0.4)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, 5e-3, 10e-3)
    pp.create_load(net, 2, 5e-3, 10e-3)

    pp.create_line_from_parameters(net, 0, 1, 0.5, 0.642, 0.083, 210, 1, alpha=0.00403)
    pp.create_line_from_parameters(net, 1, 2, 0.5, 0.642, 0.083, 210, 1, alpha=0.00403)

    vm_res_20 = [1, 0.9727288676, 0.95937328755]
    va_res_20 = [0, 2.2103403814, 3.3622612327]
    vm_res_80 = [1, 0.96677572771, 0.95062498477]
    va_res_80 = [0, 2.7993156134, 4.2714451629]

    pp.runpp(net)

    assert np.allclose(net.res_bus.vm_pu, vm_res_20, rtol=0, atol=1e-6)
    assert np.allclose(net.res_bus.va_degree, va_res_20, rtol=0, atol=1e-6)

    net.line["temperature_degree_celsius"] = 80
    pp.set_user_pf_options(net, consider_line_temperature=True)
    pp.runpp(net)

    assert np.allclose(net.res_bus.vm_pu, vm_res_80, rtol=0, atol=1e-6)
    assert np.allclose(net.res_bus.va_degree, va_res_80, rtol=0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
