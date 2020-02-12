# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
from pandapower.auxiliary import _check_connectivity, _add_ppc_options
from pandapower.pd2ppc import _pd2ppc
from pandapower.test.consistency_checks import rundcpp_with_consistency_checks
from pandapower.test.loadflow.result_test_network_generator import result_test_network_generator_dcpp
from pandapower.test.toolbox import add_grid_connection, create_test_line, assert_net_equal


def test_rundcpp_init():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b3 = pp.create_bus(net, vn_kv=0.4)
    tidx = pp.create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.25 MVA 20/0.4 kV")
    pp.rundcpp(net)


def test_rundcpp_init_auxiliary_buses():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=20.)
    b4 = pp.create_bus(net, vn_kv=10.)
    tidx = pp.create_transformer3w(net, b2, b3, b4, std_type='63/25/38 MVA 110/20/10 kV')
    pp.create_load(net, b3, p_mw=5)
    pp.create_load(net, b4, p_mw=5)
    pp.create_xward(net, b4, 1, 1, 1, 1, 0.1, 0.1, 1.0)
    net.trafo3w.shift_lv_degree.at[tidx] = 120
    net.trafo3w.shift_mv_degree.at[tidx] = 80
    pp.rundcpp(net)
    va = net.res_bus.va_degree.at[b2]
    pp.rundcpp(net)
    assert np.allclose(va - net.trafo3w.shift_mv_degree.at[tidx], net.res_bus.va_degree.at[b3],
                       atol=2)
    assert np.allclose(va - net.trafo3w.shift_lv_degree.at[tidx], net.res_bus.va_degree.at[b4],
                       atol=2)


# ToDo: Bugs because of float datatypes -> Check Travis on linux machines...
def test_result_iter():
    for net in result_test_network_generator_dcpp():
        try:
            rundcpp_with_consistency_checks(net)
        except (AssertionError):
            raise UserWarning("Consistency Error after adding %s" % net.last_added_case)
        except(pp.LoadflowNotConverged):
            raise UserWarning("Power flow did not converge after adding %s" % net.last_added_case)


def test_two_open_switches():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b3 = pp.create_bus(net, vn_kv=20.)
    l2 = create_test_line(net, b2, b3)
    create_test_line(net, b3, b1)
    pp.create_switch(net, b2, l2, et="l", closed=False)
    pp.create_switch(net, b3, l2, et="l", closed=False)
    pp.rundcpp(net)
    assert np.isnan(net.res_line.i_ka.at[l2]) or net.res_line.i_ka.at[l2] == 0


def get_isolated(net):
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=False,
                     trafo_model="t", check_connectivity=False,
                     mode="pf", r_switch=0.0, init="flat",
                     enforce_q_lims=False, recycle=None)
    ppc, ppci = _pd2ppc(net)
    return _check_connectivity(ppc)


def test_test_sn_mva():
    test_net_gen1 = result_test_network_generator_dcpp(sn_mva=1)
    test_net_gen2 = result_test_network_generator_dcpp(sn_mva=2)
    for net1, net2 in zip(test_net_gen1, test_net_gen2):
        pp.rundcpp(net1)
        pp.rundcpp(net2)
        try:
            assert_net_equal(net1, net2)
        except:
            raise UserWarning("Result difference due to sn_mva after adding %s" % net1.last_added_case)


def test_single_bus_network():
    net = pp.create_empty_network()
    b = pp.create_bus(net, vn_kv=20.)
    pp.create_ext_grid(net, b)

    pp.runpp(net)
    assert net.converged

    pp.rundcpp(net)
    assert net.converged


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
