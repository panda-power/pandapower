# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
from numpy import allclose, isclose
from pandapower.pf.runpp_3ph import runpp_3ph
from pandapower.results import get_relevant_elements
import pandapower as pp


def runpp_with_consistency_checks(net, **kwargs):
    pp.runpp(net, **kwargs)
    consistency_checks(net)
    return True

def runpp_3ph_with_consistency_checks(net, **kwargs):
    runpp_3ph(net, **kwargs)
    consistency_checks_3ph(net)
    return True

def rundcpp_with_consistency_checks(net, **kwargs):
    pp.rundcpp(net, **kwargs)
    consistency_checks(net, test_q=False)
    return True

def consistency_checks(net, rtol=1e-3, test_q=True):
    indices_consistent(net)
    branch_loss_consistent_with_bus_feed_in(net, rtol)
    element_power_consistent_with_bus_power(net, rtol, test_q)

def indices_consistent(net):
    elements = get_relevant_elements()
    for element in elements:
        e_idx = net[element].index
        res_idx = net["res_" + element].index
        assert len(e_idx) == len(res_idx), "length of %s bus and res_%s indices do not match"%(element, element)
        assert all(e_idx == res_idx), "%s bus and res_%s indices do not match"%(element, element)


def branch_loss_consistent_with_bus_feed_in(net, atol=1e-2):
    """
    The surpluss of bus feed summed over all buses always has to be equal to the sum of losses in
    all branches.
    """
    # Active Power
    bus_surplus_p = -net.res_bus.p_mw.sum()
    bus_surplus_q = -net.res_bus.q_mvar.sum()

    branch_loss_p = net.res_line.pl_mw.values.sum() + net.res_trafo.pl_mw.values.sum() + \
                    net.res_trafo3w.pl_mw.values.sum() + net.res_impedance.pl_mw.values.sum() + \
                    net.res_dcline.pl_mw.values.sum()
    branch_loss_q = net.res_line.ql_mvar.values.sum() + net.res_trafo.ql_mvar.values.sum() + \
                    net.res_trafo3w.ql_mvar.values.sum() + net.res_impedance.ql_mvar.values.sum() + \
                    net.res_dcline.q_to_mvar.values.sum() + net.res_dcline.q_from_mvar.values.sum()

    try:
        assert isclose(bus_surplus_p, branch_loss_p, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MW, but power generation at the buses exceeds the feedin by %.4f MW"%(branch_loss_p, bus_surplus_p))
    try:
        assert isclose(bus_surplus_q, branch_loss_q, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MVar, but power generation at the buses exceeds the feedin by %.4f MVar"%(branch_loss_q, bus_surplus_q))


def element_power_consistent_with_bus_power(net, rtol=1e-2, test_q=True):
    """
    The bus feed-in at each node has to be equal to the sum of the element feed ins at each node.
    """
    bus_p = pd.Series(data=0., index=net.bus.index)
    bus_q = pd.Series(data=0., index=net.bus.index)

    for idx, tab in net.ext_grid.iterrows():
        if tab.in_service:
            bus_p.at[tab.bus] -= net.res_ext_grid.p_mw.at[idx]
            bus_q.at[tab.bus] -= net.res_ext_grid.q_mvar.at[idx]

    for idx, tab in net.gen.iterrows():
        if tab.in_service:
            bus_p.at[tab.bus] -= net.res_gen.p_mw.at[idx]
            bus_q.at[tab.bus] -= net.res_gen.q_mvar.at[idx]

    for idx, tab in net.load.iterrows():
        bus_p.at[tab.bus] += net.res_load.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_load.q_mvar.at[idx]

    for idx, tab in net.sgen.iterrows():
        bus_p.at[tab.bus] -= net.res_sgen.p_mw.at[idx]
        bus_q.at[tab.bus] -= net.res_sgen.q_mvar.at[idx]

    for idx, tab in net.storage.iterrows():
        bus_p.at[tab.bus] += net.res_storage.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_storage.q_mvar.at[idx]

    for idx, tab in net.shunt.iterrows():
        bus_p.at[tab.bus] += net.res_shunt.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_shunt.q_mvar.at[idx]

    for idx, tab in net.ward.iterrows():
        bus_p.at[tab.bus] += net.res_ward.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_ward.q_mvar.at[idx]

    for idx, tab in net.xward.iterrows():
        bus_p.at[tab.bus] += net.res_xward.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_xward.q_mvar.at[idx]

    assert allclose(net.res_bus.p_mw.values, bus_p.values, equal_nan=True, rtol=rtol)
    if test_q:
        assert allclose(net.res_bus.q_mvar.values, bus_q.values, equal_nan=True, rtol=rtol)


def consistency_checks_3ph(net, rtol=2e-3):
    indices_consistent_3ph(net)
    branch_loss_consistent_with_bus_feed_in_3ph(net, rtol)
    element_power_consistent_with_bus_power_3ph(net, rtol)

def indices_consistent_3ph(net):
    elements = get_relevant_elements("pf_3ph")
    for element in elements:
        e_idx = net[element].index
        res_idx = net["res_" + element+"_3ph"].index
        assert len(e_idx) == len(res_idx), "length of %s bus and res_%s indices do not match"%(element, element)
        assert all(e_idx == res_idx), "%s bus and res_%s indices do not match"%(element, element)


def branch_loss_consistent_with_bus_feed_in_3ph(net, atol=1e-2):
    """
    The surpluss of bus feed summed over all buses always has to be equal to the sum of losses in
    all branches.
    """
    # Active Power
    bus_surplus_p_a = net.res_bus_3ph.p_a_mw.sum()
    bus_surplus_q_a = net.res_bus_3ph.q_a_mvar.sum()
    bus_surplus_p_b = net.res_bus_3ph.p_b_mw.sum()
    bus_surplus_q_b = net.res_bus_3ph.q_b_mvar.sum()
    bus_surplus_p_c = net.res_bus_3ph.p_c_mw.sum()
    bus_surplus_q_c = net.res_bus_3ph.q_c_mvar.sum()

    branch_loss_p_a = net.res_line_3ph.p_a_l_mw.sum() + net.res_trafo_3ph.p_a_l_mw.sum() + \
                    net.res_trafo3w.pl_mw.sum()/3 + net.res_impedance.pl_mw.sum()/3 + \
                    net.res_dcline.pl_mw.sum()/3
    branch_loss_q_a = net.res_line_3ph.q_a_l_mvar.sum() + net.res_trafo_3ph.q_a_l_mvar.sum() + \
                    net.res_trafo3w.ql_mvar.sum()/3 + net.res_impedance.ql_mvar.sum() + \
                    net.res_dcline.q_to_mvar.sum()/3 + net.res_dcline.q_from_mvar.sum()
    branch_loss_p_b = net.res_line_3ph.p_b_l_mw.sum() + net.res_trafo_3ph.p_b_l_mw.sum() + \
                    net.res_trafo3w.pl_mw.sum()/3 + net.res_impedance.pl_mw.sum()/3 + \
                    net.res_dcline.pl_mw.sum()/3
    branch_loss_q_b = net.res_line_3ph.q_b_l_mvar.sum() + net.res_trafo_3ph.q_b_l_mvar.sum() + \
                    net.res_trafo3w.ql_mvar.sum()/3 + net.res_impedance.ql_mvar.sum() + \
                    net.res_dcline.q_to_mvar.sum()/3 + net.res_dcline.q_from_mvar.sum()
    branch_loss_p_c = net.res_line_3ph.p_c_l_mw.sum() + net.res_trafo_3ph.p_c_l_mw.sum() + \
                    net.res_trafo3w.pl_mw.sum()/3 + net.res_impedance.pl_mw.sum()/3 + \
                    net.res_dcline.pl_mw.sum()/3
    branch_loss_q_c = net.res_line_3ph.q_c_l_mvar.sum() + net.res_trafo_3ph.q_c_l_mvar.sum() + \
                    net.res_trafo3w.ql_mvar.sum()/3 + net.res_impedance.ql_mvar.sum() + \
                    net.res_dcline.q_to_mvar.sum()/3 + net.res_dcline.q_from_mvar.sum()

    try:
        assert isclose(bus_surplus_p_a, branch_loss_p_a, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MW, but power generation at the buses exceeds the feedin by %.4f MW"%(branch_loss_p_a, bus_surplus_p_a))
    try:
        assert isclose(bus_surplus_q_a, branch_loss_q_a, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MVar, but power generation at the buses exceeds the feedin by %.4f MVar"%(branch_loss_q_a, bus_surplus_q_a))
    try:
        assert isclose(bus_surplus_p_b, branch_loss_p_b, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MW, but power generation at the buses exceeds the feedin by %.4f MW"%(branch_loss_p_b, bus_surplus_p_b))
    try:
        assert isclose(bus_surplus_q_b, branch_loss_q_b, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MVar, but power generation at the buses exceeds the feedin by %.4f MVar"%(branch_loss_q_b, bus_surplus_q_b))
    try:
        assert isclose(bus_surplus_p_c, branch_loss_p_c, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MW, but power generation at the buses exceeds the feedin by %.4f MW"%(branch_loss_p_c, bus_surplus_p_c))
    try:
        assert isclose(bus_surplus_q_c, branch_loss_q_c, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MVar, but power generation at the buses exceeds the feedin by %.4f MVar"%(branch_loss_q_c, bus_surplus_q_c))


def element_power_consistent_with_bus_power_3ph(net, rtol=1e-2):
    """
    The bus feed-in at each node has to be equal to the sum of the element feed ins at each node.
    """
    bus_p_a = pd.Series(data=0., index=net.bus.index)
    bus_q_a = pd.Series(data=0., index=net.bus.index)
    bus_p_b = pd.Series(data=0., index=net.bus.index)
    bus_q_b = pd.Series(data=0., index=net.bus.index)
    bus_p_c = pd.Series(data=0., index=net.bus.index)
    bus_q_c = pd.Series(data=0., index=net.bus.index)

    for idx, tab in net.ext_grid.iterrows():
        bus_p_a.at[tab.bus] += net.res_ext_grid_3ph.p_a_mw.at[idx]
        bus_q_a.at[tab.bus] += net.res_ext_grid_3ph.q_a_mvar.at[idx]
        bus_p_b.at[tab.bus] += net.res_ext_grid_3ph.p_b_mw.at[idx]
        bus_q_b.at[tab.bus] += net.res_ext_grid_3ph.q_b_mvar.at[idx]
        bus_p_c.at[tab.bus] += net.res_ext_grid_3ph.p_c_mw.at[idx]
        bus_q_c.at[tab.bus] += net.res_ext_grid_3ph.q_c_mvar.at[idx]

    for idx, tab in net.load.iterrows():
        bus_p_a.at[tab.bus] -= net.res_load_3ph.p_mw.at[idx]/3
        bus_q_a.at[tab.bus] -= net.res_load_3ph.q_mvar.at[idx] /3
        bus_p_b.at[tab.bus] -= net.res_load_3ph.p_mw.at[idx]/3
        bus_q_b.at[tab.bus] -= net.res_load_3ph.q_mvar.at[idx] /3
        bus_p_c.at[tab.bus] -= net.res_load_3ph.p_mw.at[idx]/3
        bus_q_c.at[tab.bus] -= net.res_load_3ph.q_mvar.at[idx] /3

    for idx, tab in net.asymmetric_load.iterrows():
        bus_p_a.at[tab.bus] -= net.res_asymmetric_load_3ph.p_a_mw.at[idx]
        bus_q_a.at[tab.bus] -= net.res_asymmetric_load_3ph.q_a_mvar.at[idx]
        bus_p_b.at[tab.bus] -= net.res_asymmetric_load_3ph.p_b_mw.at[idx]
        bus_q_b.at[tab.bus] -= net.res_asymmetric_load_3ph.q_b_mvar.at[idx]
        bus_p_c.at[tab.bus] -= net.res_asymmetric_load_3ph.p_c_mw.at[idx]
        bus_q_c.at[tab.bus] -= net.res_asymmetric_load_3ph.q_c_mvar.at[idx]

    for idx, tab in net.sgen.iterrows():
        bus_p_a.at[tab.bus] += net.res_sgen_3ph.p_mw.at[idx] / 3
        bus_q_a.at[tab.bus] += net.res_sgen_3ph.q_mvar.at[idx] / 3
        bus_p_b.at[tab.bus] += net.res_sgen_3ph.p_mw.at[idx] / 3
        bus_q_b.at[tab.bus] += net.res_sgen_3ph.q_mvar.at[idx] / 3
        bus_p_c.at[tab.bus] += net.res_sgen_3ph.p_mw.at[idx] / 3
        bus_q_c.at[tab.bus] += net.res_sgen_3ph.q_mvar.at[idx] / 3

    assert allclose(net.res_bus_3ph.p_a_mw.values, bus_p_a.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_3ph.q_a_mvar.values, bus_q_a.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_3ph.p_b_mw.values, bus_p_b.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_3ph.q_b_mvar.values, bus_q_b.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_3ph.p_c_mw.values, bus_p_c.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_3ph.q_c_mvar.values, bus_q_c.values, equal_nan=True, rtol=rtol)
    