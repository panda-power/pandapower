# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
from numpy import allclose, isclose

import pandapower as pp


def runpp_with_consistency_checks(net, **kwargs):
    pp.runpp(net, **kwargs)
    consistency_checks(net)
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
    for element in ["bus", "load", "ext_grid", "sgen", "trafo", "trafo3w", "line", "shunt",
                    "ward", "xward", "impedance", "gen", "dcline", "storage"]:
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
