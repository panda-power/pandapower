# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np

from pandapower.pypower.idx_bus import PD, QD
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci
from pandapower.pf.pfsoln_numba import pfsoln
from pandapower.results import _copy_results_ppci_to_ppc, _extract_results_se, reset_results
from pandapower.auxiliary import _add_pf_options, get_values, _clean_up

def _calc_power_flow(ppci, V):
    # store results for all elements
    # calculate branch results (in ppc_i)
    baseMVA, bus, gen, branch, ref, pv, pq, _, _, _, ref_gens = _get_pf_variables_from_ppci(ppci)
    Ybus, Yf, Yt = ppci['internal']['Ybus'], ppci['internal']['Yf'], ppci['internal']['Yt']
    ppci['bus'], ppci['gen'], ppci['branch'] = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, ref_gens)
    
    # calculate bus power injections
    Sbus = np.multiply(V, np.conj(Ybus * V))
    ppci["bus"][:, PD] = -Sbus.real  # saved in per unit, injection -> demand
    ppci["bus"][:, QD] = -Sbus.imag  # saved in per unit, injection -> demand
    return ppci


def _extract_result_ppci_to_pp(net, ppc, ppci):
    # convert to pandapower indices
    ppc = _copy_results_ppci_to_ppc(ppci, ppc, mode="se")

    # extract results from ppc
    try:
        _add_pf_options(net, tolerance_mva=1e-8, trafo_loading="current",
                        numba=True, ac=True, algorithm='nr', max_iteration="auto")
    except:
        pass
    # writes res_bus.vm_pu / va_degree and res_line
    _extract_results_se(net, ppc)

    # restore backup of previous results
    _rename_results(net)

    # additionally, write bus power demand results (these are not written in _extract_results)
    mapping_table = net["_pd2ppc_lookups"]["bus"]
    net.res_bus_est.index = net.bus.index
    net.res_bus_est.p_mw = get_values(ppc["bus"][:, 2], net.bus.index.values,
                                      mapping_table)
    net.res_bus_est.q_mvar = get_values(ppc["bus"][:, 3], net.bus.index.values,
                                        mapping_table)

    _clean_up(net)
    # delete results which are not correctly calculated
    for k in list(net.keys()):
        if k.startswith("res_") and k.endswith("_est") and \
                k not in ("res_bus_est", "res_line_est", "res_trafo_est", "res_trafo3w_est"):
            del net[k]
    return net


def _copy_power_flow_results(net):
    """
    copy old power flow results (if they exist) into res_*_power_flow tables for backup
    :param net: pandapower grid
    :return:
    """
    elements_to_init = ["bus", "ext_grid", "line", "load", "sgen", "trafo", "trafo3w",
                        "shunt", "impedance", "gen", "ward", "xward", "dcline"]
    for element in elements_to_init:
        res_name = "res_" + element
        res_name_pf = res_name + "_power_flow"
        if res_name in net:
            net[res_name_pf] = (net[res_name]).copy()
    reset_results(net)


def _rename_results(net):
    """
    write result tables to result tables for estimation (e.g., res_bus -> res_bus_est)
    reset backed up result tables (e.g., res_bus_power_flow -> res_bus)
    :param net: pandapower grid
    :return:
    """
    elements_to_init = ["bus", "ext_grid", "line", "load", "sgen", "trafo", "trafo3w",
                        "shunt", "impedance", "gen", "ward", "xward", "dcline"]
    # rename res_* tables to res_*_est and then res_*_power_flow to res_*
    for element in elements_to_init:
        res_name = "res_" + element
        res_name_pf = res_name + "_power_flow"
        res_name_est = res_name + "_est"
        net[res_name_est] = net[res_name]
        if res_name_pf in net:
            net[res_name] = net[res_name_pf]
        else:
            del net[res_name]
            
def eppci2pp(net, ppc, eppci):
    # calculate the branch power flow and bus power injection based on the estimated voltage vector
    eppci = _calc_power_flow(eppci, eppci.V)

    # extract the result from ppci to ppc and pandpower network
    net = _extract_result_ppci_to_pp(net, ppc, eppci)
    return net

