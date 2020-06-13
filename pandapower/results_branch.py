# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd
from pandapower.auxiliary import _sum_by_group, I_from_SV_elementwise, sequence_to_phase, S_from_VI_elementwise
from pandapower.auxiliary import _sum_by_group
from pandapower.pypower.idx_brch import F_BUS, T_BUS, PF, QF, PT, QT, BR_R
from pandapower.pypower.idx_bus import BASE_KV, VM, VA



def _get_branch_results(net, ppc, bus_lookup_aranged, pq_buses, suffix=None):
    """
    Extract the branch results and writes it in the Dataframe net.res_line,  net.res_trafo, net.res_trafo3w...

    INPUT:

        **net** - pandapower net

        **ppc** (np.array) - ppc structure

        **bus_lookup_aranged** - consecutive aranged bus lookup

        **pq_buses** - the PQ type buses in ppc

    OPTIONAL:

        **suffix** (str, None) - a suffix like "_sc" to write to res_line_sc for example (short circuit)

    """

    i_ft, s_ft = _get_branch_flows(ppc)

    _get_line_results(net, ppc, i_ft, suffix=suffix)
    _get_trafo_results(net, ppc, s_ft, i_ft, suffix=suffix)
    _get_trafo3w_results(net, ppc, s_ft, i_ft, suffix=suffix)
    _get_impedance_results(net, ppc, i_ft, suffix=suffix)
    _get_xward_branch_results(net, ppc, bus_lookup_aranged, pq_buses, suffix=suffix)
    _get_switch_results(net, i_ft, suffix=suffix)


def _get_branch_results_3ph(net, ppc0, ppc1, ppc2, bus_lookup_aranged, pq_buses):
    """
    Extract the bus results and writes it in the Dataframe net.res_line and net.res_trafo.

    INPUT:

        **results** - the result of runpf loadflow calculation

        **p** - the dict to dump the "res_line" and "res_trafo" Dataframe

    """
    I012_f, S012_f, V012_f, I012_t, S012_t, V012_t = _get_branch_flows_3ph(ppc0, ppc1, ppc2)
    _get_line_results_3ph(net, ppc0, ppc1, ppc2, I012_f, V012_f, I012_t, V012_t)
    _get_trafo_results_3ph(net, ppc0, ppc1, ppc2, I012_f, V012_f, I012_t, V012_t)
    # _get_trafo3w_results(net, ppc, s_ft, i_ft)
    # _get_impedance_results(net, ppc, i_ft)
    # _get_xward_branch_results(net, ppc, bus_lookup_aranged, pq_buses)
    # _get_switch_results(net, i_ft)

																			 
def _get_branch_flows(ppc):
    br_idx = ppc["branch"][:, (F_BUS, T_BUS)].real.astype(int)
    vm_ft = ppc["bus"][br_idx, VM] * ppc["bus"][br_idx, BASE_KV]
    s_ft = np.sqrt(ppc["branch"][:, (PF, PT)].real ** 2 +
                   ppc["branch"][:, (QF, QT)].real ** 2)
    i_ft = s_ft / vm_ft / np.sqrt(3)
    return i_ft, s_ft


def _get_branch_flows_3ph(ppc0, ppc1, ppc2):
    br_from_idx = ppc1["branch"][:, F_BUS].real.astype(int)
    br_to_idx = ppc1["branch"][:, T_BUS].real.astype(int)
    V012_f = np.array([(ppc["bus"][br_from_idx, VM] * ppc["bus"][br_from_idx, BASE_KV] *
                         np.exp(1j * np.deg2rad(ppc["bus"][br_from_idx, VA]))).flatten() for ppc in [ppc0, ppc1, ppc2]])
    V012_t = np.array([(ppc["bus"][br_to_idx, VM] * ppc["bus"][br_to_idx, BASE_KV] *
                         np.exp(1j * np.deg2rad(ppc["bus"][br_to_idx, VA]))).flatten() for ppc in [ppc0, ppc1, ppc2]])
    S012_f = np.array([((ppc["branch"][:, PF].real +
                    1j * ppc["branch"][:, QF].real) )
                    for ppc in [ppc0, ppc1, ppc2]])
    S012_t = np.array([((ppc["branch"][:, PT].real +
                    1j * ppc["branch"][:, QT].real) )
                    for ppc in [ppc0, ppc1, ppc2]])
    I012_f = I_from_SV_elementwise(S012_f , V012_f / np.sqrt(3))
    I012_t = I_from_SV_elementwise(S012_t , V012_t / np.sqrt(3))

    return I012_f, S012_f, V012_f, I012_t, S012_t, V012_t


def _get_line_results(net, ppc, i_ft, suffix=None):
    # create res_line_vals which are written to the pandas dataframe
    if "line" not in net._pd2ppc_lookups["branch"]:
        return
    ac = net["_options"]["ac"]

    f, t = net._pd2ppc_lookups["branch"]["line"]
    pf_mw = ppc["branch"][f:t, PF].real
    q_from_mvar = ppc["branch"][f:t, QF].real
    p_from_mw = pf_mw

    pt_mw = ppc["branch"][f:t, PT].real
    q_to_mvar = ppc["branch"][f:t, QT].real
    p_to_mw = pt_mw

    if ac:
        pl_mw = pf_mw + pt_mw
        ql_mvar = q_from_mvar + q_to_mvar
    else:
        pl_mw = np.zeros_like(pf_mw)
        ql_mvar = np.zeros_like(q_from_mvar)

    with np.errstate(invalid='ignore'):
        i_ka = np.max(i_ft[f:t], axis=1)
    i_from_ka = i_ft[f:t][:, 0]
    i_to_ka = i_ft[f:t][:, 1]
    line_df = net["line"]
    i_max = line_df["max_i_ka"].values * line_df["df"].values * line_df["parallel"].values
    from_bus = ppc["branch"][f:t, F_BUS].real.astype(int)
    to_bus = ppc["branch"][f:t, T_BUS].real.astype(int)

    # write to line
    res_line_df = net["res_line"] if suffix is None else net["res_line%s"%suffix]

    res_line_df["p_from_mw"].values[:] = p_from_mw
    res_line_df["q_from_mvar"].values[:] = q_from_mvar
    res_line_df["p_to_mw"].values[:] = p_to_mw
    res_line_df["q_to_mvar"].values[:] = q_to_mvar
    res_line_df["pl_mw"].values[:] = pl_mw
    res_line_df["ql_mvar"].values[:] = ql_mvar
    res_line_df["i_from_ka"].values[:] = i_from_ka
    res_line_df["i_to_ka"].values[:] = i_to_ka
    res_line_df["i_ka"].values[:] = i_ka
    res_line_df["vm_from_pu"].values[:] = ppc["bus"][from_bus, VM]
    res_line_df["va_from_degree"].values[:] = ppc["bus"][from_bus, VA]
    res_line_df["vm_to_pu"].values[:] = ppc["bus"][to_bus, VM]
    res_line_df["va_to_degree"].values[:] = ppc["bus"][to_bus, VA]
    res_line_df["loading_percent"].values[:] = i_ka / i_max * 100

    # if consider_line_temperature, add resulting r_ohm_per_km to net.res_line
    if net["_options"]["consider_line_temperature"]:
        base_kv = ppc["bus"][from_bus, BASE_KV]
        baseR = np.square(base_kv) / net.sn_mva
        length_km = line_df.length_km.values
        parallel = line_df.parallel.values
        res_line_df["r_ohm_per_km"] = ppc["branch"][f:t, BR_R].real / length_km * baseR * parallel


def _get_line_results_3ph(net, ppc0, ppc1, ppc2, I012_f, V012_f, I012_t, V012_t):
    # create res_line_vals which are written to the pandas dataframe
    ac = net["_options"]["ac"]

    if not "line" in net._pd2ppc_lookups["branch"]:
        return

    f, t = net._pd2ppc_lookups["branch"]["line"]
    I012_from_ka = I012_f[:, f:t]
    I012_to_ka = I012_t[:, f:t]
    line_df = net["line"]
    i_max_phase = line_df["max_i_ka"].values * line_df["df"].values * line_df["parallel"].values

    Vabc_f, Vabc_t, Iabc_f, Iabc_t = [sequence_to_phase(X012) for X012 in
                                      [V012_f[:, f:t], V012_t[:, f:t], I012_f[:, f:t], I012_t[:, f:t]]]
    Sabc_f, Sabc_t = [S_from_VI_elementwise(*Xabc_tup) / np.sqrt(3) for Xabc_tup in
                      [(Vabc_f, Iabc_f), (Vabc_t, Iabc_t)]]
    # Todo: Check why the sqrt(3) is necessary in the previous line as opposed to _get_line_results()
    Pabcf_mw = Sabc_f.real
    Qabcf_mvar = Sabc_f.imag
    Pabct_mw = Sabc_t.real
    Qabct_mvar = Sabc_t.imag
    if ac:
        Pabcl_mw = Pabcf_mw + Pabct_mw
        Qabcl_mvar = Qabcf_mvar + Qabct_mvar
    else:
        Pabcl_mw = np.zeros_like(Pabcf_mw)
        Qabcl_mvar = np.zeros_like(Qabct_mvar)
    
    #geting complex values of the sequence current    
    Iabc_f_ka_complex = sequence_to_phase(I012_from_ka)
    Iabc_t_ka_complex = sequence_to_phase(I012_to_ka)
    
    Iabc_f_ka = np.abs(Iabc_f_ka_complex)
    Iabc_t_ka = np.abs(Iabc_t_ka_complex)
    Iabc_ka = np.maximum.reduce([Iabc_t_ka, Iabc_f_ka])
    
    In_f_ka_complex = Iabc_f_ka_complex.sum(axis=0)
    In_f_ka = np.abs(In_f_ka_complex)
    In_f_ia_n_degree = np.angle(In_f_ka_complex).flatten()*180/np.pi
    In_t_ka_complex = Iabc_t_ka_complex.sum(axis=0)
    In_t_ia_n_degree = np.angle(In_t_ka_complex).flatten()*180/np.pi
    In_t_ka = np.abs(In_t_ka_complex)
    In_ka = np.maximum.reduce([In_t_ka, In_f_ka])

    # write to line
    net["res_line_3ph"]["p_a_from_mw"] = Pabcf_mw[0, :].flatten()
    net["res_line_3ph"]["p_b_from_mw"] = Pabcf_mw[1, :].flatten()
    net["res_line_3ph"]["p_c_from_mw"] = Pabcf_mw[2, :].flatten()
    net["res_line_3ph"]["q_a_from_mvar"] = Qabcf_mvar[0, :].flatten()
    net["res_line_3ph"]["q_b_from_mvar"] = Qabcf_mvar[1, :].flatten()
    net["res_line_3ph"]["q_c_from_mvar"] = Qabcf_mvar[2, :].flatten()
    net["res_line_3ph"]["p_a_to_mw"] = Pabct_mw[0, :].flatten()
    net["res_line_3ph"]["p_b_to_mw"] = Pabct_mw[1, :].flatten()
    net["res_line_3ph"]["p_c_to_mw"] = Pabct_mw[2, :].flatten()
    net["res_line_3ph"]["q_a_to_mvar"] = Qabct_mvar[0, :].flatten()
    net["res_line_3ph"]["q_b_to_mvar"] = Qabct_mvar[1, :].flatten()
    net["res_line_3ph"]["q_c_to_mvar"] = Qabct_mvar[2, :].flatten()
    net["res_line_3ph"]["p_a_l_mw"] = Pabcl_mw[0, :].flatten()
    net["res_line_3ph"]["p_b_l_mw"] = Pabcl_mw[1, :].flatten()
    net["res_line_3ph"]["p_c_l_mw"] = Pabcl_mw[2, :].flatten()
    net["res_line_3ph"]["q_a_l_mvar"] = Qabcl_mvar[0, :].flatten()
    net["res_line_3ph"]["q_b_l_mvar"] = Qabcl_mvar[1, :].flatten()
    net["res_line_3ph"]["q_c_l_mvar"] = Qabcl_mvar[2, :].flatten()
    net["res_line_3ph"]["i_a_from_ka"] = Iabc_f_ka[0, :].flatten()
    net["res_line_3ph"]["i_b_from_ka"] = Iabc_f_ka[1, :].flatten()
    net["res_line_3ph"]["i_c_from_ka"] = Iabc_f_ka[2, :].flatten()
    net["res_line_3ph"]["i_a_to_ka"] = Iabc_t_ka[0, :].flatten()
    net["res_line_3ph"]["i_b_to_ka"] = Iabc_t_ka[1, :].flatten()
    net["res_line_3ph"]["i_c_to_ka"] = Iabc_t_ka[2, :].flatten()
    net["res_line_3ph"]["i_a_ka"] = Iabc_ka[0, :]
    net["res_line_3ph"]["i_b_ka"] = Iabc_ka[1, :]
    net["res_line_3ph"]["i_c_ka"] = Iabc_ka[2, :]
    net["res_line_3ph"]["i_n_from_ka"] = In_f_ka
    net["res_line_3ph"]["i_n_to_ka"] = In_t_ka
    net["res_line_3ph"]["i_n_ka"] = In_ka
    net["res_line_3ph"]["loading_percentA"] = Iabc_ka[0, :] / i_max_phase * 100
    net["res_line_3ph"]["loading_percentB"] = Iabc_ka[1, :] / i_max_phase * 100
    net["res_line_3ph"]["loading_percentC"] = Iabc_ka[2, :] / i_max_phase * 100
    net["res_line_3ph"]["loading_percent"] = Iabc_ka.max(axis=0) / i_max_phase * 100
    net["res_line_3ph"].index = net["line"].index


def _get_trafo_results(net, ppc, s_ft, i_ft, suffix=None):
    if "trafo" not in net._pd2ppc_lookups["branch"]:
        return
    ac = net["_options"]["ac"]
    trafo_loading = net["_options"]["trafo_loading"]
    f, t = net._pd2ppc_lookups["branch"]["trafo"]
    p_hv_mw = ppc["branch"][f:t, PF].real
    p_lv_mw = ppc["branch"][f:t, PT].real

    if ac:
        q_hv_mvar = ppc["branch"][f:t, QF].real
        q_lv_mvar = ppc["branch"][f:t, QT].real
        pl_mw = p_hv_mw + p_lv_mw
        ql_mvar = q_hv_mvar + q_lv_mvar
    else:
        q_hv_mvar = np.zeros_like(p_hv_mw)
        q_lv_mvar = np.zeros_like(p_lv_mw)
        pl_mw = np.zeros_like(p_lv_mw)
        ql_mvar = np.zeros_like(p_lv_mw)

    i_hv_ka = i_ft[:, 0][f:t]
    i_lv_ka = i_ft[:, 1][f:t]
    if trafo_loading == "current":
        # calculate loading with rated current
        trafo_df = net["trafo"]
        vns = np.vstack([trafo_df["vn_hv_kv"].values, trafo_df["vn_lv_kv"].values]).T
        lds_trafo = i_ft[f:t] * vns * np.sqrt(3) / trafo_df["sn_mva"].values[:, np.newaxis] * 100.
        with np.errstate(invalid='ignore'):
            ld_trafo = np.max(lds_trafo, axis=1)
    elif trafo_loading == "power":
        # calculate loading with rated loading
        ld_trafo = np.max(s_ft[f:t] / net["trafo"]["sn_mva"].values[:, np.newaxis] * 100., axis=1)
    else:
        raise ValueError(
            "Unknown transformer loading parameter %s - choose 'current' or 'power'" % trafo_loading)
    if any(net["trafo"]["df"].values <= 0):
        raise UserWarning('Transformer rating factor df must be positive. Transformers with false '
                          'rating factors: %s' % net["trafo"].query('df<=0').index.tolist())

    loading_percent = ld_trafo / net["trafo"]["parallel"].values / net["trafo"]["df"].values

    hv_buses = ppc["branch"][f:t, F_BUS].real.astype(int)
    lv_buses = ppc["branch"][f:t, T_BUS].real.astype(int)

    # write results to trafo dataframe
    res_trafo_df = net["res_trafo"] if suffix is None else net["res_trafo%s" % suffix]

    res_trafo_df["p_hv_mw"].values[:] = p_hv_mw
    res_trafo_df["q_hv_mvar"].values[:] = q_hv_mvar
    res_trafo_df["p_lv_mw"].values[:] = p_lv_mw
    res_trafo_df["q_lv_mvar"].values[:] = q_lv_mvar
    res_trafo_df["pl_mw"].values[:] = pl_mw
    res_trafo_df["ql_mvar"].values[:] = ql_mvar
    res_trafo_df["i_hv_ka"].values[:] = i_hv_ka
    res_trafo_df["i_lv_ka"].values[:] = i_lv_ka
    res_trafo_df["vm_hv_pu"].values[:] = ppc["bus"][hv_buses, VM]
    res_trafo_df["va_hv_degree"].values[:] = ppc["bus"][hv_buses, VA]
    res_trafo_df["vm_lv_pu"].values[:] = ppc["bus"][lv_buses, VM]
    res_trafo_df["va_lv_degree"].values[:] = ppc["bus"][lv_buses, VA]
    res_trafo_df["loading_percent"].values[:] = loading_percent



def _get_trafo_results_3ph(net, ppc0, ppc1, ppc2, I012_f, V012_f, I012_t, V012_t):
    ac = net["_options"]["ac"]
    trafo_loading = net["_options"]["trafo_loading"]

    if not "trafo" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["trafo"]
    I012_hv_ka = I012_f[:, f:t]
    I012_lv_ka = I012_t[:, f:t]
    trafo_df = net["trafo"]

    Vabc_hv, Vabc_lv, Iabc_hv, Iabc_lv = [sequence_to_phase(X012) for X012 in
                                      [V012_f[:, f:t], V012_t[:, f:t], I012_f[:, f:t], I012_t[:, f:t]]]
    Sabc_hv, Sabc_lv = [S_from_VI_elementwise(*Xabc_tup) / np.sqrt(3) for Xabc_tup in
                      [(Vabc_hv, Iabc_hv), (Vabc_lv, Iabc_lv)]]
    # Todo: Check why the sqrt(3) is necessary in the previous line as opposed to _get_line_results()
    Pabc_hv_mw = Sabc_hv.real
    Qabc_hv_mvar = Sabc_hv.imag
    Pabc_lv_mw = Sabc_lv.real
    Qabc_lv_mvar = Sabc_lv.imag
    if ac:
        Pabcl_mw = Pabc_hv_mw + Pabc_lv_mw
        Qabcl_mvar = Qabc_hv_mvar + Qabc_lv_mvar
    else:
        Pabcl_mw = np.zeros_like(Pabc_hv_mw)
        Qabcl_mvar = np.zeros_like(Qabc_lv_mvar)
        
    Iabc_hv_ka = np.abs(sequence_to_phase(I012_hv_ka))
    Iabc_lv_ka = np.abs(sequence_to_phase(I012_lv_ka))
    
    #geting complex values of the sequence current    
#    Iabc_hv_ka_complex = sequence_to_phase(I012_hv_ka)
#    Iabc_lv_ka_complex = sequence_to_phase(I012_lv_ka)
#    
#    Iabc_hv_ka = np.abs(Iabc_hv_ka_complex)
#    Iabc_lv_ka = np.abs(Iabc_lv_ka_complex)
#   
#    In_hv_ka_complex = Iabc_hv_ka_complex.sum(axis=0)
#    In_hv_ka = np.abs(In_hv_ka_complex)
#    In_hv_ia_n_degree = np.angle(In_hv_ka_complex).flatten()*180/np.pi
#    In_lv_ka_complex = Iabc_lv_ka_complex.sum(axis=0)
#    In_lv_ka = np.abs(In_lv_ka_complex)
#    In_lv_ia_n_degree = np.angle(In_lv_ka_complex).flatten()*180/np.pi


    if trafo_loading == "current":
        trafo_df = net["trafo"]
        vns = np.vstack([trafo_df["vn_hv_kv"].values, trafo_df["vn_lv_kv"].values]).T
        ld_trafo = np.maximum.reduce([np.asarray(Iabc_hv_ka) * vns[:, 0], np.asarray(Iabc_lv_ka) * vns[:, 1]])
        ld_trafo = ld_trafo * np.sqrt(3) / trafo_df["sn_mva"].values * 100.
    elif trafo_loading == "power":
        ld_trafo = np.maximum.reduce([np.abs(Sabc_hv), np.abs(Sabc_lv)])
        ld_trafo = ld_trafo / net["trafo"]["sn_mva"].values[:, np.newaxis] * 3. * 100.
    else:
        raise ValueError(
            "Unknown transformer loading parameter %s - choose 'current' or 'power'" % trafo_loading)
    if any(net["trafo"]["df"].values <= 0):
        raise UserWarning('Transformer rating factor df must be positive. Transformers with false '
                          'rating factors: %s' % net["trafo"].query('df<=0').index.tolist())
    loading_percent = ld_trafo / net["trafo"]["parallel"].values / net["trafo"]["df"].values

    # write results to trafo dataframe
    res_trafo_df = net["res_trafo_3ph"]
    res_trafo_df["p_a_hv_mw"] = Pabc_hv_mw[0, :].flatten()
    res_trafo_df["p_b_hv_mw"] = Pabc_hv_mw[1, :].flatten()
    res_trafo_df["p_c_hv_mw"] = Pabc_hv_mw[2, :].flatten()
    res_trafo_df["q_a_hv_mvar"] = Qabc_hv_mvar[0, :].flatten()
    res_trafo_df["q_b_hv_mvar"] = Qabc_hv_mvar[1, :].flatten()
    res_trafo_df["q_c_hv_mvar"] = Qabc_hv_mvar[2, :].flatten()
    res_trafo_df["p_a_lv_mw"] = Pabc_lv_mw[0, :].flatten()
    res_trafo_df["p_b_lv_mw"] = Pabc_lv_mw[1, :].flatten()
    res_trafo_df["p_c_lv_mw"] = Pabc_lv_mw[2, :].flatten()
    res_trafo_df["q_a_lv_mvar"] = Qabc_lv_mvar[0, :].flatten()
    res_trafo_df["q_b_lv_mvar"] = Qabc_lv_mvar[1, :].flatten()
    res_trafo_df["q_c_lv_mvar"] = Qabc_lv_mvar[2, :].flatten()
    res_trafo_df["p_a_l_mw"] = Pabcl_mw[0, :].flatten()
    res_trafo_df["p_b_l_mw"] = Pabcl_mw[1, :].flatten()
    res_trafo_df["p_c_l_mw"] = Pabcl_mw[2, :].flatten()
    res_trafo_df["q_a_l_mvar"] = Qabcl_mvar[0, :].flatten()
    res_trafo_df["q_b_l_mvar"] = Qabcl_mvar[1, :].flatten()
    res_trafo_df["q_c_l_mvar"] = Qabcl_mvar[2, :].flatten()
    res_trafo_df["i_a_hv_ka"] = Iabc_hv_ka[0, :].flatten()
    res_trafo_df["i_b_hv_ka"] = Iabc_hv_ka[1, :].flatten()
    res_trafo_df["i_c_hv_ka"] = Iabc_hv_ka[2, :].flatten()
#    res_trafo_df["i_n_hv_ka"] = In_hv_ka.flatten()
    res_trafo_df["i_a_lv_ka"] = Iabc_lv_ka[0, :].flatten()
    res_trafo_df["i_b_lv_ka"] = Iabc_lv_ka[1, :].flatten()
    res_trafo_df["i_c_lv_ka"] = Iabc_lv_ka[2, :].flatten()
#    res_trafo_df["i_n_lv_ka"] = In_lv_ka.flatten()
    res_trafo_df["loading_percentA"] = loading_percent[0, :]
    res_trafo_df["loading_percentB"] = loading_percent[1, :]
    res_trafo_df["loading_percentC"] = loading_percent[2, :]
    res_trafo_df["loading_percent"] = loading_percent.max(axis=0)
    res_trafo_df.index = net["trafo"].index.values



def _get_trafo3w_lookups(net):
    f, t = net._pd2ppc_lookups["branch"]["trafo3w"]
    hv = int(f + (t - f) / 3)
    mv = int(f + 2 * (t - f) / 3)
    lv = t
    return f, hv, mv, lv


def _get_trafo3w_results(net, ppc, s_ft, i_ft, suffix=None):
    if "trafo3w" not in net._pd2ppc_lookups["branch"]:
        return

    trafo_loading = net["_options"]["trafo_loading"]
    ac = net["_options"]["ac"]
    f, hv, mv, lv = _get_trafo3w_lookups(net)

    phv_mw = ppc["branch"][f:hv, PF].real
    pmv_mw = ppc["branch"][hv:mv, PT].real
    plv_mw = ppc["branch"][mv:lv, PT].real

    p_hv_mw = phv_mw
    p_mv_mw = pmv_mw
    p_lv_mw = plv_mw

    if ac:
        q_hv_mvar = ppc["branch"][f:hv, QF].real
        q_mv_mvar = ppc["branch"][hv:mv, QT].real
        q_lv_mvar = ppc["branch"][mv:lv, QT].real
        pl_mw = phv_mw + pmv_mw + plv_mw
        ql_mvar = q_hv_mvar + q_mv_mvar + q_lv_mvar
    else:
        zeros = np.zeros_like(phv_mw)
        q_hv_mvar = zeros
        q_mv_mvar = zeros
        q_lv_mvar = zeros
        pl_mw = zeros
        ql_mvar = zeros

    i_h = i_ft[:, 0][f:hv]
    i_m = i_ft[:, 1][hv:mv]
    i_l = i_ft[:, 1][mv:lv]

    t3 = net["trafo3w"]
    if trafo_loading == "current":
        ld_h = i_h * t3["vn_hv_kv"].values * np.sqrt(3) / t3["sn_hv_mva"].values * 100
        ld_m = i_m * t3["vn_mv_kv"].values * np.sqrt(3) / t3["sn_mv_mva"].values * 100
        ld_l = i_l * t3["vn_lv_kv"].values * np.sqrt(3) / t3["sn_lv_mva"].values * 100
        with np.errstate(invalid='ignore'):
            ld_trafo = np.max(np.vstack([ld_h, ld_m, ld_l]), axis=0)
    elif trafo_loading == "power":
        ld_h = s_ft[:, 0][f:hv] / t3["sn_hv_mva"].values * 100.
        ld_m = s_ft[:, 1][hv:mv] / t3["sn_mv_mva"].values * 100.
        ld_l = s_ft[:, 1][mv:lv] / t3["sn_lv_mva"].values * 100.
        ld_trafo = np.max(np.vstack([ld_h, ld_m, ld_l]), axis=0)
    else:
        raise ValueError(
            "Unknown transformer loading parameter %s - choose 'current' or 'power'" % trafo_loading)
    loading_percent = ld_trafo

    hv_buses = ppc["branch"][f:hv, F_BUS].real.astype(int)
    aux_buses = ppc["branch"][f:hv, T_BUS].real.astype(int)
    mv_buses = ppc["branch"][hv:mv, T_BUS].real.astype(int)
    lv_buses = ppc["branch"][mv:lv, T_BUS].real.astype(int)

    # write results to trafo3w dataframe
    res_trafo3w_df = net["res_trafo3w"] if suffix is None else net["res_trafo3w%s"%suffix]

    res_trafo3w_df["p_hv_mw"].values[:] = p_hv_mw
    res_trafo3w_df["q_hv_mvar"].values[:] = q_hv_mvar
    res_trafo3w_df["p_mv_mw"].values[:] = p_mv_mw
    res_trafo3w_df["q_mv_mvar"].values[:] = q_mv_mvar
    res_trafo3w_df["p_lv_mw"].values[:] = p_lv_mw
    res_trafo3w_df["q_lv_mvar"].values[:] = q_lv_mvar
    res_trafo3w_df["pl_mw"].values[:] = pl_mw
    res_trafo3w_df["ql_mvar"].values[:] = ql_mvar
    res_trafo3w_df["i_hv_ka"].values[:] = i_h
    res_trafo3w_df["i_mv_ka"].values[:] = i_m
    res_trafo3w_df["i_lv_ka"].values[:] = i_l
    res_trafo3w_df["vm_hv_pu"].values[:] = ppc["bus"][hv_buses, VM]
    res_trafo3w_df["va_hv_degree"].values[:] = ppc["bus"][hv_buses, VA]
    res_trafo3w_df["vm_mv_pu"].values[:] = ppc["bus"][mv_buses, VM]
    res_trafo3w_df["va_mv_degree"].values[:] = ppc["bus"][mv_buses, VA]
    res_trafo3w_df["vm_lv_pu"].values[:] = ppc["bus"][lv_buses, VM]
    res_trafo3w_df["va_lv_degree"].values[:] = ppc["bus"][lv_buses, VA]
    res_trafo3w_df["va_internal_degree"].values[:] = ppc["bus"][aux_buses, VA]
    res_trafo3w_df["vm_internal_pu"].values[:] = ppc["bus"][aux_buses, VM]
    res_trafo3w_df["loading_percent"].values[:] = loading_percent



def _get_impedance_results(net, ppc, i_ft, suffix=None):
    ac = net["_options"]["ac"]

    if not "impedance" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["impedance"]
    pf_mw = ppc["branch"][f:t, (PF)].real
    pt_mw = ppc["branch"][f:t, (PT)].real
    p_from_mw = pf_mw
    p_to_mw = pt_mw

    if ac:
        q_from_mvar = ppc["branch"][f:t, (QF)].real
        q_to_mvar = ppc["branch"][f:t, (QT)].real
        ql_mvar = q_from_mvar + q_to_mvar
        pl_mw = pf_mw + pt_mw
    else:
        zeros = np.zeros_like(p_from_mw)
        # this looks like a pyramid
        q_from_mvar = zeros
        q_to_mvar = zeros
        ql_mvar = zeros
        pl_mw = zeros
        # zeros

    i_from_ka = i_ft[f:t][:, 0]
    i_to_ka = i_ft[f:t][:, 1]

    # write to impedance
    res_impediance_df = net["res_impedance"] if suffix is None else net["res_impedance%s"%suffix]

    res_impediance_df["p_from_mw"].values[:] = p_from_mw
    res_impediance_df["q_from_mvar"].values[:] = q_from_mvar
    res_impediance_df["p_to_mw"].values[:] = p_to_mw
    res_impediance_df["q_to_mvar"].values[:] = q_to_mvar
    res_impediance_df["pl_mw"].values[:] = pl_mw
    res_impediance_df["ql_mvar"].values[:] = ql_mvar
    res_impediance_df["i_from_ka"].values[:] = i_from_ka
    res_impediance_df["i_to_ka"].values[:] = i_to_ka



def _get_xward_branch_results(net, ppc, bus_lookup_aranged, pq_buses, suffix=None):
    ac = net["_options"]["ac"]

    if not "xward" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["xward"]
    p_branch_xward = ppc["branch"][f:t, PF].real
    net["res_xward"]["p_mw"].values[:] = net["res_xward"]["p_mw"].values + p_branch_xward
    if ac:
        q_branch_xward = ppc["branch"][f:t, QF].real
        net["res_xward"]["q_mvar"].values[:] = net["res_xward"]["q_mvar"].values + q_branch_xward
    else:
        q_branch_xward = np.zeros(len(p_branch_xward))
    b_pp, p, q = _sum_by_group(net["xward"]["bus"].values, p_branch_xward, q_branch_xward)
    b_ppc = bus_lookup_aranged[b_pp]

    pq_buses[b_ppc, 0] += p
    pq_buses[b_ppc, 1] += q
    aux_buses = net["_pd2ppc_lookups"]["bus"][net["_pd2ppc_lookups"]["aux"]["xward"]]

    res_xward_df = net["res_xward"] if suffix is None else net["res_xward%s" % suffix]

    res_xward_df["va_internal_degree"].values[:] = ppc["bus"][aux_buses, VA]
    res_xward_df["vm_internal_pu"].values[:] = ppc["bus"][aux_buses, VM]
    res_xward_df.index = net["xward"].index



def _get_switch_results(net, i_ft, suffix=None):
    if not "switch" in net._pd2ppc_lookups["branch"]:
        return
    f, t = net._pd2ppc_lookups["branch"]["switch"]
    with np.errstate(invalid='ignore'):
        i_ka = np.max(i_ft[f:t], axis=1)

    res_switch_df = "res_switch" if suffix is None else "res_switch%s" % suffix
    net[res_switch_df] = pd.DataFrame(data=i_ka, columns=["i_ka"],
                                      index=net.switch[net._impedance_bb_switches].index)
