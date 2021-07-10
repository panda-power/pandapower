# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd

from pandapower.auxiliary import _sum_by_group
from pandapower.pypower.idx_bus import BASE_KV
from pandapower.pypower.idx_gen import GEN_BUS, MBASE
from pandapower.shortcircuit.idx_brch import IKSS_F, IKSS_T, IP_F, IP_T, ITH_F, ITH_T
from pandapower.shortcircuit.idx_bus import C_MIN, C_MAX, KAPPA, R_EQUIV, IKSS1, IP, ITH,\
    X_EQUIV, IKSS2, IKCV, M, R_EQUIV_OHM, X_EQUIV_OHM, V_G, K_SG, SKSS
from pandapower.shortcircuit.impedance import _calc_zbus_diag

from pandapower.pypower.pfsoln import pfsoln as pfsoln_pypower
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci


def _calc_ikss(net, ppci, bus_idx):
    fault = net._options["fault"]
    case = net._options["case"]
    c = ppci["bus"][bus_idx, C_MIN] if case == "min" else ppci["bus"][bus_idx, C_MAX]
    ppci["internal"]["baseI"] = ppci["bus"][:, BASE_KV] * np.sqrt(3) / ppci["baseMVA"]

    # Only for test, should correspondant to PF result
    baseZ = ppci["bus"][bus_idx, BASE_KV] ** 2 / ppci["baseMVA"]
    ppci["bus"][bus_idx, R_EQUIV_OHM] = baseZ * ppci["bus"][bus_idx, R_EQUIV]
    ppci["bus"][bus_idx, X_EQUIV_OHM] = baseZ * ppci["bus"][bus_idx, X_EQUIV]

    z_equiv = abs(ppci["bus"][bus_idx, R_EQUIV] + ppci["bus"][bus_idx, X_EQUIV] * 1j)
    if fault == "3ph":
        ppci["bus"][bus_idx, IKSS1] = c / z_equiv / ppci["bus"][bus_idx, BASE_KV] / np.sqrt(3) * ppci["baseMVA"]
    elif fault == "2ph":
        ppci["bus"][bus_idx, IKSS1] = c / z_equiv / ppci["bus"][bus_idx, BASE_KV] / 2 * ppci["baseMVA"]

    if fault == "3ph":
        ppci["bus"][bus_idx, SKSS] = np.sqrt(3) * ppci["bus"][bus_idx, IKSS1] * ppci["bus"][bus_idx, BASE_KV]
    elif fault == "2ph":
        ppci["bus"][bus_idx, SKSS] = ppci["bus"][bus_idx, IKSS1] * ppci["bus"][bus_idx, BASE_KV] / np.sqrt(3)

    # Correct voltage of generator bus inside power station
    if np.any(~np.isnan(ppci["bus"][:, K_SG])):
        gen_bus_idx = bus_idx[~np.isnan(ppci["bus"][bus_idx, K_SG])]
        ppci["bus"][gen_bus_idx, IKSS1] *=\
            (ppci["bus"][gen_bus_idx, V_G] / ppci["bus"][gen_bus_idx, BASE_KV])
        ppci["bus"][gen_bus_idx, SKSS] *=\
            (ppci["bus"][gen_bus_idx, V_G] / ppci["bus"][gen_bus_idx, BASE_KV])

    _current_source_current(net, ppci)


def _calc_ikss_1ph(net, ppci, ppci_0, bus_idx):
    case = net._options["case"]
    c = ppci["bus"][bus_idx, C_MIN] if case == "min" else ppci["bus"][bus_idx, C_MAX]
    ppci["internal"]["baseI"] = ppci["bus"][:, BASE_KV] * np.sqrt(3) / ppci["baseMVA"]
    ppci_0["internal"]["baseI"] = ppci_0["bus"][:, BASE_KV] * np.sqrt(3) / ppci_0["baseMVA"]

    z_equiv = abs((ppci["bus"][bus_idx, R_EQUIV] + ppci["bus"][bus_idx, X_EQUIV] * 1j) * 2 +
                (ppci_0["bus"][bus_idx, R_EQUIV] + ppci_0["bus"][bus_idx, X_EQUIV] * 1j))

    # Only for test, should correspondant to PF result
    baseZ = ppci["bus"][bus_idx, BASE_KV] ** 2 / ppci["baseMVA"]
    ppci["bus"][bus_idx, R_EQUIV_OHM] = baseZ * ppci['bus'][bus_idx, R_EQUIV]
    ppci["bus"][bus_idx, X_EQUIV_OHM] = baseZ * ppci['bus'][bus_idx, X_EQUIV]
    ppci_0["bus"][bus_idx, R_EQUIV_OHM] = baseZ * ppci_0['bus'][bus_idx, R_EQUIV]
    ppci_0["bus"][bus_idx, X_EQUIV_OHM] = baseZ * ppci_0['bus'][bus_idx, X_EQUIV]

    ppci_0["bus"][bus_idx, IKSS1] = c / z_equiv / ppci_0["bus"][bus_idx, BASE_KV] * np.sqrt(3) * ppci_0["baseMVA"]
    ppci["bus"][bus_idx, IKSS1] = c / z_equiv / ppci["bus"][bus_idx, BASE_KV] * np.sqrt(3) * ppci["baseMVA"]

    _current_source_current(net, ppci)


def _current_source_current(net, ppci):
    ppci["bus"][:, IKCV] = 0
    ppci["bus"][:, IKSS2] = 0
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    if not False in net.sgen.current_source.values:
        sgen = net.sgen[net._is_elements["sgen"]]
    else:
        sgen = net.sgen[net._is_elements["sgen"] & net.sgen.current_source]
    if len(sgen) == 0:
        return
    if any(pd.isnull(sgen.sn_mva)):
        raise ValueError("sn_mva needs to be specified for all sgens in net.sgen.sn_mva")
    baseI = ppci["internal"]["baseI"]
    sgen_buses = sgen.bus.values
    sgen_buses_ppc = bus_lookup[sgen_buses]
    if not "k" in sgen:
        raise ValueError("Nominal to short-circuit current has to specified in net.sgen.k")
    i_sgen_pu = sgen.sn_mva.values / net.sn_mva * sgen.k.values
    buses, ikcv_pu, _ = _sum_by_group(sgen_buses_ppc, i_sgen_pu, i_sgen_pu)
    ppci["bus"][buses, IKCV] = ikcv_pu
    if net["_options"]["inverse_y"]:
        Zbus = ppci["internal"]["Zbus"]
        ppci["bus"][:, IKSS2] = abs(1 / np.diag(Zbus) * np.dot(Zbus, ppci["bus"][:, IKCV] * -1j) / baseI)
    else:
        ybus_fact = ppci["internal"]["ybus_fact"]
        diagZ = _calc_zbus_diag(net, ppci)
        ppci["bus"][:, IKSS2] = abs(ybus_fact(ppci["bus"][:, IKCV] * -1j) / diagZ / baseI)
    ppci["bus"][buses, IKCV] /= baseI[buses]


def _calc_ip(net, ppci):
    ip = np.sqrt(2) * (ppci["bus"][:, KAPPA] * ppci["bus"][:, IKSS1] + ppci["bus"][:, IKSS2])
    ppci["bus"][:, IP] = ip


def _calc_ith(net, ppci):
    tk_s = net["_options"]["tk_s"]
    kappa = ppci["bus"][:, KAPPA]
    f = 50
    n = 1
    m = (np.exp(4 * f * tk_s * np.log(kappa - 1)) - 1) / (2 * f * tk_s * np.log(kappa - 1))
    m[np.where(kappa > 1.99)] = 0
    ppci["bus"][:, M] = m
    ith = (ppci["bus"][:, IKSS1] + ppci["bus"][:, IKSS2]) * np.sqrt(m + n)
    ppci["bus"][:, ITH] = ith


# TODO: Ib for generation close bus
# def _calc_ib_generator(net, ppci):
#     # Zbus = ppci["internal"]["Zbus"]
#     # baseI = ppci["internal"]["baseI"]
#     tk_s = net._options['tk_s']
#     c = 1.1

#     z_equiv = ppci["bus"][:, R_EQUIV] + ppci["bus"][:, X_EQUIV] * 1j
#     I_ikss = c / z_equiv / ppci["bus"][:, BASE_KV] / np.sqrt(3) * ppci["baseMVA"]

#     # calculate voltage source branch current
#     # I_ikss = ppci["bus"][:, IKSS1]
#     # V_ikss = (I_ikss * baseI) * Zbus

#     gen = net["gen"][net._is_elements["gen"]]
#     gen_vn_kv = gen.vn_kv.values

#     # Check difference ext_grid and gen
#     gen_buses = ppci['gen'][:, GEN_BUS].astype(np.int64)
#     gen_mbase = ppci['gen'][:, MBASE]
#     gen_i_rg = gen_mbase / (np.sqrt(3) * gen_vn_kv)

#     gen_buses_ppc, gen_sn_mva, I_rG = _sum_by_group(gen_buses, gen_mbase, gen_i_rg)

#     # shunt admittance of generator buses and generator short circuit current
#     # YS = ppci["bus"][gen_buses_ppc, GS] + ppci["bus"][gen_buses_ppc, BS] * 1j
#     # I_kG = V_ikss.T[:, gen_buses_ppc] * YS / baseI[gen_buses_ppc]

#     xdss_pu = gen.xdss_pu.values
#     rdss_pu = gen.rdss_pu.values
#     cosphi = gen.cos_phi.values
#     X_dsss = xdss_pu * np.square(gen_vn_kv) / gen_mbase
#     R_dsss = rdss_pu * np.square(gen_vn_kv) / gen_mbase

#     K_G = ppci['bus'][gen_buses, BASE_KV] / gen_vn_kv * c / (1 + xdss_pu * np.sin(np.arccos(cosphi)))
#     Z_G = (R_dsss + 1j * X_dsss)

#     I_kG = c * ppci['bus'][gen_buses, BASE_KV] / np.sqrt(3) / (Z_G * K_G) * ppci["baseMVA"]

#     dV_G = 1j * X_dsss * K_G * I_kG
#     V_Is = c * ppci['bus'][gen_buses, BASE_KV] / np.sqrt(3)

#     # I_kG_contribution = I_kG.sum(axis=1)
#     # ratio_SG_ikss = I_kG_contribution / I_ikss
#     # close_to_SG = ratio_SG_ikss > 5e-2

#     close_to_SG = I_kG / I_rG > 2

#     if tk_s == 2e-2:
#         mu = 0.84 + 0.26 * np.exp(-0.26 * abs(I_kG) / I_rG)
#     elif tk_s == 5e-2:
#         mu = 0.71 + 0.51 * np.exp(-0.3 * abs(I_kG) / I_rG)
#     elif tk_s == 10e-2:
#         mu = 0.62 + 0.72 * np.exp(-0.32 * abs(I_kG) / I_rG)
#     elif tk_s >= 25e-2:
#         mu = 0.56 + 0.94 * np.exp(-0.38 * abs(I_kG) / I_rG)
#     else:
#         raise UserWarning('not implemented for other tk_s than 20ms, 50ms, 100ms and >=250ms')

#     mu = np.clip(mu, 0, 1)

#     I_ikss_G = abs(I_ikss - np.sum((1 - mu) * I_kG, axis=1))

#     # I_ikss_G = I_ikss - np.sum(abs(V_ikss.T[:, gen_buses_ppc]) * (1-mu) * I_kG, axis=1)

#     I_ikss_G = abs(I_ikss - np.sum(dV_G / V_Is * (1 - mu) * I_kG, axis=1))

#     return I_ikss_G


def _calc_branch_currents(net, ppci, bus_idx):
    n_sc_bus = np.shape(bus_idx)[0]

    case = net._options["case"]
    minmax = np.nanmin if case == "min" else np.nanmax

    Yf = ppci["internal"]["Yf"]
    Yt = ppci["internal"]["Yt"]
    baseI = ppci["internal"]["baseI"]
    n_bus = ppci["bus"].shape[0]
    fb = np.real(ppci["branch"][:, 0]).astype(int)
    tb = np.real(ppci["branch"][:, 1]).astype(int)

    # calculate voltage source branch current
    if net["_options"]["inverse_y"]:
        Zbus = ppci["internal"]["Zbus"]
        V_ikss = (ppci["bus"][:, IKSS1] * baseI) * Zbus
        V_ikss = V_ikss[:, bus_idx]
    else:
        ybus_fact = ppci["internal"]["ybus_fact"]
        V_ikss = np.zeros((n_bus, n_sc_bus), dtype=np.complex)
        for ix, b in enumerate(bus_idx):
            ikss = np.zeros(n_bus, dtype=np.complex)
            ikss[b] = ppci["bus"][b, IKSS1] * baseI[b]
            V_ikss[:, ix] = ybus_fact(ikss)

    ikss1_all_f = np.conj(Yf.dot(V_ikss))
    ikss1_all_t = np.conj(Yt.dot(V_ikss))
    ikss1_all_f[abs(ikss1_all_f) < 1e-10] = 0.
    ikss1_all_t[abs(ikss1_all_t) < 1e-10] = 0.

    # add current source branch current if there is one
    current_sources = any(ppci["bus"][:, IKCV]) > 0
    if current_sources:
        current = np.tile(-ppci["bus"][:, IKCV], (n_sc_bus, 1))
        for ix, b in enumerate(bus_idx):
            current[ix, b] += ppci["bus"][b, IKSS2]

        # calculate voltage source branch current
        if net["_options"]["inverse_y"]:
            Zbus = ppci["internal"]["Zbus"]
            V = np.dot((current * baseI), Zbus).T
        else:
            ybus_fact = ppci["internal"]["ybus_fact"]
            V = np.zeros((n_bus, n_sc_bus), dtype=np.complex)
            for ix, b in enumerate(bus_idx):
                V[:, ix] = ybus_fact(current[ix, :] * baseI[b])

        fb = np.real(ppci["branch"][:, 0]).astype(int)
        tb = np.real(ppci["branch"][:, 1]).astype(int)
        ikss2_all_f = np.conj(Yf.dot(V))
        ikss2_all_t = np.conj(Yt.dot(V))

        ikss_all_f = abs(ikss1_all_f + ikss2_all_f)
        ikss_all_t = abs(ikss1_all_t + ikss2_all_t)
    else:
        ikss_all_f = abs(ikss1_all_f)
        ikss_all_t = abs(ikss1_all_t)

    if net._options["return_all_currents"]:
        ppci["internal"]["branch_ikss_f"] = ikss_all_f / baseI[fb, None]
        ppci["internal"]["branch_ikss_t"] = ikss_all_t / baseI[tb, None]
    else:
        ikss_all_f[abs(ikss_all_f) < 1e-10] = np.nan
        ikss_all_t[abs(ikss_all_t) < 1e-10] = np.nan
        ppci["branch"][:, IKSS_F] = np.nan_to_num(minmax(ikss_all_f, axis=1) / baseI[fb])
        ppci["branch"][:, IKSS_T] = np.nan_to_num(minmax(ikss_all_t, axis=1) / baseI[tb])

    if net._options["ip"]:
        kappa = ppci["bus"][:, KAPPA]
        if current_sources:
            ip_all_f = np.sqrt(2) * (ikss1_all_f * kappa[bus_idx] + ikss2_all_f)
            ip_all_t = np.sqrt(2) * (ikss1_all_t * kappa[bus_idx] + ikss2_all_t)
        else:
            ip_all_f = np.sqrt(2) * ikss1_all_f * kappa[bus_idx]
            ip_all_t = np.sqrt(2) * ikss1_all_t * kappa[bus_idx]

        if net._options["return_all_currents"]:
            ppci["internal"]["branch_ip_f"] = abs(ip_all_f) / baseI[fb, None]
            ppci["internal"]["branch_ip_t"] = abs(ip_all_t) / baseI[tb, None]
        else:
            ip_all_f[abs(ip_all_f) < 1e-10] = np.nan
            ip_all_t[abs(ip_all_t) < 1e-10] = np.nan
            ppci["branch"][:, IP_F] = np.nan_to_num(minmax(abs(ip_all_f), axis=1) / baseI[fb])
            ppci["branch"][:, IP_T] = np.nan_to_num(minmax(abs(ip_all_t), axis=1) / baseI[tb])

    if net._options["ith"]:
        n = 1
        m = ppci["bus"][bus_idx, M]
        ith_all_f = ikss_all_f * np.sqrt(m + n)
        ith_all_t = ikss_all_t * np.sqrt(m + n)

        if net._options["return_all_currents"]:
            ppci["internal"]["branch_ith_f"] = ith_all_f / baseI[fb, None]
            ppci["internal"]["branch_ith_t"] = ith_all_t / baseI[tb, None]
        else:
            ppci["branch"][:, ITH_F] = np.nan_to_num(minmax(ith_all_f, axis=1) / baseI[fb])
            ppci["branch"][:, ITH_T] = np.nan_to_num(minmax(ith_all_t, axis=1) / baseI[fb])

    # Update bus index for branch results
    if net._options["return_all_currents"]:
        ppci["internal"]["br_res_ks_ppci_bus"] = bus_idx
