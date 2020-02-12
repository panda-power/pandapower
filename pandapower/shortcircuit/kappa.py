# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import networkx as nx
import numpy as np

from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X
from pandapower.pypower.idx_bus import BUS_I, GS, BS, BASE_KV

from pandapower.shortcircuit.idx_bus import KAPPA, R_EQUIV, X_EQUIV
from pandapower.shortcircuit.impedance import _calc_ybus, _calc_zbus, _calc_rx

def _add_kappa_to_ppc(net, ppc):
    if not net._options["kappa"]:
        return
    topology = net._options["topology"]
    kappa_method = net._options["kappa_method"]
    if topology == "radial":
        kappa = _kappa(ppc["bus"][:, R_EQUIV] / ppc["bus"][:, X_EQUIV])
    elif kappa_method in ["C", "c"]:
        kappa = _kappa_method_c(net, ppc)
    elif kappa_method in ["B", "b"]:
        kappa = _kappa_method_b(net, ppc)
    else:
        raise ValueError("Unknown kappa method %s - specify B or C"%kappa_method)
    ppc["bus"][:, KAPPA] = kappa

def _kappa_method_c(net, ppc):
    if net.f_hz == 50:
        fc = 20
    elif net.f_hz == 60:
        fc = 24
    else:
        raise ValueError("Frequency has to be 50 Hz or 60 Hz according to the standard")
    ppc_c = copy.deepcopy(ppc)
    ppc_c["branch"][:, BR_X] *= fc / net.f_hz

    zero_conductance = np.where(ppc["bus"][:,GS] == 0)
    ppc["bus"][zero_conductance, BS] *= net.f_hz / fc

    conductance = np.where(ppc["bus"][:,GS] != 0)
    z_shunt = 1 / (ppc_c["bus"][conductance, GS] + 1j * ppc_c["bus"][conductance, BS])
    y_shunt = 1 / (z_shunt.real + 1j * z_shunt.imag * fc / net.f_hz)
    ppc_c["bus"][conductance, GS] = y_shunt.real[0]
    ppc_c["bus"][conductance, BS] = y_shunt.imag[0]
    _calc_ybus(ppc_c)
    _calc_zbus(ppc_c)
    _calc_rx(net, ppc_c)
    rx_equiv_c = ppc_c["bus"][:, R_EQUIV] / ppc_c["bus"][:, X_EQUIV] * fc / net.f_hz
    return _kappa(rx_equiv_c)

def _kappa(rx):
    return 1.02 + .98 * np.exp(-3 * rx)

def _kappa_method_b(net, ppc):
    topology = net._options["topology"]
    kappa_max = np.full(ppc["bus"].shape[0], 2.)
    lv_buses = np.where(ppc["bus"][:, BASE_KV] < 1.)
    if len(lv_buses) > 0:
        kappa_max[lv_buses] = 1.8

    if topology == "meshed":
        kappa_korr = np.full(ppc["bus"].shape[0], 1.15)
    else:
        kappa_korr = np.full(ppc["bus"].shape[0], 1.)
    if topology == "auto":
        kappa_korr = np.full(ppc["bus"].shape[0], 1.)
        mg = nxgraph_from_ppc(net, ppc)
        for bidx in ppc["bus"][:, BUS_I].astype(int):
            paths = list(nx.all_simple_paths(mg, bidx, "earth"))
            if len(paths) > 1:
                kappa_korr[bidx] = 1.15
                for path in paths:
                    r = sum([mg[b1][b2][0]["r"] for b1, b2 in zip(path, path[1:])])
                    x = sum([mg[b1][b2][0]["x"] for b1, b2 in zip(path, path[1:])])
                    if r / x < .3:
                        kappa_korr[bidx] = 1.
                        break
    rx_equiv = ppc["bus"][:, R_EQUIV] / ppc["bus"][:, X_EQUIV]
    return np.clip(kappa_korr * _kappa(rx_equiv), 1, kappa_max)

def nxgraph_from_ppc(net, ppc):
    bus_lookup = net._pd2ppc_lookups["bus"]
    mg = nx.MultiGraph()
    mg.add_nodes_from(ppc["bus"][:, BUS_I].astype(int))
    mg.add_edges_from((int(branch[T_BUS]), int(branch[F_BUS]),
                       {"r": branch[BR_R], "x": branch[BR_X]}) for branch in ppc["branch"].real)
    mg.add_node("earth")
    vs_buses_pp = list(set(net["ext_grid"][net._is_elements["ext_grid"]].bus.values) |
                       set(net["gen"][net._is_elements["gen"]].bus))
    vs_buses = bus_lookup[vs_buses_pp]
    z = 1 / (ppc["bus"][vs_buses, GS] + ppc["bus"][vs_buses, BS] * 1j)
    mg.add_edges_from(("earth", int(bus), {"r": z.real, "x": z.imag})
                        for bus, z in zip(vs_buses, z))
    return mg
