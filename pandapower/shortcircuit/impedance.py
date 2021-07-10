# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import warnings

import numpy as np
from scipy.sparse.linalg import inv as inv_sparse
from scipy.linalg import inv

from pandapower.shortcircuit.idx_bus import R_EQUIV, X_EQUIV
from pandapower.pypower.idx_bus import BASE_KV
from pandapower.auxiliary import _clean_up

try:
    from pandapower.pf.makeYbus_numba import makeYbus
except ImportError:
    from pandapower.pypower.makeYbus import makeYbus


def _calc_rx(net, ppci, bus_idx):
    # Vectorized for multiple bus
    r_fault = net["_options"]["r_fault_ohm"]
    x_fault = net["_options"]["x_fault_ohm"]
    if r_fault > 0 or x_fault > 0:
        base_r = np.square(ppci["bus"][bus_idx, BASE_KV]) / ppci["baseMVA"]
        fault_impedance = (r_fault + x_fault * 1j) / base_r
    else:
        fault_impedance = 0 + 0j

    if net["_options"]["inverse_y"]:
        Zbus = ppci["internal"]["Zbus"]
        z_equiv = np.diag(Zbus)[bus_idx] + fault_impedance
    else:
        z_equiv = _calc_zbus_diag(net, ppci, bus_idx) + fault_impedance
    ppci["bus"][bus_idx, R_EQUIV] = z_equiv.real
    ppci["bus"][bus_idx, X_EQUIV] = z_equiv.imag


def _calc_ybus(ppci):
    Ybus, Yf, Yt = makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    if np.isnan(Ybus.data).any():
        raise ValueError("nan value detected in Ybus matrix - check calculation parameters for nan values")
    ppci["internal"]["Yf"] = Yf
    ppci["internal"]["Yt"] = Yt
    ppci["internal"]["Ybus"] = Ybus


def _calc_zbus(net, ppci):
    try:
        Ybus = ppci["internal"]["Ybus"]
        sparsity = Ybus.nnz / Ybus.shape[0]**2
        if sparsity < 0.002:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ppci["internal"]["Zbus"] = inv_sparse(Ybus).toarray()
        else:
            ppci["internal"]["Zbus"] = inv(Ybus.toarray())
    except Exception as e:
        _clean_up(net, res=False)
        raise (e)


def _calc_zbus_diag(net, ppci, bus_idx=None):
    ybus_fact = ppci["internal"]["ybus_fact"]
    n_ppci_bus = ppci["bus"].shape[0]

    if bus_idx is None:
        diagZ = np.zeros(n_ppci_bus, dtype=np.complex)
        for i in range(n_ppci_bus):
            b = np.zeros(n_ppci_bus, dtype=np.complex)
            b[i] = 1 + 0j
            diagZ[i] = ybus_fact(b)[i]
        ppci["internal"]["diagZ"] = diagZ
        return diagZ
    else:
        diagZ = np.zeros(bus_idx.shape[0], dtype=np.complex)
        for ix, b in enumerate(bus_idx):
            rhs = np.zeros(n_ppci_bus, dtype=np.complex)
            rhs[b] = 1 + 0j
            diagZ[ix] = ybus_fact(rhs)[b]
        return diagZ
