# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import math
import numpy as np
import copy
import pandas as pd
from packaging import version
import pandapower.auxiliary as aux
from pandapower.pd2ppc import _init_ppc
from pandapower.build_bus import _build_bus_ppc
from pandapower.build_gen import _build_gen_ppc
from pandapower.pd2ppc import _ppc2ppci
from pandapower.pypower.idx_brch import BR_B, BR_R, BR_X, F_BUS, T_BUS, branch_cols, BR_STATUS, \
    SHIFT, TAP
from pandapower.pypower.idx_bus import BASE_KV, BS, GS
from pandapower.build_branch import _calc_tap_from_dataframe, _transformer_correction_factor, \
    _calc_nominal_ratio_from_dataframe
from pandapower.build_branch import _switch_branches, _branches_with_oos_buses, \
    _initialize_branch_lookup


def _pd2ppc_zero(net):
    """
    Builds the ppc data structure for zero impedance system. Includes the impedance values of
    lines and transformers, but no load or generation data.

    For short-circuit calculation, the short-circuit impedance of external grids is also considered.
    """
    # select elements in service (time consuming, so we do it once)
    net["_is_elements"] = aux._select_is_elements_numba(net)

    ppc = _init_ppc(net)

    _build_bus_ppc(net, ppc)
    _build_gen_ppc(net, ppc)
    _add_ext_grid_sc_impedance_zero(net, ppc)
    _build_branch_ppc_zero(net, ppc)

    # adds auxilary buses for open switches at branches
    _switch_branches(net, ppc)

    # add auxilary buses for out of service buses at in service lines.
    # Also sets lines out of service if they are connected to two out of service buses
    _branches_with_oos_buses(net, ppc)
    if hasattr(net, "_isolated_buses"):
        ppc["bus"][net._isolated_buses, 1] = 4.
    # generates "internal" ppci format (for powerflow calc) from "external" ppc format and updates
    # the bus lookup
    # Note: Also reorders buses and gens in ppc
    ppci = _ppc2ppci(ppc, net)
    net._ppc0 = ppc
    return ppc, ppci


def _build_branch_ppc_zero(net, ppc):
    """
    Takes the empty ppc network and fills it with the zero imepdance branch values. The branch
    datatype will be np.complex 128 afterwards.

    .. note:: The order of branches in the ppc is:
            1. Lines
            2. Transformers

    **INPUT**:
        **net** -The pandapower format network

        **ppc** - The PYPOWER format network to fill in values

    """
    length = _initialize_branch_lookup(net)
    lookup = net._pd2ppc_lookups["branch"]
    mode = net._options["mode"]
    ppc["branch"] = np.zeros(shape=(length, branch_cols), dtype=np.complex128)
    if mode == "sc":
        from pandapower.shortcircuit.idx_brch import branch_cols_sc
        branch_sc = np.empty(shape=(length, branch_cols_sc), dtype=float)
        branch_sc.fill(np.nan)
        ppc["branch"] = np.hstack((ppc["branch"], branch_sc))
    ppc["branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -360, 360])
    _add_line_sc_impedance_zero(net, ppc)
    _add_trafo_sc_impedance_zero(net, ppc)
    if "trafo3w" in lookup:
        raise NotImplemented(
            "Three winding transformers are not implemented for unbalanced calculations")


def _add_trafo_sc_impedance_zero(net, ppc, trafo_df=None):
    if trafo_df is None:
        trafo_df = net["trafo"]
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    if not "trafo" in branch_lookup:
        return
    mode = net["_options"]["mode"]
    f, t = branch_lookup["trafo"]
    trafo_df["_ppc_idx"] = range(f, t)
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    buses_all, gs_all, bs_all = np.array([], dtype=int), np.array([]), np.array([])
    for vector_group, trafos in trafo_df.groupby("vector_group"):
        ppc_idx = trafos["_ppc_idx"].values.astype(int)
        ppc["branch"][ppc_idx, BR_STATUS] = 0

        if vector_group in ["Yy", "Yd", "Dy", "Dd"]:
            continue

        vk_percent = trafos["vk_percent"].values.astype(float)
        vkr_percent = trafos["vkr_percent"].values.astype(float)
        sn_mva = trafos["sn_mva"].values.astype(float)
        vk0_percent = trafos["vk0_percent"].values.astype(float)
        vkr0_percent = trafos["vkr0_percent"].values.astype(float)
        lv_buses = trafos["lv_bus"].values.astype(int)
        hv_buses = trafos["hv_bus"].values.astype(int)
        lv_buses_ppc = bus_lookup[lv_buses]
        hv_buses_ppc = bus_lookup[hv_buses]
        mag0_percent = trafos.mag0_percent.values.astype(float)
        mag0_rx = trafos["mag0_rx"].values.astype(float)
        si0_hv_partial = trafos.si0_hv_partial.values.astype(float)
        parallel = trafos.parallel.values.astype(float)
        in_service = trafos["in_service"].astype(int)

        ppc["branch"][ppc_idx, F_BUS] = hv_buses_ppc
        ppc["branch"][ppc_idx, T_BUS] = lv_buses_ppc

        vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafos)
        vn_lv = ppc["bus"][lv_buses_ppc, BASE_KV]
        ratio = _calc_nominal_ratio_from_dataframe(ppc, trafos, vn_trafo_hv, vn_trafo_lv,
                                                   bus_lookup)
        ppc["branch"][ppc_idx, TAP] = ratio
        ppc["branch"][ppc_idx, SHIFT] = shift

        # zero seq. transformer impedance
        tap_lv = np.square(
            vn_trafo_lv / vn_lv) * net.sn_mva  # adjust for low voltage side voltage converter
        z_sc = vk0_percent / 100. / sn_mva * tap_lv
        r_sc = vkr0_percent / 100. / sn_mva * tap_lv
        z_sc = z_sc.astype(float)
        r_sc = r_sc.astype(float)
        x_sc = np.sign(z_sc) * np.sqrt(z_sc ** 2 - r_sc ** 2)
        z0_k = (r_sc + x_sc * 1j) / parallel
        if mode == "sc":
            from pandapower.shortcircuit.idx_bus import C_MAX
            cmax = net._ppc["bus"][lv_buses_ppc, C_MAX]
            kt = _transformer_correction_factor(vk_percent, vkr_percent, sn_mva, cmax)
            z0_k *= kt
        y0_k = 1 / z0_k
        # zero sequence transformer magnetising impedance
        z_m = vk0_percent * mag0_percent / 100. / sn_mva * tap_lv
        x_m = z_m / np.sqrt(mag0_rx ** 2 + 1)
        r_m = x_m * mag0_rx
        r0_trafo_mag = r_m / parallel
        x0_trafo_mag = x_m / parallel
        z0_mag = r0_trafo_mag + x0_trafo_mag * 1j

        if vector_group == "Dyn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            gs_all = np.hstack([gs_all, y0_k.real * in_service])
            bs_all = np.hstack([bs_all, y0_k.imag * in_service])

        elif vector_group == "YNd":
            buses_all = np.hstack([buses_all, hv_buses_ppc])
            gs_all = np.hstack([gs_all, y0_k.real * in_service])
            bs_all = np.hstack([bs_all, y0_k.imag * in_service])

        elif vector_group == "Yyn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            y = 1 / (z0_mag + z0_k).astype(complex)
            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])

        elif vector_group == "YNyn":
            ppc["branch"][ppc_idx, BR_STATUS] = in_service
            # convert the t model to pi model
            z1 = si0_hv_partial * z0_k
            z2 = (1 - si0_hv_partial) * z0_k
            z3 = z0_mag

            z_temp = z1 * z2 + z2 * z3 + z1 * z3
            za = z_temp / z2
            zb = z_temp / z1
            zc = z_temp / z3

            ppc["branch"][ppc_idx, BR_R] = zc.real
            ppc["branch"][ppc_idx, BR_X] = zc.imag
            y = 2 / za
            ppc["branch"][ppc_idx, BR_B] = y.imag - y.real * 1j
            # add a shunt element parallel to zb if the leakage impedance distribution is unequal
            # TODO: this only necessary if si0_hv_partial!=0.5 --> test
            zs = (za * zb) / (za - zb)
            ys = 1 / zs.astype(complex)
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            gs_all = np.hstack([gs_all, ys.real * in_service])
            bs_all = np.hstack([bs_all, ys.imag * in_service])
        elif vector_group == "YNy":
            buses_all = np.hstack([buses_all, hv_buses_ppc])
            y = 1 / (z0_mag + z0_k).astype(complex)
            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])
        elif vector_group[-1].isdigit():
            raise ValueError(
                "Unknown transformer vector group %s - please specify vector group without phase shift number. Phase shift can be specified in net.trafo.shift_degree" % vector_group)
        else:
            raise ValueError(
                "Transformer vector group %s is unknown / not implemented" % vector_group)

    buses, gs, bs = aux._sum_by_group(buses_all, gs_all, bs_all)
    ppc["bus"][buses, GS] += gs
    ppc["bus"][buses, BS] += bs
    del net.trafo["_ppc_idx"]


def _add_ext_grid_sc_impedance_zero(net, ppc):
    mode = net["_options"]["mode"]

    if mode == "sc":
        from pandapower.shortcircuit.idx_bus import C_MAX, C_MIN
        case = net._options["case"]
    else:
        case = "max"
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    eg = net["ext_grid"][net._is_elements["ext_grid"]]
    if len(eg) == 0:
        return
    eg_buses = eg.bus.values
    eg_buses_ppc = bus_lookup[eg_buses]

    if mode == "sc":
        c = ppc["bus"][eg_buses_ppc, C_MAX] if case == "max" else ppc["bus"][eg_buses_ppc, C_MIN]
    else:
        c = 1.
    if not "s_sc_%s_mva" % case in eg:
        raise ValueError(
            "short circuit apparent power s_sc_%s_mva needs to be specified for " % case +
            "external grid")
    s_sc = eg["s_sc_%s_mva" % case].values
    if not "rx_%s" % case in eg:
        raise ValueError("short circuit R/X rate rx_%s needs to be specified for external grid" %
                         case)
    rx = eg["rx_%s" % case].values

    z_grid = c / s_sc
    x_grid = z_grid / np.sqrt(rx ** 2 + 1)
    r_grid = rx * x_grid
    eg["r"] = r_grid
    eg["x"] = x_grid

    # ext_grid zero sequence impedance
    if case == "max":
        x0_grid = net.ext_grid["x0x_%s" % case] * x_grid
        r0_grid = net.ext_grid["r0x0_%s" % case] * x0_grid
    elif case == "min":
        x0_grid = net.ext_grid["x0x_%s" % case] * x_grid
        r0_grid = net.ext_grid["r0x0_%s" % case] * x0_grid

    r0_grid_np, x0_grid_np = r0_grid.values, x0_grid.values
    y0_grid_np = 1 / (r0_grid_np + x0_grid_np * 1j)

    buses, gs, bs = aux._sum_by_group(eg_buses_ppc, y0_grid_np.real, y0_grid_np.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs


def _add_line_sc_impedance_zero(net, ppc):
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    if not "line" in branch_lookup:
        return
    line = net["line"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    length = line["length_km"].values
    parallel = line["parallel"].values

    fb = bus_lookup[line["from_bus"].values]
    tb = bus_lookup[line["to_bus"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_mva
    f, t = branch_lookup["line"]
    # line zero sequence impedance
    ppc["branch"][f:t, F_BUS] = fb
    ppc["branch"][f:t, T_BUS] = tb
    ppc["branch"][f:t, BR_R] = line["r0_ohm_per_km"].values * length / baseR / parallel
    ppc["branch"][f:t, BR_X] = line["x0_ohm_per_km"].values * length / baseR / parallel
    ppc["branch"][f:t, BR_B] = (2 * net["f_hz"] * math.pi * line[
        "c0_nf_per_km"].values * 1e-9 * baseR * length * parallel)
    ppc["branch"][f:t, BR_STATUS] = line["in_service"].astype(int)
