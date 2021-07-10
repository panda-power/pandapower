# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import math
import numpy as np
from itertools import product

import pandapower.auxiliary as aux
from pandapower.build_bus import _build_bus_ppc
from pandapower.build_gen import _build_gen_ppc
#from pandapower.pd2ppc import _ppc2ppci, _init_ppc
from pandapower.pypower.idx_brch import BR_B, BR_R, BR_X, F_BUS, T_BUS, branch_cols, BR_STATUS, SHIFT, TAP
from pandapower.pypower.idx_bus import BASE_KV, BS, GS, BUS_TYPE
from pandapower.build_branch import _calc_tap_from_dataframe, _transformer_correction_factor, _calc_nominal_ratio_from_dataframe,\
     get_trafo_values, _trafo_df_from_trafo3w, _calc_branch_values_from_trafo_df
from pandapower.build_branch import _switch_branches, _branches_with_oos_buses, _initialize_branch_lookup, _end_temperature_correction_factor

def _pd2ppc_zero(net, sequence=0):
    from pandapower.pd2ppc import _ppc2ppci, _init_ppc

    """
    Builds the ppc data structure for zero impedance system. Includes the impedance values of
    lines and transformers, but no load or generation data.

    For short-circuit calculation, the short-circuit impedance of external grids is also considered.
    """
    # select elements in service (time consuming, so we do it once)
    net["_is_elements"] = aux._select_is_elements_numba(net, sequence=sequence)

    ppc = _init_ppc(net, sequence)

    _build_bus_ppc(net, ppc)
    _build_gen_ppc(net, ppc)
    _add_gen_sc_impedance_zero(net, ppc)
    _add_ext_grid_sc_impedance_zero(net, ppc)
    _build_branch_ppc_zero(net, ppc)

    # adds auxilary buses for open switches at branches
    _switch_branches(net, ppc)

    # add auxilary buses for out of service buses at in service lines.
    # Also sets lines out of service if they are connected to two out of service buses
    _branches_with_oos_buses(net, ppc)
    if hasattr(net, "_isolated_buses"):
        ppc["bus"][net._isolated_buses, BUS_TYPE] = 4.

    # generates "internal" ppci format (for powerflow calc) from "external" ppc format and updates the bus lookup
    # Note: Also reorders buses and gens in ppc
    ppci = _ppc2ppci(ppc, net)
    #net._ppc0 = ppc    <--Obsolete. now covered in _init_ppc
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
        ppc["branch"] = np.hstack((ppc["branch"], branch_sc ))
    ppc["branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -360, 360])

    _add_line_sc_impedance_zero(net, ppc)
    _add_trafo_sc_impedance_zero(net, ppc)
    if mode == "sc":
        _add_trafo3w_sc_impedance_zero(net, ppc)
    else:
        if "trafo3w" in lookup:
            raise NotImplementedError("Three winding transformers are not implemented for unbalanced calculations")


def _add_trafo_sc_impedance_zero(net, ppc, trafo_df=None):
    if trafo_df is None:
        trafo_df = net["trafo"]
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    if not "trafo" in branch_lookup:
        return
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    mode = net["_options"]["mode"]
    trafo_model = net["_options"]["trafo_model"]
    f, t = branch_lookup["trafo"]
    trafo_df["_ppc_idx"] = range(f, t)
    bus_lookup = net["_pd2ppc_lookups"]["bus"]

    hv_bus = get_trafo_values(trafo_df, "hv_bus").astype(int)
    lv_bus = get_trafo_values(trafo_df, "lv_bus").astype(int)
    in_service = get_trafo_values(trafo_df, "in_service").astype(int)
    ppc["branch"][f:t, F_BUS] = bus_lookup[hv_bus]
    ppc["branch"][f:t, T_BUS] = bus_lookup[lv_bus]
    buses_all, gs_all, bs_all = np.array([], dtype=int), np.array([]), \
                                np.array([])
    if mode == "sc":
        # Should be considered as connected for all in_service branches
        ppc["branch"][f:t, BR_X] = 1e20
        ppc["branch"][f:t, BR_R] = 1e20
        ppc["branch"][f:t, BR_B] = 0
        ppc["branch"][f:t, BR_STATUS] = in_service
    else:
        ppc["branch"][f:t, BR_STATUS] = 0

    if not "vector_group" in trafo_df:
        raise ValueError("Vector Group of transformer needs to be specified for zero \
                         sequence modelling \n Try : net.trafo[\"vector_group\"] = 'Dyn'" )

    for vector_group, trafos in trafo_df.groupby("vector_group"):
        ppc_idx = trafos["_ppc_idx"].values.astype(int)

        if vector_group in ["Yy", "Yd", "Dy", "Dd"]:
            continue

        vk_percent = trafos["vk_percent"].values.astype(float)
        vkr_percent = trafos["vkr_percent"].values.astype(float)
        sn_mva = trafos["sn_mva"].values.astype(float)
        # Just put pos seq parameter if zero seq parameter is zero
        if not "vk0_percent" in trafos:
            raise ValueError("Short circuit voltage of transformer Vk0 needs to be specified for zero \
                             sequence modelling \n Try : net.trafo[\"vk0_percent\"] = net.trafo[\"vk_percent\"]" )
        vk0_percent = trafos["vk0_percent"].values.astype(float) if \
            trafos["vk0_percent"].values.astype(float).all() != 0. else \
            trafos["vk_percent"].values.astype(float)
        # Just put pos seq parameter if zero seq parameter is zero
        if not "vkr0_percent" in trafos:
            raise ValueError("Real part of short circuit voltage Vk0(Real) needs to be specified for transformer \
                             modelling \n Try : net.trafo[\"vkr0_percent\"] = net.trafo[\"vkr_percent\"]" )
        vkr0_percent = trafos["vkr0_percent"].values.astype(float) if \
            trafos["vkr0_percent"].values.astype(float).all() != 0. else \
            trafos["vkr_percent"].values.astype(float)
        lv_buses = trafos["lv_bus"].values.astype(int)
        hv_buses = trafos["hv_bus"].values.astype(int)
        lv_buses_ppc = bus_lookup[lv_buses]
        hv_buses_ppc = bus_lookup[hv_buses]
        if not "mag0_percent" in trafos:
            # For Shell Type transformers vk0 = vk * 1
            #        and mag0_percent = 10 ... 100  Zm0/ Zsc0
            # --pg 50 DigSilent Power Factory Transformer manual
            raise ValueError("Magnetizing impedance to vk0 ratio needs to be specified for transformer \
                             modelling  \n Try : net.trafo[\"mag0_percent\"] = 100" )
        mag0_ratio = trafos.mag0_percent.values.astype(float)
        if not "mag0_rx" in trafos:
            raise ValueError("Magnetizing impedance R/X ratio needs to be specified for transformer \
                             modelling \n Try : net.trafo[\"mag0_rx\"] = 0 " )
        mag0_rx = trafos["mag0_rx"].values.astype(float)
        if not "si0_hv_partial" in trafos:
            raise ValueError("Zero sequence short circuit impedance partition towards HV side needs to be specified for transformer \
                             modelling \n Try : net.trafo[\"si0_hv_partial\"] = 0.9 " )
        si0_hv_partial = trafos.si0_hv_partial.values.astype(float)
        parallel = trafos.parallel.values.astype(float)
        in_service = trafos["in_service"].astype(int)

        ppc["branch"][ppc_idx, F_BUS] = hv_buses_ppc
        ppc["branch"][ppc_idx, T_BUS] = lv_buses_ppc

        vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafos)
        vn_lv = ppc["bus"][lv_buses_ppc, BASE_KV]
        ratio = _calc_nominal_ratio_from_dataframe(ppc, trafos, vn_trafo_hv, \
                                                   vn_trafo_lv, bus_lookup)
        ppc["branch"][ppc_idx, TAP] = ratio
        ppc["branch"][ppc_idx, SHIFT] = shift

        # zero seq. transformer impedance
        tap_lv = np.square(vn_trafo_lv / vn_lv) * net.sn_mva
        if mode == 'pf_3ph':
            # =============================================================================
            #     Changing base from transformer base to Network base to get Zpu(Net)
            #     Zbase = (kV).squared/S_mva
            #     Zpu(Net)={Zpu(trafo) * Zb(trafo)} / {Zb(Net)}
            #        Note:
            #             Network base voltage is Line-Neutral voltage in each phase
            #             Line-Neutral voltage= Line-Line Voltage(vn_lv) divided by sq.root(3)
            # =============================================================================
            tap_lv = np.square(vn_trafo_lv / vn_lv) * (3 * net.sn_mva)

        z_sc = vk0_percent / 100. / sn_mva * tap_lv
        r_sc = vkr0_percent / 100. / sn_mva * tap_lv
        z_sc = z_sc.astype(float)
        r_sc = r_sc.astype(float)
        x_sc = np.sign(z_sc) * np.sqrt(z_sc ** 2 - r_sc ** 2)
        # TODO: This equation needs to be checked!
        # z0_k = (r_sc + x_sc * 1j) / parallel  * max(1, ratio) **2
        z0_k = (r_sc + x_sc * 1j) / parallel
        y0_k = 1 / z0_k #adding admittance for "pi" model
        if mode == "sc":# or trafo_model == "pi":
            from pandapower.shortcircuit.idx_bus import C_MAX
            cmax = net._ppc["bus"][lv_buses_ppc, C_MAX]
            kt = _transformer_correction_factor(vk_percent, vkr_percent, \
                                                sn_mva, cmax)
            z0_k *= kt
            y0_k = 1 / z0_k
        # =============================================================================
        #       Transformer magnetising impedance for zero sequence
        # =============================================================================
        z_m = z_sc * mag0_ratio
        x_m = z_m / np.sqrt(mag0_rx ** 2 + 1)
        r_m = x_m * mag0_rx
        r0_trafo_mag = r_m / parallel
        x0_trafo_mag = x_m / parallel
        z0_mag = r0_trafo_mag + x0_trafo_mag * 1j
        # =============================================================================
        #         Star - Delta conversion ( T model to Pi Model)
        #      ----------- |__zc=ZAB__|-----------------
        #            _|                   _|
        #     za=ZAN|_|                  |_| zb=ZBN
        #            |                    |
        # =============================================================================
        z1 = si0_hv_partial * z0_k
        z2 = (1 - si0_hv_partial) * z0_k
        z3 = z0_mag
        z_temp = z1 * z2 + z2 * z3 + z1 * z3
        za = z_temp / z2
#        za = z_temp / (z2+z3)
        zb = z_temp / z1
#        zb = z_temp / (z1+z3)
        zc = z_temp / z3 # ZAB  Transfer impedance
#        zc = z_temp / (z1+z2)  # ZAB  Transfer impedance
        YAB = 1 / zc.astype(complex)
        YAN = 1 / za.astype(complex)
        YBN = 1 / zb.astype(complex)

#        YAB_AN = (zc + za) /(zc * za).astype(complex)  # Series conn YAB and YAN
#        YAB_BN = (zc + zb) / (zc * zb).astype(complex)  # Series conn YAB and YBN

        YAB_AN = 1 / (zc + za).astype(complex)  # Series conn YAB and YAN
        YAB_BN = 1 / (zc + zb).astype(complex)  # Series conn YAB and YBN

        # y0_k = 1 / z0_k #adding admittance for "pi" model
        if vector_group == "Dyn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            if trafo_model == "pi":
                y = y0_k # pi model
            else:
                y = (YAB + YBN).astype(complex)  # T model
            gs_all = np.hstack([gs_all, y.real * in_service]) * int(ppc["baseMVA"])
            bs_all = np.hstack([bs_all, y.imag * in_service]) * int(ppc["baseMVA"])

        elif vector_group == "YNd":
            buses_all = np.hstack([buses_all, hv_buses_ppc])
            if trafo_model == "pi":
                y = y0_k * ppc["baseMVA"] # pi model
            else:
                y = (YAB_BN + YAN).astype(complex) #T model
            gs_all = np.hstack([gs_all, y.real * in_service]) * int(ppc["baseMVA"])
            bs_all = np.hstack([bs_all, y.imag * in_service]) * int(ppc["baseMVA"])

        elif vector_group == "Yyn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            if trafo_model == "pi":
                y = 1/(z0_mag+z0_k).astype(complex) #pi model
            else:
#                y = (YAB_AN + YBN).astype(complex) #T model
                y = (YAB + YAB_BN + YBN).astype(complex) # T model

            gs_all = np.hstack([gs_all, y.real * in_service]) * int(ppc["baseMVA"])
            bs_all = np.hstack([bs_all, y.imag * in_service]) * int(ppc["baseMVA"])

        elif vector_group == "YNyn":
            ppc["branch"][ppc_idx, BR_STATUS] = in_service
            # zc = ZAB
            ppc["branch"][ppc_idx, BR_R] = zc.real
            ppc["branch"][ppc_idx, BR_X] = zc.imag

            buses_all = np.hstack([buses_all, hv_buses_ppc])
            gs_all = np.hstack([gs_all, YAN.real * in_service \
                                * int(ppc["baseMVA"])])
            bs_all = np.hstack([bs_all, YAN.imag * in_service \
                                * int(ppc["baseMVA"])])

            buses_all = np.hstack([buses_all, lv_buses_ppc])
            gs_all = np.hstack([gs_all, YBN.real * in_service \
                                * int(ppc["baseMVA"])])
            bs_all = np.hstack([bs_all, YBN.imag * in_service \
                                * int(ppc["baseMVA"])])

        elif vector_group == "YNy":
            buses_all = np.hstack([buses_all, hv_buses_ppc])
            if trafo_model == "pi":
                y = 1/(z0_mag+z0_k).astype(complex)* int(ppc["baseMVA"])#pi model
            else:
                y = (YAB_BN + YAN).astype(complex) * int(ppc["baseMVA"])  #T model
            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])

        elif vector_group == "Yzn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            #            y = 1/(z0_mag+z0_k).astype(complex)* int(ppc["baseMVA"])#T model
            #            y= (za+zb+zc)/((za+zc)*zb).astype(complex)* int(ppc["baseMVA"])#pi model
            y = (YAB_AN + YBN).astype(complex) * int(ppc["baseMVA"])  #T model
            gs_all = np.hstack([gs_all, (1.1547) * y.real * in_service \
                                * int(ppc["baseMVA"])])
            bs_all = np.hstack([bs_all, (1.1547) * y.imag * in_service \
                                * int(ppc["baseMVA"])])

        elif vector_group[-1].isdigit():
            raise ValueError("Unknown transformer vector group %s -\
                             please specify vector group without \
                             phase shift number. Phase shift can be \
                             specified in net.trafo.shift_degree" % vector_group)
        else:
            raise ValueError("Transformer vector group %s is unknown\
                    / not implemented for three phase load flow" % vector_group)

    buses, gs, bs = aux._sum_by_group(buses_all, gs_all, bs_all)
    ppc["bus"][buses, GS] += gs
    ppc["bus"][buses, BS] += bs
    del net.trafo["_ppc_idx"]


def _add_gen_sc_impedance_zero(net, ppc):
    mode = net["_options"]["mode"]
    if mode == 'pf_3ph':
        return

    eg = net["gen"][net._is_elements["gen"]]
    if len(eg) == 0:
        return
    eg_buses = eg.bus.values
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    eg_buses_ppc = bus_lookup[eg_buses]

    y0_gen = 1 / (1e3 + 1e3*1j)
    # buses, gs, bs = aux._sum_by_group(eg_buses_ppc, y0_gen.real, y0_gen.imag)
    ppc["bus"][eg_buses_ppc, GS] += y0_gen.real
    ppc["bus"][eg_buses_ppc, BS] += y0_gen.imag

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
        c = 1.1
    if not "s_sc_%s_mva" % case in eg:
        raise ValueError("short circuit apparent power s_sc_%s_mva needs to be specified for "% case +
                         "external grid" )
    s_sc = eg["s_sc_%s_mva" % case].values
    if not "rx_%s" % case in eg:
        raise ValueError("short circuit R/X rate rx_%s needs to be specified for external grid" %
                         case)
    rx = eg["rx_%s" % case].values

    z_grid = c / s_sc
    if mode == 'pf_3ph':
        z_grid = c / (s_sc/3)
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
    y0_grid = 1 / (r0_grid + x0_grid*1j)

    buses, gs, bs = aux._sum_by_group(eg_buses_ppc, y0_grid.values.real, y0_grid.values.imag)
    ppc["bus"][buses, GS] = gs
    ppc["bus"][buses, BS] = bs


def _add_line_sc_impedance_zero(net, ppc):
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    mode = net["_options"]["mode"]
    if not "line" in branch_lookup:
        return
    line = net["line"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    length = line["length_km"].values
    parallel = line["parallel"].values

    fb = bus_lookup[line["from_bus"].values]
    tb = bus_lookup[line["to_bus"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_mva
    if mode == 'pf_3ph':
        baseR = np.square(ppc["bus"][fb, BASE_KV]) / (3*net.sn_mva)
    f, t = branch_lookup["line"]
    # line zero sequence impedance
    ppc["branch"][f:t, F_BUS] = fb
    ppc["branch"][f:t, T_BUS] = tb
    ppc["branch"][f:t, BR_R] = line["r0_ohm_per_km"].values * length / baseR / parallel
    if mode == "sc":
        # temperature correction
        if net["_options"]["case"] == "min":
            ppc["branch"][f:t, BR_R] *= _end_temperature_correction_factor(net, short_circuit=True)
    ppc["branch"][f:t, BR_X] = line["x0_ohm_per_km"].values * length / baseR / parallel
    ppc["branch"][f:t, BR_B] = (2 * net["f_hz"] * math.pi * line["c0_nf_per_km"].values * 1e-9 * baseR * length * parallel)
    ppc["branch"][f:t, BR_STATUS] = line["in_service"].astype(int)


def _add_trafo3w_sc_impedance_zero(net, ppc):
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    if not "trafo3w" in branch_lookup:
        return
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    branch = ppc["branch"]
    f, t = net["_pd2ppc_lookups"]["branch"]["trafo3w"]
    trafo_df = _trafo_df_from_trafo3w(net, sequence=0)
    hv_bus = get_trafo_values(trafo_df, "hv_bus").astype(int)
    lv_bus = get_trafo_values(trafo_df, "lv_bus").astype(int)
    in_service = get_trafo_values(trafo_df, "in_service").astype(int)
    branch[f:t, F_BUS] = bus_lookup[hv_bus]
    branch[f:t, T_BUS] = bus_lookup[lv_bus]

    r, x, _, ratio, shift = _calc_branch_values_from_trafo_df(net, ppc, trafo_df, sequence=0)

    n_t3 = net.trafo3w.shape[0]
    for t3_ix in np.arange(n_t3):
        t3 = net.trafo3w.iloc[t3_ix, :]

        if t3.vector_group.lower() in set(map(lambda vg: "".join(vg), product("dy", repeat=3))):
            x[[t3_ix, t3_ix+n_t3, t3_ix+n_t3*2]] = 1e10
            r[[t3_ix, t3_ix+n_t3, t3_ix+n_t3*2]] = 1e10
        elif t3.vector_group.lower() == "ynyd":
            # Correction for YnYD
            # z3->y3
            ys = 1 / ((x[t3_ix+n_t3*2] * 1j + r[t3_ix+n_t3*2]) * ratio[t3_ix+n_t3*2] ** 2)
            aux_bus = bus_lookup[lv_bus[t3_ix]]
            ppc["bus"][aux_bus, BS] += ys.imag
            ppc["bus"][aux_bus, GS] += ys.real

            # Set y2/y3 to almost 0 to avoid isolated bus
            x[[t3_ix+n_t3, t3_ix+n_t3*2]] = 1e10
            r[[t3_ix+n_t3, t3_ix+n_t3*2]] = 1e10
        elif t3.vector_group.lower() == "yynd":
            # z3->y3
            ys = 1 / ((x[t3_ix+n_t3*2] * 1j + r[t3_ix+n_t3*2]) * ratio[t3_ix+n_t3*2] ** 2)
            aux_bus = bus_lookup[lv_bus[t3_ix]]
            ppc["bus"][aux_bus, BS] += ys.imag
            ppc["bus"][aux_bus, GS] += ys.real

            # Set y1/y3 to almost 0 to avoid isolated bus
            x[[t3_ix, t3_ix+n_t3*2]] = 1e10
            r[[t3_ix, t3_ix+n_t3*2]] = 1e10
        elif t3.vector_group.lower() == "ynynd":
            # z3->y3
            ys = 1 / ((x[t3_ix+n_t3*2] * 1j + r[t3_ix+n_t3*2]) * ratio[t3_ix+n_t3*2] ** 2)
            aux_bus = bus_lookup[lv_bus[t3_ix]]
            ppc["bus"][aux_bus, BS] += ys.imag
            ppc["bus"][aux_bus, GS] += ys.real

            # Set y3 to almost 0 to avoid isolated bus
            x[t3_ix+n_t3*2] = 1e10
            r[t3_ix+n_t3*2] = 1e10
        else:
            raise UserWarning(f"{t3.vector_group} not supported yet for trafo3w!")

    branch[f:t, BR_R] = r
    branch[f:t, BR_X] = x
    branch[f:t, BR_B] = 0
    branch[f:t, TAP] = ratio
    branch[f:t, SHIFT] = shift
    branch[f:t, BR_STATUS] = in_service
