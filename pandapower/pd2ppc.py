# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import numpy as np
import math
import pandapower.auxiliary as aux
import warnings
from pandapower.build_branch import _initialize_branch_lookup,\
_build_branch_ppc, _switch_branches, _branches_with_oos_buses,\
_calc_tap_from_dataframe,_calc_nominal_ratio_from_dataframe,\
_transformer_correction_factor
from pandapower.build_bus import _build_bus_ppc, _calc_pq_elements_and_add_on_ppc, \
_calc_shunts_and_add_on_ppc, _add_gen_impedances_ppc, _add_motor_impedances_ppc
from pandapower.build_gen import _build_gen_ppc, _check_voltage_setpoints_at_same_bus, \
    _check_voltage_angles_at_same_bus, _check_for_reference_bus
from pandapower.opf.make_objective import _make_objective
from pandapower.pypower.idx_area import PRICE_REF_BUS
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_STATUS, branch_cols, \
    TAP, SHIFT, BR_R, BR_X, BR_B
from pandapower.pypower.idx_bus import NONE, BUS_I, BUS_TYPE, BASE_KV, GS, BS
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS
from pandapower.pypower.run_userfcn import run_userfcn


def _pd2ppc(net, sequence=None):
    """
    Converter Flow:
        1. Create an empty pypower datatructure
        2. Calculate loads and write the bus matrix
        3. Build the gen (Infeeder)- Matrix
        4. Calculate the line parameter and the transformer parameter,
           and fill it in the branch matrix.
           Order: 1st: Line values, 2nd: Trafo values
        5. if opf: make opf objective (gencost)
        6. convert internal ppci format for pypower powerflow / 
        opf without out of service elements and rearanged buses

    INPUT:
        **net** - The pandapower format network
        **sequence** - Used for three phase analysis
        ( 0 - Zero Sequence
          1 - Positive Sequence
          2 - Negative Sequence
        ) 

    OUTPUT:
        **ppc** - The simple matpower format network. Which consists of:
                  ppc = {
                        "baseMVA": 1., *float*
                        "version": 2,  *int*
                        "bus": np.array([], dtype=float),
                        "branch": np.array([], dtype=np.complex128),
                        "gen": np.array([], dtype=float),
                        "gencost" =  np.array([], dtype=float), only for OPF
                        "internal": {
                              "Ybus": np.array([], dtype=np.complex128)
                              , "Yf": np.array([], dtype=np.complex128)
                              , "Yt": np.array([], dtype=np.complex128)
                              , "branch_is": np.array([], dtype=bool)
                              , "gen_is": np.array([], dtype=bool)
                              }
        **ppci** - The "internal" pypower format network for PF calculations
        
    """
    # select elements in service (time consuming, so we do it once)
    net["_is_elements"] = aux._select_is_elements_numba(net, sequence=sequence)

    # Gets network configurations
    mode = net["_options"]["mode"]
    check_connectivity = net["_options"]["check_connectivity"]
    calculate_voltage_angles = net["_options"]["calculate_voltage_angles"]

    ppc = _init_ppc(net, mode=mode, sequence=sequence)

    # generate ppc['bus'] and the bus lookup
    _build_bus_ppc(net, ppc)
    if sequence == 0:
        # Adds external grid impedance for 3ph and sc calculations in ppc0
        _add_ext_grid_sc_impedance_zero(net, ppc)
        # Calculates ppc0 branch impedances from branch elements
        _build_branch_ppc_zero(net, ppc)
    else:
        # Calculates ppc1/ppc2 branch impedances from branch elements  
        _build_branch_ppc(net, ppc)

    # Adds P and Q for loads / sgens in ppc['bus'] (PQ nodes)
    if mode == "sc":
        _add_gen_impedances_ppc(net, ppc)
        _add_motor_impedances_ppc(net, ppc)
    else:
        _calc_pq_elements_and_add_on_ppc(net, ppc, sequence=sequence)
        # adds P and Q for shunts, wards and xwards (to PQ nodes)
        _calc_shunts_and_add_on_ppc(net, ppc)

    # adds auxilary buses for open switches at branches
    _switch_branches(net, ppc)

    # Adds auxilary buses for in service lines with out of service buses.
    # Also deactivates lines if they are connected to two out of service buses
    _branches_with_oos_buses(net, ppc)

    if check_connectivity:
        if sequence in [None, 1, 2]:
            # sets islands (multiple isolated nodes) out of service
            if "opf" in mode:
                net["_isolated_buses"], _, _ = aux._check_connectivity_opf(ppc)
            else:
                net["_isolated_buses"], _, _ = aux._check_connectivity(ppc)
            net["_is_elements_final"] = aux._select_is_elements_numba(net,
                                                                      net._isolated_buses, sequence)
        else:
            ppc["bus"][net._isolated_buses, BUS_TYPE] = NONE
        net["_is_elements"] = net["_is_elements_final"]
    else:
        # sets buses out of service, which aren't connected to branches / REF buses
        aux._set_isolated_buses_out_of_service(net, ppc)

    _build_gen_ppc(net, ppc)

    if "pf" in mode:
        _check_for_reference_bus(ppc)

    aux._replace_nans_with_default_limits(net, ppc)

    # generates "internal" ppci format (for powerflow calc) 
    # from "external" ppc format and updates the bus lookup
    # Note: Also reorders buses and gens in ppc
    ppci = _ppc2ppci(ppc, net)

    if mode == "pf":
        # check if any generators connected to the same bus have different voltage setpoints
        _check_voltage_setpoints_at_same_bus(ppc)
        if calculate_voltage_angles:
            _check_voltage_angles_at_same_bus(net, ppci)

    if mode == "opf":
        # make opf objective
        ppci = _make_objective(ppci, net)

    return ppc, ppci


def _init_ppc(net, mode="pf", sequence=None):
    # init empty ppc
    ppc = {"baseMVA": net.sn_mva
        , "version": 2
        , "bus": np.array([], dtype=float)
        , "branch": np.array([], dtype=np.complex128)
        , "gen": np.array([], dtype=float)
        , "internal": {
            "Ybus": np.array([], dtype=np.complex128)
            , "Yf": np.array([], dtype=np.complex128)
            , "Yt": np.array([], dtype=np.complex128)
            , "branch_is": np.array([], dtype=bool)
            , "gen_is": np.array([], dtype=bool)
            , "DLF": np.array([], dtype=np.complex128)
            , "buses_ord_bfs_nets": np.array([], dtype=float)
        }
           }
    if mode == "opf":
        # additional fields in ppc
        ppc["gencost"] = np.array([], dtype=float)
    net["_ppc"] = ppc

    if sequence is None:
        net["_ppc"] = ppc
    else:
        ppc["sequence"] = int(sequence)
        net["_ppc%s" % sequence] = ppc
    return ppc


def _ppc2ppci(ppc, net, ppci=None):
    """
    Creates the ppci which is used to run the power flow / OPF...
    The ppci is similar to the ppc except that:
    1. it contains no out of service elements
    2. buses are sorted

    Parameters
    ----------
    ppc - the ppc
    net - the pandapower net

    Returns
    -------
    ppci - the "internal" ppc

    """
    # get empty ppci
    if ppci is None:
        ppci = _init_ppc(net, mode=net["_options"]["mode"])
    # BUS Sorting and lookups
    # get bus_lookup
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # get OOS busses and place them at the end of the bus array
    # (there are no OOS busses in the ppci)
    oos_busses = ppc['bus'][:, BUS_TYPE] == NONE
    ppci['bus'] = ppc['bus'][~oos_busses]
    # in ppc the OOS busses are included and at the end of the array
    ppc['bus'] = np.vstack([ppc['bus'][~oos_busses], ppc['bus'][oos_busses]])

    # generate bus_lookup_ppc_ppci (ppc -> ppci lookup)
    ppc_former_order = (ppc['bus'][:, BUS_I]).astype(int)
    aranged_buses = np.arange(len(ppc["bus"]))

    # lookup ppc former order -> consecutive order
    e2i = np.zeros(len(ppc["bus"]), dtype=int)
    e2i[ppc_former_order] = aranged_buses

    # save consecutive indices in ppc and ppci
    ppc['bus'][:, BUS_I] = aranged_buses
    ppci['bus'][:, BUS_I] = ppc['bus'][:len(ppci['bus']), BUS_I]

    # update lookups (pandapower -> ppci internal)
    _update_lookup_entries(net, bus_lookup, e2i, "bus")

    if 'areas' in ppc:
        if len(ppc["areas"]) == 0:  # if areas field is empty
            del ppc['areas']  # delete it (so it's ignored)

    # bus types
    bt = ppc["bus"][:, BUS_TYPE]

    # update branch, gen and areas bus numbering
    ppc['gen'][:, GEN_BUS] = e2i[np.real(ppc["gen"][:, GEN_BUS]).astype(int)].copy()
    ppc["branch"][:, F_BUS] = e2i[np.real(ppc["branch"][:, F_BUS]).astype(int)].copy()
    ppc["branch"][:, T_BUS] = e2i[np.real(ppc["branch"][:, T_BUS]).astype(int)].copy()

    # Note: The "update branch, gen and areas bus numbering" does the same as:
    # ppc['gen'][:, GEN_BUS] = get_indices(ppc['gen'][:, GEN_BUS], bus_lookup_ppc_ppci)
    # ppc["branch"][:, F_BUS] = get_indices(ppc["branch"][:, F_BUS], bus_lookup_ppc_ppci)
    # ppc["branch"][:, T_BUS] = get_indices( ppc["branch"][:, T_BUS], bus_lookup_ppc_ppci)
    # but faster...

    if 'areas' in ppc:
        ppc["areas"][:, PRICE_REF_BUS] = \
            e2i[np.real(ppc["areas"][:, PRICE_REF_BUS]).astype(int)].copy()

    # initialize gen lookups
    for element, (f, t) in net._gen_order.items():
        _build_gen_lookups(net, element, f, t)

    # determine which buses, branches, gens are connected and
    # in-service
    n2i = ppc["bus"][:, BUS_I].astype(int)
    bs = (bt != NONE)  # bus status

    gs = ((ppc["gen"][:, GEN_STATUS] > 0) &  # gen status
          bs[n2i[np.real(ppc["gen"][:, GEN_BUS]).astype(int)]])
    ppci["internal"]["gen_is"] = gs

    brs = (np.real(ppc["branch"][:, BR_STATUS]).astype(int) &  # branch status
           bs[n2i[np.real(ppc["branch"][:, F_BUS]).astype(int)]] &
           bs[n2i[np.real(ppc["branch"][:, T_BUS]).astype(int)]]).astype(bool)
    ppci["internal"]["branch_is"] = brs

    if 'areas' in ppc:
        ar = bs[n2i[ppc["areas"][:, PRICE_REF_BUS].astype(int)]]
        # delete out of service areas
        ppci["areas"] = ppc["areas"][ar]

    # select in service elements from ppc and put them in ppci
    ppci["branch"] = ppc["branch"][brs]

    ppci["gen"] = ppc["gen"][gs]

    if 'dcline' in ppc:
        ppci['dcline'] = ppc['dcline']
    # execute userfcn callbacks for 'ext2int' stage
    if 'userfcn' in ppci:
        ppci = run_userfcn(ppci['userfcn'], 'ext2int', ppci)

    if net._pd2ppc_lookups["ext_grid"] is not None:
        ref_gens = np.setdiff1d(net._pd2ppc_lookups["ext_grid"], np.array([-1]))
    else:
        ref_gens = np.array([])
    if np.any(net.gen.slack.values[net._is_elements["gen"]]):
        slack_gens = np.array(net.gen.index)[net._is_elements["gen"] \
                                             & net.gen["slack"].values]
        ref_gens = np.append(ref_gens, net._pd2ppc_lookups["gen"][slack_gens])
    ppci["internal"]["ref_gens"] = ref_gens.astype(int)
    return ppci


def _update_lookup_entries(net, lookup, e2i, element):
    valid_bus_lookup_entries = lookup >= 0
    # update entries
    lookup[valid_bus_lookup_entries] = e2i[lookup[valid_bus_lookup_entries]]
    aux._write_lookup_to_net(net, element, lookup)


def _build_gen_lookups(net, element, f, t):
    in_service = net._is_elements[element]
    if "controllable" in element:
        pandapower_index = net[element.split("_")[0]].index.values[in_service]
    else:
        pandapower_index = net[element].index.values[in_service]
    ppc_index = np.arange(f, t)
    if len(pandapower_index) > 0:
        _init_lookup(net, element, pandapower_index, ppc_index)


def _init_lookup(net, lookup_name, pandapower_index, ppc_index):
    # init lookup
    lookup = -np.ones(max(pandapower_index) + 1, dtype=int)

    # update lookup
    lookup[pandapower_index] = ppc_index

    aux._write_lookup_to_net(net, lookup_name, lookup)


def _build_branch_ppc_zero(net, ppc):
    """
    Takes an empty ppc0 and puts zero sequence branch impedances. The branch
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
    ppc["branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, \
                                      1, -360, 360])
    # Adds zero sequence impedances of lines in ppc0
    _add_line_sc_impedance_zero(net, ppc)
    # Adds zero sequence impedances of transformers in ppc0
    _add_trafo_sc_impedance_zero(net, ppc)
    if "trafo3w" in lookup:
        raise NotImplementedError("Three winding transformers are not \
                                  implemented for unbalanced calculations")


def _add_trafo_sc_impedance_zero(net, ppc, trafo_df=None):
    if trafo_df is None:
        trafo_df = net["trafo"]
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    if not "trafo" in branch_lookup:
        return
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    mode = net["_options"]["mode"]
    f, t = branch_lookup["trafo"]
    trafo_df["_ppc_idx"] = range(f, t)
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    buses_all, gs_all, bs_all = np.array([], dtype=int), np.array([]), \
                                np.array([])
    if not "vector_group" in trafo_df:
        raise ValueError("Vector Group of transformer needs to be specified for zero \
                         sequence modelling \n Try : net.trafo[\"vector_group\"] = 'Dyn'" )

    for vector_group, trafos in trafo_df.groupby("vector_group"):
        ppc_idx = trafos["_ppc_idx"].values.astype(int)
        ppc["branch"][ppc_idx, BR_STATUS] = 0

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
        z0_k = (r_sc + x_sc * 1j) / parallel
        if mode == "sc":
            from pandapower.shortcircuit.idx_bus import C_MAX
            cmax = net._ppc["bus"][lv_buses_ppc, C_MAX]
            kt = _transformer_correction_factor(vk_percent, vkr_percent, \
                                                sn_mva, cmax)
            z0_k *= kt
        #        y0_k = 1 / z0_k   --- No longer needed since we are using Pi model
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
        zb = z_temp / z1
        zc = z_temp / z3  # ZAB  Transfer impedance
        YAB = 1 / zc.astype(complex)
        YAN = 1 / za.astype(complex)
        YBN = 1 / zb.astype(complex)
        YAB_AN = 1 / (zc + za).astype(complex)  # Series conn YAB and YAN
        YAB_BN = 1 / (zc + zb).astype(complex)  # Series conn YAB and YBN

        if vector_group == "Dyn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])

            y = (YAB + YBN).astype(complex) * int(ppc["baseMVA"])  # pi model
            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])

        elif vector_group == "YNd":
            buses_all = np.hstack([buses_all, hv_buses_ppc])

            #            gs_all = np.hstack([gs_all, y0_k.real*in_service])#T model
            #            bs_all = np.hstack([bs_all, y0_k.imag*in_service])

            y = (YAB_BN + YAN).astype(complex) * int(ppc["baseMVA"])  # pi model
            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])

        elif vector_group == "Yyn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            #            y = 1/(z0_mag+z0_k).astype(complex)* int(ppc["baseMVA"])#T model

            y = (YAB_AN + YBN).astype(complex) * int(ppc["baseMVA"])  # pi model
            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])

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
            #            y = 1/(z0_mag+z0_k).astype(complex)* int(ppc["baseMVA"])#T model
            y = (YAB_BN + YAN).astype(complex) * int(ppc["baseMVA"])  # pi model
            gs_all = np.hstack([gs_all, y.real * in_service])
            bs_all = np.hstack([bs_all, y.imag * in_service])

        elif vector_group == "Yzn":
            buses_all = np.hstack([buses_all, lv_buses_ppc])
            #            y = 1/(z0_mag+z0_k).astype(complex)* int(ppc["baseMVA"])#T model
            #            y= (za+zb+zc)/((za+zc)*zb).astype(complex)* int(ppc["baseMVA"])#pi model
            y = (YAB_AN + YBN).astype(complex) * int(ppc["baseMVA"])  # pi model
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
        c = ppc["bus"][eg_buses_ppc, C_MAX] if case == "max" else \
            ppc["bus"][eg_buses_ppc, C_MIN]
    else:
        c = 1.1
    if not "s_sc_%s_mva" % case in eg:
        raise ValueError("Short Circuit apparent power s_sc_%s_mva for zero sequence needs to be specified for "% case +
                         "external grid \n Try: net.ext_grid['s_sc_max_mva'] = 1000" )
    else:
        s_sc = eg["s_sc_%s_mva" % case].values/ppc['baseMVA']
    if not "rx_%s" % case in eg:
        raise ValueError("Positive sequence short circuit R/X rate rx_%s needs to be specified\
                         for external grid \n Try: net.ext_grid['rx_max'] = 0.1" % case)
    rx = eg["rx_%s" % case].values
    z_grid = c / s_sc
    if mode == 'pf_3ph':
        z_grid = c / (s_sc / 3)
    x_grid = z_grid / np.sqrt(rx ** 2 + 1)
    r_grid = rx * x_grid
    eg["r"] = r_grid
    eg["x"] = x_grid

    if not "x0x_%s" % case in eg:
        raise ValueError("Zero sequence to positive sequence reactance ratio x0x_%s needs to be specified for "% case +
                         "external grid \n Try : net.ext_grid[\"x0x_max\"] = 1.0" )
    if not "r0x0_%s" % case in eg:
        raise ValueError("Zero sequence short circuit R0/X0 rate r0x0_%s needs to be specified\
                         for external grid \n Try : net.ext_grid[\"r0x0_max\"] = 0.1" % case )

    # ext_grid zero sequence impedance
    if case == "max":
        x0_grid = net.ext_grid["x0x_%s" % case] * x_grid
        r0_grid = net.ext_grid["r0x0_%s" % case] * x0_grid
    elif case == "min":
        x0_grid = net.ext_grid["x0x_%s" % case] * x_grid
        r0_grid = net.ext_grid["r0x0_%s" % case] * x0_grid
    y0_grid = 1 / (r0_grid + x0_grid * 1j)
    buses, gs, bs = aux._sum_by_group(eg_buses_ppc, np.real(np.array(y0_grid)), np.imag(np.array(y0_grid)))
    ppc["bus"][buses, GS] = gs * ppc['baseMVA']
    ppc["bus"][buses, BS] = bs * ppc['baseMVA']   

def _add_line_sc_impedance_zero(net, ppc):
    branch_lookup = net["_pd2ppc_lookups"]["branch"]
    mode = net["_options"]["mode"]
    if not "line" in branch_lookup:
        return
#    line = net["line"].loc[net._is_elements["line_is_idx"].values.tolist()]
    line = net["line"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    length = line["length_km"].values
    parallel = line["parallel"].values

    fb = bus_lookup[line["from_bus"].values]
    tb = bus_lookup[line["to_bus"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_mva
    if mode == 'pf_3ph':
        baseR = np.square(ppc["bus"][fb, BASE_KV]) / (3 * net.sn_mva)
    f, t = branch_lookup["line"]
    # line zero sequence impedance
    ppc["branch"][f:t, F_BUS] = fb
    ppc["branch"][f:t, T_BUS] = tb
    # Just putting pos seq resistance if zero seq resistance is zero 
    if not "r0_ohm_per_km" in line:
        raise ValueError("Zero sequence resistance R0 needs to be specified\
                         for line \n Try : net.line[\"r0_ohm_per_km\"] = net.line[\"r_ohm_per_km\"].values * 4" )
    ppc["branch"][f:t, BR_R] = line["r0_ohm_per_km"].values * length / baseR / parallel if \
            line["r0_ohm_per_km"].values.all() != 0 else \
            line["r_ohm_per_km"].values * 4 * length / baseR / parallel 
    # Just putting pos seq inducatance if zero seq inductance is zero
    if not "x0_ohm_per_km" in line:
        raise ValueError("Zero sequence reactance X0 needs to be specified\
                         for line \n Try : net.line[\"x0_ohm_per_km\"] = net.line[\"x_ohm_per_km\"].values * 4" )
    ppc["branch"][f:t, BR_X] = \
        line["x0_ohm_per_km"].values * length / baseR / parallel if \
            line["x0_ohm_per_km"].values.all() != 0 else \
            line["x_ohm_per_km"].values * 4 *  length / baseR / parallel
    # Just putting pos seq capacitance if zero seq capacitance is zero
    if not "c0_nf_per_km" in line:
        raise ValueError("Zero sequence capacitance C0 needs to be specified\
                         for line \n Try : net.line[\"c0_nf_per_km\"] = net.line[\"c_nf_per_km\"].values * 0.25" )
    ppc["branch"][f:t, BR_B] = \
        (2 * net["f_hz"] * math.pi * line["c0_nf_per_km"].values * 1e-9 * baseR \
         * length * parallel) if \
            line["c0_nf_per_km"].values.all() != 0 else \
            (2 * net["f_hz"] * math.pi * (line["c_nf_per_km"].values * 0.25) * 1e-9 \
             * baseR * length * parallel)
    ppc["branch"][f:t, BR_STATUS] = line["in_service"].astype(int)