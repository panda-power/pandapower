# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandapower.auxiliary as aux
from pandapower.build_branch import _switch_branches, _branches_with_oos_buses, _build_branch_ppc
from pandapower.build_bus import _build_bus_ppc, _calc_pq_elements_and_add_on_ppc, \
_calc_shunts_and_add_on_ppc, _add_ext_grid_sc_impedance, _add_motor_impedances_ppc
from pandapower.build_gen import _build_gen_ppc, _check_voltage_setpoints_at_same_bus, \
    _check_voltage_angles_at_same_bus, _check_for_reference_bus
from pandapower.opf.make_objective import _make_objective
from pandapower.pypower.idx_area import PRICE_REF_BUS
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_STATUS
from pandapower.pypower.idx_bus import NONE, BUS_I, BUS_TYPE
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS
from pandapower.pypower.run_userfcn import run_userfcn


def _pd2ppc_recycle(net, sequence, recycle):
    key = "_ppc" if sequence is None else "_ppc%d" % sequence
    if not recycle or not net.get(key, None):
        return _pd2ppc(net, sequence=sequence)

    ppc = net[key]
    ppc["success"] = False
    ppc["iterations"] = 0.
    ppc["et"] = 0.

    if "bus_pq" in recycle and recycle["bus_pq"]:
        # update pq values in bus
        _calc_pq_elements_and_add_on_ppc(net, ppc, sequence=sequence)

    # if "trafo" in recycle and recycle["trafo"]:
    #     # update trafo in branch and Ybus
    #     lookup = net._pd2ppc_lookups["branch"]
    #     if "trafo" in lookup:
    #         _calc_trafo_parameter(net, ppc)
    #     if "trafo3w" in lookup:
    #         _calc_trafo3w_parameter(net, ppc)

    if "gen" in recycle and recycle["gen"]:
        # updates the ppc["gen"] part
        _build_gen_ppc(net, ppc)
        ppc["gen"] = np.nan_to_num(ppc["gen"])

    ppci = _ppc2ppci(ppc, net)
    ppci["internal"] = net[key]["internal"]
    net[key] = ppc

    return ppc, ppci


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
    _build_bus_ppc(net, ppc, sequence=sequence)
    if sequence == 0:
        from pandapower.pd2ppc_zero import _add_ext_grid_sc_impedance_zero, _build_branch_ppc_zero
        # Adds external grid impedance for 3ph and sc calculations in ppc0
        _add_ext_grid_sc_impedance_zero(net, ppc)
        # Calculates ppc0 branch impedances from branch elements
        _build_branch_ppc_zero(net, ppc)
    else:
        # Calculates ppc1/ppc2 branch impedances from branch elements
        _build_branch_ppc(net, ppc)

    # Adds P and Q for loads / sgens in ppc['bus'] (PQ nodes)
    if mode == "sc":
        _add_ext_grid_sc_impedance(net, ppc)
        # Generator impedance are seperately added in sc module
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