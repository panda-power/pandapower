# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from collections import defaultdict, Counter
from itertools import chain

import numpy as np
import pandas as pd

from pandapower.auxiliary import _sum_by_group, phase_to_sequence
from pandapower.pypower.idx_bus import BUS_I, BASE_KV, PD, QD, GS, BS, VMAX, VMIN, BUS_TYPE, NONE, VM, VA, \
    CID, CZD, bus_cols, REF

try:
    from numba import jit
except ImportError:
    from .pf.no_numba import jit


@jit(nopython=True, cache=False)
def ds_find(ar, bus):  # pragma: no cover
    while True:
        p = ar[bus]
        if p == bus:
            break
        bus = p
    return p


@jit(nopython=True, cache=False)
def ds_union(ar, bus1, bus2, bus_is_pv):  # pragma: no cover
    root1 = ds_find(ar, bus1)
    root2 = ds_find(ar, bus2)
    if root1 == root2:
        return
    if bus_is_pv[root2]:
        ar[root1] = root2
    else:
        ar[root2] = root1


@jit(nopython=True, cache=False)
def ds_create(ar, switch_bus, switch_elm, switch_et_bus, switch_closed, switch_z_ohm,
              bus_is_pv, bus_in_service):  # pragma: no cover
    for i in range(len(switch_bus)):
        if not switch_closed[i] or not switch_et_bus[i] or switch_z_ohm[i] > 0:
            continue
        bus1 = switch_bus[i]
        bus2 = switch_elm[i]
        if bus_in_service[bus1] and bus_in_service[bus2]:
            ds_union(ar, bus1, bus2, bus_is_pv)


@jit(nopython=True, cache=False)
def fill_bus_lookup(ar, bus_lookup, bus_index):
    for i in range(len(bus_index)):
        bus_lookup[bus_index[i]] = i
    for b in bus_index:
        ds = ds_find(ar, b)
        bus_lookup[b] = bus_lookup[ar[ds]]


def create_bus_lookup_numba(net, bus_index, bus_is_idx, gen_is_mask, eg_is_mask):
    max_bus_idx = np.max(bus_index)
    # extract numpy arrays of switch table data
    switch = net["switch"]
    switch_bus = switch["bus"].values
    switch_elm = switch["element"].values
    switch_et_bus = switch["et"].values == "b"
    switch_closed = switch["closed"].values
    switch_z_ohm = switch['z_ohm'].values
    # create array for fast checking if a bus is in_service
    bus_in_service = np.zeros(max_bus_idx + 1, dtype=bool)
    bus_in_service[bus_is_idx] = True
    # create array for fast checking if a bus is pv bus
    bus_is_pv = np.zeros(max_bus_idx + 1, dtype=bool)
    bus_is_pv[net["ext_grid"]["bus"].values[eg_is_mask]] = True
    bus_is_pv[net["gen"]["bus"].values[gen_is_mask]] = True
    # create array that represents the disjoint set
    ar = np.arange(max_bus_idx + 1)
    ds_create(ar, switch_bus, switch_elm, switch_et_bus, switch_closed, switch_z_ohm, bus_is_pv, bus_in_service)
    # finally create and fill bus lookup
    bus_lookup = -np.ones(max_bus_idx + 1, dtype=int)
    fill_bus_lookup(ar, bus_lookup, bus_index)
    return bus_lookup


class DisjointSet(dict):
    def add(self, item):
        self[item] = item

    def find(self, item):
        parent = self[item]
        if self[parent] != parent:
            parent = self.find(parent)
            self[item] = parent
        return parent

    def union(self, item1, item2):
        p1 = self.find(item1)
        p2 = self.find(item2)
        self[p1] = p2


def create_consecutive_bus_lookup(net, bus_index):
    # create a mapping from arbitrary pp-index to a consecutive index starting at zero (ppc-index)
    consec_buses = np.arange(len(bus_index))
    # bus_lookup as dict:
    # bus_lookup = dict(zip(bus_index, consec_buses))
    # bus lookup as mask from pandapower -> pypower
    bus_lookup = -np.ones(max(bus_index) + 1, dtype=int)
    bus_lookup[bus_index] = consec_buses
    return bus_lookup


def create_bus_lookup_numpy(net, bus_index, bus_is_idx, gen_is_mask, eg_is_mask, closed_bb_switch_mask):
    bus_lookup = create_consecutive_bus_lookup(net, bus_index)
    net._fused_bb_switches = closed_bb_switch_mask & (net["switch"]["z_ohm"].values <= 0)
    if net._fused_bb_switches.any():
        # Note: this might seem a little odd - first constructing a pp to ppc mapping without
        # fused busses and then update the entries. The alternative (to construct the final
        # mapping at once) would require to determine how to fuse busses and which busses
        # are not part of any opened bus-bus switches first. It turns out, that the latter takes
        # quite some time in the average usecase, where #busses >> #bus-bus switches.

        # Find PV / Slack nodes -> their bus must be kept when fused with a PQ node
        pv_list = [net["ext_grid"]["bus"].values[eg_is_mask], net["gen"]["bus"].values[gen_is_mask]]
        pv_ref = np.unique(np.hstack(pv_list))
        # get the pp-indices of the buses which are connected to a switch to be fused
        fbus = net["switch"]["bus"].values[net._fused_bb_switches]
        tbus = net["switch"]["element"].values[net._fused_bb_switches]

        # create a mapping to map each bus to itself at frist ...
        ds = DisjointSet({e: e for e in chain(fbus, tbus)})

        # ... to follow each bus along a possible chain of switches to its final bus and update the
        # map
        for f, t in zip(fbus, tbus):
            ds.union(f, t)

        # now we can find out how to fuse each bus by looking up the final bus of the chain they
        # are connected to
        v = defaultdict(set)
        for a in ds:
            v[ds.find(a)].add(a)
        # build sets of buses which will be fused
        disjoint_sets = [e for e in v.values() if len(e) > 1]

        # check if PV buses need to be fused
        # if yes: the sets with PV buses must be found (which is slow)
        # if no: the check can be omitted
        if any(i in fbus or i in tbus for i in pv_ref):
            # for every disjoint set
            for dj in disjoint_sets:
                # check if pv buses are in the disjoint set dj
                pv_buses_in_set = set(pv_ref) & dj
                nr_pv_bus = len(pv_buses_in_set)
                if nr_pv_bus == 0:
                    # no pv buses. Use any bus in dj
                    map_to = bus_lookup[dj.pop()]
                else:
                    # one pv bus. Get bus from pv_buses_in_set
                    map_to = bus_lookup[pv_buses_in_set.pop()]
                for bus in dj:
                    # update lookup
                    bus_lookup[bus] = map_to
        else:
            # no PV buses in set
            for dj in disjoint_sets:
                # use any bus in set
                map_to = bus_lookup[dj.pop()]
                for bus in dj:
                    # update bus lookup
                    bus_lookup[bus] = map_to
    return bus_lookup


def create_bus_lookup(net, bus_index, bus_is_idx, gen_is_mask, eg_is_mask, numba):
    switches_with_pos_z_ohm = net["switch"]["z_ohm"].values > 0
    if switches_with_pos_z_ohm.any() or numba == False:
        # if there are any closed bus-bus switches find them
        closed_bb_switch_mask = ((net["switch"]["closed"].values == True) &
                                 (net["switch"]["et"].values == "b") &
                                 np.in1d(net["switch"]["bus"].values, bus_is_idx) &
                                 np.in1d(net["switch"]["element"].values, bus_is_idx))

    if switches_with_pos_z_ohm.any():
        net._impedance_bb_switches = closed_bb_switch_mask & switches_with_pos_z_ohm
    else:
        net._impedance_bb_switches = np.zeros(switches_with_pos_z_ohm.shape)

    if numba:
        bus_lookup = create_bus_lookup_numba(net, bus_index, bus_is_idx,
                                             gen_is_mask, eg_is_mask)
    else:
        bus_lookup = create_bus_lookup_numpy(net, bus_index, bus_is_idx,
                                             gen_is_mask, eg_is_mask, closed_bb_switch_mask)
    return bus_lookup


def get_voltage_init_vector(net, init_v, mode, sequence=None):
    if isinstance(init_v, str):
        if init_v == "results":
            res_table = "res_bus" if sequence is None else "res_bus_3ph"

            # init voltage possible if bus results are available
            if res_table not in net or not net[res_table].index.equals(net.bus.index):
                # cannot init from results, since sorting of results is different from element table
                # TO BE REVIEWED! Why there was no raise before this commit?
                raise UserWarning("Init from results not possible. Index of %s do not match with bus. "
                                  "You should sort res_bus before calling runpp." % res_table)

            if res_table == "res_bus_3ph":
                vm = net.res_bus_3ph[["vm_a_pu", "vm_b_pu", "vm_c_pu"]].values.T
                va = net.res_bus_3ph[["va_a_degree", "va_b_degree", "va_c_degree"]].values.T

                voltage_vector = phase_to_sequence(vm * np.exp(1j * np.pi * va / 180.))[sequence, :]

                if mode == "magnitude":
                    return np.abs(voltage_vector)
                elif mode == "angle":
                    return np.angle(voltage_vector) * 180 / np.pi
                else:
                    raise UserWarning(str(mode)+" for initialization not available!")
            else:
                if mode == "magnitude":
                    return net[res_table]["vm_pu"].values.copy()
                elif mode == "angle":
                    return net[res_table]["va_degree"].values.copy()
                else:
                    raise UserWarning(str(mode)+" for initialization not available!")
        if init_v == "flat":
            return None
    elif isinstance(init_v, (float, np.ndarray, list)) and sequence is None or sequence == 1:
        return init_v
    elif isinstance(init_v, pd.Series) and sequence is None or sequence == 1:
        if init_v.index.equals(net.bus.index):
            return init_v.loc[net.bus.index]
        else:
            raise UserWarning("Voltage starting vector indices do not match bus indices")


def _build_bus_ppc(net, ppc, sequence=None):
    """
    Generates the ppc["bus"] array and the lookup pandapower indices -> ppc indices
    """
    init_vm_pu = net["_options"]["init_vm_pu"]
    init_va_degree = net["_options"]["init_va_degree"]
    mode = net["_options"]["mode"]
    numba = net["_options"]["numba"] if "numba" in net["_options"] else False

    # get bus indices
    nr_xward = len(net.xward)
    nr_trafo3w = len(net.trafo3w)
    aux = dict()
    if nr_xward > 0 or nr_trafo3w > 0:
        bus_indices = [net["bus"].index.values, np.array([], dtype=np.int64)]
        max_idx = max(net["bus"].index) + 1
        if nr_xward > 0:
            aux_xward = np.arange(max_idx, max_idx + nr_xward, dtype=np.int64)
            aux["xward"] = aux_xward
            bus_indices.append(aux_xward)
        if nr_trafo3w:
            aux_trafo3w = np.arange(max_idx + nr_xward, max_idx + nr_xward + nr_trafo3w)
            aux["trafo3w"] = aux_trafo3w
            bus_indices.append(aux_trafo3w)
        bus_index = np.concatenate(bus_indices)
    else:
        bus_index = net["bus"].index.values
    # get in service elements

    if mode == "nx":
        bus_lookup = create_consecutive_bus_lookup(net, bus_index)
    else:
        _is_elements = net["_is_elements"]
        eg_is_mask = _is_elements['ext_grid']
        gen_is_mask = _is_elements['gen']
        bus_is_idx = _is_elements['bus_is_idx']
        bus_lookup = create_bus_lookup(net, bus_index, bus_is_idx,
                                       gen_is_mask, eg_is_mask, numba=numba)

    n_bus_ppc = len(bus_index)
    # init ppc with empty values
    ppc["bus"] = np.zeros(shape=(n_bus_ppc, bus_cols), dtype=float)
    ppc["bus"][:, :15] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0., 0.])  # changes of
    # voltage limits (2 and 0) must be considered in check_opf_data

    if sequence is not None and sequence != 1:
        ppc["bus"][:, VM] = 0.

    if mode == "sc":
        from pandapower.shortcircuit.idx_bus import bus_cols_sc
        bus_sc = np.empty(shape=(n_bus_ppc, bus_cols_sc), dtype=float)
        bus_sc.fill(np.nan)
        ppc["bus"] = np.hstack((ppc["bus"], bus_sc))

    # apply consecutive bus numbers
    ppc["bus"][:, BUS_I] = np.arange(n_bus_ppc)

    n_bus = len(net.bus.index)
    # init voltages from net
    ppc["bus"][:n_bus, BASE_KV] = net["bus"]["vn_kv"].values
    # set buses out of service (BUS_TYPE == 4)
    if nr_xward > 0 or nr_trafo3w > 0:
        in_service = np.concatenate([net["bus"]["in_service"].values,
                                     net["xward"]["in_service"].values,
                                     net["trafo3w"]["in_service"].values])
    else:
        in_service = net["bus"]["in_service"].values
    ppc["bus"][~in_service, BUS_TYPE] = NONE
    if mode != "nx":
        set_reference_buses(net, ppc, bus_lookup, mode)
    vm_pu = get_voltage_init_vector(net, init_vm_pu, "magnitude", sequence=sequence)
    if vm_pu is not None:
        ppc["bus"][:n_bus, VM] = vm_pu

    va_degree = get_voltage_init_vector(net, init_va_degree, "angle", sequence=sequence)
    if va_degree is not None:
        ppc["bus"][:n_bus, VA] = va_degree

    if mode == "sc":
        _add_c_to_ppc(net, ppc)

    if net._options["mode"] == "opf":
        if "max_vm_pu" in net.bus:
            ppc["bus"][:n_bus, VMAX] = net["bus"].max_vm_pu.values
        else:
            ppc["bus"][:n_bus, VMAX] = 2.  # changes of VMAX must be considered in check_opf_data
        if "min_vm_pu" in net.bus:
            ppc["bus"][:n_bus, VMIN] = net["bus"].min_vm_pu.values
        else:
            ppc["bus"][:n_bus, VMIN] = 0.  # changes of VMIN must be considered in check_opf_data

    if len(net.xward):
        _fill_auxiliary_buses(net, ppc, bus_lookup, "xward", "bus", aux)

    if len(net.trafo3w):
        _fill_auxiliary_buses(net, ppc, bus_lookup, "trafo3w", "hv_bus", aux)
    net["_pd2ppc_lookups"]["bus"] = bus_lookup
    net["_pd2ppc_lookups"]["aux"] = aux


def _fill_auxiliary_buses(net, ppc, bus_lookup, element, bus_column, aux):
    element_bus_idx = bus_lookup[net[element][bus_column].values]
    aux_idx = bus_lookup[aux[element]]
    ppc["bus"][aux_idx, BASE_KV] = ppc["bus"][element_bus_idx, BASE_KV]
    if net._options["mode"] == "opf":
        ppc["bus"][aux_idx, VMIN] = ppc["bus"][element_bus_idx, VMIN]
        ppc["bus"][aux_idx, VMAX] = ppc["bus"][element_bus_idx, VMAX]

    if net._options["init_vm_pu"] == "results":
        ppc["bus"][aux_idx, VM] = net["res_%s" % element]["vm_internal_pu"].values
    else:
        ppc["bus"][aux_idx, VM] = ppc["bus"][element_bus_idx, VM]
    if net._options["init_va_degree"] == "results":
        ppc["bus"][aux_idx, VA] = net["res_%s" % element]["va_internal_degree"].values
    else:
        ppc["bus"][aux_idx, VA] = ppc["bus"][element_bus_idx, VA]


def set_reference_buses(net, ppc, bus_lookup, mode):
    if mode == "nx":
        return
    eg_buses = bus_lookup[net.ext_grid.bus.values[net._is_elements["ext_grid"]]]
    ppc["bus"][eg_buses, BUS_TYPE] = REF
    if mode == "sc":
        gen_slacks = net._is_elements["gen"]  # generators are slacks for short-circuit calculation
    else:
        gen_slacks = net._is_elements["gen"] & net.gen["slack"].values
    if gen_slacks.any():
        slack_buses = net.gen["bus"].values[gen_slacks]
        ppc["bus"][bus_lookup[slack_buses], BUS_TYPE] = REF


def _calc_pq_elements_and_add_on_ppc(net, ppc, sequence= None):
    # init values
    b, p, q = np.array([], dtype=int), np.array([]), np.array([])

    _is_elements = net["_is_elements"]
    voltage_depend_loads = net["_options"]["voltage_depend_loads"]
    mode = net["_options"]["mode"]
    pq_elements = ["load", "motor", "sgen", "storage", "ward", "xward"]
    for element in pq_elements:
        tab = net[element]
        if len(tab):
            if element == "load" and voltage_depend_loads:
                cz = tab["const_z_percent"].values / 100.
                ci = tab["const_i_percent"].values / 100.
                if ((cz + ci) > 1).any():
                    raise ValueError("const_z_percent + const_i_percent need to be less or equal to " +
                                     "100%!")

                # cumulative sum of constant-current loads
                b_zip = tab["bus"].values
                load_counter = Counter(b_zip)

                bus_lookup = net["_pd2ppc_lookups"]["bus"]
                b_zip = bus_lookup[b_zip]
                load_counter = {bus_lookup[k]: v for k, v in load_counter.items()}
                b_zip, ci_sum, cz_sum = _sum_by_group(b_zip, ci, cz)

                for bus, no_loads in load_counter.items():
                    ci_sum[b_zip == bus] /= no_loads
                    cz_sum[b_zip == bus] /= no_loads

                ppc["bus"][b_zip, CID] = ci_sum
                ppc["bus"][b_zip, CZD] = cz_sum
            active = _is_elements[element]
            sign = -1 if element == "sgen" else 1
            if element == "motor":
                p_mw, q_mvar = _get_motor_pq(net)
                p = np.hstack([p, p_mw])
                q = np.hstack([q, q_mvar])
            elif element.endswith("ward"):
                p = np.hstack([p, tab["ps_mw"].values * active * sign])
                q = np.hstack([q, tab["qs_mvar"].values * active * sign])
            else:
                scaling = tab["scaling"].values
                p = np.hstack([p, tab["p_mw"].values * active * scaling * sign])
                q = np.hstack([q, tab["q_mvar"].values * active * scaling * sign])
            b = np.hstack([b, tab["bus"].values])

    for element in ["asymmetric_load", "asymmetric_sgen"]:
        if len(net[element]) > 0 and mode == "pf":
            p_mw, q_mvar = _get_symmetric_pq_of_unsymetric_element(net, element)
            sign = -1 if element.endswith("sgen") else 1
            p = np.hstack([p, p_mw * sign])
            q = np.hstack([q, q_mvar * sign ])
            b = np.hstack([b, net[element]["bus"].values])

    # sum up p & q of bus elements
    if b.size:
        bus_lookup = net["_pd2ppc_lookups"]["bus"]
        b = bus_lookup[b]
        b, vp, vq = _sum_by_group(b, p, q)
        ppc["bus"][b, PD] = vp
        ppc["bus"][b, QD] = vq


def _get_symmetric_pq_of_unsymetric_element(net, element):
    scale = net["_is_elements"][element] * net[element]["scaling"].values.T
    q_mvar = np.sum(net[element][["q_a_mvar", "q_b_mvar", "q_c_mvar"]].values, axis=1)
    p_mw = np.sum(net[element][["p_a_mw", "p_b_mw", "p_c_mw"]].values, axis=1)
    return p_mw*scale, q_mvar*scale

def _get_motor_pq(net):
    tab = net["motor"]
    active = net._is_elements["motor"]
    scale = tab["loading_percent"].values/100 *tab["scaling"].values*active

    efficiency = tab["efficiency_percent"].values
    p_mech = tab["pn_mech_mw"].values
    cos_phi = tab["cos_phi"].values

    p_mw = p_mech / efficiency * 100 * scale
    s_mvar = p_mw / cos_phi
    q_mvar = np.sqrt(s_mvar**2 - p_mw**2)
    return p_mw, q_mvar

def _calc_shunts_and_add_on_ppc(net, ppc):
    # init values
    b, p, q = np.array([], dtype=int), np.array([]), np.array([])
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # get in service elements
    _is_elements = net["_is_elements"]

    s = net["shunt"]
    if len(s) > 0:
        vl = _is_elements["shunt"]
        v_ratio = (ppc["bus"][bus_lookup[s["bus"].values], BASE_KV] / s["vn_kv"].values) ** 2
        q = np.hstack([q, s["q_mvar"].values * s["step"].values * v_ratio * vl])
        p = np.hstack([p, s["p_mw"].values * s["step"].values * v_ratio * vl])
        b = np.hstack([b, s["bus"].values])

    w = net["ward"]
    if len(w) > 0:
        vl = _is_elements["ward"]
        q = np.hstack([q, w["qz_mvar"].values * vl])
        p = np.hstack([p, w["pz_mw"].values * vl])
        b = np.hstack([b, w["bus"].values])

    xw = net["xward"]
    if len(xw) > 0:
        vl = _is_elements["xward"]
        q = np.hstack([q, xw["qz_mvar"].values * vl])
        p = np.hstack([p, xw["pz_mw"].values * vl])
        b = np.hstack([b, xw["bus"].values])

    loss_location = net._options["trafo3w_losses"].lower()
    trafo3w = net["trafo3w"]
    if loss_location == "star" and len(trafo3w) > 0:
        pfe_mw = trafo3w["pfe_kw"].values * 1e-3
        i0 = trafo3w["i0_percent"].values
        sn_mva = trafo3w["sn_hv_mva"].values

        q_mvar = (sn_mva * i0 / 100.) ** 2 - pfe_mw ** 2
        q_mvar[q_mvar < 0] = 0
        q_mvar = np.sqrt(q_mvar)

        vn_hv_trafo = trafo3w["vn_hv_kv"].values
        vn_hv_bus = ppc["bus"][bus_lookup[trafo3w.hv_bus.values], BASE_KV]
        v_ratio = (vn_hv_bus / vn_hv_trafo) ** 2

        q = np.hstack([q, q_mvar * v_ratio])
        p = np.hstack([p, pfe_mw * v_ratio])
        b = np.hstack([b, net._pd2ppc_lookups["aux"]["trafo3w"]])

    # if array is not empty
    if b.size:
        b = bus_lookup[b]
        b, vp, vq = _sum_by_group(b, p, q)

        ppc["bus"][b, GS] = vp
        ppc["bus"][b, BS] = -vq

# Short circuit relevant routines
def _add_ext_grid_sc_impedance(net, ppc):
    from pandapower.shortcircuit.idx_bus import C_MAX, C_MIN
    mode = net._options["mode"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    if mode == "sc":
        case = net._options["case"]
    else:
        case = "max"
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
                          "external grid \n Try: net.ext_grid['s_sc_max_mva'] = 1000" )
    s_sc = eg["s_sc_%s_mva" % case].values/ppc['baseMVA']
    if not "rx_%s" % case in eg:
        raise ValueError("short circuit R/X rate rx_%s needs to be specified for external grid \n Try: net.ext_grid['rx_max'] = 0.1" %
                          case)
    rx = eg["rx_%s" % case].values

    z_grid = c / s_sc
    if mode == 'pf_3ph':
        z_grid = c / (s_sc/3)  # 3 phase power divided to get 1 ph power
    x_grid = z_grid / np.sqrt(rx ** 2 + 1)
    r_grid = rx * x_grid
    eg["r"] = r_grid
    eg["x"] = x_grid

    y_grid = 1 / (r_grid + x_grid * 1j)
    buses, gs, bs = _sum_by_group(eg_buses_ppc, y_grid.real, y_grid.imag)
    if mode == "sc":
        ppc["bus"][buses, GS] += gs * ppc['baseMVA']
        ppc["bus"][buses, BS] += bs * ppc['baseMVA']
    else:
        ppc["bus"][buses, GS] = gs * ppc['baseMVA']
        ppc["bus"][buses, BS] = bs * ppc['baseMVA']
    return gs * ppc['baseMVA'], bs * ppc['baseMVA']

def _add_motor_impedances_ppc(net, ppc):
    if net._options["case"] == "min":
        return
    motor = net["motor"][net._is_elements["motor"]]
    if motor.empty:
        return
    for par in ["vn_kv", "lrc_pu", "efficiency_n_percent", "cos_phi_n", "rx", "pn_mech_mw"]:
        if any(pd.isnull(motor[par])):
            raise UserWarning("%s needs to be specified for all motors in net.motor.%s" % (par, par))
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    motor_buses_ppc = bus_lookup[motor.bus.values]
    vn_net = ppc["bus"][motor_buses_ppc, BASE_KV]

    efficiency = motor.efficiency_n_percent.values
    cos_phi = motor.cos_phi_n.values
    p_mech = motor.pn_mech_mw.values
    vn_kv = motor.vn_kv.values
    lrc = motor.lrc_pu.values
    rx = motor.rx.values

    s_motor = p_mech / (efficiency/100 * cos_phi)
    z_motor_ohm = 1 / lrc * vn_kv**2 / s_motor
    z_motor_pu = z_motor_ohm / vn_net**2

    x_motor_pu = z_motor_pu / np.sqrt(rx ** 2 + 1)
    r_motor_pu = rx * x_motor_pu
    y_motor_pu = 1 / (r_motor_pu + x_motor_pu * 1j)

    buses, gs, bs = _sum_by_group(motor_buses_ppc, y_motor_pu.real, y_motor_pu.imag)
    ppc["bus"][buses, GS] += gs
    ppc["bus"][buses, BS] += bs


def _add_c_to_ppc(net, ppc):
    from pandapower.shortcircuit.idx_bus import C_MAX, C_MIN
    ppc["bus"][:, C_MAX] = 1.1
    ppc["bus"][:, C_MIN] = 1.
    lv_buses = np.where(ppc["bus"][:, BASE_KV] < 1.)
    if len(lv_buses) > 0:
        lv_tol_percent = net["_options"]["lv_tol_percent"]
        if lv_tol_percent == 10:
            c_ns = 1.1
        elif lv_tol_percent == 6:
            c_ns = 1.05
        else:
            raise ValueError("Voltage tolerance in the low voltage grid has" +
                             " to be either 6% or 10% according to IEC 60909")
        ppc["bus"][lv_buses, C_MAX] = c_ns
        ppc["bus"][lv_buses, C_MIN] = .95
