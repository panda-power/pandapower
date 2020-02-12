# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from pandapower.pypower.idx_bus import VM, VA
from pandapower.pypower.idx_gen import PG, QG

from pandapower.auxiliary import _sum_by_group


def _get_gen_results(net, ppc, bus_lookup_aranged, pq_bus):
    ac = net["_options"]["ac"]

    eg_end = len(net['ext_grid'])
    gen_end = eg_end + len(net['gen'])

    if eg_end > 0:
        b, p, q = _get_ext_grid_results(net, ppc)
    else:
        b, p, q = [], [], []# np.array([]), np.array([]), np.array([])

    # get results for gens
    if gen_end > eg_end:
        b, p, q = _get_pp_gen_results(net, ppc, b, p, q)

    if len(net.dcline) > 0:
        _get_dcline_results(net)
        b = np.hstack([b, net.dcline[["from_bus", "to_bus"]].values.flatten()])
        p = np.hstack([p, net.res_dcline[["p_from_mw", "p_to_mw"]].values.flatten()])
        q = np.hstack([q, net.res_dcline[["q_from_mvar", "q_to_mvar"]].values.flatten()])

    if not ac:
        q = np.zeros(len(p))
    b_sum, p_sum, q_sum = _sum_by_group(b, p, q)
    b = bus_lookup_aranged[b_sum.astype(int)]
    pq_bus[b, 0] -= p_sum
    pq_bus[b, 1] -= q_sum


def _get_ext_grid_results(net, ppc):
    ac = net["_options"]["ac"]

    # get results for external grids
    eg_is_mask = net["_is_elements"]['ext_grid']
    ext_grid_lookup = net["_pd2ppc_lookups"]["ext_grid"]

    n_res_eg = len(net['ext_grid'])
    # indices of in service gens in the ppc
    eg_is_idx = net["ext_grid"].index.values[eg_is_mask]
    gen_idx_ppc = ext_grid_lookup[eg_is_idx]

    # read results from ppc for these buses
    p = np.zeros(n_res_eg)
    q = np.zeros(n_res_eg)
    p[eg_is_mask] = ppc["gen"][gen_idx_ppc, PG]
    # store result in net['res']
    net["res_ext_grid"]["p_mw"] = p

    # if ac PF q results are also available
    if ac:
        q[eg_is_mask] = ppc["gen"][gen_idx_ppc, QG]
        net["res_ext_grid"]["q_mvar"] = q

    # get bus values for pq_bus
    b = net['ext_grid'].bus.values
    # copy index for results
    net["res_ext_grid"].index = net['ext_grid'].index

    return b, p, q


def _get_p_q_gen_results(net, ppc):
    gen_is = net["_is_elements"]["gen"]
    gen_lookup = net["_pd2ppc_lookups"]["gen"]
    gen_is_idx = np.array(net["gen"].index)[gen_is]
    # indices of in service gens in the ppc
    if np.any(gen_is):
        gen_idx_ppc = gen_lookup[gen_is_idx]
    else:
        gen_idx_ppc = []

    # read results from ppc for these buses
    n_res_gen = len(net['gen'])
    p_gen = np.zeros(n_res_gen)
    p_gen[gen_is] = ppc["gen"][gen_idx_ppc, PG]
    q_gen = None
    if net["_options"]["ac"]:
        q_gen = np.zeros(n_res_gen)
        q_gen[gen_is] = ppc["gen"][gen_idx_ppc, QG]
        net["res_gen"]["q_mvar"].values[:] = q_gen

    net["res_gen"]["p_mw"].values[:] = p_gen
    return p_gen, q_gen


def _get_v_gen_resuts(net, ppc):
    # lookups for ppc
    bus_lookup = net["_pd2ppc_lookups"]["bus"]

    # in service gens
    gen_is = net["_is_elements"]['gen']
    bus_idx_ppc = bus_lookup[net["gen"]["bus"].values[gen_is]]

    n_res_gen = len(net['gen'])

    # voltage magnitudes
    v_pu = np.zeros(n_res_gen)
    v_pu[gen_is] = ppc["bus"][bus_idx_ppc][:, VM]

    # voltage angles
    v_a = np.zeros(n_res_gen)
    v_a[gen_is] = ppc["bus"][bus_idx_ppc][:, VA]

    net["res_gen"]["vm_pu"].values[:] = v_pu
    net["res_gen"]["va_degree"].values[:] = v_a
    return v_pu, v_a


def _get_pp_gen_results(net, ppc, b, p, q):
    p_gen, q_gen = _get_p_q_gen_results(net, ppc)
    _get_v_gen_resuts(net, ppc)

    b = np.hstack([b, net['gen'].bus.values])

    p = np.hstack([p, p_gen])
    if net["_options"]["ac"]:
        q = np.hstack([q, q_gen])

    return b, p, q


def _get_dcline_results(net):
    dc_gens = net.gen.index[(len(net.gen) - len(net.dcline) * 2):]
    from_gens = net.res_gen.loc[dc_gens[1::2]]
    to_gens = net.res_gen.loc[dc_gens[::2]]

    net.res_dcline.p_from_mw = - from_gens.p_mw.values
    net.res_dcline.p_to_mw = - to_gens.p_mw.values
    net.res_dcline.pl_mw = - (to_gens.p_mw.values + from_gens.p_mw.values)

    net.res_dcline.q_from_mvar = - from_gens.q_mvar.values
    net.res_dcline.q_to_mvar = - to_gens.q_mvar.values

    net.res_dcline.vm_from_pu = from_gens.vm_pu.values
    net.res_dcline.vm_to_pu = to_gens.vm_pu.values
    net.res_dcline.va_from_degree = from_gens.va_degree.values
    net.res_dcline.va_to_degree = to_gens.va_degree.values

    net.res_dcline.index = net.dcline.index
