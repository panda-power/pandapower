# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
from numpy import complex128
from pandapower.pypower.idx_bus import VM, VA,BASE_KV
from pandapower.pypower.idx_gen import PG, QG, GEN_BUS

from pandapower.auxiliary import _sum_by_group, sequence_to_phase, _sum_by_group_nvals, \
    I_from_SV_elementwise, S_from_VI_elementwise, SVabc_from_SV012

from pandapower.auxiliary import _sum_by_group
from pandapower.pypower.idx_bus import VM, VA
from pandapower.pypower.idx_gen import PG, QG


def _get_gen_results(net, ppc, bus_lookup_aranged, pq_bus):
    ac = net["_options"]["ac"]

    eg_end = sum(net['ext_grid'].in_service)
    gen_end = eg_end + len(net['gen'])

    if eg_end > 0:
        b, p, q = _get_ext_grid_results(net, ppc)
    else:
        b, p, q = [], [], []  # np.array([]), np.array([]), np.array([])

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


def _get_gen_results_3ph(net, ppc0, ppc1, ppc2, bus_lookup_aranged, pq_bus):
    ac = net["_options"]["ac"]

    eg_end = len(net['ext_grid'])
    gen_end = eg_end + len(net['gen'][net['_is_elements']['gen']])

    b, pA, qA, pB, qB, pC, qC = _get_ext_grid_results_3ph(net, ppc0, ppc1, ppc2)

    # get results for gens
    if gen_end > eg_end:
        b, pA, qA, pB, qB, pC, qC = _get_pp_gen_results_3ph(net, ppc0, ppc1, ppc2, b, pA, qA, pB, qB, pC, qC)

    if len(net.dcline) > 0:
        _get_dcline_results(net)
        b = np.hstack([b, net.dcline[["from_bus", "to_bus"]].values.flatten()])
        pDC = -net.res_dcline[["p_from_mw", "p_to_mw"]].values.flatten() / 3
        pA = np.hstack([pA, pDC])
        pB = np.hstack([pB, pDC])
        pC = np.hstack([pC, pDC])
        qDC = -net.res_dcline[["q_from_mvar", "q_to_mvar"]].values.flatten() / 3
        qA = np.hstack([qA, qDC])
        qB = np.hstack([qB, qDC])
        qC = np.hstack([qC, qDC])

    if not ac:
        qA, qB, qC = np.copy((np.zeros(len(pA)),)*3)

    b_pp, pA_sum, qA_sum, pB_sum, qB_sum, pC_sum, qC_sum = _sum_by_group_nvals(b.astype(int), pA, qA, pB, qB, pC, qC)
    b_ppc = bus_lookup_aranged[b_pp]
    pq_bus[b_ppc, 0] += pA_sum
    pq_bus[b_ppc, 1] += qA_sum
    pq_bus[b_ppc, 2] += pB_sum
    pq_bus[b_ppc, 3] += qB_sum
    pq_bus[b_ppc, 4] += pC_sum
    pq_bus[b_ppc, 5] += qC_sum


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

def _get_ext_grid_results_3ph(net, ppc0, ppc1, ppc2):
    ac = net["_options"]["ac"]

    # get results for external grids
    eg_is_mask = net["_is_elements"]['ext_grid']
    ext_grid_lookup = net["_pd2ppc_lookups"]["ext_grid"]

    n_res_eg = len(net['ext_grid'])
    # indices of in service gens in the ppc
    eg_is_idx = net["ext_grid"].index.values[eg_is_mask]
    eg_idx_ppc = ext_grid_lookup[eg_is_idx]
    """ # 2 ext_grids Fix: Instead of the generator index, bus indices of the generators are used"""
    eg_bus_idx_ppc = np.real(ppc1["gen"][eg_idx_ppc, GEN_BUS]).astype(int)
    # read results from ppc for these buses
    V012 = np.array(np.zeros((3, n_res_eg)),dtype = np.complex128)
    V012[:, eg_is_idx] = np.array([ppc["bus"][eg_bus_idx_ppc, VM] * ppc["bus"][eg_bus_idx_ppc, BASE_KV]
                                      * np.exp(1j * np.deg2rad(ppc["bus"][eg_bus_idx_ppc, VA]))
                                      for ppc in [ppc0, ppc1, ppc2]])
    
    S012 = np.array(np.zeros((3, n_res_eg)),dtype = np.complex128)
    S012[:, eg_idx_ppc] = np.array([(ppc["gen"][eg_idx_ppc, PG] + 1j \
                                   * ppc["gen"][eg_idx_ppc, QG]) \
                                    for ppc in [ppc0, ppc1, ppc2]])

    Sabc, Vabc = SVabc_from_SV012(S012, V012/ np.sqrt(3), n_res=n_res_eg, idx=eg_idx_ppc)

    pA, pB, pC = map(lambda x: x.flatten(), np.real(Sabc))
    qA, qB, qC = map(lambda x: x.flatten(), np.imag(Sabc))

    # store result in net['res']
    net["res_ext_grid_3ph"]["p_a_mw"] = pA
    net["res_ext_grid_3ph"]["p_b_mw"] = pB
    net["res_ext_grid_3ph"]["p_c_mw"] = pC
    net["res_ext_grid_3ph"]["q_a_mvar"] = qA
    net["res_ext_grid_3ph"]["q_b_mvar"] = qB
    net["res_ext_grid_3ph"]["q_c_mvar"] = qC

    # get bus values for pq_bus
    b = net['ext_grid'].bus.values
    # copy index for results
    net["res_ext_grid_3ph"].index = net['ext_grid'].index

    return b, pA, qA, pB, qB, pC, qC


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

def _get_p_q_gen_results_3ph(net, ppc0, ppc1, ppc2):
    _is_elements = net["_is_elements"]
    ac = net["_options"]["ac"]
    gen_is_mask = _is_elements['gen']
    gen_lookup = net["_pd2ppc_lookups"]["gen"]
    gen_is_idx = net["gen"].index[gen_is_mask]
    # indices of in service gens in the ppc
    if np.any(_is_elements["gen"]):
        gen_idx_ppc = gen_lookup[gen_is_idx]
    else:
        gen_idx_ppc = []

    # read results from ppc for these buses
    n_res_gen = len(net['gen'])
    gen_idx_ppc = gen_lookup[gen_is_idx]
    """ # 2 ext_grids Fix: Instead of the generator index, bus indices of the generators are used"""
    gen_bus_idx_ppc = np.real(ppc1["gen"][gen_idx_ppc, GEN_BUS]).astype(int)

    V012 = np.array(np.zeros((3, n_res_gen)))
    V012[:, gen_is_idx] = np.array([ppc["bus"][gen_bus_idx_ppc, VM]
                                      * np.exp(1j * np.deg2rad(ppc["bus"][gen_bus_idx_ppc, VA]))
                                      for ppc in [ppc0, ppc1, ppc2]])

    S012 = np.array(np.zeros((3, n_res_gen)))
    S012[:, gen_is_idx] = np.array(
        [-(ppc["gen"][gen_idx_ppc, PG] + 1j * ppc["gen"][gen_idx_ppc, QG]) for ppc in [ppc0, ppc1, ppc2]])
    I012 = np.array(np.zeros((3, n_res_gen)))
    I012[:, gen_is_idx] = I_from_SV_elementwise(S012[:, gen_is_idx], V012[:, gen_is_idx])

    Vabc = sequence_to_phase(V012)
    Iabc = sequence_to_phase(I012)
    Sabc = S_from_VI_elementwise(Vabc, Iabc) * 1e3
    pA, pB, pC = map(lambda x: x.flatten(), np.real(Sabc))
    qA, qB, qC = map(lambda x: x.flatten(), np.imag(Sabc))

    net["res_gen_3ph"]["p_a_mw"] = pA
    net["res_gen_3ph"]["p_b_mw"] = pB
    net["res_gen_3ph"]["p_c_mw"] = pC
    net["res_gen_3ph"]["q_a_mvar"] = qA
    net["res_gen_3ph"]["q_b_mvar"] = qB
    net["res_gen_3ph"]["q_c_mvar"] = qC

    return pA, qA, pB, qB, pC, qC



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


def _get_v_gen_results_3ph(net, ppc0, ppc1, ppc2):
    # lookups for ppc
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    gen_lookup = net["_pd2ppc_lookups"]["gen"]
    
    # in service gens
    gen_is_mask = net["_is_elements"]['gen']
    gen_is_idx = net["gen"].index[gen_is_mask]
    bus_idx_ppc = bus_lookup[net["gen"]["bus"].values[gen_is_mask]]

    n_res_gen = len(net['gen'])
    gen_idx_ppc = gen_lookup[gen_is_idx]
    """ # 2 ext_grids Fix: Instead of the generator index, bus indices of the generators are used"""
    gen_bus_idx_ppc = np.real(ppc1["gen"][gen_idx_ppc, GEN_BUS]).astype(int)
    V012 = np.array(np.zeros((3, n_res_gen)))
    V012[:, gen_is_mask] = np.array([ppc["bus"][gen_bus_idx_ppc, VM]
                                      * np.exp(1j * np.deg2rad(ppc["bus"][gen_bus_idx_ppc, VA]))
                                      for ppc in [ppc0, ppc1, ppc2]])
    VABC = sequence_to_phase(V012)

    # voltage magnitudes
    vA_pu, vB_pu, vC_pu = np.copy((np.zeros(n_res_gen),) * 3)
    vA_pu[gen_idx_ppc] = np.abs(VABC[0, gen_idx_ppc])
    vB_pu[gen_idx_ppc] = np.abs(VABC[1, gen_idx_ppc])
    vC_pu[gen_idx_ppc] = np.abs(VABC[2, gen_idx_ppc])

    # voltage angles
    vA_a, vB_a, vC_a = np.copy((np.zeros(n_res_gen),) * 3)
    vA_a[gen_idx_ppc] = np.rad2deg(np.angle(VABC[0, gen_idx_ppc]))
    vB_a[gen_idx_ppc] = np.rad2deg(np.angle(VABC[1, gen_idx_ppc]))
    vC_a[gen_idx_ppc] = np.rad2deg(np.angle(VABC[2, gen_idx_ppc]))

    net["res_gen_3ph"]["vmA_pu"] = vA_pu
    net["res_gen_3ph"]["vmB_pu"] = vB_pu
    net["res_gen_3ph"]["vmC_pu"] = vC_pu
    net["res_gen_3ph"]["vaA_degree"] = vA_a
    net["res_gen_3ph"]["vaB_degree"] = vB_a
    net["res_gen_3ph"]["vaC_degree"] = vC_a
    return vA_pu, vA_a, vB_pu, vB_a, vC_pu, vC_a


def _get_pp_gen_results(net, ppc, b, p, q):
    p_gen, q_gen = _get_p_q_gen_results(net, ppc)
    _get_v_gen_resuts(net, ppc)

    b = np.hstack([b, net['gen'].bus.values])

    p = np.hstack([p, p_gen])
    if net["_options"]["ac"]:
        q = np.hstack([q, q_gen])

    return b, p, q

def _get_pp_gen_results_3ph(net, ppc0, ppc1, ppc2, b, pA, qA, pB, qB, pC, qC):
    pA_gen, qA_gen, pB_gen, qB_gen, pC_gen, qC_gen = _get_p_q_gen_results_3ph(net, ppc0, ppc1, ppc2)
    _get_v_gen_results_3ph(net, ppc0, ppc1, ppc2)

    ac = net["_options"]["ac"]

    net["res_gen_3ph"].index = net['gen'].index
    b = np.hstack([b, net['gen'].bus.values])

    pA = np.hstack([pA, pA_gen])
    pB = np.hstack([pB, pB_gen])
    pC = np.hstack([pC, pC_gen])
    if ac:
        qA = np.hstack([qA, qA_gen])
        qB = np.hstack([qB, qB_gen])
        qC = np.hstack([qC, qC_gen])

    return b, pA, qA, pB, qB, pC, qC

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
