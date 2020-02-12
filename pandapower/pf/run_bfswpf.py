# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



from time import time  # alternatively use import timeit.default_timer as time

import numpy as np
import scipy as sp
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, BR_STATUS, SHIFT
from pandapower.pypower.idx_bus import BUS_I, BUS_TYPE, GS, BS
from pandapower.pypower.idx_gen import GEN_BUS, QG, QMAX, QMIN, GEN_STATUS, VG
from pandapower.pypower.makeSbus import makeSbus
from scipy.sparse import csr_matrix, csgraph
from six import iteritems

from pandapower.auxiliary import ppException
from pandapower.pypower.bustypes import bustypes
from pandapower.pypower.newtonpf import _evaluate_Fx, _check_for_convergence
from pandapower.pypower.pfsoln import pfsoln
from pandapower.pf.run_newton_raphson_pf import _get_Y_bus
from pandapower.pf.runpf_pypower import _import_numba_extensions_if_flag_is_true
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci


class LoadflowNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass


def _make_bibc_bcbv(bus, branch, graph):
    """
    performs depth-first-search bus ordering and creates Direct Load Flow (DLF) matrix
    which establishes direct relation between bus current injections and voltage drops from each bus to the root bus

    :param ppc: matpower-type case data
    :return: DLF matrix DLF = BIBC * BCBV where
                    BIBC - Bus Injection to Branch-Current
                    BCBV - Branch-Current to Bus-Voltage
            ppc with bfs ordering
            original bus names bfs ordered (used to convert voltage array back to normal)
    """

    nobus = bus.shape[0]
    nobranch = branch.shape[0]

    # reference bus is assumed as root bus for a radial network
    refs = bus[bus[:, BUS_TYPE] == 3, BUS_I]
    norefs = len(refs)

    G = graph.copy()  # network graph

    # dictionary with impedance values keyed by branch tuple (frombus, tobus)
    # TODO use list or array, not both
    branches_lst = list(zip(branch[:, F_BUS].real.astype(int), branch[:, T_BUS].real.astype(int)))
    branches_arr = branch[:, F_BUS:T_BUS + 1].real.astype(int)
    branches_ind_dict = dict(zip(zip(branches_arr[:, 0], branches_arr[:, 1]), range(0, nobranch)))
    branches_ind_dict.update(dict(zip(zip(branches_arr[:, 1], branches_arr[:, 0]), range(0, nobranch))))

    tap = branch[:, TAP]  # * np.exp(1j * np.pi / 180 * branch[:, SHIFT])
    z_ser = (branch[:, BR_R].real + 1j * branch[:, BR_X].real) * tap  # series impedance
    z_brch_dict = dict(zip(branches_lst, z_ser))

    # initialization of lists for building sparse BIBC and BCBV matrices
    rowi_BIBC = []
    coli_BIBC = []
    data_BIBC = []
    data_BCBV = []

    buses_ordered_bfs_nets = []
    for ref in refs:
        # ordering buses according to breadth-first-search (bfs)
        buses_ordered_bfs, predecs_bfs = csgraph.breadth_first_order(G, ref, directed=False, return_predecessors=True)
        buses_ordered_bfs_nets.append(buses_ordered_bfs)
        branches_ordered_bfs = list(zip(predecs_bfs[buses_ordered_bfs[1:]], buses_ordered_bfs[1:]))
        G_tree = csgraph.breadth_first_tree(G, ref, directed=False)

        # if multiple networks get subnetwork branches
        if norefs > 1:
            branches_sub_mask = (np.in1d(branches_arr[:, 0], buses_ordered_bfs) &
                                 np.in1d(branches_arr[:, 1], buses_ordered_bfs))
            branches = np.sort(branches_arr[branches_sub_mask, :], axis=1)
        else:
            branches = np.sort(branches_arr, axis=1)

        # identify loops if graph is not a tree
        branches_loops = []
        if G_tree.nnz < branches.shape[0]:
            G_tree_nnzs = G_tree.nonzero()
            branches_tree = np.sort(np.array([G_tree_nnzs[0], G_tree_nnzs[1]]).T, axis=1)
            branches_loops = (set(zip(branches[:, 0], branches[:, 1])) -
                              set(zip(branches_tree[:, 0], branches_tree[:, 1])))

        # #------ building BIBC and BCBV martrices ------
        # branches in trees
        brchi = 0
        for brch in branches_ordered_bfs:
            tree_down, predecs = csgraph.breadth_first_order(G_tree, brch[1], directed=True, return_predecessors=True)
            if len(tree_down) == 1:  # If at leaf
                pass
            if brch in z_brch_dict:
                z_br = z_brch_dict[brch]
            else:
                z_br = z_brch_dict[brch[::-1]]
            rowi_BIBC += [branches_ind_dict[brch]] * len(tree_down)
            coli_BIBC += list(tree_down)
            data_BCBV += [z_br] * len(tree_down)
            data_BIBC += [1] * len(tree_down)

        # branches from loops
        for loop_i, brch_loop in enumerate(branches_loops):
            path_lens, path_preds = csgraph.shortest_path(G_tree, directed=False,
                                                          indices=brch_loop, return_predecessors=True)
            init, end = brch_loop
            loop = [end]
            while init != end:
                end = path_preds[0, end]
                loop.append(end)

            loop_size = len(loop)
            coli_BIBC += [nobus + loop_i] * loop_size
            for i in range(len(loop)):
                brch = (loop[i - 1], loop[i])
                if np.argwhere(buses_ordered_bfs == brch[0]) < np.argwhere(buses_ordered_bfs == brch[1]):
                    brch_direct = 1
                else:
                    brch_direct = -1
                data_BIBC.append(brch_direct)

                if brch in branches_ind_dict:
                    rowi_BIBC.append(branches_ind_dict[brch])
                else:
                    rowi_BIBC.append(branches_ind_dict[brch[::-1]])

                if brch in z_brch_dict:
                    data_BCBV.append(z_brch_dict[brch] * brch_direct)
                else:
                    data_BCBV.append(z_brch_dict[brch[::-1]] * brch_direct)

                brchi += 1

    # construction of the BIBC matrix
    # column indices correspond to buses: assuming root bus is always 0 after ordering indices are subtracted by 1
    BIBC = csr_matrix((data_BIBC, (rowi_BIBC, np.array(coli_BIBC) - norefs)),
                      shape=(nobranch, nobranch))
    BCBV = csr_matrix((data_BCBV, (rowi_BIBC, np.array(coli_BIBC) - norefs)),
                      shape=(nobranch, nobranch)).transpose()

    if BCBV.shape[0] > nobus - 1:  # if nbrch > nobus - 1 -> network has loops
        DLF_loop = BCBV * BIBC
        # DLF = [A  M.T ]
        #       [M  N   ]
        A = DLF_loop[0:nobus - 1, 0:nobus - 1]
        M = DLF_loop[nobus - 1:, 0:nobus - 1]
        N = DLF_loop[nobus - 1:, nobus - 1:].A
        # considering the fact that number of loops is relatively small, N matrix is expected to be small and dense
        # ...in that case dense version is more efficient, i.e. N is transformed to dense and
        # inverted using sp.linalg.inv(N)
        DLF = A - M.T * csr_matrix(sp.linalg.inv(N)) * M  # Kron's Reduction
    else:  # no loops -> radial network
        DLF = BCBV * BIBC

    return DLF, buses_ordered_bfs_nets


def _get_bibc_bcbv(ppci, options, bus, branch, graph):
    recycle = options["recycle"]

    if recycle is not None and recycle["bfsw"] and ppci["internal"]["DLF"].size:
        DLF, buses_ordered_bfs_nets = ppci["internal"]['DLF'], \
                                      ppci["internal"]['buses_ord_bfs_nets']
    else:
        ## build matrices
        DLF, buses_ordered_bfs_nets = _make_bibc_bcbv(bus, branch, graph)
        if recycle is not None and recycle["bfsw"]:
            ppci["internal"]['DLF'], \
            ppci["internal"]['buses_ord_bfs_nets'] = DLF, buses_ordered_bfs_nets

    return ppci, DLF, buses_ordered_bfs_nets


def _makeYsh_bfsw(bus, branch, baseMVA):
    # compute shunt admittance
    # if Psh is the real power consumed by the shunt at V = 1.0 p.u. and Qsh is the reactive power injected by
    # the shunt at V = 1.0 p.u. then Psh - j Qsh = V * conj(Ysh * V) = conj(Ysh) = Gs - j Bs,
    # vector of shunt admittances
    nobus = bus.shape[0]

    Ysh = (bus[:, GS] + 1j * bus[:, BS]) / baseMVA

    # Line charging susceptance BR_B is also added as shunt admittance:
    # summation of charging susceptances per each bus
    stat = branch[:, BR_STATUS]  ## ones at in-service branches
    Ys = stat / (branch[:, BR_R] + 1j * branch[:, BR_X])
    ysh = (- branch[:, BR_B].imag + 1j * (branch[:, BR_B].real)) / 2
    tap = branch[:, TAP]  # * np.exp(1j * np.pi / 180 * branch[:, SHIFT])

    ysh_f = Ys * (1 - tap) / (tap * np.conj(tap)) + ysh / (tap * np.conj(tap))
    ysh_t = Ys * (tap - 1) / tap + ysh

    Gch = (np.bincount(branch[:, F_BUS].real.astype(int), weights=ysh_f.real, minlength=nobus) +
           np.bincount(branch[:, T_BUS].real.astype(int), weights=ysh_t.real, minlength=nobus))
    Bch = (np.bincount(branch[:, F_BUS].real.astype(int), weights=ysh_f.imag, minlength=nobus) +
           np.bincount(branch[:, T_BUS].real.astype(int), weights=ysh_t.imag, minlength=nobus))

    Ysh += Gch + 1j * Bch

    return Ysh


def _bfswpf(DLF, bus, gen, branch, baseMVA, Ybus, Sbus, V0, ref, pv, pq, buses_ordered_bfs_nets,
            options, **kwargs):
    """
    distribution power flow solution according to [1]
    :param DLF: direct-Load-Flow matrix which relates bus current injections to voltage drops from the root bus

    :param bus: buses martix
    :param gen: generators matrix
    :param branch: branches matrix
    :param baseMVA:
    :param Ybus: bus admittance matrix
    :param Sbus: vector of power injections
    :param V0: initial voltage state vector
    :param ref: reference bus index
    :param pv: PV buses indices
    :param pq: PQ buses indices
    :param buses_ordered_bfs_nets: buses ordered according to breadth-first search

    :return: power flow result
    """
    enforce_q_lims = options["enforce_q_lims"]
    tolerance_mva = options["tolerance_mva"]
    max_iteration = options["max_iteration"]
    voltage_depend_loads = options["voltage_depend_loads"]
    # setting options
    max_it = max_iteration  # maximum iterations
    verbose = kwargs["VERBOSE"]  # verbose is set in run._runpppf() #

    # tolerance for the inner loop for PV nodes
    if 'tolerance_mva_pv' in kwargs:
        tol_mva_inner = kwargs['tolerance_mva_pv']
    else:
        tol_mva_inner = 1.e-2

    if 'max_iter_pv' in kwargs:
        max_iter_pv = kwargs['max_iter_pv']
    else:
        max_iter_pv = 20

    nobus = bus.shape[0]
    ngen = gen.shape[0]

    mask_root = ~ (bus[:, BUS_TYPE] == 3)  # mask for eliminating root bus
    norefs = len(ref)

    Ysh = _makeYsh_bfsw(bus, branch, baseMVA)

    # detect generators on PV buses which have status ON
    gen_pv = np.in1d(gen[:, GEN_BUS], pv) & (gen[:, GEN_STATUS] > 0)
    qg_lim = np.zeros(ngen, dtype=bool)  # initialize generators which violated Q limits

    Iinj = np.conj(Sbus / V0) - Ysh * V0   # Initial current injections

    # initiate reference voltage vector
    V_ref = np.ones(nobus, dtype=complex)
    for neti, buses_ordered_bfs in enumerate(buses_ordered_bfs_nets):
        V_ref[buses_ordered_bfs] *= V0[ref[neti]]
    V = V0.copy()

    n_iter = 0
    converged = 0
    if verbose:
        print(' -- AC Power Flow (Backward/Forward sweep)\n')

    while not converged and n_iter < max_it:
        n_iter_inner = 0
        n_iter += 1

        deltaV = DLF * Iinj[mask_root]
        V[mask_root] = V_ref[mask_root] + deltaV

        # ##
        # inner loop for considering PV buses
        # TODO improve PV buses inner loop
        inner_loop_converged = False
        while not inner_loop_converged and len(pv) > 0:

            pvi = pv - norefs  # internal PV buses indices, assuming reference node is always 0

            Vmis = (np.abs(gen[gen_pv, VG])) ** 2 - (np.abs(V[pv])) ** 2
            # TODO improve getting values from sparse DLF matrix - DLF[pvi, pvi] is unefficient
            dQ = (Vmis / (2 * DLF[pvi, pvi].A1.imag)).flatten()

            gen[gen_pv, QG] += dQ

            if enforce_q_lims:  # check Q violation limits
                ## find gens with violated Q constraints
                qg_max_lim = (gen[:, QG] > gen[:, QMAX]) & gen_pv
                qg_min_lim = (gen[:, QG] < gen[:, QMIN]) & gen_pv

                if qg_min_lim.any():
                    gen[qg_min_lim, QG] = gen[qg_min_lim, QMIN]
                    bus[gen[qg_min_lim, GEN_BUS].astype(int), BUS_TYPE] = 1  # convert to PQ bus

                if qg_max_lim.any():
                    gen[qg_max_lim, QG] = gen[qg_max_lim, QMAX]
                    bus[gen[qg_max_lim, GEN_BUS].astype(int), BUS_TYPE] = 1  # convert to PQ bus

                # TODO: correct: once all the PV buses are converted to PQ buses, conversion back to PV is not possible
                qg_lim_new = qg_min_lim | qg_max_lim
                if qg_lim_new.any():
                    pq2pv = (qg_lim != qg_lim_new) & qg_lim
                    # convert PQ to PV bus
                    if pq2pv.any():
                        bus[gen[qg_max_lim, GEN_BUS].astype(int), BUS_TYPE] = 2  # convert to PV bus

                    qg_lim = qg_lim_new.copy()
                    ref, pv, pq = bustypes(bus, gen)

            # avoid calling makeSbus, update only Sbus for pv nodes
            Sbus = (makeSbus(baseMVA, bus, gen, vm=abs(V))
                    if voltage_depend_loads else
                    makeSbus(baseMVA, bus, gen))
            Iinj = np.conj(Sbus / V) - Ysh * V
            deltaV = DLF * Iinj[mask_root]
            V[mask_root] = V_ref[mask_root] + deltaV

            if n_iter_inner > max_iter_pv:
                raise LoadflowNotConverged(" FBSW Power Flow did not converge - inner iterations for PV nodes "
                                           "reached maximum value of {0}!".format(max_iter_pv))

            n_iter_inner += 1

            if np.all(np.abs(dQ) < tol_mva_inner):  # inner loop termination criterion
                inner_loop_converged = True

        # testing termination criterion -
        if voltage_depend_loads:
            Sbus = makeSbus(baseMVA, bus, gen, vm=abs(V))
        F = _evaluate_Fx(Ybus, V, Sbus, pv, pq)
        # check tolerance
        converged = _check_for_convergence(F, tolerance_mva)

        if converged and verbose:
            print("\nFwd-back sweep power flow converged in "
                  "{0} iterations.\n".format(n_iter))

        # updating injected currents
        Iinj = np.conj(Sbus / V) - Ysh * V

    return V, converged


def _get_options(options):
    enforce_q_lims = options['enforce_q_lims']
    tolerance_mva = options['tolerance_mva']
    max_iteration = options['max_iteration']
    calculate_voltage_angles = options['calculate_voltage_angles']
    numba = options["numba"]

    return enforce_q_lims, tolerance_mva, max_iteration, calculate_voltage_angles, numba


def _run_bfswpf(ppci, options, **kwargs):
    """
    SPARSE version of distribution power flow solution according to [1]
    :References:
    [1] Jen-Hao Teng, "A Direct Approach for Distribution System Load Flow Solutions",
    IEEE Transactions on Power Delivery, vol. 18, no. 3, pp. 882-887, July 2003.

    :param ppci: matpower-style case data
    :param options: pf options
    :return: results (pypower style), success (flag about PF convergence)
    """
    time_start = time()  # starting pf calculation timing

    baseMVA, bus, gen, branch, ref, pv, pq, _, gbus, V0, ref_gens = _get_pf_variables_from_ppci(ppci)

    enforce_q_lims, tolerance_mva, max_iteration, calculate_voltage_angles, numba = _get_options(options)

    numba, makeYbus = _import_numba_extensions_if_flag_is_true(numba)

    nobus = bus.shape[0]
    nobranch = branch.shape[0]

    # generate Sbus
    Sbus = makeSbus(baseMVA, bus, gen)
    # generate results for original bus ordering
    # Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    ppci, Ybus, Yf, Yt = _get_Y_bus(ppci, options, makeYbus, baseMVA, bus, branch)

    # creating network graph from list of branches
    bus_from = branch[:, F_BUS].real.astype(int)
    bus_to = branch[:, T_BUS].real.astype(int)
    G = csr_matrix((np.ones(nobranch), (bus_from, bus_to)),
                   shape=(nobus, nobus))
    # create spanning trees using breadth-first-search
    # TODO add efficiency warning if a network is heavy-meshed
    G_trees = []
    for refbus in ref:
        G_trees.append(csgraph.breadth_first_tree(G, refbus, directed=False))

        # depth-first-search bus ordering and generating Direct Load Flow matrix DLF = BCBV * BIBC
        ppci, DLF, buses_ordered_bfs_nets = _get_bibc_bcbv(ppci, options, bus, branch, G)

    # if there are trafos with phase-shift calculate Ybus without phase-shift for bfswpf
    any_trafo_shift = (branch[:, SHIFT] != 0).any()
    if any_trafo_shift:
        branch_noshift = branch.copy()
        branch_noshift[:, SHIFT] = 0
        Ybus_noshift, Yf_noshift, _ = makeYbus(baseMVA, bus, branch_noshift)
    else:
        Ybus_noshift = Ybus.copy()

    # #-----  run the power flow  -----
    V_final, success = _bfswpf(DLF, bus, gen, branch, baseMVA, Ybus_noshift,
                               Sbus, V0, ref, pv, pq, buses_ordered_bfs_nets,
                               options, **kwargs)

    # if phase-shifting trafos are present adjust final state vector angles accordingly
    if calculate_voltage_angles and any_trafo_shift:
        brch_shift_mask = branch[:, SHIFT] != 0
        trafos_shift = dict(list(zip(list(zip(branch[brch_shift_mask, F_BUS].real.astype(int),
                                              branch[brch_shift_mask, T_BUS].real.astype(int))),
                                     branch[brch_shift_mask, SHIFT].real)))
        for trafo_ind, shift_degree in iteritems(trafos_shift):
            neti = 0
            # if multiple reference nodes, find in which network trafo is located
            if len(ref) > 0:
                for refbusi in range(len(ref)):
                    if trafo_ind[0] in buses_ordered_bfs_nets[refbusi]:
                        neti = refbusi
                        break
            G_tree = G_trees[neti]
            buses_ordered_bfs = buses_ordered_bfs_nets[neti]
            if (np.argwhere(buses_ordered_bfs == trafo_ind[0]) <
                    np.argwhere(buses_ordered_bfs == trafo_ind[1])):
                lv_bus = trafo_ind[1]
                shift_degree *= -1
            else:
                lv_bus = trafo_ind[0]

            buses_shifted_from_root = csgraph.breadth_first_order(G_tree, lv_bus,
                                                                  directed=True, return_predecessors=False)
            V_final[buses_shifted_from_root] *= np.exp(1j * np.pi / 180 * shift_degree)

    # #----- output results to ppc ------
    ppci["et"] = time() - time_start  # pf time end

    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V_final, ref, ref_gens)
    # bus, gen, branch = pfsoln_bfsw(baseMVA, bus, gen, branch, V_final, ref, pv, pq, BIBC, ysh_f,ysh_t,Iinj, Sbus)

    ppci["success"] = success

    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch

    return ppci, success
