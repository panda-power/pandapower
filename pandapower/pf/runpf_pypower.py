# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



"""Runs a power flow.
"""

from time import time
from packaging import version
from numpy import flatnonzero as find, r_, zeros, argmax, real, setdiff1d

from pandapower.pypower.idx_bus import PD, QD, BUS_TYPE, PQ, REF
from pandapower.pypower.idx_gen import PG, QG, QMAX, QMIN, GEN_BUS, GEN_STATUS
from pandapower.pypower.bustypes import bustypes
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.pfsoln import pfsoln
from pandapower.pf.run_newton_raphson_pf import _run_dc_pf
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci, _store_results_from_pf_in_ppci

from pandapower.pypower.makeB import makeB
from pandapower.pypower.ppoption import ppoption
from pandapower.pypower.fdpf import fdpf
from pandapower.pypower.gausspf import gausspf

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _runpf_pypower(ppci, options, **kwargs):
    """
    This is a modified version* of runpf() to run the algorithms gausspf and fdpf from PYPOWER.
    See PYPOWER documentation for more information.

    * mainly the verbose functions and ext2int() int2ext() were deleted
    """

    ##-----  run the power flow  -----
    t0 = time()
    # ToDo: Options should be extracted in every subfunction not here...
    init_va_degree, ac, numba, recycle, ppopt = _get_options(options, **kwargs)

    if ac:  # AC formulation
        if init_va_degree == "dc":
            ppci = _run_dc_pf(ppci)
            success = True

        ppci, success, bus, gen, branch, it = _ac_runpf(ppci, ppopt, numba, recycle)
    else:  ## DC formulation
        ppci = _run_dc_pf(ppci)
        success = True

    et = time() - t0
    ppci = _store_results_from_pf_in_ppci(ppci, bus, gen, branch, success, it, et)
    return ppci, success


def _get_options(options, **kwargs):
    init_va_degree = options["init_va_degree"]
    ac = options["ac"]
    recycle = options["recycle"]
    numba = options["numba"]
    enforce_q_lims = options["enforce_q_lims"]
    tolerance_mva = options["tolerance_mva"]
    algorithm = options["algorithm"]
    max_iteration = options["max_iteration"]

    # algorithms implemented within pypower
    algorithm_pypower_dict = {'nr': 1, 'fdbx': 2, 'fdxb': 3, 'gs': 4}

    ppopt = ppoption(ENFORCE_Q_LIMS=enforce_q_lims, PF_TOL=tolerance_mva,
                     PF_ALG=algorithm_pypower_dict[algorithm], **kwargs)
    ppopt['PF_MAX_IT'] = max_iteration
    ppopt['PF_MAX_IT_GS'] = max_iteration
    ppopt['PF_MAX_IT_FD'] = max_iteration
    ppopt['VERBOSE'] = 0
    return init_va_degree, ac, numba, recycle, ppopt


def _ac_runpf(ppci, ppopt, numba, recycle):
    numba, makeYbus = _import_numba_extensions_if_flag_is_true(numba)
    if ppopt["ENFORCE_Q_LIMS"]:
        return _run_ac_pf_with_qlims_enforced(ppci, recycle, makeYbus, ppopt)
    else:
        return _run_ac_pf_without_qlims_enforced(ppci, recycle, makeYbus, ppopt)


def _import_numba_extensions_if_flag_is_true(numba):
    ## check if numba is available and the corresponding flag
    if numba:
        try:
            from numba._version import version_version as nb_version
            # get numba Version (in order to use it it must be > 0.25)
            if version.parse(nb_version) < version.parse("0.25"):
                logger.warning('Warning: Numba version too old -> Upgrade to a version > 0.25. Numba is disabled\n')
                numba = False

        except ImportError:
            # raise UserWarning('numba cannot be imported. Call runpp() with numba=False!')
            logger.warning('Warning: Numba cannot be imported. Numba is disabled. Call runpp() with Numba=False!\n')
            numba = False

    if numba:
        from pandapower.pf.makeYbus_numba import makeYbus
    else:
        from pandapower.pypower.makeYbus import makeYbus

    return numba, makeYbus


def _get_Y_bus(ppci, recycle, makeYbus, baseMVA, bus, branch):
    if recycle is not None and not recycle["trafo"] and ppci["internal"]["Ybus"].size:
        Ybus, Yf, Yt = ppci["internal"]['Ybus'], ppci["internal"]['Yf'], ppci["internal"]['Yt']
    else:
        ## build admittance matrices
        Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
        ppci["internal"]['Ybus'], ppci["internal"]['Yf'], ppci["internal"]['Yt'] = Ybus, Yf, Yt

    return ppci, Ybus, Yf, Yt


def _run_ac_pf_without_qlims_enforced(ppci, recycle, makeYbus, ppopt):
    baseMVA, bus, gen, branch, ref, pv, pq, _, gbus, V0, ref_gens = _get_pf_variables_from_ppci(ppci)

    ppci, Ybus, Yf, Yt = _get_Y_bus(ppci, recycle, makeYbus, baseMVA, bus, branch)

    ## compute complex bus power injections [generation - load]
    Sbus = makeSbus(baseMVA, bus, gen)

    ## run the power flow
    V, success, it = _call_power_flow_function(baseMVA, bus, branch, Ybus, Sbus, V0, ref, pv, pq, ppopt)

    ## update data matrices with solution
    bus, gen, branch = pfsoln(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, ref_gens)

    return ppci, success, bus, gen, branch, it


def _run_ac_pf_with_qlims_enforced(ppci, recycle, makeYbus, ppopt):
    baseMVA, bus, gen, branch, ref, pv, pq, on, gbus, V0, _ = _get_pf_variables_from_ppci(ppci)

    qlim = ppopt["ENFORCE_Q_LIMS"]
    limited = []  ## list of indices of gens @ Q lims
    fixedQg = zeros(gen.shape[0])  ## Qg of gens at Q limits
    it = 0
    while True:
        ppci, success, bus, gen, branch, it_inner = _run_ac_pf_without_qlims_enforced(ppci, recycle,
                                                                                      makeYbus, ppopt)
        it += it_inner

        ## find gens with violated Q constraints
        gen_status = gen[:, GEN_STATUS] > 0
        qg_max_lim = gen[:, QG] > gen[:, QMAX]
        qg_min_lim = gen[:, QG] < gen[:, QMIN]

        non_refs = (gen[:, QMAX] != 0.) & (gen[:, QMIN] != 0.)
        mx = find(gen_status & qg_max_lim & non_refs)
        mn = find(gen_status & qg_min_lim & non_refs)


        if len(mx) > 0 or len(mn) > 0:  ## we have some Q limit violations
            ## one at a time?
            if qlim == 2:  ## fix largest violation, ignore the rest
                k = argmax(r_[gen[mx, QG] - gen[mx, QMAX],
                              gen[mn, QMIN] - gen[mn, QG]])
                if k > len(mx):
                    mn = mn[k - len(mx)]
                    mx = []
                else:
                    mx = mx[k]
                    mn = []

                    ## save corresponding limit values
            fixedQg[mx] = gen[mx, QMAX]
            fixedQg[mn] = gen[mn, QMIN]
            mx = r_[mx, mn].astype(int)

            ## convert to PQ bus
            gen[mx, QG] = fixedQg[mx]  ## set Qg to binding
            for i in mx:  ## [one at a time, since they may be at same bus]
                gen[i, GEN_STATUS] = 0  ## temporarily turn off gen,
                bi = gen[i, GEN_BUS].astype(int)  ## adjust load accordingly,
                bus[bi, [PD, QD]] = (bus[bi, [PD, QD]] - gen[i, [PG, QG]])

            if len(ref) > 1 and any(bus[gen[mx, GEN_BUS].astype(int), BUS_TYPE] == REF):
                raise ValueError('Sorry, pandapower cannot enforce Q '
                                 'limits for slack buses in systems '
                                 'with multiple slacks.')

            changed_gens = gen[mx, GEN_BUS].astype(int)
            bus[setdiff1d(changed_gens, ref), BUS_TYPE] = PQ  ## & set bus type to PQ

            ## update bus index lists of each type of bus
            ref, pv, pq = bustypes(bus, gen)

            limited = r_[limited, mx].astype(int)
        else:
            break  ## no more generator Q limits violated

    if len(limited) > 0:
        ## restore injections from limited gens [those at Q limits]
        gen[limited, QG] = fixedQg[limited]  ## restore Qg value,
        for i in limited:  ## [one at a time, since they may be at same bus]
            bi = gen[i, GEN_BUS].astype(int)  ## re-adjust load,
            bus[bi, [PD, QD]] = bus[bi, [PD, QD]] + gen[i, [PG, QG]]
            gen[i, GEN_STATUS] = 1  ## and turn gen back on

    return ppci, success, bus, gen, branch, it


def _call_power_flow_function(baseMVA, bus, branch, Ybus, Sbus, V0, ref, pv, pq, ppopt):
    alg = ppopt["PF_ALG"]
    # alg == 1 was deleted = nr -> moved as own pandapower solver
    if alg == 2 or alg == 3:
        Bp, Bpp = makeB(baseMVA, bus, real(branch), alg)
        V, success, it = fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, ppopt)
    elif alg == 4:
        V, success, it = gausspf(Ybus, Sbus, V0, ref, pv, pq, ppopt)
    else:
        raise ValueError('Only PYPOWERS fast-decoupled, and '
                         'Gauss-Seidel power flow algorithms currently '
                         'implemented.\n')

    return V, success, it
