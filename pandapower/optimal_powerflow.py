# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import warnings
from sys import stdout

from pandapower.pypower.add_userfcn import add_userfcn
from pandapower.pypower.ppoption import ppoption
from scipy.sparse import csr_matrix as sparse

from pandapower.auxiliary import ppException, _clean_up, _add_auxiliary_elements
from pandapower.pypower.idx_bus import VM
from pandapower.pypower.opf import opf
from pandapower.pypower.printpf import printpf
from pandapower.pd2ppc import _pd2ppc
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.results import _copy_results_ppci_to_ppc, reset_results, \
    _extract_results


class OPFNotConverged(ppException):
    """
    Exception being raised in case optimal powerflow did not converge.
    """
    pass


def _optimal_powerflow(net, verbose, suppress_warnings, **kwargs):
    ac = net["_options"]["ac"]
    init = net["_options"]["init"]

    ppopt = ppoption(VERBOSE=verbose, OPF_FLOW_LIM=2, PF_DC=not ac, INIT=init, **kwargs)
    net["OPF_converged"] = False
    net["converged"] = False
    _add_auxiliary_elements(net)
    reset_results(net)

    ppc, ppci = _pd2ppc(net)

    if not ac:
        ppci["bus"][:, VM] = 1.0
    net["_ppc_opf"] = ppci
    if len(net.dcline) > 0:
        ppci = add_userfcn(ppci, 'formulation', _add_dcline_constraints, args=net)

    if init == "pf":
        ppci = _run_pf_before_opf(net, ppci)
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = opf(ppci, ppopt)
    else:
        result = opf(ppci, ppopt)
#    net["_ppc_opf"] = result

    if verbose:
        ppopt['OUT_ALL'] = 1
        printpf(baseMVA=result["baseMVA"], bus=result["bus"], gen=result["gen"], fd=stdout,
                branch=result["branch"],  success=result["success"], et=result["et"], ppopt=ppopt)

    if verbose:
        ppopt['OUT_ALL'] = 1
        printpf(baseMVA=result["baseMVA"], bus=result["bus"], gen=result["gen"], fd=stdout,
                branch=result["branch"],  success=result["success"], et=result["et"], ppopt=ppopt)

    if not result["success"]:
        raise OPFNotConverged("Optimal Power Flow did not converge!")

    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    mode = net["_options"]["mode"]
    result = _copy_results_ppci_to_ppc(result, ppc, mode=mode)

#    net["_ppc_opf"] = result
    net["OPF_converged"] = True
    _extract_results(net, result)
    _clean_up(net)


def _add_dcline_constraints(om, net):
    # from numpy import hstack, diag, eye, zeros
    ppc = om.get_ppc()
    ndc = net.dcline.in_service.sum()  ## number of in-service DC lines
    if ndc > 0:
        ng = ppc['gen'].shape[0]  ## number of total gens
        Adc = sparse((ndc, ng))
        gen_lookup = net._pd2ppc_lookups["gen"]

        dcline_gens_from = net.gen.index[-2 * ndc::2]
        dcline_gens_to = net.gen.index[-2 * ndc + 1::2]
        for i, (f, t, loss, active) in enumerate(zip(dcline_gens_from, dcline_gens_to,
                                                     net.dcline.loss_percent.values,
                                                     net.dcline.in_service.values)):
            if active:
                Adc[i, gen_lookup[f]] = 1. + loss / 100
                Adc[i, gen_lookup[t]] = 1.

        ## constraints
        nL0 = -net.dcline.loss_mw.values # absolute losses
        #    L1  = -net.dcline.loss_percent.values * 1e-2 #relative losses
        #    Adc = sparse(hstack([zeros((ndc, ng)), diag(1-L1), eye(ndc)]))

        ## add them to the model
        om = om.add_constraints('dcline', Adc, nL0, nL0, ['Pg'])


def _run_pf_before_opf(net, ppci):
#    net._options["numba"] = True
    net._options["tolerance_mva"] = 1e-8
    net._options["max_iteration"] = 10
    net._options["algorithm"] = "nr"
    return _run_newton_raphson_pf(ppci, net["_options"])
