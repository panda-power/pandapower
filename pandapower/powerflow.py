# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from numpy import nan_to_num

from pandapower.auxiliary import ppException, _clean_up, _add_auxiliary_elements
from pandapower.build_branch import _calc_trafo_parameter, _calc_trafo3w_parameter
from pandapower.build_gen import _build_gen_ppc
from pandapower.pd2ppc import _pd2ppc, _calc_pq_elements_and_add_on_ppc, _ppc2ppci
from pandapower.pf.ppci_variables import _get_pf_variables_from_ppci
from pandapower.pf.run_bfswpf import _run_bfswpf
from pandapower.pf.run_dc_pf import _run_dc_pf
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.pf.runpf_pypower import _runpf_pypower
from pandapower.pypower.idx_bus import VM
from pandapower.pypower.makeYbus import makeYbus as makeYbus_pypower
from pandapower.pypower.pfsoln import pfsoln as pfsoln_pypower
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc, reset_results, verify_results, \
    _ppci_bus_to_ppc, _ppci_other_to_ppc


class AlgorithmUnknown(ppException):
    """
    Exception being raised in case optimal powerflow did not converge.
    """
    pass


class LoadflowNotConverged(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass


def _powerflow(net, **kwargs):
    """
    Gets called by runpp or rundcpp with different arguments.
    """

    # get infos from options
    init_results = net["_options"]["init_results"]
    ac = net["_options"]["ac"]
    algorithm = net["_options"]["algorithm"]

    net["converged"] = False
    net["OPF_converged"] = False
    _add_auxiliary_elements(net)

    if not ac or init_results:
        verify_results(net)
    else:
        reset_results(net)

    # TODO remove this when zip loads are integrated for all PF algorithms
    if algorithm not in ['nr', 'bfsw']:
        net["_options"]["voltage_depend_loads"] = False

    # convert pandapower net to ppc
    ppc, ppci = _pd2ppc(net)

    # store variables
    net["_ppc"] = ppc

    if not "VERBOSE" in kwargs:
        kwargs["VERBOSE"] = 0

    # ----- run the powerflow -----
    result = _run_pf_algorithm(ppci, net["_options"], **kwargs)
    # read the results (=ppci with results) to net
    _ppci_to_net(result, net)


def _recycled_powerflow(net, **kwargs):
    options = net["_options"]
    options["recycle"] = kwargs.get("recycle", None)
    options["init_vm_pu"] = "results"
    options["init_va_degree"] = "results"
    algorithm = options["algorithm"]
    ac = options["ac"]
    ppci = {"bus": net["_ppc"]["internal"]["bus"],
            "gen": net["_ppc"]["internal"]["gen"],
            "branch": net["_ppc"]["internal"]["branch"],
            "baseMVA": net["_ppc"]["internal"]["baseMVA"],
            "internal": net["_ppc"]["internal"],
            }
    if not ac:
        # DC recycle
        result = _run_dc_pf(ppci)
        _ppci_to_net(result, net)
        return
    if not algorithm in ['nr', 'iwamoto_nr'] and ac:
        raise ValueError("recycle is only available with Newton-Raphson power flow. Choose algorithm='nr'")

    recycle = options["recycle"]
    ppc = net["_ppc"]
    if "bus_pq" in recycle and recycle["bus_pq"]:
        # update pq values in bus
        _calc_pq_elements_and_add_on_ppc(net, ppc)

    if "trafo" in recycle and recycle["trafo"]:
        # update trafo in branch and Ybus
        lookup = net._pd2ppc_lookups["branch"]
        if "trafo" in lookup:
            _calc_trafo_parameter(net, ppc)
        if "trafo3w" in lookup:
            _calc_trafo3w_parameter(net, ppc)

    if "gen" in recycle and recycle["gen"]:
        # updates the ppc["gen"] part
        _build_gen_ppc(net, ppc)
        ppc["gen"] = nan_to_num(ppc["gen"])

    ppci = _ppc2ppci(ppc, net, ppci=ppci)
    ppci["internal"] = net["_ppc"]["internal"]
    net["_ppc"] = ppc

    # run the Newton-Raphson power flow
    result = _run_newton_raphson_pf(ppci, options)
    if options["only_v_results"]:
        _ppci_bus_to_ppc(result, ppc)
        _ppci_other_to_ppc(result, ppc, options["mode"])
        return
    # read the results from  result (==ppci) to net
    _ppci_to_net(result, net)


def _run_pf_algorithm(ppci, options, **kwargs):
    algorithm = options["algorithm"]
    ac = options["ac"]

    if ac:
        # ----- run the powerflow -----
        if ppci["branch"].shape[0] == 0:
            result = _pf_without_branches(ppci, options)
        elif algorithm == 'bfsw':  # forward/backward sweep power flow algorithm
            result = _run_bfswpf(ppci, options, **kwargs)[0]
        elif algorithm in ['nr', 'iwamoto_nr']:
            result = _run_newton_raphson_pf(ppci, options)
        elif algorithm in ['fdbx', 'fdxb', 'gs']:  # algorithms existing within pypower
            result = _runpf_pypower(ppci, options, **kwargs)[0]
        else:
            raise AlgorithmUnknown("Algorithm {0} is unknown!".format(algorithm))
    else:
        result = _run_dc_pf(ppci)

    return result


def _ppci_to_net(result, net):
    # reads the results from result (== ppci with results) to pandapower net

    mode = net["_options"]["mode"]
    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    ppc = net["_ppc"]
    result = _copy_results_ppci_to_ppc(result, ppc, mode)

    # raise if PF was not successful. If DC -> success is always 1
    if result["success"] != 1:
        _clean_up(net, res=False)
        algorithm = net["_options"]["algorithm"]
        max_iteration = net["_options"]["max_iteration"]
        raise LoadflowNotConverged("Power Flow {0} did not converge after "
                                   "{1} iterations!".format(algorithm, max_iteration))
    else:
        net["_ppc"] = result
        net["converged"] = True

    _extract_results(net, result)
    _clean_up(net)


def _pf_without_branches(ppci, options):
    Ybus, Yf, Yt = makeYbus_pypower(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    baseMVA, bus, gen, branch, ref, _, pq, _, _, V0, ref_gens = _get_pf_variables_from_ppci(ppci)
    V = ppci["bus"][:, VM]
    bus, gen, branch = pfsoln_pypower(baseMVA, bus, gen, branch, Ybus, Yf, Yt, V, ref, ref_gens)
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    ppci["success"] = True
    ppci["iterations"] = 1
    ppci["et"] = 0
    return ppci
