import math

import numpy as np

from pandapower.auxiliary import _clean_up
from pandapower.converter import logger
from pandapower.pypower.idx_brch import PF, PT, QF, QT, BR_STATUS
from pandapower.pypower.idx_bus import VA, VM
from pandapower.pypower.idx_gen import PG, QG
from pandapower.results import _extract_results, _copy_results_ppci_to_ppc


def read_pm_results_to_net(net, ppc, ppci, result_pm):
    """
    reads power models results from result_pm to ppc / ppci and then to pandapower net
    """
    # read power models results from result_pm to result (== ppc with results)
    result, multinetwork = pm_results_to_ppc_results(net, ppc, ppci, result_pm)
    net._pm_result = result_pm
    success = ppc["success"]
    if success:
        if not multinetwork:
            # results are extracted from a single time step to pandapower dataframes
            _extract_results(net, result)
        _clean_up(net)
        net["OPF_converged"] = True
    else:
        _clean_up(net, res=False)
        logger.warning("OPF did not converge!")


def pm_results_to_ppc_results(net, ppc, ppci, result_pm):
    options = net._options
    # status if result is from multiple grids
    multinetwork = False
    sol = result_pm["solution"]
    ppci["obj"] = result_pm["objective"]
    termination_status = str(result_pm["termination_status"])
    ppci["success"] = "LOCALLY_SOLVED" in termination_status or "OPTIMAL" in termination_status
    ppci["et"] = result_pm["solve_time"]
    ppci["f"] = result_pm["objective"]

    if "multinetwork" in sol and sol["multinetwork"]:
        multinetwork = True
        ppc["obj"] = ppci["obj"]
        ppc["success"] = ppci["success"]
        ppc["et"] = ppci["et"]
        ppc["f"] = ppci["f"]
        return ppc, multinetwork

    for i, bus in sol["bus"].items():
        bus_idx = int(i) - 1
        ppci["bus"][bus_idx, VM] = bus["vm"]
        # replace nans with 0.(in case of SOCWR model for example
        ppci["bus"][bus_idx, VA] = 0.0 if bus["va"] == None else math.degrees(bus["va"])

    for i, gen in sol["gen"].items():
        gen_idx = int(i) - 1
        ppci["gen"][gen_idx, PG] = gen["pg"]
        ppci["gen"][gen_idx, QG] = gen["qg"]

    # read Q from branch results (if not DC calculation)
    if "branch" in sol:
        dc_results = sol["branch"]["1"]["qf"] is None or np.isnan(sol["branch"]["1"]["qf"])
        # read branch status results (OTS)
        branch_status = "br_status" in sol["branch"]["1"]
        for i, branch in sol["branch"].items():
            br_idx = int(i) - 1
            ppci["branch"][br_idx, PF] = branch["pf"]
            ppci["branch"][br_idx, PT] = branch["pt"]
            if not dc_results:
                ppci["branch"][br_idx, QF] = branch["qf"]
                ppci["branch"][br_idx, QT] = branch["qt"]
            if branch_status:
                ppci["branch"][br_idx, BR_STATUS] = branch["br_status"] > 0.5

    result = _copy_results_ppci_to_ppc(ppci, ppc, options["mode"])
    return result, multinetwork


def read_ots_results(net):
    """
    Reads the branch_status variable from ppc to pandapower net

    INPUT

        **net** - pandapower net
    """
    ppc = net._ppc
    for element, (f, t) in net._pd2ppc_lookups["branch"].items():
        # for trafo, line, trafo3w
        res = "res_" + element
        if "in_service" not in net[res]:
            # copy in service state from inputs
            net[res].loc[:, "in_service"] = None
            net[res].loc[:, "in_service"] = net[res].loc[:, "in_service"].values
        branch_status = ppc["branch"][f:t, BR_STATUS].real

        net[res]["in_service"].values[:] = branch_status


def read_tnep_results(net):
    ne_branch = net._pm_result["solution"]["ne_branch"]
    line_idx = net["res_ne_line"].index
    for pm_branch_idx, branch_data in ne_branch.items():
        # get pandapower index from power models index
        pp_idx = line_idx[int(pm_branch_idx) - 1]
        # built is a float, which is not exactly 1.0 or 0. sometimes
        net["res_ne_line"].loc[pp_idx, "built"] = branch_data["built"] > 0.5
