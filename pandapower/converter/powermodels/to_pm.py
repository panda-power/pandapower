import json
import math
import os
import tempfile
from os import remove
from os.path import isfile

import numpy as np
import pandas as pd

from pandapower.auxiliary import _add_ppc_options, _add_opf_options, _add_auxiliary_elements
from pandapower.build_branch import _calc_line_parameter
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.idx_brch import ANGMIN, ANGMAX, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, TAP, SHIFT, \
    branch_cols, F_BUS, T_BUS, BR_STATUS
from pandapower.pypower.idx_bus import ZONE, VA, BASE_KV, BS, GS, BUS_I, BUS_TYPE, VMAX, VMIN, VM, PD, QD
from pandapower.pypower.idx_cost import MODEL, NCOST, COST
from pandapower.pypower.idx_gen import PG, QG, GEN_BUS, VG, GEN_STATUS, QMAX, QMIN, PMIN, PMAX
from pandapower.results import init_results

# const value in branch for tnep
CONSTRUCTION_COST = 23
try:
    import pplog as logging
except ImportError:
    import logging


def convert_pp_to_pm(net, pm_file_path=None, correct_pm_network_data=True, calculate_voltage_angles=True, ac=True,
                     trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                     pp_to_pm_callback=None, pm_model="ACPPowerModel", pm_solver="ipopt",
                     pm_mip_solver="cbc", pm_nl_solver="ipopt"):
    """
    Converts a pandapower net to a PowerModels.jl datastructure and saves it to a json file

    INPUT:

    **net** - pandapower net

    OPTIONAL:
    **pm_file_path** (str, None) - file path to *.json file to store pm data to

    **correct_pm_network_data** (bool, True) - correct some input data (e.g. angles, p.u. conversion)

    **delta** (float, 1e-8) - (small) offset to set for "hard" OPF limits.

    **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

    **pm_model** (str, "ACPPowerModel") - model to use. Default is AC model

    **pm_solver** (str, "ipopt") - default solver to use.

    **pm_nl_solver** (str, "ipopt") - default nonlinear solver to use.

    **pm_mip_solver** (str, "cbc") - default mip solver to use.

    **correct_pm_network_data** (bool, True) - checks if network data is correct. If not tries to correct it

    Returns
    -------
    **pm** (json str) - PowerModels.jl data structure
    """

    net._options = {}

    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, pm_solver=pm_solver, pm_model=pm_model,
                     correct_pm_network_data=correct_pm_network_data, pm_mip_solver=pm_mip_solver,
                     pm_nl_solver=pm_nl_solver)

    net, pm, ppc, ppci = convert_to_pm_structure(net)
    buffer_file = dump_pm_json(pm, pm_file_path)
    if pm_file_path is None and isfile(buffer_file):
        remove(buffer_file)
    return pm


logger = logging.getLogger(__name__)


def convert_to_pm_structure(net):
    net["OPF_converged"] = False
    net["converged"] = False
    _add_auxiliary_elements(net)
    init_results(net)
    ppc, ppci = _pd2ppc(net)
    ppci = build_ne_branch(net, ppci)
    net["_ppc_opf"] = ppci
    pm = ppc_to_pm(net, ppci)
    pm = add_pm_options(pm, net)
    net._pm = pm
    return net, pm, ppc, ppci


def dump_pm_json(pm, buffer_file=None):
    # dump pm dict to buffer_file (*.json)
    if buffer_file is None:
        # if no buffer file is provided a random file name is generated
        temp_name = next(tempfile._get_candidate_names())
        buffer_file = os.path.join(tempfile.gettempdir(), "pp_to_pm_" + temp_name + ".json")
    logger.debug("writing PowerModels data structure to %s" % buffer_file)

    with open(buffer_file, 'w') as outfile:
        json.dump(pm, outfile)
    return buffer_file


def _pp_element_to_pm(net, pm, element, pd_bus, qd_bus, load_idx):
    bus_lookup = net._pd2ppc_lookups["bus"]

    for idx in net[element].index:
        if "controllable" in net[element] and net[element].at[idx, "controllable"]:
            continue

        pp_bus = net[element].at[idx, "bus"]
        pm_bus = bus_lookup[pp_bus] + 1

        scaling = net[element].at[idx, "scaling"]
        if element == "sgen":
            pd = -net[element].at[idx, "p_mw"] * scaling
            qd = -net[element].at[idx, "q_mvar"] * scaling
        else:
            pd = net[element].at[idx, "p_mw"] * scaling
            qd = net[element].at[idx, "q_mvar"] * scaling
        in_service = net[element].at[idx, "in_service"]

        pm["load"][str(load_idx)] = {"pd": pd.item(), "qd": qd.item(), "load_bus": pm_bus.item(),
                                     "status": int(in_service), "index": load_idx}
        if pm_bus not in pd_bus:
            pd_bus[pm_bus] = pd
            qd_bus[pm_bus] = qd
        else:
            pd_bus[pm_bus] += pd
            qd_bus[pm_bus] += qd

        load_idx += 1
    return load_idx


def get_branch_angles(row, correct_pm_network_data):
    angmin = row[ANGMIN].real
    angmax = row[ANGMAX].real
    # check if angles are too small for PowerModels OPF (recommendation from Carleton Coffrin himself)
    if correct_pm_network_data:
        if angmin < -60.:
            logger.debug("changed voltage angle minimum of branch {}, "
                         "to -60 from {} degrees".format(int(row[0].real), angmin))
            angmin = -60.
        if angmax > 60.:
            logger.debug("changed voltage angle maximum of branch {} to 60. "
                         "from {} degrees".format(int(row[0].real), angmax))
            angmax = 60.
    # convert to rad
    angmin = math.radians(angmin)
    angmax = math.radians(angmax)
    return angmin, angmax


def ppc_to_pm(net, ppci):
    # create power models dict. Similar to matpower case file. ne_branch is for a tnep case
    pm = {"gen": dict(), "branch": dict(), "bus": dict(), "dcline": dict(), "load": dict(), "storage": dict(),
          "ne_branch": dict(), "switch": dict(),
          "baseMVA": ppci["baseMVA"], "source_version": "2.0.0", "shunt": dict(),
          "sourcetype": "matpower", "per_unit": True, "name": net.name}
    load_idx = 1
    shunt_idx = 1
    # PowerModels has a load model -> add loads and sgens to pm["load"]

    # temp dicts which hold the sum of p, q of loads + sgens
    pd_bus = dict()
    qd_bus = dict()
    load_idx = _pp_element_to_pm(net, pm, "load", pd_bus, qd_bus, load_idx)
    load_idx = _pp_element_to_pm(net, pm, "sgen", pd_bus, qd_bus, load_idx)
    load_idx = _pp_element_to_pm(net, pm, "storage", pd_bus, qd_bus, load_idx)
    correct_pm_network_data = net._options["correct_pm_network_data"]

    for row in ppci["bus"]:
        bus = dict()
        idx = int(row[BUS_I]) + 1
        bus["index"] = idx
        bus["bus_i"] = idx
        bus["zone"] = int(row[ZONE])
        bus["bus_type"] = int(row[BUS_TYPE])
        bus["vmax"] = row[VMAX]
        bus["vmin"] = row[VMIN]
        bus["va"] = row[VA]
        bus["vm"] = row[VM]
        bus["base_kv"] = row[BASE_KV]

        pd = row[PD]
        qd = row[QD]

        # pd and qd are the PQ values in the ppci, if they are equal to the sum in load data is consistent
        if idx in pd_bus:
            pd -= pd_bus[idx]
            qd -= qd_bus[idx]
        # if not we have to add more loads wit the remaining value
        pq_mismatch = not np.allclose(pd, 0.) or not np.allclose(qd, 0.)
        if pq_mismatch:
            # This will be called if ppc PQ != sum at bus.
            logger.info("PQ mismatch. Adding another load at idx {}".format(load_idx))
            pm["load"][str(load_idx)] = {"pd": pd, "qd": qd, "load_bus": idx,
                                         "status": True, "index": load_idx}
            load_idx += 1
        # if bs or gs != 0. -> shunt element at this bus
        bs = row[BS]
        gs = row[GS]
        if not np.allclose(bs, 0.) or not np.allclose(gs, 0.):
            pm["shunt"][str(shunt_idx)] = {"gs": gs, "bs": bs, "shunt_bus": idx,
                                           "status": True, "index": shunt_idx}
            shunt_idx += 1
        pm["bus"][str(idx)] = bus

    n_lines = net.line.in_service.sum()
    for idx, row in enumerate(ppci["branch"], start=1):
        branch = dict()
        branch["index"] = idx
        branch["transformer"] = bool(idx > n_lines)
        branch["br_r"] = row[BR_R].real
        branch["br_x"] = row[BR_X].real
        branch["g_fr"] = - row[BR_B].imag / 2.0
        branch["g_to"] = - row[BR_B].imag / 2.0
        branch["b_fr"] = row[BR_B].real / 2.0
        branch["b_to"] = row[BR_B].real / 2.0
        branch["rate_a"] = row[RATE_A].real if row[RATE_A] > 0 else row[RATE_B].real
        branch["rate_b"] = row[RATE_B].real
        branch["rate_c"] = row[RATE_C].real
        branch["f_bus"] = int(row[F_BUS].real) + 1
        branch["t_bus"] = int(row[T_BUS].real) + 1
        branch["br_status"] = int(row[BR_STATUS].real)
        branch["angmin"], branch["angmax"] = get_branch_angles(row, correct_pm_network_data)
        branch["tap"] = row[TAP].real
        branch["shift"] = math.radians(row[SHIFT].real)
        pm["branch"][str(idx)] = branch

    for idx, row in enumerate(ppci["gen"], start=1):
        gen = dict()
        gen["pg"] = row[PG]
        gen["qg"] = row[QG]
        gen["gen_bus"] = int(row[GEN_BUS]) + 1
        gen["vg"] = row[VG]
        gen["qmax"] = row[QMAX]
        gen["gen_status"] = int(row[GEN_STATUS])
        gen["qmin"] = row[QMIN]
        gen["pmin"] = row[PMIN]
        gen["pmax"] = row[PMAX]
        gen["index"] = idx
        pm["gen"][str(idx)] = gen

    if "ne_branch" in ppci:
        for idx, row in enumerate(ppci["ne_branch"], start=1):
            branch = dict()
            branch["index"] = idx
            branch["transformer"] = False
            branch["br_r"] = row[BR_R].real
            branch["br_x"] = row[BR_X].real
            branch["g_fr"] = - row[BR_B].imag / 2.0
            branch["g_to"] = - row[BR_B].imag / 2.0
            branch["b_fr"] = row[BR_B].real / 2.0
            branch["b_to"] = row[BR_B].real / 2.0
            branch["rate_a"] = row[RATE_A].real if row[RATE_A] > 0 else row[RATE_B].real
            branch["rate_b"] = row[RATE_B].real
            branch["rate_c"] = row[RATE_C].real
            branch["f_bus"] = int(row[F_BUS].real) + 1
            branch["t_bus"] = int(row[T_BUS].real) + 1
            branch["br_status"] = int(row[BR_STATUS].real)
            branch["angmin"], branch["angmax"] = get_branch_angles(row, correct_pm_network_data)
            branch["tap"] = row[TAP].real
            branch["shift"] = math.radians(row[SHIFT].real)
            branch["construction_cost"] = row[CONSTRUCTION_COST].real
            pm["ne_branch"][str(idx)] = branch

    if len(ppci["gencost"]) > len(ppci["gen"]):
        logger.warning("PowerModels.jl does not consider reactive power cost - costs are ignored")
        ppci["gencost"] = ppci["gencost"][:ppci["gen"].shape[0], :]
    for idx, row in enumerate(ppci["gencost"], start=1):
        gen = pm["gen"][str(idx)]
        gen["model"] = int(row[MODEL])
        if gen["model"] == 1:
            gen["ncost"] = int(row[NCOST])
            gen["cost"] = row[COST:COST + gen["ncost"] * 2].tolist()
        elif gen["model"] == 2:
            gen["ncost"] = 3
            gen["cost"] = [0] * 3
            costs = row[COST:]
            if len(costs) > 3:
                logger.info(costs)
                raise ValueError("Maximum quadratic cost function allowed")
            gen["cost"][-len(costs):] = costs
    return pm


def add_pm_options(pm, net):
    # read values from net_options if present else use default values
    pm["pm_solver"] = net._options["pm_solver"] if "pm_solver" in net._options else "ipopt"
    pm["pm_mip_solver"] = net._options["pm_mip_solver"] if "pm_mip_solver" in net._options else "cbc"
    pm["pm_nl_solver"] = net._options["pm_nl_solver"] if "pm_nl_solver" in net._options else "ipopt"
    pm["pm_model"] = net._options["pm_model"] if "pm_model" in net._options else "DCPPowerModel"
    pm["pm_log_level"] = net._options["pm_log_level"] if "pm_log_level" in net._options else 0

    if "pm_time_limits" in net._options and isinstance(net._options["pm_time_limits"], dict):
        # write time limits to power models data structure
        for key, val in net._options["pm_time_limits"].items():
            pm[key] = val
    else:
        pm["pm_time_limit"], pm["pm_nl_time_limit"], pm["pm_mip_time_limit"] = np.inf, np.inf, np.inf
    pm["correct_pm_network_data"] = net._options["correct_pm_network_data"]
    return pm


def build_ne_branch(net, ppc):
    # this is only used by pm tnep
    if "ne_line" in net:
        length = len(net["ne_line"])
        ppc["ne_branch"] = np.zeros(shape=(length, branch_cols + 1), dtype=np.complex128)
        ppc["ne_branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -60, 60])
        # create branch array ne_branch like the common branch array in the ppc
        net._pd2ppc_lookups["ne_branch"] = dict()
        net._pd2ppc_lookups["ne_branch"]["ne_line"] = (0, length)
        _calc_line_parameter(net, ppc, "ne_line", "ne_branch")
        ppc["ne_branch"][:, CONSTRUCTION_COST] = net["ne_line"].loc[:, "construction_cost"].values
    return ppc


def init_ne_line(net, new_line_index, construction_costs=None):
    """
    init function for new line dataframe, which specifies the possible new lines being built by power models tnep opt

    Parameters
    ----------
    net - pp net
    new_line_index (list) - indices of new lines. These are copied to the new dataframe net["ne_line"] from net["line"]
    construction_costs (list, 0.) - costs of newly constructed lines

    Returns
    -------

    """
    # init dataframe
    net["ne_line"] = net["line"].loc[new_line_index, :]
    # add costs, if None -> init with zeros
    construction_costs = np.zeros(len(new_line_index)) if construction_costs is None else construction_costs
    net["ne_line"].loc[new_line_index, "construction_cost"] = construction_costs
    # set in service, but only in ne line dataframe
    net["ne_line"].loc[new_line_index, "in_service"] = True
    # init res_ne_line to save built status afterwards
    net["res_ne_line"] = pd.DataFrame(data=0, index=new_line_index, columns=["built"], dtype=int)
