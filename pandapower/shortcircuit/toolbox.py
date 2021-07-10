# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import networkx as nx
from copy import deepcopy

import pandapower as pp
from pandapower.shortcircuit import calc_sc
from pandapower.create import _get_index_with_check
from pandapower.topology import create_nxgraph

__all__ = ["detect_power_station_unit", "calc_sc_on_line"]

def detect_power_station_unit(net, mode="auto",
                              max_gen_voltage_kv=80, max_distance_km=0.01):
    """
    Identifies the power station units configuration (gen and trafo) according to IEC 60909.
    Updates the power_station_trafo in the gen table

    INPUT:
        **net** - panpdapower net

        **mode** (str, ("auto", trafo""))

        **max_gen_voltage_level** (float)

        **max_distance_km** (float)
    """
    logger.info("This function will overwrites the value 'power_station_trafo' in gen table")
    net.gen["power_station_trafo"] = np.nan

    required_gen = net.gen.loc[net.bus.loc[net.gen.bus.values, "vn_kv"].values < max_gen_voltage_kv,:]
    gen_bus = required_gen.loc[:, "bus"].values

    if mode.lower() == "auto":
        required_trafo = net.trafo.loc[net.bus.loc[net.trafo.lv_bus.values, "vn_kv"].values < max_gen_voltage_kv, :]
    elif mode.lower() == "trafo":
        if "power_station_unit" in net.trafo.columns:
            required_trafo = net.trafo.loc[net.trafo.power_station_unit, :]
        else:
            logger.warning("Using mode 'trafo' requires 'power_station_unit' defined for trafo! Using 'auto' mode instead!")
            required_trafo = net.trafo.loc[net.bus.loc[net.trafo.lv_bus.values, "vn_kv"].values < max_gen_voltage_kv, :]

    else:
        raise UserWarning(f"Unsupported modes: {mode}")

    trafo_lv_bus = net.trafo.loc[required_trafo.index, "lv_bus"].values
    trafo_hv_bus = net.trafo.loc[required_trafo.index, "hv_bus"].values

    g = create_nxgraph(net, respect_switches=True,
                       nogobuses=None, notravbuses=trafo_hv_bus)

    for t_ix in required_trafo.index:
        t_lv_bus = required_trafo.at[t_ix, "lv_bus"]
        bus_dist = pd.Series(nx.single_source_dijkstra_path_length(g, t_lv_bus, weight='weight'))

        connected_bus_at_lv_side = bus_dist[bus_dist < max_distance_km].index.values
        gen_bus_at_lv_side = np.intersect1d(connected_bus_at_lv_side, gen_bus)

        if len(gen_bus_at_lv_side) == 1:
            # Check parallel trafo
            if not len(np.intersect1d(connected_bus_at_lv_side, trafo_lv_bus)) == 1:
                raise UserWarning("Failure in power station units detection! Parallel trafos on generator detected!")
            if np.in1d(net.gen.bus.values, gen_bus_at_lv_side).sum() > 1:
                raise UserWarning("More than 1 gen detected at the lv side of a power station trafo")
            net.gen.loc[np.in1d(net.gen.bus.values, gen_bus_at_lv_side),
                        "power_station_trafo"] = t_ix


def _create_element_from_exisiting(net, ele_type, ele_ix):
    net[ele_type] = net[ele_type].append(pd.Series(net[ele_type].loc[ele_ix, :].to_dict(),
                                         name=_get_index_with_check(net, ele_type, None)))
    return net[ele_type].index.to_numpy()[-1]


def _create_aux_net(net, line_ix, distance_to_bus0):
    if distance_to_bus0 < 0 or distance_to_bus0 > 1:
        raise UserWarning("Calculating SC current on line failed! distance_to_bus0 must be between 0-1!")
    aux_net = deepcopy(net)

    # Create auxiliary bus
    aux_bus = pp.create_bus(aux_net, vn_kv=aux_net.bus.at[aux_net.line.at[line_ix, "from_bus"], "vn_kv"],
                            name="aux_bus_sc_calc")

    # Create auxiliary line, while preserve the original index
    aux_line0 = _create_element_from_exisiting(aux_net, "line", line_ix)
    aux_line1 = _create_element_from_exisiting(aux_net, "line", line_ix)

    ## Update distance and auxiliary bus
    aux_net.line.at[aux_line0, "length_km"] = distance_to_bus0 * aux_net.line.at[line_ix, "length_km"]
    aux_net.line.at[aux_line0, "to_bus"] = aux_bus
    aux_net.line.at[aux_line0, "name"] += "_aux_line0"

    aux_net.line.at[aux_line1, "length_km"] = (1 - distance_to_bus0) * aux_net.line.at[line_ix, "length_km"]
    aux_net.line.at[aux_line1, "from_bus"] = aux_bus
    aux_net.line.at[aux_line1, "name"] += "_aux_line1"

    ## Disable original line
    aux_net.line.at[line_ix, "in_service"] = False

    ## Update line switch
    for switch_ix in aux_net.switch.query(f" et == 'l' and element == {line_ix}").index:
        aux_switch_ix = _create_element_from_exisiting(aux_net, "switch", switch_ix)
        if aux_net.switch.at[aux_switch_ix, "bus"] == aux_net.line.at[line_ix, "from_bus"]:
            # The from side switch connected to aux_line0
            aux_net.switch.at[aux_switch_ix, "element"] =  aux_line0
        else:
            # The to side switch connected to aux_line1
            aux_net.switch.at[aux_switch_ix, "element"] =  aux_line1
    return aux_net, aux_bus


def calc_sc_on_line(net, line_ix, distance_to_bus0, **kwargs):
    """
    Calculate the shortcircuit in the middle of the line, returns a modified network
    with the shortcircuit calculation results and the bus added

    INPUT:
        **net** - panpdapower net

        **line_ix** (int) - The line of the shortcircuit

        **distance_to_bus0** (float) - The position of the shortcircuit should be between 0-1

    OPTIONAL:
        **kwargs**** - the parameters required for the pandapower calc_sc function
    """
    # Update network
    aux_net, aux_bus = _create_aux_net(net, line_ix, distance_to_bus0)

    pp.rundcpp(aux_net)
    calc_sc(aux_net, bus=aux_bus, **kwargs)

    # Return the new net and the aux bus
    return aux_net, aux_bus
