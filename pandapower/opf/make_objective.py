# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from numpy import zeros, array
from pandapower.pypower.idx_cost import MODEL, NCOST, COST, PW_LINEAR, POLYNOMIAL
from pandapower.pypower.idx_gen import PMIN, PMAX

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def _make_objective(ppci, net):
    is_quadratic, q_costs = _init_gencost(ppci, net)
    if len(net.pwl_cost):
        ppci["gencost"][:, MODEL] = PW_LINEAR
        ppci["gencost"][:, NCOST] = 2
        ppci["gencost"][:, COST + 2] = 1

        _fill_gencost_pwl(ppci, net)
        if is_quadratic:
            raise ValueError("Piecewise linear costs can not be mixed with quadratic costs")
        elif len(net.poly_cost):
            _add_linear_costs_as_pwl_cost(ppci, net)
    elif len(net.poly_cost):
        ppci["gencost"][:, MODEL] = POLYNOMIAL
        _fill_gencost_poly(ppci, net, is_quadratic, q_costs)
    else:
        logger.warning("no costs are given - overall generated power is minimized")
        ppci["gencost"][:, MODEL] = POLYNOMIAL
        ppci["gencost"][:, NCOST] = 2
        ppci["gencost"][:, COST] = 1
    return ppci


def _get_gen_index(net, et, element):
    if et == "dcline":
        dc_idx = net.dcline.index.get_loc(element)
        element = len(net.gen.index) - 2*len(net.dcline) + dc_idx*2 + 1
        et = "gen"
    lookup = "%s_controllable" % et if et in ["load", "sgen", "storage"] else et
    try:
        return int(net._pd2ppc_lookups[lookup][int(element)])
    except:
        return None


def _map_costs_to_gen(net, cost):
    gens = array([_get_gen_index(net, et, element)
                  for et, element in zip(cost.et.values, cost.element.values)])
    cost_is = array([gen is not None for gen in gens])
    cost = cost[cost_is]
    gens = gens[cost_is].astype(int)
    signs = array([-1 if element in ["load", "storage", "dcline"] else 1 for element in cost.et])
    return gens, cost, signs


def _init_gencost(ppci, net):
    is_quadratic = net.poly_cost[["cp2_eur_per_mw2", "cq2_eur_per_mvar2"]].values.any()
    q_costs = net.poly_cost[["cq1_eur_per_mvar", "cq2_eur_per_mvar2"]].values.any() or \
        "q" in net.pwl_cost.power_type.values
    rows = len(ppci["gen"])*2 if q_costs else len(ppci["gen"])
    if len(net.pwl_cost):
        nr_points = {len(p) for p in net.pwl_cost.points.values}
        points = max(nr_points)
        if is_quadratic:
            raise ValueError("Quadratic costs can be mixed with piecewise linear costs")
        columns = COST + (max(points, 2) + 1)*2
    else:
        columns = COST + 3 if is_quadratic else COST + 2
    ppci["gencost"] = zeros((rows, columns), dtype=float)
    return is_quadratic, q_costs


def _fill_gencost_poly(ppci, net, is_quadratic, q_costs):
    gens, cost, signs = _map_costs_to_gen(net, net.poly_cost)
    c0 = cost["cp0_eur"].values
    c1 = cost["cp1_eur_per_mw"].values
    signs = array([-1 if element in ["load", "storage", "dcline"] else 1 for element in cost.et])
    if is_quadratic:
        c2 = cost["cp2_eur_per_mw2"]
        ppci["gencost"][gens, NCOST] = 3
        ppci["gencost"][gens, COST] = c2 * signs
        ppci["gencost"][gens, COST + 1] = c1 * signs
        ppci["gencost"][gens, COST + 2] = c0 * signs
    else:
        ppci["gencost"][gens, NCOST] = 2
        ppci["gencost"][gens, COST] = c1 * signs
        ppci["gencost"][gens, COST + 1] = c0 * signs
    if q_costs:
        gens_q = gens + len(ppci["gen"])
        c0 = cost["cq0_eur"].values
        c1 = cost["cq1_eur_per_mvar"].values
        signs = array([-1 if element in ["load", "storage"] else 1 for element in cost.et])
        if is_quadratic:
            c2 = cost["cq2_eur_per_mvar2"]
            ppci["gencost"][gens_q, NCOST] = 3
            ppci["gencost"][gens_q, COST] = c2 * signs
            ppci["gencost"][gens_q, COST + 1] = c1 * signs
            ppci["gencost"][gens_q, COST + 2] = c0 * signs
        else:
            ppci["gencost"][gens_q, NCOST] = 2
            ppci["gencost"][gens_q, COST] = c1 * signs
            ppci["gencost"][gens_q, COST + 1] = c0 * signs


def _fill_gencost_pwl(ppci, net):
    for power_mode, cost in net.pwl_cost.groupby("power_type"):
        gens, cost, signs = _map_costs_to_gen(net, cost)
        if power_mode == "q":
            gens += len(ppci["gen"])
        for gen, points, sign in zip(gens, cost.points.values, signs):
            costs = costs_from_areas(points, sign)
            ppci["gencost"][gen, COST:COST+len(costs)] = costs
            ppci["gencost"][gen, NCOST] = len(costs) / 2


def costs_from_areas(points, sign):
    costs = []
    c0 = 0
    last_upper = None
    for lower, upper, slope in points:
        if last_upper is None:
            costs.append(lower)
            c = c0 + lower * slope * sign
            c0 = c
            costs.append(c)
        if last_upper is not None and last_upper != lower:
            raise ValueError("Non-consecutive cost function areas")
        last_upper = upper
        costs.append(upper)
        c = c0 + (upper - lower) * slope * sign
        c0 = c
        costs.append(c)
    return costs


def _add_linear_costs_as_pwl_cost(ppci, net):
    gens, cost, signs = _map_costs_to_gen(net, net.poly_cost)
    ppci["gencost"][gens, NCOST] = 2
    pmin = ppci["gen"][gens, PMIN]
    pmax = ppci["gen"][gens, PMAX]
    ppci["gencost"][gens, COST] = pmin
    ppci["gencost"][gens, COST + 1] = pmin * cost.cp1_eur_per_mw.values * signs
    ppci["gencost"][gens, COST + 2] = pmax
    ppci["gencost"][gens, COST + 3] = pmax * cost.cp1_eur_per_mw.values * signs
