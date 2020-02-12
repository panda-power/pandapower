# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandas as pd
from collections import UserDict

from pandapower.auxiliary import _select_is_elements_numba, _add_ppc_options, _add_auxiliary_elements
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.idx_brch import branch_cols
from pandapower.pypower.idx_bus import bus_cols
import pandapower.pypower.idx_bus as idx_bus
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pf.run_newton_raphson_pf import _run_dc_pf
from pandapower.run import rundcpp
from pandapower.build_branch import get_is_lines
from pandapower.create import create_buses, create_line_from_parameters

from pandapower.estimation.util import estimate_voltage_vector
from pandapower.estimation.idx_bus import *
from pandapower.estimation.idx_brch import *
from pandapower.estimation.results import _copy_power_flow_results

try:
    import pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)
ZERO_INJECTION_STD_DEV = 0.001


def _initialize_voltage(net, init, calculate_voltage_angles):
    v_start, delta_start = None, None
    if init == 'results':
        v_start, delta_start = 'results', 'results'
    elif init == 'slack':
        res_bus = estimate_voltage_vector(net)
        v_start = res_bus.vm_pu.values
        if calculate_voltage_angles:
            delta_start = res_bus.va_degree.values
    elif init != 'flat':
        raise UserWarning("Unsupported init value. Using flat initialization.")
    return v_start, delta_start


def _init_ppc(net, v_start, delta_start, calculate_voltage_angles):
    # select elements in service and convert pandapower ppc to ppc
    net._options = {}
    _add_ppc_options(net, check_connectivity=False, init_vm_pu=v_start, init_va_degree=delta_start,
                     trafo_model="pi", mode="pf", enforce_q_lims=False,
                     calculate_voltage_angles=calculate_voltage_angles, switch_rx_ratio=2,
                     recycle=dict(_is_elements=False, ppc=False, Ybus=False))
    net["_is_elements"] = _select_is_elements_numba(net)
    _add_auxiliary_elements(net)
    ppc, ppci = _pd2ppc(net)

    # do dc power flow for phase shifting transformers
    if np.any(net.trafo.shift_degree):
        vm_backup = ppci["bus"][:, 7].copy()
        pq_backup = ppci["bus"][:, [2, 3]].copy()
        ppci["bus"][:, [2, 3]] = 0.
        ppci = _run_dc_pf(ppci)
        ppci["bus"][:, 7] = vm_backup
        ppci["bus"][:, [2, 3]] = pq_backup

    return ppc, ppci


def _add_measurements_to_ppci(net, ppci, zero_injection):
    """
    Add pandapower measurements to the ppci structure by adding new columns
    :param net: pandapower net
    :param ppci: generated ppci
    :return: ppc with added columns
    """
    meas = net.measurement.copy(deep=False)
    if meas.empty:
        raise Exception("No measurements are available in pandapower Network! Abort estimation!")

    # Convert side from string to bus id
    meas["side"] = meas.apply(lambda row:
                              net['line'].at[row["element"], row["side"]+"_bus"] if
                              row["side"] in ("from", "to") else
                              net[row["element_type"]].at[row["element"], row["side"]+'_bus'] if
                              row["side"] in ("hv", "mv", "lv") else row["side"], axis=1)

    # convert p, q, i measurement to p.u., u already in p.u.
    meas.loc[meas.measurement_type=="p", ["value", "std_dev"]] /= ppci["baseMVA"]
    meas.loc[meas.measurement_type=="q", ["value", "std_dev"]] /= ppci["baseMVA"]
    
    if not meas.query("measurement_type=='i'").empty:
        meas_i_mask = (meas.measurement_type=='i')
        base_i_ka = ppci["baseMVA"] / net.bus.loc[(meas.side.fillna(meas.element))[meas_i_mask].values, 
                        "vn_kv"].values
        meas.loc[meas_i_mask, "value"] /= base_i_ka/np.sqrt(3)
        meas.loc[meas_i_mask, "std_dev"] /= base_i_ka/np.sqrt(3)

    if not meas.query("(measurement_type=='ia' )| (measurement_type=='va')").empty:
        meas_dg_mask = (meas.measurement_type=='ia')|(meas.measurement_type=='va')
        meas.loc[meas_dg_mask, "value"] = np.deg2rad(meas.loc[meas_dg_mask, "value"])
        meas.loc[meas_dg_mask, "std_dev"] = np.deg2rad(meas.loc[meas_dg_mask, "std_dev"])

    # Get elements mapping from pandapower to ppc
    map_bus = net["_pd2ppc_lookups"]["bus"]
    meas_bus = meas[(meas['element_type'] == 'bus')]
    if (map_bus[meas_bus['element'].values.astype(int)] >= ppci["bus"].shape[0]).any():
        std_logger.warning("Measurement defined in pp-grid does not exist in ppci! Will be deleted!")
        meas_bus = meas_bus[map_bus[meas_bus['element'].values.astype(int)] < ppci["bus"].shape[0]]

    # mapping to dict instead of np array ensures good performance for large indices
    # (e.g., 999999999 requires a large np array even if there are only 2 buses)
    # downside is loop comprehension to access the map
    map_line, map_trafo, map_trafo3w = None, None, None
    branch_mask = ppci['internal']['branch_is']
    if "line" in net["_pd2ppc_lookups"]["branch"]:
        map_line = {line_ix: br_ix for line_ix, br_ix in
                    zip(net.line.index, range(*net["_pd2ppc_lookups"]["branch"]["line"])) if branch_mask[br_ix]}

    if "trafo" in net["_pd2ppc_lookups"]["branch"]:
        trafo_ix_start, trafo_ix_end = net["_pd2ppc_lookups"]["branch"]["trafo"]
        trafo_ix_offset = np.sum(~branch_mask[:trafo_ix_start])
        trafo_ix_start, trafo_ix_end = trafo_ix_start - trafo_ix_offset, trafo_ix_end - trafo_ix_offset
        map_trafo = {trafo_ix: br_ix for trafo_ix, br_ix in
                     zip(net.trafo.index, range(trafo_ix_start, trafo_ix_end))
                     if branch_mask[br_ix+trafo_ix_offset]}

    if "trafo3w" in net["_pd2ppc_lookups"]["branch"]:
        trafo3w_ix_start, trafo3w_ix_end = net["_pd2ppc_lookups"]["branch"]["trafo3w"]
        trafo3w_ix_offset = np.sum(~branch_mask[:trafo3w_ix_start])
        num_trafo3w = net.trafo3w.shape[0]
        trafo3w_ix_start, trafo3w_ix_end = trafo3w_ix_start - trafo3w_ix_offset, trafo3w_ix_end - trafo3w_ix_offset
        map_trafo3w = {trafo3w_ix: {'hv': br_ix, 'mv': br_ix+num_trafo3w, 'lv': br_ix+2*num_trafo3w}
                        for trafo3w_ix, br_ix in
                       zip(net.trafo3w.index, range(trafo3w_ix_start, trafo3w_ix_start+num_trafo3w))
                       if branch_mask[br_ix+trafo3w_ix_offset]}

    # set measurements for ppc format
    # add 9 columns to ppc[bus] for Vm, Vm std dev, P, P std dev, Q, Q std dev,
    # pandapower measurement indices V, P, Q
    bus_append = np.full((ppci["bus"].shape[0], bus_cols_se), np.nan, dtype=ppci["bus"].dtype)
    
    v_measurements = meas_bus[(meas_bus.measurement_type == "v")]
    if len(v_measurements):
        bus_positions = map_bus[v_measurements.element.values.astype(int)]
        bus_append[bus_positions, VM] = v_measurements.value.values
        bus_append[bus_positions, VM_STD] = v_measurements.std_dev.values
        bus_append[bus_positions, VM_IDX] = v_measurements.index.values

    va_measurements = meas_bus[(meas_bus.measurement_type == "va")]
    if len(va_measurements):
        bus_positions = map_bus[va_measurements.element.values.astype(int)]
        bus_append[bus_positions, VA] = va_measurements.value.values
        bus_append[bus_positions, VA_STD] = va_measurements.std_dev.values
        bus_append[bus_positions, VA_IDX] = va_measurements.index.values

    p_measurements = meas_bus[(meas_bus.measurement_type == "p")]
    if len(p_measurements):
        bus_positions = map_bus[p_measurements.element.values.astype(int)]
        unique_bus_positions = np.unique(bus_positions)
        if len(unique_bus_positions) < len(bus_positions):
            std_logger.debug("P Measurement duplication will be automatically merged!")
            for bus in unique_bus_positions:
                p_meas_on_bus = p_measurements.iloc[np.argwhere(bus_positions==bus).ravel(), :]
                bus_append[bus, P] = p_meas_on_bus.value.sum()
                bus_append[bus, P_STD] = p_meas_on_bus.std_dev.max()
                bus_append[bus, P_IDX] = p_meas_on_bus.index[0]
        else:
            bus_append[bus_positions, P] = p_measurements.value.values
            bus_append[bus_positions, P_STD] = p_measurements.std_dev.values
            bus_append[bus_positions, P_IDX] = p_measurements.index.values

    q_measurements = meas_bus[(meas_bus.measurement_type == "q")]
    if len(q_measurements):
        bus_positions = map_bus[q_measurements.element.values.astype(int)]
        unique_bus_positions = np.unique(bus_positions)
        if len(unique_bus_positions) < len(bus_positions):
            std_logger.debug("Q Measurement duplication will be automatically merged!")
            for bus in unique_bus_positions:
                q_meas_on_bus = q_measurements.iloc[np.argwhere(bus_positions==bus).ravel(), :]
                bus_append[bus, Q] = q_meas_on_bus.value.sum()
                bus_append[bus, Q_STD] = q_meas_on_bus.std_dev.max()
                bus_append[bus, Q_IDX] = q_meas_on_bus.index[0]
        else:
            bus_positions = map_bus[q_measurements.element.values.astype(int)]
            bus_append[bus_positions, Q] = q_measurements.value.values
            bus_append[bus_positions, Q_STD] = q_measurements.std_dev.values
            bus_append[bus_positions, Q_IDX] = q_measurements.index.values

    #add zero injection measurement and labels defined in parameter zero_injection
    bus_append = _add_zero_injection(net, ppci, bus_append, zero_injection)
    # add virtual measurements for artificial buses, which were created because
    # of an open line switch. p/q are 0. and std dev is 1. (small value)
    new_in_line_buses = np.setdiff1d(np.arange(ppci["bus"].shape[0]), map_bus[map_bus >= 0])
    bus_append[new_in_line_buses, 2] = 0.
    bus_append[new_in_line_buses, 3] = 1.
    bus_append[new_in_line_buses, 4] = 0.
    bus_append[new_in_line_buses, 5] = 1.

    # add 15 columns to mpc[branch] for Im_from, Im_from std dev, Im_to, Im_to std dev,
    # P_from, P_from std dev, P_to, P_to std dev, Q_from, Q_from std dev,  Q_to, Q_to std dev,
    # pandapower measurement index I, P, Q
    branch_append = np.full((ppci["branch"].shape[0], branch_cols_se),
                            np.nan, dtype=ppci["branch"].dtype)

    if map_line is not None:
        i_measurements = meas[(meas.measurement_type == "i") & (meas.element_type == "line") &\
                              meas.element.isin(map_line)]
        if len(i_measurements):
            meas_from = i_measurements[(i_measurements.side.values.astype(int) ==
                                        net.line.from_bus[i_measurements.element]).values]
            meas_to = i_measurements[(i_measurements.side.values.astype(int) ==
                                      net.line.to_bus[i_measurements.element]).values]
            ix_from = [map_line[l] for l in meas_from.element.values.astype(int)]
            ix_to = [map_line[l] for l in meas_to.element.values.astype(int)]
            branch_append[ix_from, IM_FROM] = meas_from.value.values
            branch_append[ix_from, IM_FROM_STD] = meas_from.std_dev.values
            branch_append[ix_from, IM_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, IM_TO] = meas_to.value.values
            branch_append[ix_to, IM_TO_STD] = meas_to.std_dev.values
            branch_append[ix_to, IM_TO_IDX] = meas_to.index.values
            
        ia_measurements = meas[(meas.measurement_type == "ia") & (meas.element_type == "line") & \
                                  meas.element.isin(map_line)]
        if len(ia_measurements):
            dg_meas_from = ia_measurements[(ia_measurements.side.values.astype(int) ==
                                            net.line.from_bus[ia_measurements.element]).values]
            dg_meas_to = ia_measurements[(ia_measurements.side.values.astype(int) ==
                                          net.line.to_bus[ia_measurements.element]).values]
            ix_from = [map_line[l] for l in dg_meas_from.element.values.astype(int)]
            ix_to = [map_line[l] for l in dg_meas_to.element.values.astype(int)]
            branch_append[ix_from, IA_FROM] = dg_meas_from.value.values
            branch_append[ix_from, IA_FROM_STD] = dg_meas_from.std_dev.values
            branch_append[ix_from, IA_FROM_IDX] = dg_meas_from.index.values
            branch_append[ix_to, IA_TO] = dg_meas_to.value.values
            branch_append[ix_to, IA_TO_STD] = dg_meas_to.std_dev.values
            branch_append[ix_to, IA_TO_IDX] = dg_meas_to.index.values

        p_measurements = meas[(meas.measurement_type == "p") & (meas.element_type == "line") &
                              meas.element.isin(map_line)]
        if len(p_measurements):
            meas_from = p_measurements[(p_measurements.side.values.astype(int) ==
                                        net.line.from_bus[p_measurements.element]).values]
            meas_to = p_measurements[(p_measurements.side.values.astype(int) ==
                                      net.line.to_bus[p_measurements.element]).values]
            ix_from = [map_line[l] for l in meas_from.element.values.astype(int)]
            ix_to = [map_line[l] for l in meas_to.element.values.astype(int)]
            branch_append[ix_from, P_FROM] = meas_from.value.values
            branch_append[ix_from, P_FROM_STD] = meas_from.std_dev.values
            branch_append[ix_from, P_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, P_TO] = meas_to.value.values
            branch_append[ix_to, P_TO_STD] = meas_to.std_dev.values
            branch_append[ix_to, P_TO_IDX] = meas_to.index.values

        q_measurements = meas[(meas.measurement_type == "q") & (meas.element_type == "line") &
                              meas.element.isin(map_line)]
        if len(q_measurements):
            meas_from = q_measurements[(q_measurements.side.values.astype(int) ==
                                        net.line.from_bus[q_measurements.element]).values]
            meas_to = q_measurements[(q_measurements.side.values.astype(int) ==
                                      net.line.to_bus[q_measurements.element]).values]
            ix_from = [map_line[l] for l in meas_from.element.values.astype(int)]
            ix_to = [map_line[l] for l in meas_to.element.values.astype(int)]
            branch_append[ix_from, Q_FROM] = meas_from.value.values
            branch_append[ix_from, Q_FROM_STD] = meas_from.std_dev.values
            branch_append[ix_from, Q_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, Q_TO] = meas_to.value.values
            branch_append[ix_to, Q_TO_STD] = meas_to.std_dev.values
            branch_append[ix_to, Q_TO_IDX] = meas_to.index.values

    # TODO review in 2019 -> is this a use case? create test with switches on lines
    # determine number of lines in ppci["branch"]
    # out of service lines and lines with open switches at both ends are not in the ppci
    # _is_elements = net["_is_elements"]
    # if "line" not in _is_elements:
    #     get_is_lines(net)
    # lines_is = _is_elements['line']
    # bus_is_idx = _is_elements['bus_is_idx']
    # slidx = (net["switch"]["closed"].values == 0) \
    #         & (net["switch"]["et"].values == "l") \
    #         & (np.in1d(net["switch"]["element"].values, lines_is.index)) \
    #         & (np.in1d(net["switch"]["bus"].values, bus_is_idx))
    # ppci_lines = len(lines_is) - np.count_nonzero(slidx)

    if map_trafo is not None:
        i_tr_measurements = meas[(meas.measurement_type == "i") & (meas.element_type == "trafo") &
                                 meas.element.isin(map_trafo)]
        if len(i_tr_measurements):
            meas_from = i_tr_measurements[(i_tr_measurements.side.values.astype(int) ==
                                           net.trafo.hv_bus[i_tr_measurements.element]).values]
            meas_to = i_tr_measurements[(i_tr_measurements.side.values.astype(int) ==
                                         net.trafo.lv_bus[i_tr_measurements.element]).values]
            ix_from = [map_trafo[t] for t in meas_from.element.values.astype(int)]
            ix_to = [map_trafo[t] for t in meas_to.element.values.astype(int)]
            branch_append[ix_from, IM_FROM] = meas_from.value.values
            branch_append[ix_from, IM_FROM_STD] = meas_from.std_dev.values
            branch_append[ix_from, IM_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, IM_TO] = meas_to.value.values
            branch_append[ix_to, IM_TO_STD] = meas_to.std_dev.values
            branch_append[ix_to, IM_TO_IDX] = meas_to.index.values

        i_tr_dg_measurements = meas[(meas.measurement_type == "ia") & (meas.element_type == "trafo") &
                                     meas.element.isin(map_trafo)]
        if len(i_tr_dg_measurements):
            dg_meas_from = i_tr_dg_measurements[(i_tr_dg_measurements.side.values.astype(int) ==
                                           net.trafo.hv_bus[i_tr_dg_measurements.element]).values]
            dg_meas_to = i_tr_dg_measurements[(i_tr_dg_measurements.side.values.astype(int) ==
                                         net.trafo.lv_bus[i_tr_dg_measurements.element]).values]
            ix_from = [map_trafo[t] for t in dg_meas_from.element.values.astype(int)]
            ix_to = [map_trafo[t] for t in dg_meas_to.element.values.astype(int)]
            branch_append[ix_from, IA_FROM] = dg_meas_from.value.values
            branch_append[ix_from, IA_FROM_STD] = dg_meas_from.std_dev.values
            branch_append[ix_from, IA_FROM_IDX] = dg_meas_from.index.values
            branch_append[ix_to, IA_TO] = dg_meas_to.value.values
            branch_append[ix_to, IA_TO_STD] = dg_meas_to.std_dev.values
            branch_append[ix_to, IA_TO_IDX] = dg_meas_to.index.values

        p_tr_measurements = meas[(meas.measurement_type == "p") & (meas.element_type == "trafo") &
                                 meas.element.isin(map_trafo)]
        if len(p_tr_measurements):
            meas_from = p_tr_measurements[(p_tr_measurements.side.values.astype(int) ==
                                           net.trafo.hv_bus[p_tr_measurements.element]).values]
            meas_to = p_tr_measurements[(p_tr_measurements.side.values.astype(int) ==
                                         net.trafo.lv_bus[p_tr_measurements.element]).values]
            ix_from = [map_trafo[t] for t in meas_from.element.values.astype(int)]
            ix_to = [map_trafo[t] for t in meas_to.element.values.astype(int)]
            branch_append[ix_from, P_FROM] = meas_from.value.values
            branch_append[ix_from, P_FROM_STD] = meas_from.std_dev.values
            branch_append[ix_from, P_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, P_TO] = meas_to.value.values
            branch_append[ix_to, P_TO_STD] = meas_to.std_dev.values
            branch_append[ix_to, P_TO_IDX] = meas_to.index.values

        q_tr_measurements = meas[(meas.measurement_type == "q") & (meas.element_type == "trafo") &
                                 meas.element.isin(map_trafo)]
        if len(q_tr_measurements):
            meas_from = q_tr_measurements[(q_tr_measurements.side.values.astype(int) ==
                                           net.trafo.hv_bus[q_tr_measurements.element]).values]
            meas_to = q_tr_measurements[(q_tr_measurements.side.values.astype(int) ==
                                         net.trafo.lv_bus[q_tr_measurements.element]).values]
            ix_from = [map_trafo[t] for t in meas_from.element.values.astype(int)]
            ix_to = [map_trafo[t] for t in meas_to.element.values.astype(int)]
            branch_append[ix_from, Q_FROM] = meas_from.value.values
            branch_append[ix_from, Q_FROM_STD] = meas_from.std_dev.values
            branch_append[ix_from, Q_FROM_IDX] = meas_from.index.values
            branch_append[ix_to, Q_TO] = meas_to.value.values
            branch_append[ix_to, Q_TO_STD] = meas_to.std_dev.values
            branch_append[ix_to, Q_TO_IDX] = meas_to.index.values

    # Add measurements for trafo3w
    if map_trafo3w is not None:
        i_tr3w_measurements = meas[(meas.measurement_type == "i") & (meas.element_type == "trafo3w") &
                                   meas.element.isin(map_trafo3w)]
        if len(i_tr3w_measurements):
            meas_hv = i_tr3w_measurements[(i_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.hv_bus[i_tr3w_measurements.element]).values]
            meas_mv = i_tr3w_measurements[(i_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.mv_bus[i_tr3w_measurements.element]).values]
            meas_lv = i_tr3w_measurements[(i_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.lv_bus[i_tr3w_measurements.element]).values]
            ix_hv = [map_trafo3w[t]['hv'] for t in meas_hv.element.values.astype(int)]
            ix_mv = [map_trafo3w[t]['mv'] for t in meas_mv.element.values.astype(int)]
            ix_lv = [map_trafo3w[t]['lv'] for t in meas_lv.element.values.astype(int)]
            branch_append[ix_hv, IM_FROM] = meas_hv.value.values
            branch_append[ix_hv, IM_FROM_STD] = meas_hv.std_dev.values
            branch_append[ix_hv, IM_FROM_IDX] = meas_hv.index.values
            branch_append[ix_mv, IM_TO] = meas_mv.value.values
            branch_append[ix_mv, IM_TO_STD] = meas_mv.std_dev.values
            branch_append[ix_mv, IM_TO_IDX] = meas_mv.index.values
            branch_append[ix_lv, IM_TO] = meas_lv.value.values
            branch_append[ix_lv, IM_TO_STD] = meas_lv.std_dev.values
            branch_append[ix_lv, IM_TO_IDX] = meas_lv.index.values

    # Add degree_measurements for trafo3w
    if map_trafo3w is not None:
        i_tr3w_dg_measurements = meas[(meas.measurement_type == "ia") & (meas.element_type == "trafo3w") &
                                   meas.element.isin(map_trafo3w)]
        if len(i_tr3w_dg_measurements):
            dg_meas_hv = i_tr3w_dg_measurements[(i_tr3w_dg_measurements.side.values.astype(int) ==
                                           net.trafo3w.hv_bus[i_tr3w_dg_measurements.element]).values]
            dg_meas_mv = i_tr3w_dg_measurements[(i_tr3w_dg_measurements.side.values.astype(int) ==
                                           net.trafo3w.mv_bus[i_tr3w_dg_measurements.element]).values]
            dg_meas_lv = i_tr3w_dg_measurements[(i_tr3w_dg_measurements.side.values.astype(int) ==
                                           net.trafo3w.lv_bus[i_tr3w_dg_measurements.element]).values]
            ix_hv = [map_trafo3w[t]['hv'] for t in dg_meas_hv.element.values.astype(int)]
            ix_mv = [map_trafo3w[t]['mv'] for t in dg_meas_mv.element.values.astype(int)]
            ix_lv = [map_trafo3w[t]['lv'] for t in dg_meas_lv.element.values.astype(int)]
            branch_append[ix_hv, IA_FROM] = dg_meas_hv.value.values
            branch_append[ix_hv, IA_FROM_STD] = dg_meas_hv.std_dev.values
            branch_append[ix_hv, IA_FROM_IDX] = dg_meas_hv.index.values
            branch_append[ix_mv, IA_TO] = dg_meas_mv.value.values
            branch_append[ix_mv, IA_TO_STD] = dg_meas_mv.std_dev.values
            branch_append[ix_mv, IA_TO_IDX] = dg_meas_mv.index.values
            branch_append[ix_lv, IA_TO] = dg_meas_lv.value.values
            branch_append[ix_lv, IA_TO_STD] = dg_meas_lv.std_dev.values
            branch_append[ix_lv, IA_TO_IDX] = dg_meas_lv.index.values

        p_tr3w_measurements = meas[(meas.measurement_type == "p") & (meas.element_type == "trafo3w") &
                                   meas.element.isin(map_trafo3w)]
        if len(p_tr3w_measurements):
            meas_hv = p_tr3w_measurements[(p_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.hv_bus[p_tr3w_measurements.element]).values]
            meas_mv = p_tr3w_measurements[(p_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.mv_bus[p_tr3w_measurements.element]).values]
            meas_lv = p_tr3w_measurements[(p_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.lv_bus[p_tr3w_measurements.element]).values]
            ix_hv = [map_trafo3w[t]['hv'] for t in meas_hv.element.values.astype(int)]
            ix_mv = [map_trafo3w[t]['mv'] for t in meas_mv.element.values.astype(int)]
            ix_lv = [map_trafo3w[t]['lv'] for t in meas_lv.element.values.astype(int)]
            branch_append[ix_hv, P_FROM] = meas_hv.value.values
            branch_append[ix_hv, P_FROM_STD] = meas_hv.std_dev.values
            branch_append[ix_hv, P_FROM_IDX] = meas_hv.index.values
            branch_append[ix_mv, P_TO] = meas_mv.value.values
            branch_append[ix_mv, P_TO_STD] = meas_mv.std_dev.values
            branch_append[ix_mv, P_TO_IDX] = meas_mv.index.values
            branch_append[ix_lv, P_TO] = meas_lv.value.values
            branch_append[ix_lv, P_TO_STD] = meas_lv.std_dev.values
            branch_append[ix_lv, P_TO_IDX] = meas_lv.index.values

        q_tr3w_measurements = meas[(meas.measurement_type == "q") & (meas.element_type == "trafo3w") &
                                   meas.element.isin(map_trafo3w)]
        if len(q_tr3w_measurements):
            meas_hv = q_tr3w_measurements[(q_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.hv_bus[q_tr3w_measurements.element]).values]
            meas_mv = q_tr3w_measurements[(q_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.mv_bus[q_tr3w_measurements.element]).values]
            meas_lv = q_tr3w_measurements[(q_tr3w_measurements.side.values.astype(int) ==
                                           net.trafo3w.lv_bus[q_tr3w_measurements.element]).values]
            ix_hv = [map_trafo3w[t]['hv'] for t in meas_hv.element.values.astype(int)]
            ix_mv = [map_trafo3w[t]['mv'] for t in meas_mv.element.values.astype(int)]
            ix_lv = [map_trafo3w[t]['lv'] for t in meas_lv.element.values.astype(int)]
            branch_append[ix_hv, Q_FROM] = meas_hv.value.values
            branch_append[ix_hv, Q_FROM_STD] = meas_hv.std_dev.values
            branch_append[ix_hv, Q_FROM_IDX] = meas_hv.index.values
            branch_append[ix_mv, Q_TO] = meas_mv.value.values
            branch_append[ix_mv, Q_TO_STD] = meas_mv.std_dev.values
            branch_append[ix_mv, Q_TO_IDX] = meas_mv.index.values
            branch_append[ix_lv, Q_TO] = meas_lv.value.values
            branch_append[ix_lv, Q_TO_STD] = meas_lv.std_dev.values
            branch_append[ix_lv, Q_TO_IDX] = meas_lv.index.values
    
    # Check append or update
    if ppci["bus"].shape[1] == bus_cols:
        ppci["bus"] = np.hstack((ppci["bus"], bus_append))
    else:
        ppci["bus"][:, bus_cols: bus_cols+bus_cols_se] = bus_append
    
    if ppci["branch"].shape[1] == branch_cols:
        ppci["branch"] = np.hstack((ppci["branch"], branch_append))
    else:
        ppci["branch"][:, branch_cols: branch_cols+branch_cols_se] = branch_append
    return ppci


def _add_zero_injection(net, ppci, bus_append, zero_injection): 
    """
    Add zero injection labels to the ppci structure and add virtual measurements to those buses
    :param net: pandapower net
    :param ppci: generated ppci
    :param bus_append: added columns to the ppci bus with zero injection label
    :param zero_injection: parameter to control which bus to be identified as zero injection
    :return bus_append: added columns
    """   
    bus_append[:, ZERO_INJ_FLAG] = False   
    if zero_injection is not None:
        # identify aux bus to zero injection
        if net._pd2ppc_lookups['aux']:
            aux_bus_lookup = np.concatenate([v for k,v in net._pd2ppc_lookups['aux'].items() if k != 'xward'])
            aux_bus = net._pd2ppc_lookups['bus'][aux_bus_lookup]
            bus_append[aux_bus, ZERO_INJ_FLAG] = True

        if isinstance(zero_injection, str):
            if zero_injection == 'auto':
                # identify bus without elements and pq measurements as zero injection
                zero_inj_bus_mask = (ppci["bus"][:, 1] == 1) & (ppci["bus"][:, 2:6]==0).all(axis=1) &\
                    np.isnan(bus_append[:, P:(Q_STD+1)]).all(axis=1)
                bus_append[zero_inj_bus_mask, ZERO_INJ_FLAG] = True
            elif zero_injection != "aux_bus":
                raise UserWarning("zero injection parameter is not correctly initialized")
        elif hasattr(zero_injection, '__iter__'):
            zero_inj_bus = net._pd2ppc_lookups['bus'][zero_injection]
            bus_append[zero_inj_bus, ZERO_INJ_FLAG] = True

        zero_inj_bus = np.argwhere(bus_append[:, ZERO_INJ_FLAG]).ravel()
        bus_append[zero_inj_bus, P] = 0
        bus_append[zero_inj_bus, P_STD] = ZERO_INJECTION_STD_DEV
        bus_append[zero_inj_bus, Q] = 0
        bus_append[zero_inj_bus, Q_STD] = ZERO_INJECTION_STD_DEV
    return bus_append


def _build_measurement_vectors(ppci, update_meas_only=False):
    """
    Building measurement vector z, pandapower to ppci measurement mapping and covariance matrix R
    :param ppci: generated ppci which contains the measurement columns
    :param branch_cols: number of columns in original ppci["branch"] without measurements
    :param bus_cols: number of columns in original ppci["bus"] without measurements
    :return: both created vectors
    """
    p_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + P])
    p_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_FROM])
    p_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + P_TO])
    q_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + Q])
    q_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + Q_FROM])
    q_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + Q_TO])
    v_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + VM])
    v_degree_bus_not_nan = ~np.isnan(ppci["bus"][:, bus_cols + VA])
    i_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IM_FROM])
    i_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IM_TO])
    i_degree_line_f_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IA_FROM])
    i_degree_line_t_not_nan = ~np.isnan(ppci["branch"][:, branch_cols + IA_TO])
    # piece together our measurement vector z
    z = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P],
                        ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM],
                        ppci["branch"][p_line_t_not_nan, branch_cols + P_TO],
                        ppci["bus"][q_bus_not_nan, bus_cols + Q],
                        ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM],
                        ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO],
                        ppci["bus"][v_bus_not_nan, bus_cols + VM],
                        ppci["bus"][v_degree_bus_not_nan, bus_cols + VA],
                        ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM],
                        ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO],
                        ppci["branch"][i_degree_line_f_not_nan, branch_cols + IA_FROM],
                        ppci["branch"][i_degree_line_t_not_nan, branch_cols + IA_TO]
                        )).real.astype(np.float64)
    if not update_meas_only:
        # conserve the pandapower indices of measurements in the ppci order
        pp_meas_indices = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P_IDX],
                                          ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM_IDX],
                                          ppci["branch"][p_line_t_not_nan, branch_cols + P_TO_IDX],
                                          ppci["bus"][q_bus_not_nan, bus_cols + Q_IDX],
                                          ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM_IDX],
                                          ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO_IDX],
                                          ppci["bus"][v_bus_not_nan, bus_cols + VM_IDX],
                                          ppci["bus"][v_degree_bus_not_nan, bus_cols + VA_IDX],
                                          ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM_IDX],
                                          ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO_IDX],
                                          ppci["branch"][i_degree_line_f_not_nan, branch_cols + IA_FROM_IDX],
                                          ppci["branch"][i_degree_line_t_not_nan, branch_cols + IA_TO_IDX]
                                          )).real.astype(int)
        # Covariance matrix R
        r_cov = np.concatenate((ppci["bus"][p_bus_not_nan, bus_cols + P_STD],
                                ppci["branch"][p_line_f_not_nan, branch_cols + P_FROM_STD],
                                ppci["branch"][p_line_t_not_nan, branch_cols + P_TO_STD],
                                ppci["bus"][q_bus_not_nan, bus_cols + Q_STD],
                                ppci["branch"][q_line_f_not_nan, branch_cols + Q_FROM_STD],
                                ppci["branch"][q_line_t_not_nan, branch_cols + Q_TO_STD],
                                ppci["bus"][v_bus_not_nan, bus_cols + VM_STD],
                                ppci["bus"][v_degree_bus_not_nan, bus_cols + VA_STD],
                                ppci["branch"][i_line_f_not_nan, branch_cols + IM_FROM_STD],
                                ppci["branch"][i_line_t_not_nan, branch_cols + IM_TO_STD],
                                ppci["branch"][i_degree_line_f_not_nan, branch_cols + IA_FROM_STD],
                                ppci["branch"][i_degree_line_t_not_nan, branch_cols + IA_TO_STD]
                                )).real.astype(np.float64)
        meas_mask = np.concatenate([p_bus_not_nan,
                                    p_line_f_not_nan,
                                    p_line_t_not_nan,
                                    q_bus_not_nan,
                                    q_line_f_not_nan,
                                    q_line_t_not_nan,
                                    v_bus_not_nan,
                                    v_degree_bus_not_nan,
                                    i_line_f_not_nan,
                                    i_line_t_not_nan,
                                    i_degree_line_f_not_nan,
                                    i_degree_line_t_not_nan])
        any_i_meas = np.any(np.r_[i_line_f_not_nan, i_line_t_not_nan])
        any_degree_meas = np.any(np.r_[v_degree_bus_not_nan,
                                       i_degree_line_f_not_nan, 
                                       i_degree_line_t_not_nan]) 
        return z, pp_meas_indices, r_cov, meas_mask, any_i_meas, any_degree_meas
    else:
        return z


def pp2eppci(net, v_start=None, delta_start=None, calculate_voltage_angles=True, zero_injection="aux_bus",
             ppc=None, eppci=None):
    # initialize result tables if not existent
    _copy_power_flow_results(net)
    if isinstance(eppci, ExtendedPPCI):
        eppci.data = _add_measurements_to_ppci(net, eppci.data, zero_injection)
        eppci.update_meas()
        return net, ppc, eppci
    else:
        # initialize ppc
        ppc, ppci = _init_ppc(net, v_start, delta_start, calculate_voltage_angles)

        # add measurements to ppci structure
        # Finished converting pandapower network to ppci
        ppci = _add_measurements_to_ppci(net, ppci, zero_injection)
        return net, ppc, ExtendedPPCI(ppci)


class ExtendedPPCI(UserDict):
    def __init__(self, ppci):
        self.data = ppci

        # Measurement relevant parameters
        self.z = None
        self.r_cov = None
        self.pp_meas_indices = None 
        self.non_nan_meas_mask = None 
        self.non_nan_meas_selector = None
        self.any_i_meas = False
        self.any_degree_meas = False
        self._initialize_meas()

        # check slack bus
        self.non_slack_buses = np.argwhere(ppci["bus"][:, idx_bus.BUS_TYPE] != 3).ravel()
        self.non_slack_bus_mask = (ppci['bus'][:, idx_bus.BUS_TYPE] != 3).ravel()
        self.num_non_slack_bus = np.sum(self.non_slack_bus_mask)
        self.delta_v_bus_mask = np.r_[self.non_slack_bus_mask,
                                      np.ones(self.non_slack_bus_mask.shape[0], dtype=bool)].ravel()
        self.delta_v_bus_selector = np.flatnonzero(self.delta_v_bus_mask)

        # Initialize state variable
        self.v_init = ppci["bus"][:, idx_bus.VM]
        self.delta_init = np.radians(ppci["bus"][:, idx_bus.VA])
        self.E_init = np.r_[self.delta_init[self.non_slack_bus_mask], self.v_init]
        self.v = self.v_init.copy()
        self.delta = self.delta_init.copy()
        self.E = self.E_init.copy()


    def _initialize_meas(self):
        # calculate relevant vectors from ppci measurements
        self.z, self.pp_meas_indices, self.r_cov, self.non_nan_meas_mask, self.any_i_meas, self.any_degree_meas =\
             _build_measurement_vectors(self, update_meas_only=False)
        self.non_nan_meas_selector = np.flatnonzero(self.non_nan_meas_mask)
    
    def update_meas(self):
        self.z = _build_measurement_vectors(self, update_meas_only=True)

    @property
    def V(self):
        return self.v * np.exp(1j * self.delta)
    
    def reset(self):
        self.v, self.delta, self.E = self.v_init.copy(), self.delta_init.copy(), self.E_init.copy()

    def update_E(self, E):
        self.E = E
        self.v = E[self.num_non_slack_bus:]
        self.delta[self.non_slack_buses] = E[:self.num_non_slack_bus]
    
    def E2V(self, E):
        self.update_E(E)
        return self.V
        
    def get_Y(self):
        # Using recycled version if available
        if "Ybus" in self["internal"] and self["internal"]["Ybus"].size:
            Ybus, Yf, Yt = self["internal"]['Ybus'], self["internal"]['Yf'], self["internal"]['Yt']
        else:
            ## build admittance matrices
            Ybus, Yf, Yt = makeYbus(self['baseMVA'], self['bus'], self['branch'])
            self["internal"]['Ybus'], self["internal"]['Yf'], self["internal"]['Yt'] = Ybus, Yf, Yt
        return Ybus, Yf, Yt