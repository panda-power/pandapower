# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd

import pandapower as pp
from pandapower.pd2ppc import _pd2ppc
from pandapower.run import rundcpp

from pandas import DataFrame, notnull, isnull
from pandapower.topology import create_nxgraph, connected_component


def estimate_voltage_vector(net):
    """
    Function initializes the voltage vector of net with a rough estimation. All buses are set to the
    slack bus voltage. Transformer differences in magnitude and phase shifting are accounted for.
    :param net: pandapower network
    :return: pandas dataframe with estimated vm_pu and va_degree
    """
    res_bus = DataFrame(index=net.bus.index, columns=["vm_pu", "va_degree"])
    net_graph = create_nxgraph(net, include_trafos=False)
    for _, ext_grid in net.ext_grid.iterrows():
        area = list(connected_component(net_graph, ext_grid.bus))
        res_bus.vm_pu.loc[area] = ext_grid.vm_pu
        res_bus.va_degree.loc[area] = ext_grid.va_degree
    trafos = net.trafo[net.trafo.in_service == 1]
    trafo_index = trafos.index.tolist()
    while len(trafo_index):
        for tix in trafo_index:
            trafo = trafos.loc[tix]
            if notnull(res_bus.vm_pu.at[trafo.hv_bus]) and isnull(res_bus.vm_pu.at[trafo.lv_bus]):
                try:
                    area = list(connected_component(net_graph, trafo.lv_bus))
                    shift = trafo.shift_degree if "shift_degree" in trafo else 0
                    ratio = (trafo.vn_hv_kv / trafo.vn_lv_kv) / (net.bus.vn_kv.at[trafo.hv_bus]
                                                                 / net.bus.vn_kv.at[trafo.lv_bus])
                    res_bus.vm_pu.loc[area] = res_bus.vm_pu.at[trafo.hv_bus] * ratio
                    res_bus.va_degree.loc[area] = res_bus.va_degree.at[trafo.hv_bus] - shift
                except KeyError:
                    raise UserWarning("An out-of-service bus is connected to an in-service "
                                      "transformer. Please set the transformer out of service or"
                                      "put the bus into service. Treat results with caution!")
                trafo_index.remove(tix)
            elif notnull(res_bus.vm_pu.at[trafo.hv_bus]):
                # parallel transformer, lv buses are already set from previous transformer
                trafo_index.remove(tix)
            if len(trafo_index) == len(trafos):
                # after the initial run we could not identify any areas correctly, it's probably a transmission grid
                # with slack on the LV bus and multiple transformers/gens. do flat init and return
                res_bus.vm_pu.loc[res_bus.vm_pu.isnull()] = 1.
                res_bus.va_degree.loc[res_bus.va_degree.isnull()] = 0.
                return res_bus
    return res_bus


def _get_bus_ppc_mapping(net, bus_to_be_fused):
    bus_with_elements = set(net.load.bus).union(set(net.sgen.bus)).union(
                    set(net.shunt.bus)).union(set(net.gen.bus)).union(
                    set(net.ext_grid.bus)).union(set(net.ward.bus)).union(
                    set(net.xward.bus))
    # Run dc pp to get the ppc we need
    try:
        pp.rundcpp(net)
    except:
        pass

    bus_ppci = pd.DataFrame(data=net._pd2ppc_lookups['bus'], columns=["bus_ppci"])
    bus_ppci['bus_with_elements'] = bus_ppci.index.isin(bus_with_elements)
    existed_bus = bus_ppci[bus_ppci.index.isin(net.bus.index)]
    bus_ppci['vn_kv'] = net.bus.loc[existed_bus.index, 'vn_kv']
    ppci_bus_with_elements = bus_ppci.groupby('bus_ppci')['bus_with_elements'].sum()
    bus_ppci.loc[:, 'elements_in_cluster'] = ppci_bus_with_elements[bus_ppci['bus_ppci'].values].values 
    bus_ppci['bus_to_be_fused'] = False
    if bus_to_be_fused is not None:
        bus_ppci.loc[bus_to_be_fused, 'bus_to_be_fused'] = True
        bus_cluster_to_be_fused_mask = bus_ppci.groupby('bus_ppci')['bus_to_be_fused'].any()
        bus_ppci.loc[bus_cluster_to_be_fused_mask[bus_ppci['bus_ppci'].values].values, 'bus_to_be_fused'] = True    
    return bus_ppci


def set_bb_switch_impedance(net, bus_to_be_fused=None, z_ohm=0.1):
    """
     Assuming a substation with multiple buses connected to each other with bb switch
     if multiple of them have an element (load, sgen usw.) then it caused problem on 
     e.g. State Estimation, since they will be automatically fused in ppc
     A possiblity would be to properly seperate the substation to multiple clusters (ppc) bus
     according to the number of buses with elements, in order to estimate the p, q on all 
     the busses. 
     These method implemented a forward and backward substitution:
     1. Find the mapping of pd2ppc bus, and identify the ppc bus, which contains multiple bus with elements
     2. Skip the first bus with elements with the same ppc bus, and finding all the switch connected to the
       rest pandapower bus and set impedance
     3. Update the pd2ppc lookup
     4. if it caused some bus without elements to detach from any of the bus with connected element, then the 
       switch caused it will be reset to fuse mode
     5. Iterate this process until all buses with elements are seperated to their own ppc bus, while no bus is 
       detached in order to get the voltage on bus right
    :param net: pandapower net
    :param z_ohm: float z_ohm to be setted for the switch
    :return: None
    """
    if 'z_ohm' in net.switch:
        net.switch['z_ohm_ori'] = net.switch['z_ohm']

    lookup = _get_bus_ppc_mapping(net, bus_to_be_fused)
    for _ in range(int(lookup.elements_in_cluster.max())-1):     
        bus_to_be_handled = lookup[((lookup['elements_in_cluster']>=2)&\
                                    lookup['bus_with_elements'])&(~lookup['bus_to_be_fused'])]
        bus_to_be_handled = bus_to_be_handled[bus_to_be_handled['bus_ppci'].duplicated(keep='first')].index
        imp_switch_sel = (net.switch.et=='b')&(net.switch.closed)&\
            (net.switch.bus.isin(bus_to_be_handled)|net.switch.element.isin(bus_to_be_handled))
        net.switch.loc[imp_switch_sel, 'z_ohm'] = z_ohm

        #check if fused buses were isolated by the switching type change
        lookup_discon = _get_bus_ppc_mapping(net, bus_to_be_fused)
        imp_switch = net.switch.loc[imp_switch_sel, :]
        detached_bus = lookup_discon.loc[lookup_discon.elements_in_cluster==0, :].index
        # Find the cause of isolated bus
        switch_to_be_fused = (imp_switch.bus.isin(bus_to_be_handled)&imp_switch.element.isin(detached_bus))|\
                             (imp_switch.bus.isin(detached_bus)&imp_switch.element.isin(bus_to_be_handled))
        if not switch_to_be_fused.values.any():
            return
        net.switch.loc[imp_switch[switch_to_be_fused].index, 'z_ohm'] = 0  
        lookup = _get_bus_ppc_mapping(net, bus_to_be_fused)
        if lookup.elements_in_cluster.max() == 1:
            return


def reset_bb_switch_impedance(net):
    """
    Reset the z_ohm of the switch to its original state, undo the operation by set_bb_switch_impedance 
    :param net: pandapower net
    :return: None
    """
    if "z_ohm_ori" in net.switch:
        net.switch["z_ohm"] = net.switch["z_ohm_ori"] 
        net.switch.drop("z_ohm_ori", axis=1, inplace=True)


def add_virtual_meas_from_loadflow(net, v_std_dev=0.01, p_std_dev=0.03, q_std_dev=0.03,
                                   seed=14, with_random_error=False):
    np.random.seed(seed)

    bus_meas_types = {'v': 'vm_pu', 'p': 'p_mw', 'q': 'q_mvar'}
    branch_meas_type = {'line':{'side': ('from', 'to'), 
                                'meas_type': ('p_mw', 'q_mvar')},
                        'trafo':{'side': ('hv', 'lv'), 
                                 'meas_type': ('p_mw', 'q_mvar')},
                        'trafo3w': {'side': ('hv', 'mv', 'lv'), 
                                    'meas_type': ('p_mw', 'q_mvar')}} 
    for bus_ix, bus_res in net.res_bus.iterrows():
        for meas_type in bus_meas_types.keys():
            meas_value = float(bus_res[bus_meas_types[meas_type]])
            if meas_type in ('p', 'q'):
                meas_value *= -1
                pp.create_measurement(net, meas_type=meas_type, element_type='bus', element=bus_ix,
                                      value=meas_value, std_dev=1)
            else:
                pp.create_measurement(net, meas_type=meas_type, element_type='bus', element=bus_ix,
                                      value=meas_value, std_dev=v_std_dev)

    for br_type in branch_meas_type.keys():
        if not net['res_'+br_type].empty:
            for br_ix, br_res in net['res_'+br_type].iterrows():
                for side in branch_meas_type[br_type]['side']:
                    for meas_type in branch_meas_type[br_type]['meas_type']:
                        pp.create_measurement(net, meas_type=meas_type[0], element_type=br_type, 
                                              element=br_ix, side=side,
                                              value=br_res[meas_type[0]+'_'+side+meas_type[1:]], std_dev=1)
    
    add_virtual_meas_error(net, v_std_dev, p_std_dev, q_std_dev, with_random_error=with_random_error)


def add_virtual_pmu_meas_from_loadflow(net, v_std_dev=0.001, i_std_dev=0.1, 
                                       p_std_dev=0.01, q_std_dev=0.01, dg_std_dev=0.1, seed=14, with_random_error=True):
    np.random.seed(seed)

    bus_meas_types = {'v': 'vm_pu', "va": "va_degree", 'p': 'p_mw', 'q': 'q_mvar'}
    branch_meas_type = {'line':{'side': ('from', 'to'), 
                                'meas_type': ('i_ka','ia_degree', 'p_mw', 'q_mvar')},
                        'trafo':{'side': ('hv', 'lv'), 
                                 'meas_type': ('i_ka','ia_degree', 'p_mw', 'q_mvar')},
                        'trafo3w': {'side': ('hv', 'mv', 'lv'), 
                                    'meas_type': ('i_ka','ia_degree', 'p_mw', 'q_mvar')}}

    # Added degree result for branches              
    for br_type in branch_meas_type.keys():
        for side in branch_meas_type[br_type]['side']:
            p, q, vm, va = net["res_"+br_type]["p_%s_mw"%side].values, \
                net["res_"+br_type]["q_%s_mvar"%side].values,\
                net["res_"+br_type]["vm_%s_pu"%side].values,\
                net["res_"+br_type]["va_%s_degree"%side].values
            S = p + q * 1j
            V = vm * np.exp(np.deg2rad(va)*1j)
            I = np.conj(S / V)
            net["res_"+br_type]["ia_%s_degree"%side] = np.rad2deg(np.angle(I))

    for bus_ix, bus_res in net.res_bus.iterrows():
        for meas_type in bus_meas_types.keys():
            meas_value = float(bus_res[bus_meas_types[meas_type]])
            if meas_type in ('p', 'q'):
                meas_value *= -1
                pp.create_measurement(net, meas_type=meas_type, element_type='bus', element=bus_ix,
                                      value=meas_value, std_dev=1)
            else:
                pp.create_measurement(net, meas_type=meas_type, element_type='bus', element=bus_ix,
                                      value=meas_value, std_dev=v_std_dev)

    for br_type in branch_meas_type.keys():
        if not net['res_'+br_type].empty:
            for br_ix, br_res in net['res_'+br_type].iterrows():
                for side in branch_meas_type[br_type]['side']:
                    for meas_type in branch_meas_type[br_type]['meas_type']:
                        pp.create_measurement(net, meas_type=meas_type.split("_")[0], element_type=br_type, 
                                              element=br_ix, side=side,
                                              value=br_res[meas_type.split("_")[0]+'_'+
                                                           side+'_'+meas_type.split("_")[1]], std_dev=1)
    if with_random_error:
        add_virtual_meas_error(net, v_std_dev, p_std_dev, q_std_dev,
                               i_std_dev, dg_std_dev, with_random_error=with_random_error)


def add_virtual_meas_error(net,  v_std_dev=0.001, i_std_dev=0.01, 
                           p_std_dev=0.03, q_std_dev=0.03, dg_std_dev=0.1, with_random_error=True):
    assert not net.measurement.empty

    r = np.random.normal(0, 1, net.measurement.shape[0]) # random error in range from -1, 1
    p_meas_mask = net.measurement.measurement_type=="p"
    q_meas_mask = net.measurement.measurement_type=="q"    
    v_meas_mask = net.measurement.measurement_type=="v" 
    i_meas_mask = net.measurement.measurement_type=="i"
    dg_meas_mask = net.measurement.measurement_type.isin(("ia", "va"))
    
    if with_random_error:
        net.measurement.loc[p_meas_mask, 'value'] += r[p_meas_mask.values] * p_std_dev * net.measurement.loc[p_meas_mask, 'value'].abs()
        net.measurement.loc[q_meas_mask, 'value'] += r[q_meas_mask.values] * q_std_dev * net.measurement.loc[q_meas_mask, 'value'].abs()
        net.measurement.loc[v_meas_mask, 'value'] += r[v_meas_mask.values] * v_std_dev
        net.measurement.loc[i_meas_mask, 'value'] += r[i_meas_mask.values] * i_std_dev / 10000
        net.measurement.loc[dg_meas_mask, 'value'] += r[dg_meas_mask.values] * dg_std_dev
        
    net.measurement.loc[p_meas_mask, 'std_dev'] = p_std_dev
    net.measurement.loc[q_meas_mask, 'std_dev'] = q_std_dev
    net.measurement.loc[v_meas_mask, 'std_dev'] = v_std_dev
    net.measurement.loc[i_meas_mask, 'std_dev'] = i_std_dev
    net.measurement.loc[dg_meas_mask, 'std_dev'] = dg_std_dev
