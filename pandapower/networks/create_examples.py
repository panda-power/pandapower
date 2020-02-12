# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd

import pandapower as pp


def example_simple():
    """
    Returns the simple example network from the pandapower tutorials.

    OUTPUT:
        net - simple example network

    EXAMPLE:

    >>> import pandapower.networks
    >>> net = pandapower.networks.example_simple()

    """
    net = pp.create_empty_network()

    # create buses
    bus1 = pp.create_bus(net, name="HV Busbar", vn_kv=110., type="b")
    bus2 = pp.create_bus(net, name="HV Busbar 2", vn_kv=110., type="b")
    bus3 = pp.create_bus(net, name="HV Transformer Bus", vn_kv=110., type="n")
    bus4 = pp.create_bus(net, name="MV Transformer Bus", vn_kv=20., type="n")
    bus5 = pp.create_bus(net, name="MV Main Bus", vn_kv=20., type="b")
    bus6 = pp.create_bus(net, name="MV Bus 1", vn_kv=20., type="b")
    bus7 = pp.create_bus(net, name="MV Bus 2", vn_kv=20., type="b")

    # create external grid
    pp.create_ext_grid(net, bus1, vm_pu=1.02, va_degree=50)

    # create transformer
    pp.create_transformer(net, bus3, bus4, name="110kV/20kV transformer",
                                   std_type="25 MVA 110/20 kV")
    # create lines
    pp.create_line(net, bus1, bus2, length_km=10,
                           std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="Line 1")
    line2 = pp.create_line(net, bus5, bus6, length_km=2.0,
                           std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line 2")
    line3 = pp.create_line(net, bus6, bus7, length_km=3.5,
                           std_type="48-AL1/8-ST1A 20.0", name="Line 3")
    line4 = pp.create_line(net, bus7, bus5, length_km=2.5,
                           std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line 4")

    # create bus-bus switches
    pp.create_switch(net, bus2, bus3, et="b", type="CB")
    pp.create_switch(net, bus4, bus5, et="b", type="CB")

    # create bus-line switches
    pp.create_switch(net, bus5, line2, et="l", type="LBS", closed=True)
    pp.create_switch(net, bus6, line2, et="l", type="LBS", closed=True)
    pp.create_switch(net, bus6, line3, et="l", type="LBS", closed=True)
    pp.create_switch(net, bus7, line3, et="l", type="LBS", closed=False)
    pp.create_switch(net, bus7, line4, et="l", type="LBS", closed=True)
    pp.create_switch(net, bus5, line4, et="l", type="LBS", closed=True)

    # create load
    pp.create_load(net, bus7, p_mw=2, q_mvar=4, scaling=0.6, name="load")

    # create generator
    pp.create_gen(net, bus6, p_mw=6, max_q_mvar=3, min_q_mvar=-3, vm_pu=1.03,
                  name="generator")

    # create static generator
    pp.create_sgen(net, bus7, p_mw=2, q_mvar=-0.5, name="static generator")

    # create shunt
    pp.create_shunt(net, bus3, q_mvar=-0.96, p_mw=0, name='Shunt')

    return net


def example_multivoltage():
    """
    Returns the multivoltage example network from the pandapower tutorials.

    OUTPUT:
        net - multivoltage example network

    EXAMPLE:

    >>> import pandapower.networks
    >>> net = pandapower.networks.example_multivoltage()

    """
    net = pp.create_empty_network()

    # --- Busses

    # HV
    # Double busbar
    pp.create_bus(net, name='Double Busbar 1', vn_kv=380, type='b')
    pp.create_bus(net, name='Double Busbar 2', vn_kv=380, type='b')

    for i in range(10):
        pp.create_bus(net, name='Bus DB T%s' % i, vn_kv=380, type='n')

    for i in range(1, 5):
        pp.create_bus(net, name='Bus DB %s' % i, vn_kv=380, type='n')

    # Single busbar
    pp.create_bus(net, name='Single Busbar', vn_kv=110, type='b')

    for i in range(1, 6):
        pp.create_bus(net, name='Bus SB %s' % i, vn_kv=110, type='n')

    for i in range(1, 6):
        for j in [1, 2]:
            pp.create_bus(net, name='Bus SB T%s.%s' % (i, j), vn_kv=110, type='n')

    # Remaining
    for i in range(1, 5):
        pp.create_bus(net, name='Bus HV%s' % i, vn_kv=110, type='n')

    # MV
    pp.create_bus(net, name='Bus MV0 20kV', vn_kv=20, type='n')

    for i in range(8):
        pp.create_bus(net, name='Bus MV%s' % i, vn_kv=10, type='n')

    # LV
    pp.create_bus(net, name='Bus LV0', vn_kv=0.4, type='n')

    for i in range(1, 6):
        pp.create_bus(net, name='Bus LV1.%s' % i, vn_kv=0.4, type='m')

    for i in range(1, 5):
        pp.create_bus(net, name='Bus LV2.%s' % i, vn_kv=0.4, type='m')

    pp.create_bus(net, name='Bus LV2.2.1', vn_kv=0.4, type='m')
    pp.create_bus(net, name='Bus LV2.2.2', vn_kv=0.4, type='m')

    # --- Lines

    # HV
    hv_lines = pd.DataFrame()
    hv_lines['line_name'] = ['HV Line%s' % i for i in range(1, 7)]
    hv_lines['from_bus'] = ['Bus SB 2', 'Bus HV1', 'Bus HV2', 'Bus HV1', 'Bus HV3', 'Bus SB 3']
    hv_lines['to_bus'] = ['Bus HV1', 'Bus HV2', 'Bus HV4', 'Bus HV4', 'Bus HV4', 'Bus HV3']
    hv_lines['std_type'] = '184-AL1/30-ST1A 110.0'
    hv_lines['length'] = [30, 20, 30, 15, 25, 30]
    hv_lines['parallel'] = [1, 1, 1, 1, 1, 2]

    for _, hv_line in hv_lines.iterrows():
        from_bus = pp.get_element_index(net, "bus", hv_line.from_bus)
        to_bus = pp.get_element_index(net, "bus", hv_line.to_bus)
        pp.create_line(net, from_bus, to_bus, length_km=hv_line.length,
                       std_type=hv_line.std_type, name=hv_line.line_name, parallel=hv_line.parallel)

    # MV
    mv_lines = pd.DataFrame()
    mv_lines['line_name'] = ['MV Line%s' % i for i in range(1, 9)]
    mv_lines['from_bus'] = ['Bus MV%s' % i for i in list(range(7)) + [0]]
    mv_lines['to_bus'] = ['Bus MV%s' % i for i in list(range(1, 8)) + [7]]
    mv_lines['length'] = 1.5
    mv_lines['std_type'] = 'NA2XS2Y 1x185 RM/25 12/20 kV'

    for _, mv_line in mv_lines.iterrows():
        from_bus = pp.get_element_index(net, "bus", mv_line.from_bus)
        to_bus = pp.get_element_index(net, "bus", mv_line.to_bus)
        pp.create_line(net, from_bus, to_bus, length_km=mv_line.length,
                       std_type=mv_line.std_type, name=mv_line.line_name)

    # LV
    lv_lines = pd.DataFrame()
    lv_line_idx = ['1.1', '1.2', '1.3', '1.4', '1.6', '2.1', '2.2', '2.3', '2.4', '2.2.1', '2.2.2']
    lv_lines['line_name'] = ['LV Line%s' % i for i in lv_line_idx]
    lv_line_idx = ['0', '1.1', '1.2', '1.3', '1.4', '0', '2.1', '2.2', '2.3', '2.2', '2.2.1']
    lv_lines['from_bus'] = ['Bus LV%s' % i for i in lv_line_idx]
    lv_line_idx = ['1.1', '1.2', '1.3', '1.4', '1.5', '2.1', '2.2', '2.3', '2.4', '2.2.1', '2.2.2']
    lv_lines['to_bus'] = ['Bus LV%s' % i for i in lv_line_idx]
    lv_lines['length'] = [0.08]*5 + [0.12]*6
    lv_lines['std_type'] = ['NAYY 4x120 SE']*7 + ['15-AL1/3-ST1A 0.4']*4

    for _, lv_line in lv_lines.iterrows():
        from_bus = pp.get_element_index(net, "bus", lv_line.from_bus)
        to_bus = pp.get_element_index(net, "bus", lv_line.to_bus)
        pp.create_line(net, from_bus, to_bus, length_km=lv_line.length,
                       std_type=lv_line.std_type, name=lv_line.line_name)

    # --- Transformer

    hv_bus = pp.get_element_index(net, "bus", "Bus DB 2")
    lv_bus = pp.get_element_index(net, "bus", "Bus SB 1")
    pp.create_transformer_from_parameters(net, hv_bus, lv_bus,
                                          sn_mva=300, vn_hv_kv=380, vn_lv_kv=110,
                                          vkr_percent=0.06, vk_percent=8, pfe_kw=0,
                                          i0_percent=0, tap_pos=0, shift_degree=0,
                                          name='EHV-HV-Trafo')

    hv_bus = pp.get_element_index(net, "bus", "Bus MV4")
    lv_bus = pp.get_element_index(net, "bus", "Bus LV0")
    pp.create_transformer_from_parameters(net, hv_bus, lv_bus,
                                          sn_mva=0.4, vn_hv_kv=10, vn_lv_kv=0.4,
                                          vkr_percent=1.325, vk_percent=4,
                                          pfe_kw=0.95, i0_percent=0.2375, tap_side="hv",
                                          tap_neutral=0, tap_min=-2, tap_max=2,
                                          tap_step_percent=2.5, tap_pos=0,
                                          shift_degree=150, name='MV-LV-Trafo')

    # Trafo3w
    hv_bus = pp.get_element_index(net, "bus", "Bus HV2")
    mv_bus = pp.get_element_index(net, "bus", "Bus MV0 20kV")
    lv_bus = pp.get_element_index(net, "bus", "Bus MV0")
    pp.create_transformer3w_from_parameters(net, hv_bus, mv_bus, lv_bus,
                                            vn_hv_kv=110, vn_mv_kv=20, vn_lv_kv=10,
                                            sn_hv_mva=40, sn_mv_mva=15, sn_lv_mva=25,
                                            vk_hv_percent=10.1, vk_mv_percent=10.1,
                                            vk_lv_percent=10.1, vkr_hv_percent=0.266667,
                                            vkr_mv_percent=0.033333, vkr_lv_percent=0.04,
                                            pfe_kw=0, i0_percent=0, shift_mv_degree=30,
                                            shift_lv_degree=30, tap_side="hv", tap_neutral=0,
                                            tap_min=-8, tap_max=8, tap_step_percent=1.25,
                                            tap_pos=0, name='HV-MV-MV-Trafo')

    # --- Static generators

    # HV
    pp.create_sgen(net, pp.get_element_index(net, "bus", 'Bus SB 5'), p_mw=20,
                   q_mvar=4, sn_mva=45, type='WP', name='Wind Park')

    # MV
    mv_sgens = pd.DataFrame()
    mv_sgens['sgen_name'] = ['Biogas plant', 'Further MV Generator', 'Industry Generator',
                             'PV Park']
    mv_sgens['bus'] = ['Bus MV6', 'Bus MV0', 'Bus MV0 20kV', 'Bus MV5']
    mv_sgens['p'] = [0.5, 0.5, 15, 2]
    mv_sgens['q'] = [0, 0.05, 3, 0.1]
    mv_sgens['sn'] = [0.75, 1, 20, 5]
    mv_sgens['type'] = ['SGEN', 'SGEN', 'SGEN', 'PV']

    for _, sgen in mv_sgens.iterrows():
        bus_idx = pp.get_element_index(net, "bus", sgen.bus)
        pp.create_sgen(net, bus_idx, p_mw=sgen.p, q_mvar=sgen.q, sn_mva=sgen.sn,
                       type=sgen.type, name=sgen.sgen_name)

    # LV
    lv_sgens = pd.DataFrame()
    lv_sgens['sgen_name'] = ['PV'] + ['PV(%s)' % i for i in range(1, 6)]
    lv_sgens['bus'] = ['Bus LV%s' % i for i in ['1.1', '1.3', '2.3', '2.4', '2.2.1', '2.2.2']]
    lv_sgens['p'] = [0.006, 0.005, 0.005, 0.005, 0.005, 0.005]
    lv_sgens['q'] = 0
    lv_sgens['sn'] = [0.012, 0.01, 0.01, 0.01, 0.01, 0.01]
    lv_sgens['type'] = 'PV'

    for _, sgen in lv_sgens.iterrows():
        bus_idx = pp.get_element_index(net, "bus", sgen.bus)
        pp.create_sgen(net, bus_idx, p_mw=sgen.p, q_mvar=sgen.q, sn_mva=sgen.sn,
                       type=sgen.type, name=sgen.sgen_name)

    # --- Loads

    # HV
    hv_loads = pd.DataFrame()
    hv_loads['load_name'] = ['MV Net %s' % i for i in range(5)]
    hv_loads['bus'] = ['Bus SB 4', 'Bus HV1', 'Bus HV2', 'Bus HV3', 'Bus HV4']
    hv_loads['p'] = 38
    hv_loads['q'] = 6

    for _, load in hv_loads.iterrows():
        bus_idx = pp.get_element_index(net, "bus", load.bus)
        pp.create_load(net, bus_idx, p_mw=load.p, q_mvar=load.q, name=load.load_name)

    # MV
    mv_loads = pd.DataFrame()
    mv_loads['load_name'] = ['Further MV-Rings', 'Industry Load'] + ['LV Net %s' % i for i in
                                                                     [1, 2, 3, 5, 6, 7]]
    mv_loads['bus'] = ['Bus MV0', 'Bus MV0 20kV'] + ['Bus MV%s' % i for i in [1, 2, 3, 5, 6, 7]]
    mv_loads['p'] = [6, 18, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    mv_loads['q'] = [2, 4, 0.1, 0.06, 0.06, 0.06, 0.06, 0.06]

    for _, load in mv_loads.iterrows():
        bus_idx = pp.get_element_index(net, "bus", load.bus)
        pp.create_load(net, bus_idx, p_mw=load.p, q_mvar=load.q, name=load.load_name)

    # LV
    lv_loads = pd.DataFrame()
    idx = ['', '(1)', '(2)', '(3)', '(4)', '(5)']
    lv_loads['load_name'] = ['Further LV-Feeders Load'] + [
        'Residential Load%s' % i for i in idx[0:5]] + ['Rural Load%s' % i for i in idx[0:6]]
    lv_loads['bus'] = ['Bus LV%s' % i for i in ['0', '1.1', '1.2', '1.3', '1.4', '1.5', '2.1',
                       '2.2', '2.3', '2.4', '2.2.1', '2.2.2']]
    lv_loads['p'] = [0.1] + [0.01]*11
    lv_loads['q'] = [0.01] + [0.03]*11

    for _, load in lv_loads.iterrows():
        bus_idx = pp.get_element_index(net, "bus", load.bus)
        pp.create_load(net, bus_idx, p_mw=load.p, q_mvar=load.q, name=load.load_name)

    # --- Other

    # Shunt
    pp.create_shunt(net, pp.get_element_index(net, "bus", 'Bus HV1'), p_mw=0, q_mvar=-0.96,
                    name='Shunt')

    # ExtGrids
    pp.create_ext_grid(net, pp.get_element_index(net, "bus", 'Double Busbar 1'), vm_pu=1.03,
                       va_degree=0, name='External grid', s_sc_max_mva=10000, rx_max=0.1,
                       rx_min=0.1)
    # Gen
    pp.create_gen(net, pp.get_element_index(net, "bus", 'Bus HV4'), vm_pu=1.03, p_mw=100,
                  name='Gas turbine')

    # Impedance
    pp.create_impedance(net, pp.get_element_index(net, "bus", 'Bus HV3'),
                        pp.get_element_index(net, "bus", 'Bus HV1'), rft_pu=0.074873,
                        xft_pu=0.198872, sn_mva=100, name='Impedance')

    # xwards
    pp.create_xward(net, pp.get_element_index(net, "bus", 'Bus HV3'), ps_mw=23.942,
                    qs_mvar=-12.24187, pz_mw=2.814571, qz_mvar=0, r_ohm=0, x_ohm=12.18951,
                    vm_pu=1.02616, name='XWard 1')
    pp.create_xward(net, pp.get_element_index(net, "bus", 'Bus HV1'), ps_mw=3.776,
                    qs_mvar=-7.769979, pz_mw=9.174917, qz_mvar=0, r_ohm=0, x_ohm=50.56217,
                    vm_pu=1.024001, name='XWard 2')

    # --- Switches

    # HV
    # Bus-bus switches
    hv_bus_sw = pd.DataFrame()
    hv_bus_sw['bus_name'] = ['DB DS%s' % i for i in range(14)] + \
                            ['DB CB%s' % i for i in range(5)] + \
                            ['SB DS%s.%s' % (i, j) for i in range(1, 6) for j in range(1, 3)] + \
                            ['SB CB%s' % i for i in range(1, 6)]
    hv_bus_sw['from_bus'] = ['Double Busbar %s' % i for i in [2, 1, 2, 1, 2, 1, 2, 1, 2, 1]] + \
                            ['Bus DB T%s' % i for i in [2, 4, 6, 8, 0, 3, 5, 7, 9]] + \
                            ['Bus SB T1.1', 'Single Busbar', 'Bus SB T2.1', 'Single Busbar',
                             'Bus SB T3.1', 'Single Busbar', 'Bus SB T4.1', 'Single Busbar',
                             'Bus SB T5.1', 'Single Busbar'] + \
                            ['Bus SB T%s.2' % i for i in range(1, 6)]
    hv_bus_sw['to_bus'] = ['Bus DB %s' % i for i in
                           ['T0', 'T1', 'T3', 'T3', 'T5', 'T5', 'T7', 'T7', 'T9', 'T9',
                            '1', '2', '3', '4', 'T1', 'T2', 'T4', 'T6', 'T8']] + \
                          ['Bus SB %s' % i for i in
                           ['1', 'T1.2', '2', 'T2.2', '3', 'T3.2', '4', 'T4.2', '5', 'T5.2']] + \
                          ['Bus SB T%s.1' % i for i in range(1, 6)]
    hv_bus_sw['type'] = ['DS']*14 + ['CB']*5 + ['DS']*10 + ['CB']*5
    hv_bus_sw['et'] = 'b'
    hv_bus_sw['closed'] = [bool(i) for i in [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                           1] + [1]*15]

    for _, switch in hv_bus_sw.iterrows():
        from_bus = pp.get_element_index(net, "bus", switch.from_bus)
        to_bus = pp.get_element_index(net, "bus", switch.to_bus)
        pp.create_switch(net, from_bus, to_bus, et=switch.et,
                         closed=switch.closed, type=switch.type, name=switch.bus_name)

    # Bus-Line switches
    hv_buses = net.bus[(net.bus.vn_kv == 380) | (net.bus.vn_kv == 110)].index
    hv_ls = net.line[(net.line.from_bus.isin(hv_buses)) & (net.line.to_bus.isin(hv_buses))]
    for _, line in hv_ls.iterrows():
        for bus in [line.from_bus, line.to_bus]:
            pp.create_switch(net, bus, line.name, et='l', closed=True, type='LBS',
                             name='Switch %s - %s' % (net.bus.name.at[bus], line['name']))

    # MV
    # Bus-line switches
    mv_buses = net.bus[(net.bus.vn_kv == 10) | (net.bus.vn_kv == 20)].index
    mv_ls = net.line[(net.line.from_bus.isin(mv_buses)) & (net.line.to_bus.isin(mv_buses))]
    for _, line in mv_ls.iterrows():
        for bus in [line.from_bus, line.to_bus]:
            pp.create_switch(net, bus, line.name, et='l', closed=True, type='LBS',
                             name='Switch %s - %s' % (net.bus.name.at[bus], line['name']))

    open_switch_id = net.switch[(net.switch.name == 'Switch Bus MV5 - MV Line5')].index
    net.switch.closed.loc[open_switch_id] = False

    # LV
    # Bus-line switches
    lv_buses = net.bus[net.bus.vn_kv == 0.4].index
    lv_ls = net.line[(net.line.from_bus.isin(lv_buses)) & (net.line.to_bus.isin(lv_buses))]
    for _, line in lv_ls.iterrows():
        for bus in [line.from_bus, line.to_bus]:
            pp.create_switch(net, bus, line.name, et='l', closed=True, type='LBS',
                             name='Switch %s - %s' % (net.bus.name.at[bus], line['name']))

    # Trafoswitches
    # HV
    pp.create_switch(net, pp.get_element_index(net, "bus", 'Bus DB 2'),
                     pp.get_element_index(net, "trafo", 'EHV-HV-Trafo'), et='t', closed=True,
                     type='LBS', name='Switch DB2 - EHV-HV-Trafo')
    pp.create_switch(net, pp.get_element_index(net, "bus", 'Bus SB 1'),
                     pp.get_element_index(net, "trafo", 'EHV-HV-Trafo'), et='t', closed=True,
                     type='LBS', name='Switch SB1 - EHV-HV-Trafo')
    # LV
    pp.create_switch(net, pp.get_element_index(net, "bus", 'Bus MV4'),
                     pp.get_element_index(net, "trafo", 'MV-LV-Trafo'), et='t', closed=True,
                     type='LBS', name='Switch MV4 - MV-LV-Trafo')
    pp.create_switch(net, pp.get_element_index(net, "bus", 'Bus LV0'),
                     pp.get_element_index(net, "trafo", 'MV-LV-Trafo'), et='t', closed=True,
                     type='LBS', name='Switch LV0 - MV-LV-Trafo')

    # --- Powerflow

    # run power flow and generate result tables
    pp.runpp(net, init='dc', calculate_voltage_angles=True, Numba=False)

    return net
