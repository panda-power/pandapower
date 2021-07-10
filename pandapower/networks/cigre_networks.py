# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from pandas import read_json
from numpy import nan
import pandapower as pp
try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


def create_cigre_network_hv(length_km_6a_6b=0.1):
    """
    Create the CIGRE HV Grid from final Report of Task Force C6.04.02:
    "Benchmark Systems for Network Integration of Renewable and Distributed Energy Resources”, 2014.

    OPTIONAL:
        **length_km_6a_6b** (float, 0.1) - Length of the line segment 9 between buses 6a and 6b
            which is intended to be optional with user-definable geometrical configuration, length
            and installation type.

    OUTPUT:
        **net** - The pandapower format network.
    """
    net_cigre_hv = pp.create_empty_network()

    # Linedata
    # Line220kV
    line_data = {'c_nf_per_km': 9.08, 'r_ohm_per_km': 0.0653,
                 'x_ohm_per_km': 0.398, 'max_i_ka': 1.14,
                 'type': 'ol'}

    pp.create_std_type(net_cigre_hv, line_data, 'Line220kV', element='line')

    # Line380kV
    line_data = {'c_nf_per_km': 11.5, 'r_ohm_per_km': 0.0328,
                 'x_ohm_per_km': 0.312, 'max_i_ka': 1.32,
                 'type': 'ol'}

    pp.create_std_type(net_cigre_hv, line_data, 'Line380kV', element='line')

    # Busses
    bus1 = pp.create_bus(net_cigre_hv, name='Bus 1', vn_kv=220, type='b', zone='CIGRE_HV')
    bus2 = pp.create_bus(net_cigre_hv, name='Bus 2', vn_kv=220, type='b', zone='CIGRE_HV')
    bus3 = pp.create_bus(net_cigre_hv, name='Bus 3', vn_kv=220, type='b', zone='CIGRE_HV')
    bus4 = pp.create_bus(net_cigre_hv, name='Bus 4', vn_kv=220, type='b', zone='CIGRE_HV')
    bus5 = pp.create_bus(net_cigre_hv, name='Bus 5', vn_kv=220, type='b', zone='CIGRE_HV')
    bus6a = pp.create_bus(net_cigre_hv, name='Bus 6a', vn_kv=220, type='b', zone='CIGRE_HV')
    bus6b = pp.create_bus(net_cigre_hv, name='Bus 6b', vn_kv=220, type='b', zone='CIGRE_HV')
    bus7 = pp.create_bus(net_cigre_hv, name='Bus 7', vn_kv=380, type='b', zone='CIGRE_HV')
    bus8 = pp.create_bus(net_cigre_hv, name='Bus 8', vn_kv=380, type='b', zone='CIGRE_HV')
    bus9 = pp.create_bus(net_cigre_hv, name='Bus 9', vn_kv=22, type='b', zone='CIGRE_HV')
    bus10 = pp.create_bus(net_cigre_hv, name='Bus 10', vn_kv=22, type='b', zone='CIGRE_HV')
    bus11 = pp.create_bus(net_cigre_hv, name='Bus 11', vn_kv=22, type='b', zone='CIGRE_HV')
    bus12 = pp.create_bus(net_cigre_hv, name='Bus 12', vn_kv=22, type='b', zone='CIGRE_HV')

    # Lines
    pp.create_line(net_cigre_hv, bus1, bus2, length_km=100,
                   std_type='Line220kV', name='Line 1-2')
    pp.create_line(net_cigre_hv, bus1, bus6a, length_km=300,
                   std_type='Line220kV', name='Line 1-6a')
    pp.create_line(net_cigre_hv, bus2, bus5, length_km=300,
                   std_type='Line220kV', name='Line 2-5')
    pp.create_line(net_cigre_hv, bus3, bus4, length_km=100,
                   std_type='Line220kV', name='Line 3-4')
    pp.create_line(net_cigre_hv, bus3, bus4, length_km=100,
                   std_type='Line220kV', name='Line 3-4_2')
    pp.create_line(net_cigre_hv, bus4, bus5, length_km=300,
                   std_type='Line220kV', name='Line 4-5')
    pp.create_line(net_cigre_hv, bus4, bus6a, length_km=300,
                   std_type='Line220kV', name='Line 4-6a')
    pp.create_line(net_cigre_hv, bus7, bus8, length_km=600,
                   std_type='Line380kV', name='Line 7-8')
    pp.create_line(net_cigre_hv, bus6a, bus6b, length_km=length_km_6a_6b,
                   std_type='Line220kV', name='Line 6a-6b')

    # Trafos
    pp.create_transformer_from_parameters(net_cigre_hv, bus7, bus1, sn_mva=1000,
                                          vn_hv_kv=380, vn_lv_kv=220, vkr_percent=0.0,
                                          vk_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=0.0, name='Trafo 1-7')
    pp.create_transformer_from_parameters(net_cigre_hv, bus8, bus3, sn_mva=1000,
                                          vn_hv_kv=380, vn_lv_kv=220, vkr_percent=0.0,
                                          vk_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=0.0, name='Trafo 3-8')

    pp.create_transformer_from_parameters(net_cigre_hv, bus1, bus9, sn_mva=1000,
                                          vn_hv_kv=220, vn_lv_kv=22, vkr_percent=0.0,
                                          vk_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=330.0, name='Trafo 9-1')
    pp.create_transformer_from_parameters(net_cigre_hv, bus2, bus10, sn_mva=1000,
                                          vn_hv_kv=220, vn_lv_kv=22, vkr_percent=0.0,
                                          vk_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=330.0, name='Trafo 10-2')
    pp.create_transformer_from_parameters(net_cigre_hv, bus3, bus11, sn_mva=1000,
                                          vn_hv_kv=220, vn_lv_kv=22, vkr_percent=0.0,
                                          vk_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=330.0, name='Trafo 11-3')
    pp.create_transformer_from_parameters(net_cigre_hv, bus6b, bus12, sn_mva=500,
                                          vn_hv_kv=220, vn_lv_kv=22, vkr_percent=0.0,
                                          vk_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=330.0, name='Trafo 12-6b')

    # Loads
    pp.create_load(net_cigre_hv, bus2, p_mw=285, q_mvar=200, name='Load 2')
    pp.create_load(net_cigre_hv, bus3, p_mw=325, q_mvar=244, name='Load 3')
    pp.create_load(net_cigre_hv, bus4, p_mw=326, q_mvar=244, name='Load 4')
    pp.create_load(net_cigre_hv, bus5, p_mw=103, q_mvar=62, name='Load 5')
    pp.create_load(net_cigre_hv, bus6a, p_mw=435, q_mvar=296, name='Load 6a')

    # External grid
    pp.create_ext_grid(net_cigre_hv, bus9, vm_pu=1.03, va_degree=0, name='Generator 9')

    # Generators
    pp.create_gen(net_cigre_hv, bus10, vm_pu=1.03, p_mw=500, name='Generator 10')
    pp.create_gen(net_cigre_hv, bus11, vm_pu=1.03, p_mw=200, name='Generator 11')
    pp.create_gen(net_cigre_hv, bus12, vm_pu=1.03, p_mw=300, name='Generator 12')

    # Shunts
    pp.create_shunt(net_cigre_hv, bus4, p_mw=0.0, q_mvar=-0.16, name='Shunt 4')
    pp.create_shunt(net_cigre_hv, bus5, p_mw=0.0, q_mvar=-0.08, name='Shunt 5')
    pp.create_shunt(net_cigre_hv, bus6a, p_mw=0.0, q_mvar=-0.18, name='Shunt 6a')

    # Bus geo data
    net_cigre_hv.bus_geodata = read_json(
        """{"x":{"0":4,"1":8,"2":20,"3":16,"4":12,"5":8,"6":12,"7":4,"8":20,"9":0,"10":8,"11":24,
        "12":16},"y":{"0":8.0,"1":8.0,"2":8.0,"3":8.0,"4":8.0,"5":6.0,"6":4.5,"7":1.0,"8":1.0,
        "9":8.0,"10":12.0,"11":8.0,"12":4.5}}""")
    # Match bus.index
    net_cigre_hv.bus_geodata = net_cigre_hv.bus_geodata.loc[net_cigre_hv.bus.index]
    return net_cigre_hv


def create_cigre_network_mv(with_der=False):
    """
    Create the CIGRE MV Grid from final Report of Task Force C6.04.02:
    "Benchmark Systems for Network Integration of Renewable and Distributed Energy Resources”, 2014.

    OPTIONAL:
        **with_der** (boolean or str, False) - Range of DER consideration, which should be in
            (False, "pv_wind", "all"). The DER types, dimensions and locations are taken from CIGRE
            CaseStudy: "DER in Medium Voltage Systems"

    OUTPUT:
        **net** - The pandapower format network.
    """
    if with_der is True:
        raise ValueError("'with_der=True' is deprecated. Please use 'with_der=pv_wind'")
    if with_der not in [False, "pv_wind", "all"]:
        raise ValueError("'with_der' is unknown. It should be in [False, 'pv_wind', 'all'].")

    net_cigre_mv = pp.create_empty_network()

    # Linedata
    line_data = {'c_nf_per_km': 151.1749, 'r_ohm_per_km': 0.501,
                 'x_ohm_per_km': 0.716, 'max_i_ka': 0.145,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_mv, line_data, name='CABLE_CIGRE_MV', element='line')

    line_data = {'c_nf_per_km': 10.09679, 'r_ohm_per_km': 0.510,
                 'x_ohm_per_km': 0.366, 'max_i_ka': 0.195,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_mv, line_data, name='OHL_CIGRE_MV', element='line')

    # Busses
    bus0 = pp.create_bus(net_cigre_mv, name='Bus 0', vn_kv=110, type='b', zone='CIGRE_MV')
    buses = pp.create_buses(net_cigre_mv, 14, name=['Bus %i' % i for i in range(1, 15)], vn_kv=20,
                            type='b', zone='CIGRE_MV')

    # Lines
    pp.create_line(net_cigre_mv, buses[0], buses[1], length_km=2.82,
                   std_type='CABLE_CIGRE_MV', name='Line 1-2')
    pp.create_line(net_cigre_mv, buses[1], buses[2], length_km=4.42,
                   std_type='CABLE_CIGRE_MV', name='Line 2-3')
    pp.create_line(net_cigre_mv, buses[2], buses[3], length_km=0.61,
                   std_type='CABLE_CIGRE_MV', name='Line 3-4')
    pp.create_line(net_cigre_mv, buses[3], buses[4], length_km=0.56,
                   std_type='CABLE_CIGRE_MV', name='Line 4-5')
    pp.create_line(net_cigre_mv, buses[4], buses[5], length_km=1.54,
                   std_type='CABLE_CIGRE_MV', name='Line 5-6')
    pp.create_line(net_cigre_mv, buses[6], buses[7], length_km=1.67,
                   std_type='CABLE_CIGRE_MV', name='Line 7-8')
    pp.create_line(net_cigre_mv, buses[7], buses[8], length_km=0.32,
                   std_type='CABLE_CIGRE_MV', name='Line 8-9')
    pp.create_line(net_cigre_mv, buses[8], buses[9], length_km=0.77,
                   std_type='CABLE_CIGRE_MV', name='Line 9-10')
    pp.create_line(net_cigre_mv, buses[9], buses[10], length_km=0.33,
                   std_type='CABLE_CIGRE_MV', name='Line 10-11')
    pp.create_line(net_cigre_mv, buses[2], buses[7], length_km=1.3,
                   std_type='CABLE_CIGRE_MV', name='Line 3-8')
    pp.create_line(net_cigre_mv, buses[11], buses[12], length_km=4.89,
                   std_type='OHL_CIGRE_MV', name='Line 12-13')
    pp.create_line(net_cigre_mv, buses[12], buses[13], length_km=2.99,
                   std_type='OHL_CIGRE_MV', name='Line 13-14')

    line6_7 = pp.create_line(net_cigre_mv, buses[5], buses[6], length_km=0.24,
                             std_type='CABLE_CIGRE_MV', name='Line 6-7')
    line4_11 = pp.create_line(net_cigre_mv, buses[10], buses[3], length_km=0.49,
                              std_type='CABLE_CIGRE_MV', name='Line 11-4')
    line8_14 = pp.create_line(net_cigre_mv, buses[13], buses[7], length_km=2.,
                              std_type='OHL_CIGRE_MV', name='Line 14-8')

    # Ext-Grid
    pp.create_ext_grid(net_cigre_mv, bus0, vm_pu=1.03, va_degree=0.,
                       s_sc_max_mva=5000, s_sc_min_mva=5000, rx_max=0.1, rx_min=0.1)

    # Trafos
    trafo0 = pp.create_transformer_from_parameters(net_cigre_mv, bus0, buses[0], sn_mva=25,
                                                   vn_hv_kv=110, vn_lv_kv=20, vkr_percent=0.16,
                                                   vk_percent=12.00107, pfe_kw=0, i0_percent=0,
                                                   shift_degree=30.0, name='Trafo 0-1')
    trafo1 = pp.create_transformer_from_parameters(net_cigre_mv, bus0, buses[11], sn_mva=25,
                                                   vn_hv_kv=110, vn_lv_kv=20, vkr_percent=0.16,
                                                   vk_percent=12.00107, pfe_kw=0, i0_percent=0,
                                                   shift_degree=30.0, name='Trafo 0-12')

    # Switches
    # S2
    pp.create_switch(net_cigre_mv, buses[5], line6_7, et='l', closed=True, type='LBS')
    pp.create_switch(net_cigre_mv, buses[6], line6_7, et='l', closed=False, type='LBS', name='S2')
    # S3
    pp.create_switch(net_cigre_mv, buses[3], line4_11, et='l', closed=False, type='LBS', name='S3')
    pp.create_switch(net_cigre_mv, buses[10], line4_11, et='l', closed=True, type='LBS')
    # S1
    pp.create_switch(net_cigre_mv, buses[7], line8_14, et='l', closed=False, type='LBS', name='S1')
    pp.create_switch(net_cigre_mv, buses[13], line8_14, et='l', closed=True, type='LBS')
    # trafos
    pp.create_switch(net_cigre_mv, bus0, trafo0, et='t', closed=True, type='CB')
    pp.create_switch(net_cigre_mv, bus0, trafo1, et='t', closed=True, type='CB')

    # Loads
    # Residential
    pp.create_load_from_cosphi(net_cigre_mv, buses[0], 15.3, 0.98, "underexcited", name='Load R1')
    pp.create_load_from_cosphi(net_cigre_mv, buses[2], 0.285, 0.97, "underexcited", name='Load R3')
    pp.create_load_from_cosphi(net_cigre_mv, buses[3], 0.445, 0.97, "underexcited", name='Load R4')
    pp.create_load_from_cosphi(net_cigre_mv, buses[4], 0.750, 0.97, "underexcited", name='Load R5')
    pp.create_load_from_cosphi(net_cigre_mv, buses[5], 0.565, 0.97, "underexcited", name='Load R6')
    pp.create_load_from_cosphi(net_cigre_mv, buses[7], 0.605, 0.97, "underexcited", name='Load R8')
    pp.create_load_from_cosphi(net_cigre_mv, buses[9], 0.490, 0.97, "underexcited", name='Load R10')
    pp.create_load_from_cosphi(net_cigre_mv, buses[10], 0.340, 0.97, "underexcited", name='Load R11')
    pp.create_load_from_cosphi(net_cigre_mv, buses[11], 15.3, 0.98, "underexcited", name='Load R12')
    pp.create_load_from_cosphi(net_cigre_mv, buses[13], 0.215, 0.97, "underexcited", name='Load R14')

    # Commercial / Industrial
    pp.create_load_from_cosphi(net_cigre_mv, buses[0], 5.1, 0.95, "underexcited", name='Load CI1')
    pp.create_load_from_cosphi(net_cigre_mv, buses[2], 0.265, 0.85, "underexcited", name='Load CI3')
    pp.create_load_from_cosphi(net_cigre_mv, buses[6], 0.090, 0.85, "underexcited", name='Load CI7')
    pp.create_load_from_cosphi(net_cigre_mv, buses[8], 0.675, 0.85, "underexcited", name='Load CI9')
    pp.create_load_from_cosphi(net_cigre_mv, buses[9], 0.080, 0.85, "underexcited", name='Load CI10')
    pp.create_load_from_cosphi(net_cigre_mv, buses[11], 5.28, 0.95, "underexcited", name='Load CI12')
    pp.create_load_from_cosphi(net_cigre_mv, buses[12], 0.04, 0.85, "underexcited", name='Load CI13')
    pp.create_load_from_cosphi(net_cigre_mv, buses[13], 0.390, 0.85, "underexcited", name='Load CI14')

    # Optional distributed energy recources
    if with_der in ["pv_wind", "all"]:
        pp.create_sgen(net_cigre_mv, buses[2], 0.02, q_mvar=0, sn_mva=0.02, name='PV 3', type='PV')
        pp.create_sgen(net_cigre_mv, buses[3], 0.020, q_mvar=0, sn_mva=0.02, name='PV 4', type='PV')
        pp.create_sgen(net_cigre_mv, buses[4], 0.030, q_mvar=0, sn_mva=0.03, name='PV 5', type='PV')
        pp.create_sgen(net_cigre_mv, buses[5], 0.030, q_mvar=0, sn_mva=0.03, name='PV 6', type='PV')
        pp.create_sgen(net_cigre_mv, buses[7], 0.030, q_mvar=0, sn_mva=0.03, name='PV 8', type='PV')
        pp.create_sgen(net_cigre_mv, buses[8], 0.030, q_mvar=0, sn_mva=0.03, name='PV 9', type='PV')
        pp.create_sgen(net_cigre_mv, buses[9], 0.040, q_mvar=0, sn_mva=0.04, name='PV 10', type='PV')
        pp.create_sgen(net_cigre_mv, buses[10], 0.010, q_mvar=0, sn_mva=0.01, name='PV 11', type='PV')
        pp.create_sgen(net_cigre_mv, buses[6], 1.5, q_mvar=0, sn_mva=1.5, name='WKA 7',
                       type='WP')
        if with_der == "all":
            pp.create_storage(net_cigre_mv, bus=buses[4], p_mw=0.6, max_e_mwh=nan, sn_mva=0.2,
                              name='Battery 1', type='Battery', max_p_mw=0.6, min_p_mw=-0.6)
            pp.create_sgen(net_cigre_mv, bus=buses[4], p_mw=0.033, sn_mva=0.033,
                           name='Residential fuel cell 1', type='Residential fuel cell')
            pp.create_sgen(net_cigre_mv, bus=buses[8], p_mw=0.310, sn_mva=0.31, name='CHP diesel 1',
                           type='CHP diesel')
            pp.create_sgen(net_cigre_mv, bus=buses[8], p_mw=0.212, sn_mva=0.212, name='Fuel cell 1',
                           type='Fuel cell')
            pp.create_storage(net_cigre_mv, bus=buses[9], p_mw=0.200, max_e_mwh=nan, sn_mva=0.2,
                              name='Battery 2', type='Battery', max_p_mw=0.2, min_p_mw=-0.2)
            pp.create_sgen(net_cigre_mv, bus=buses[9], p_mw=0.014, sn_mva=.014,
                           name='Residential fuel cell 2', type='Residential fuel cell')

    # Bus geo data
    net_cigre_mv.bus_geodata = read_json(
        """{"x":{"0":7.0,"1":4.0,"2":4.0,"3":4.0,"4":2.5,"5":1.0,"6":1.0,"7":8.0,"8":8.0,"9":6.0,
        "10":4.0,"11":4.0,"12":10.0,"13":10.0,"14":10.0},"y":{"0":16,"1":15,"2":13,"3":11,"4":9,
        "5":7,"6":3,"7":3,"8":5,"9":5,"10":5,"11":7,"12":15,"13":11,"14":5}}""")
    # Match bus.index
    net_cigre_mv.bus_geodata = net_cigre_mv.bus_geodata.loc[net_cigre_mv.bus.index]
    return net_cigre_mv


def create_cigre_network_lv():
    """
    Create the CIGRE LV Grid from final Report of Task Force C6.04.02:
    "Benchmark Systems for Network Integration of Renewable and Distributed Energy Resources”, 2014.

    OUTPUT:
        **net** - The pandapower format network.
    """
    net_cigre_lv = pp.create_empty_network()

    # Linedata
    # UG1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.162,
                 'x_ohm_per_km': 0.0832, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG1', element='line')

    # UG2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.2647,
                 'x_ohm_per_km': 0.0823, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG2', element='line')

    # UG3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.822,
                 'x_ohm_per_km': 0.0847, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG3', element='line')

    # OH1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.4917,
                 'x_ohm_per_km': 0.2847, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH1', element='line')

    # OH2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 1.3207,
                 'x_ohm_per_km': 0.321, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH2', element='line')

    # OH3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 2.0167,
                 'x_ohm_per_km': 0.3343, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH3', element='line')

    # Busses
    bus0 = pp.create_bus(net_cigre_lv, name='Bus 0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busR0 = pp.create_bus(net_cigre_lv, name='Bus R0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busR1 = pp.create_bus(net_cigre_lv, name='Bus R1', vn_kv=0.4, type='b', zone='CIGRE_LV')
    busR2 = pp.create_bus(net_cigre_lv, name='Bus R2', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR3 = pp.create_bus(net_cigre_lv, name='Bus R3', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR4 = pp.create_bus(net_cigre_lv, name='Bus R4', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR5 = pp.create_bus(net_cigre_lv, name='Bus R5', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR6 = pp.create_bus(net_cigre_lv, name='Bus R6', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR7 = pp.create_bus(net_cigre_lv, name='Bus R7', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR8 = pp.create_bus(net_cigre_lv, name='Bus R8', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR9 = pp.create_bus(net_cigre_lv, name='Bus R9', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR10 = pp.create_bus(net_cigre_lv, name='Bus R10', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR11 = pp.create_bus(net_cigre_lv, name='Bus R11', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR12 = pp.create_bus(net_cigre_lv, name='Bus R12', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR13 = pp.create_bus(net_cigre_lv, name='Bus R13', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR14 = pp.create_bus(net_cigre_lv, name='Bus R14', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR15 = pp.create_bus(net_cigre_lv, name='Bus R15', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR16 = pp.create_bus(net_cigre_lv, name='Bus R16', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR17 = pp.create_bus(net_cigre_lv, name='Bus R17', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR18 = pp.create_bus(net_cigre_lv, name='Bus R18', vn_kv=0.4, type='m', zone='CIGRE_LV')

    busI0 = pp.create_bus(net_cigre_lv, name='Bus I0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busI1 = pp.create_bus(net_cigre_lv, name='Bus I1', vn_kv=0.4, type='b', zone='CIGRE_LV')
    busI2 = pp.create_bus(net_cigre_lv, name='Bus I2', vn_kv=0.4, type='m', zone='CIGRE_LV')

    busC0 = pp.create_bus(net_cigre_lv, name='Bus C0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busC1 = pp.create_bus(net_cigre_lv, name='Bus C1', vn_kv=0.4, type='b', zone='CIGRE_LV')
    busC2 = pp.create_bus(net_cigre_lv, name='Bus C2', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC3 = pp.create_bus(net_cigre_lv, name='Bus C3', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC4 = pp.create_bus(net_cigre_lv, name='Bus C4', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC5 = pp.create_bus(net_cigre_lv, name='Bus C5', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC6 = pp.create_bus(net_cigre_lv, name='Bus C6', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC7 = pp.create_bus(net_cigre_lv, name='Bus C7', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC8 = pp.create_bus(net_cigre_lv, name='Bus C8', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC9 = pp.create_bus(net_cigre_lv, name='Bus C9', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC10 = pp.create_bus(net_cigre_lv, name='Bus C10', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC11 = pp.create_bus(net_cigre_lv, name='Bus C11', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC12 = pp.create_bus(net_cigre_lv, name='Bus C12', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC13 = pp.create_bus(net_cigre_lv, name='Bus C13', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC14 = pp.create_bus(net_cigre_lv, name='Bus C14', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC15 = pp.create_bus(net_cigre_lv, name='Bus C15', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC16 = pp.create_bus(net_cigre_lv, name='Bus C16', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC17 = pp.create_bus(net_cigre_lv, name='Bus C17', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC18 = pp.create_bus(net_cigre_lv, name='Bus C18', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC19 = pp.create_bus(net_cigre_lv, name='Bus C19', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC20 = pp.create_bus(net_cigre_lv, name='Bus C20', vn_kv=0.4, type='m', zone='CIGRE_LV')

    # Lines
    pp.create_line(net_cigre_lv, busR1, busR2, length_km=0.035, std_type='UG1',
                   name='Line R1-R2')
    pp.create_line(net_cigre_lv, busR2, busR3, length_km=0.035, std_type='UG1',
                   name='Line R2-R3')
    pp.create_line(net_cigre_lv, busR3, busR4, length_km=0.035, std_type='UG1',
                   name='Line R3-R4')
    pp.create_line(net_cigre_lv, busR4, busR5, length_km=0.035, std_type='UG1',
                   name='Line R4-R5')
    pp.create_line(net_cigre_lv, busR5, busR6, length_km=0.035, std_type='UG1',
                   name='Line R5-R6')
    pp.create_line(net_cigre_lv, busR6, busR7, length_km=0.035, std_type='UG1',
                   name='Line R6-R7')
    pp.create_line(net_cigre_lv, busR7, busR8, length_km=0.035, std_type='UG1',
                   name='Line R7-R8')
    pp.create_line(net_cigre_lv, busR8, busR9, length_km=0.035, std_type='UG1',
                   name='Line R8-R9')
    pp.create_line(net_cigre_lv, busR9, busR10, length_km=0.035, std_type='UG1',
                   name='Line R9-R10')
    pp.create_line(net_cigre_lv, busR3, busR11, length_km=0.030, std_type='UG3',
                   name='Line R3-R11')
    pp.create_line(net_cigre_lv, busR4, busR12, length_km=0.035, std_type='UG3',
                   name='Line R4-R12')
    pp.create_line(net_cigre_lv, busR12, busR13, length_km=0.035, std_type='UG3',
                   name='Line R12-R13')
    pp.create_line(net_cigre_lv, busR13, busR14, length_km=0.035, std_type='UG3',
                   name='Line R13-R14')
    pp.create_line(net_cigre_lv, busR14, busR15, length_km=0.030, std_type='UG3',
                   name='Line R14-R15')
    pp.create_line(net_cigre_lv, busR6, busR16, length_km=0.030, std_type='UG3',
                   name='Line R6-R16')
    pp.create_line(net_cigre_lv, busR9, busR17, length_km=0.030, std_type='UG3',
                   name='Line R9-R17')
    pp.create_line(net_cigre_lv, busR10, busR18, length_km=0.030, std_type='UG3',
                   name='Line R10-R18')

    pp.create_line(net_cigre_lv, busI1, busI2, length_km=0.2, std_type='UG2',
                   name='Line I1-I2')

    pp.create_line(net_cigre_lv, busC1, busC2, length_km=0.030, std_type='OH1',
                   name='Line C1-C2')
    pp.create_line(net_cigre_lv, busC2, busC3, length_km=0.030, std_type='OH1',
                   name='Line C2-C3')
    pp.create_line(net_cigre_lv, busC3, busC4, length_km=0.030, std_type='OH1',
                   name='Line C3-C4')
    pp.create_line(net_cigre_lv, busC4, busC5, length_km=0.030, std_type='OH1',
                   name='Line C4-C5')
    pp.create_line(net_cigre_lv, busC5, busC6, length_km=0.030, std_type='OH1',
                   name='Line C5-C6')
    pp.create_line(net_cigre_lv, busC6, busC7, length_km=0.030, std_type='OH1',
                   name='Line C6-C7')
    pp.create_line(net_cigre_lv, busC7, busC8, length_km=0.030, std_type='OH1',
                   name='Line C7-C8')
    pp.create_line(net_cigre_lv, busC8, busC9, length_km=0.030, std_type='OH1',
                   name='Line C8-C9')
    pp.create_line(net_cigre_lv, busC3, busC10, length_km=0.030, std_type='OH2',
                   name='Line C3-C10')
    pp.create_line(net_cigre_lv, busC10, busC11, length_km=0.030, std_type='OH2',
                   name='Line C10-C11')
    pp.create_line(net_cigre_lv, busC11, busC12, length_km=0.030, std_type='OH3',
                   name='Line C11-C12')
    pp.create_line(net_cigre_lv, busC11, busC13, length_km=0.030, std_type='OH3',
                   name='Line C11-C13')
    pp.create_line(net_cigre_lv, busC10, busC14, length_km=0.030, std_type='OH3',
                   name='Line C10-C14')
    pp.create_line(net_cigre_lv, busC5, busC15, length_km=0.030, std_type='OH2',
                   name='Line C5-C15')
    pp.create_line(net_cigre_lv, busC15, busC16, length_km=0.030, std_type='OH2',
                   name='Line C15-C16')
    pp.create_line(net_cigre_lv, busC15, busC17, length_km=0.030, std_type='OH3',
                   name='Line C15-C17')
    pp.create_line(net_cigre_lv, busC16, busC18, length_km=0.030, std_type='OH3',
                   name='Line C16-C18')
    pp.create_line(net_cigre_lv, busC8, busC19, length_km=0.030, std_type='OH3',
                   name='Line C8-C19')
    pp.create_line(net_cigre_lv, busC9, busC20, length_km=0.030, std_type='OH3',
                   name='Line C9-C20')

    # Trafos
    pp.create_transformer_from_parameters(net_cigre_lv, busR0, busR1, sn_mva=0.5, vn_hv_kv=20.0,
                                          vn_lv_kv=0.4, vkr_percent=1.0, vk_percent=4.123106,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=30.0,
                                          tap_pos=0.0, name='Trafo R0-R1')

    pp.create_transformer_from_parameters(net_cigre_lv, busI0, busI1, sn_mva=0.15, vn_hv_kv=20.0,
                                          vn_lv_kv=0.4, vkr_percent=1.003125, vk_percent=4.126896,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=30.0,
                                          tap_pos=0.0, name='Trafo I0-I1')

    pp.create_transformer_from_parameters(net_cigre_lv, busC0, busC1, sn_mva=0.3, vn_hv_kv=20.0,
                                          vn_lv_kv=0.4, vkr_percent=0.993750, vk_percent=4.115529,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=30.0,
                                          tap_pos=0.0, name='Trafo C0-C1')

    # External grid
    pp.create_ext_grid(net_cigre_lv, bus0, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=100.0,
                       s_sc_min_mva=100.0, rx_max=1.0, rx_min=1.0)

    # Loads
    pp.create_load(net_cigre_lv, busR1, p_mw=0.19, q_mvar=0.062449980, name='Load R1')
    pp.create_load(net_cigre_lv, busR11, p_mw=0.01425, q_mvar=0.004683748, name='Load R11')
    pp.create_load(net_cigre_lv, busR15, p_mw=0.0494, q_mvar=0.016236995, name='Load R15')
    pp.create_load(net_cigre_lv, busR16, p_mw=0.05225, q_mvar=0.017173744, name='Load R16')
    pp.create_load(net_cigre_lv, busR17, p_mw=0.03325, q_mvar=0.010928746, name='Load R17')
    pp.create_load(net_cigre_lv, busR18, p_mw=0.04465, q_mvar=0.014675745, name='Load R18')
    pp.create_load(net_cigre_lv, busI2, p_mw=0.0850, q_mvar=0.052678269, name='Load I2')
    pp.create_load(net_cigre_lv, busC1, p_mw=0.1080, q_mvar=0.052306787, name='Load C1')
    pp.create_load(net_cigre_lv, busC12, p_mw=0.018, q_mvar=0.008717798, name='Load C12')
    pp.create_load(net_cigre_lv, busC13, p_mw=0.018, q_mvar=0.008717798, name='Load C13')
    pp.create_load(net_cigre_lv, busC14, p_mw=0.0225, q_mvar=0.010897247, name='Load C14')
    pp.create_load(net_cigre_lv, busC17, p_mw=0.0225, q_mvar=0.010897247, name='Load C17')
    pp.create_load(net_cigre_lv, busC18, p_mw=0.0072, q_mvar=0.003487119, name='Load C18')
    pp.create_load(net_cigre_lv, busC19, p_mw=0.0144, q_mvar=0.006974238, name='Load C19')
    pp.create_load(net_cigre_lv, busC20, p_mw=0.0072, q_mvar=0.003487119, name='Load C20')

    # Switches
    pp.create_switch(net_cigre_lv, bus0, busR0, et='b', closed=True, type='CB', name='S1')
    pp.create_switch(net_cigre_lv, bus0, busI0, et='b', closed=True, type='CB', name='S2')
    pp.create_switch(net_cigre_lv, bus0, busC0, et='b', closed=True, type='CB', name='S3')

    # Bus geo data
    net_cigre_lv.bus_geodata = read_json(
        """{"x":{"0":0.2,"1":0.2,"2":-1.4583333333,"3":-1.4583333333,"4":-1.4583333333,
        "5":-1.9583333333,"6":-2.7083333333,"7":-2.7083333333,"8":-3.2083333333,"9":-3.2083333333,
        "10":-3.2083333333,"11":-3.7083333333,"12":-0.9583333333,"13":-1.2083333333,
        "14":-1.2083333333,"15":-1.2083333333,"16":-1.2083333333,"17":-2.2083333333,
        "18":-2.7083333333,"19":-3.7083333333,"20":0.2,"21":0.2,"22":0.2,"23":0.2,"24":1.9166666667,
        "25":1.9166666667,"26":1.9166666667,"27":0.5416666667,"28":0.5416666667,"29":-0.2083333333,
        "30":-0.2083333333,"31":-0.2083333333,"32":-0.7083333333,"33":3.2916666667,
        "34":2.7916666667,"35":2.2916666667,"36":3.2916666667,"37":3.7916666667,"38":1.2916666667,
        "39":0.7916666667,"40":1.7916666667,"41":0.7916666667,"42":0.2916666667,"43":-0.7083333333},
        "y":{"0":1.0,"1":1.0,"2":2.0,"3":3.0,"4":4.0,"5":5.0,"6":6.0,"7":7.0,"8":8.0,"9":9.0,
        "10":10.0,"11":11.0,"12":5.0,"13":6.0,"14":7.0,"15":8.0,"16":9.0,"17":8.0,"18":11.0,
        "19":12.0,"20":1.0,"21":2.0,"22":3.0,"23":1.0,"24":2.0,"25":3.0,"26":4.0,"27":5.0,"28":6.0,
        "29":7.0,"30":8.0,"31":9.0,"32":10.0,"33":5.0,"34":6.0,"35":7.0,"36":7.0,"37":6.0,"38":7.0,
        "39":8.0,"40":8.0,"41":9.0,"42":10.0,"43":11.0}}""")
    # Match bus.index
    net_cigre_lv.bus_geodata = net_cigre_lv.bus_geodata.loc[net_cigre_lv.bus.index]
    return net_cigre_lv
