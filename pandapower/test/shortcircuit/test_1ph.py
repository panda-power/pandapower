# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower as pp
import pandapower.shortcircuit as sc
import pandapower.test
import copy
import numpy as np
import os
import pytest

def check_results(net, vc, result):
    res_ika = net.res_bus_sc[(net.bus.zone==vc) & (net.bus.in_service)].ikss_ka.values
    if not np.allclose(result, res_ika):
        raise ValueError("Incorrect results for vector group %s"%vc, res_ika, result)

def add_network(net, vector_group):
    b1 = pp.create_bus(net, 110, zone=vector_group, index=pp.get_free_id(net.bus))
    b2 = pp.create_bus(net, 20, zone=vector_group)
    pp.create_bus(net, 20, in_service=False)
    b3 = pp.create_bus(net, 20, zone=vector_group)
    b4 = pp.create_bus(net, 20, zone=vector_group)
    pp.create_bus(net, 20)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100, s_sc_min_mva=80, rx_min=0.20, rx_max=0.35)
    net.ext_grid["r0x0_max"] = 0.4
    net.ext_grid["x0x_max"] = 1.0

    pp.create_std_type(net, {"r_ohm_per_km": 0.122, "x_ohm_per_km": 0.112, "c_nf_per_km": 304,
                         "max_i_ka": 0.421, "endtemp_degree": 70.0, "r0_ohm_per_km": 0.244,
                         "x0_ohm_per_km": 0.336, "c0_nf_per_km": 2000}, "unsymmetric_line_type")
    l1 = pp.create_line(net, b2, b3, length_km=10, std_type="unsymmetric_line_type",
                   index=pp.get_free_id(net.line)+1)
    l2 = pp.create_line(net, b3, b4, length_km=15, std_type="unsymmetric_line_type")
    pp.create_line(net, b3, b4, length_km=15, std_type="unsymmetric_line_type", in_service=False)

    transformer_type = {"i0_percent": 0.071, "pfe_kw": 29, "vkr_percent": 0.282,
            "sn_mva": 25, "vn_lv_kv": 20.0, "vn_hv_kv": 110.0, "vk_percent": 11.2,
            "shift_degree": 150, "vector_group": vector_group, "tap_side": "hv",
            "tap_neutral": 0, "tap_min": -9, "tap_max": 9, "tap_step_degree": 0,
            "tap_step_percent": 1.5, "tap_phase_shifter": False, "vk0_percent": 5,
            "vkr0_percent": 0.4, "mag0_percent": 10, "mag0_rx": 0.4,
            "si0_hv_partial": 0.9}
    pp.create_std_type(net, transformer_type, vector_group, "trafo")
    t1 = pp.create_transformer(net, b1, b2, std_type=vector_group, parallel=2,
                          index=pp.get_free_id(net.trafo)+1)
    pp.create_transformer(net, b1, b2, std_type=vector_group, in_service=False)
    pp.add_zero_impedance_parameters(net)
    return l1, l2, t1

def test_1ph_shortcircuit():
    results = {
                 "Yy":  [0.52209347337, 0.74400073149, 0.74563682772, 0.81607276962]
                ,"Yyn": [0.52209347337, 2.5145986133,  1.6737892808,  1.1117955913 ]
                ,"Yd":  [0.52209347337, 0.74400073149, 0.74563682772, 0.81607276962]
                ,"YNy": [0.6291931171,  0.74400073149, 0.74563682772, 0.81607276962]
                ,"YNyn":[0.62623661918, 2.9829679356,  1.8895041867,  1.2075537026 ]
                ,"YNd": [0.75701600162, 0.74400073149, 0.74563682772, 0.81607276962]
                ,"Dy":  [0.52209347337, 0.74400073149, 0.74563682772, 0.81607276962]
                ,"Dyn": [0.52209347337, 3.5054043285,  2.1086590382,  1.2980120038 ]
                ,"Dd":  [0.52209347337, 0.74400073149, 0.74563682772, 0.81607276962]
               }
    net = pp.create_empty_network()
    for vc in results.keys():
         add_network(net, vc)
         try:
             sc.calc_sc(net, fault="1ph", case="max")
         except:
             raise UserWarning("Did not converge after adding transformer with vector group %s"%vc)

    for vc, result in results.items():
        check_results(net, vc, result)

def test_iec60909_example_4():
    file = os.path.join(pp.pp_dir, "test", "test_files", "IEC60909-4_example.json")
    net = pp.from_json(file)
    sc.calc_sc(net, fault="1ph")
    assert np.isclose(net.res_bus_sc[net.bus.name=="Q"].ikss_ka.values[0], 10.05957231)
    assert np.isclose(net.res_bus_sc[net.bus.name=="T2LV"].ikss_ka.values[0], 34.467353142)
    assert np.isclose(net.res_bus_sc[net.bus.name=="F1"].ikss_ka.values[0], 35.53066312)
    assert np.isclose(net.res_bus_sc[net.bus.name=="F2"].ikss_ka.values[0], 34.89135137)
    assert np.isclose(net.res_bus_sc[net.bus.name=="F3"].ikss_ka.values[0], 5.0321033105)
    assert np.isclose(net.res_bus_sc[net.bus.name=="Cable/Line IC"].ikss_ka.values[0], 16.362586813)

def test_1ph_with_switches():
    net = pp.create_empty_network()
    vc = "Yy"
    l1, l2, _ = add_network(net, vc)
    sc.calc_sc(net, fault="1ph", case="max")
    pp.create_line(net, net.line.to_bus.at[l2], net.line.from_bus.at[l1], length_km=15,
                   std_type="unsymmetric_line_type", parallel=2.)
    pp.add_zero_impedance_parameters(net)
    pp.create_switch(net, bus=net.line.to_bus.at[l2], element=l2, et="l", closed=False)
    sc.calc_sc(net, fault="1ph", case="max")
    check_results(net, vc, [0.52209347338, 2.0620266652, 2.3255761263, 2.3066467489])

if __name__ == "__main__":
    pytest.main([__file__])
