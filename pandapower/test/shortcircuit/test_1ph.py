# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower as pp
import pandapower.shortcircuit as sc
import pandapower.test
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

    pp.create_ext_grid(net, b1, s_sc_max_mva=100, s_sc_min_mva=100, rx_min=0.35, rx_max=0.35)
    net.ext_grid["r0x0_max"] = 0.4
    net.ext_grid["x0x_max"] = 1.0

    net.ext_grid["r0x0_min"] = 0.4
    net.ext_grid["x0x_min"] = 1.0

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
         except Exception as e:
             raise UserWarning(str(e) + " Did not converge after adding transformer with vector group %s"%vc)

    for vc, result in results.items():
        check_results(net, vc, result)

def test_1ph_shortcircuit_min():
    results = {
                 "Yy":  [0.52209346201, 0.66632662571, 0.66756160176, 0.72517293174]
                ,"Yyn": [0.52209346201, 2.4135757259, 1.545054139, 0.99373917957]
                ,"Yd":  [0.52209346201, 0.66632662571, 0.66756160176, 0.72517293174]
                ,"YNy": [0.62316686505, 0.66632662571, 0.66756160176, 0.72517293174]
                ,"YNyn":[0.620287259, 2.9155736491, 1.7561556936, 1.0807305212]
                ,"YNd": [0.75434229157, 0.66632662571, 0.66756160176, 0.72517293174]
                ,"Dy":  [0.52209346201, 0.66632662571, 0.66756160176, 0.72517293174]
                ,"Dyn": [0.52209346201, 3.4393798093, 1.9535982949, 1.1558364456]
                ,"Dd":  [0.52209346201, 0.66632662571, 0.66756160176, 0.72517293174]
               }

    for inv_y in (False, True):
        net = pp.create_empty_network()
        for vc in results.keys():
             add_network(net, vc)
             try:
                 sc.calc_sc(net, fault="1ph", case="min", inverse_y=inv_y)
             except:
                 raise UserWarning("Did not converge after adding transformer with vector group %s"%vc)

        for vc, result in results.items():
            check_results(net, vc, result)

def test_iec60909_example_4():
    file = os.path.join(pp.pp_dir, "test", "test_files", "IEC60909-4_example.json")
    net = pp.from_json(file)
    for inv_y in (False, True):
        sc.calc_sc(net, fault="1ph", inverse_y=inv_y)
        assert np.isclose(net.res_bus_sc[net.bus.name=="Q"].ikss_ka.values[0], 10.05957231)
        assert np.isclose(net.res_bus_sc[net.bus.name=="T2LV"].ikss_ka.values[0], 34.467353142)
        assert np.isclose(net.res_bus_sc[net.bus.name=="F1"].ikss_ka.values[0], 35.53066312)
        assert np.isclose(net.res_bus_sc[net.bus.name=="F2"].ikss_ka.values[0], 34.89135137)
        assert np.isclose(net.res_bus_sc[net.bus.name=="F3"].ikss_ka.values[0], 5.0321033105)
        assert np.isclose(net.res_bus_sc[net.bus.name=="Cable/Line IC"].ikss_ka.values[0], 16.362586813)

def test_iec60909_example_4_bus_selection():
    file = os.path.join(pp.pp_dir, "test", "test_files", "IEC60909-4_example.json")
    net = pp.from_json(file)
    for inv_y in (False, True):
        sc.calc_sc(net, fault="1ph", inverse_y=inv_y,
                   bus=net.bus[net.bus.name.isin(("F1", "F2"))].index)
        assert np.isclose(net.res_bus_sc.at[net.bus[net.bus.name=="F1"].index[0],
                                            "ikss_ka"], 35.53066312)
        assert np.isclose(net.res_bus_sc.at[net.bus[net.bus.name=="F2"].index[0],
                                            "ikss_ka"], 34.89135137)

def test_iec60909_example_4_bus_selection_br_res():
    file = os.path.join(pp.pp_dir, "test", "test_files", "IEC60909-4_example.json")
    net = pp.from_json(file)
    for inv_y in (False, True):
        sc.calc_sc(net, fault="1ph", inverse_y=inv_y,
                   bus=net.bus[net.bus.name.isin(("F1", "F2"))].index,
                   branch_results=True)
        sc.calc_sc(net, fault="1ph", inverse_y=inv_y,
                   bus=net.bus[net.bus.name.isin(("F1", "F2"))].index,
                   branch_results=True, return_all_currents=True)
        assert np.isclose(net.res_bus_sc.at[net.bus[net.bus.name=="F1"].index[0],
                                            "ikss_ka"], 35.53066312)
        assert np.isclose(net.res_bus_sc.at[net.bus[net.bus.name=="F2"].index[0],
                                            "ikss_ka"], 34.89135137)

def test_1ph_with_switches():
    for inv_y in (False, True):
        net = pp.create_empty_network()
        vc = "Yy"
        l1, l2, _ = add_network(net, vc)
        sc.calc_sc(net, fault="1ph", case="max", inverse_y=inv_y)
        pp.create_line(net, net.line.to_bus.at[l2], net.line.from_bus.at[l1], length_km=15,
                       std_type="unsymmetric_line_type", parallel=2.)
        pp.add_zero_impedance_parameters(net)
        pp.create_switch(net, bus=net.line.to_bus.at[l2], element=l2, et="l", closed=False)
        sc.calc_sc(net, fault="1ph", case="max")
        check_results(net, vc, [0.52209347338, 2.0620266652, 2.3255761263, 2.3066467489])


def iec_60909_4_small(n_t3=1, num_earth=1, with_gen=False):
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=380.)
    b2 = pp.create_bus(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=110.)
    b5 = pp.create_bus(net, vn_kv=110.)
    b8 = pp.create_bus(net, vn_kv=30.)
    HG2 = pp.create_bus(net, vn_kv=10)

    pp.create_ext_grid(net, b1, s_sc_max_mva=38 * 380 * np.sqrt(3), rx_max=0.1, x0x_max=3, r0x0_max=0.15,
                       s_sc_min_mva=38 * 380 * np.sqrt(3) / 10, rx_min=0.1, x0x_min=3, r0x0_min=0.15,)
    pp.create_ext_grid(net, b5, s_sc_max_mva=16 * 110 * np.sqrt(3), rx_max=0.1, x0x_max=3.3, r0x0_max=0.2,
                       s_sc_min_mva=16 * 110 * np.sqrt(3) / 10, rx_min=0.1, x0x_min=3.3, r0x0_min=0.2)

    if num_earth == 1:
        vector_group = ("YYNd", "YNYd")
    else:
        vector_group = ("YNYNd", "YNYNd")

    if n_t3==2:
        pp.create_transformer3w_from_parameters(net,
            hv_bus=b1, mv_bus=b2, lv_bus=b8,
            vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
            sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
            pfe_kw=0, i0_percent=0,
            vk_hv_percent=21, vkr_hv_percent=.26,
            vk_mv_percent=7, vkr_mv_percent=.16,
            vk_lv_percent=10., vkr_lv_percent=.16,
            vk0_hv_percent=44.1, vkr0_hv_percent=0.26,
            vk0_mv_percent=6.2996, vkr0_mv_percent=0.03714,
            vk0_lv_percent=6.2996, vkr0_lv_percent=0.03714,
            vector_group=vector_group[0])
    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=b8,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
        pfe_kw=0, i0_percent=0,
        vk_hv_percent=21, vkr_hv_percent=.26,
        vk_mv_percent=7, vkr_mv_percent=.16,
        vk_lv_percent=10., vkr_lv_percent=.16,
        vk0_hv_percent=44.1, vkr0_hv_percent=0.26,
        vk0_mv_percent=6.2996, vkr0_mv_percent=0.03714,
        vk0_lv_percent=6.2996, vkr0_lv_percent=0.03714,
        vector_group=vector_group[1])

    pp.create_line_from_parameters(net, b2, b3, name="L1",
        c_nf_per_km=0, max_i_ka=0,  # FIXME: Optional for SC
        length_km=20, r_ohm_per_km=0.12, x_ohm_per_km=0.39,
        r0_ohm_per_km=0.32, x0_ohm_per_km=1.26, c0_nf_per_km=0, g0_us_per_km=0, endtemp_degree=80)
    pp.create_line_from_parameters(net, b2, b5, name="L3a",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39,
        r0_ohm_per_km=0.52, x0_ohm_per_km=1.86, c0_nf_per_km=0, g0_us_per_km=0, endtemp_degree=80)
    pp.create_line_from_parameters(net, b2, b5, name="L3b",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39,
        r0_ohm_per_km=0.52, x0_ohm_per_km=1.86, c0_nf_per_km=0, g0_us_per_km=0, endtemp_degree=80)
    pp.create_line_from_parameters(net, b5, b3, name="L4",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.096, x_ohm_per_km=0.388,
        r0_ohm_per_km=0.22, x0_ohm_per_km=1.1, c0_nf_per_km=0, g0_us_per_km=0, endtemp_degree=80)

    if with_gen:
        t1 = pp.create_transformer_from_parameters(net, b3, HG2, sn_mva=100,
            pfe_kw=0, i0_percent=0, vn_hv_kv=120., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5,
            vk0_percent=12, vkr0_percent=0.5, mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5,
            shift_degree=5, vector_group="YNd")
        pp.create_gen(net, HG2, p_mw=0.9 * 100, vn_kv=10.5,
                      xdss_pu=0.16, rdss_ohm=0.005, cos_phi=0.9, sn_mva=100, pg_percent=7.5,
                      slack=True, power_station_trafo=t1)

    return net

def test_iec60909_example_4_one_trafo3w():
    net = iec_60909_4_small(n_t3=1)

    # r0 = 2.378330877
    # x0 = 17.335578502
    # r = 0.59621204768
    # x = 6.0598429694
    ikss_pf = [24.4009, 8.2481, 6.1728, 10.1851]

    sc.calc_sc(net, fault="1ph")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf), atol=1e-4)

def test_iec60909_example_4_two_trafo3w():
    net = iec_60909_4_small(n_t3=2)

    ikss_pf_2t3 = [24.5772, 14.7247, 8.1060, 15.2749]

    sc.calc_sc(net, fault="1ph")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf_2t3), atol=1e-4)

def test_iec60909_example_4_two_trafo3w_two_earth():
    net = iec_60909_4_small(n_t3=2, num_earth=2)

    # r0 = 2.378330877
    # x0 = 17.335578502
    # r = 0.59621204768
    # x = 6.0598429694
    ikss_pf_max = [26.0499, 20.9472, 9.1722, 18.7457]
    ikss_pf_min = [3.9622, 8.3914, 5.0248, 6.9413]

    sc.calc_sc(net, fault="1ph", case="max")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf_max), atol=1e-4)

    sc.calc_sc(net, fault="1ph", case="min")
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf_min), atol=1e-4)

@pytest.mark.skip("1ph gen-close sc calculation still under develop")
def test_iec_60909_4_small_with_t2_1ph():
    net = iec_60909_4_small(n_t3=2, num_earth=1, with_gen=True)
    net.gen = net.gen.iloc[0:0, :]
    sc.calc_sc(net, fault="1ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss_max = [24.57717, 16.96235, 11.6109, 18.07836]
    # ikss_min = [3.5001, 8.4362, 7.4743, 7.7707]
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_max), atol=1e-4)

@pytest.mark.skip("1ph gen-close sc calculation still under develop")
def test_iec_60909_4_small_with_gen_1ph_no_ps_detection():
    net = iec_60909_4_small(n_t3=2, num_earth=1, with_gen=True)
    net.gen.power_station_trafo=np.nan
    sc.calc_sc(net, fault="1ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss_max = [24.60896, 17.2703, 12.3771, 18.4723]
    # ikss_min = [3.5001, 8.4362, 7.4743, 7.7707]
    # ip_min = [8.6843, 21.6173, 18.0242, 19.4261]
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_max), atol=1e-4)


# @pytest.mark.skip("1ph gen-close sc calculation still under develop")
def test_iec_60909_4_small_with_gen_ps_unit_1ph():
    net = iec_60909_4_small(n_t3=2, num_earth=1, with_gen=True)

    sc.calc_sc(net, fault="1ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss_max = [24.6109, 17.4363, 12.7497, 18.6883]
    # ikss_min = [3.5001, 8.4362, 7.4743, 7.7707]
    # ip_min = [8.6843, 21.6173, 18.0242, 19.4261]

    # TODO: This needs to be fixed!!
    # assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_max), atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
