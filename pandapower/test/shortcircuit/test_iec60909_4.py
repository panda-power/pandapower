# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc
from pandapower.shortcircuit.toolbox import detect_power_station_unit, calc_sc_on_line

def iec_60909_4():
    net = pp.create_empty_network()
    net.sn_mva = 23

    b1 = pp.create_bus(net, vn_kv=380.)
    b2 = pp.create_bus(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=110.)
    b4 = pp.create_bus(net, vn_kv=110.)
    b5 = pp.create_bus(net, vn_kv=110.)
    b6 = pp.create_bus(net, vn_kv=10.)
    b7 = pp.create_bus(net, vn_kv=10.)
    b8 = pp.create_bus(net, vn_kv=30.)

    HG1 = pp.create_bus(net, vn_kv=20)
    HG2 = pp.create_bus(net, vn_kv=10)  # 10.5kV?
    T_T5 = pp.create_bus(net, vn_kv=10)
    T_T6 = pp.create_bus(net, vn_kv=10)
    H = pp.create_bus(net, vn_kv=30.)

    pp.create_ext_grid(net, b1, s_sc_max_mva=38 * 380 * np.sqrt(3), rx_max=0.1)
    pp.create_ext_grid(net, b5, s_sc_max_mva=16 * 110 * np.sqrt(3), rx_max=0.1)

    # t1 = pp.create_transformer_from_parameters(net, b4, HG1, sn_mva=150,
    #     pfe_kw=0, i0_percent=0,
    #     vn_hv_kv=115., vn_lv_kv=21, vk_percent=16, vkr_percent=0.5,
    #     pt_percent=12, oltc=True)
    t1 = pp.create_transformers_from_parameters(net, [b4], [HG1], sn_mva=[150],
        pfe_kw=[0], i0_percent=[0],
        vn_hv_kv=[115.], vn_lv_kv=[21], vk_percent=[16], vkr_percent=[0.5],
        pt_percent=[12], oltc=[True])
    pp.create_gen(net, HG1, p_mw=0.85 * 150, vn_kv=21,
                  xdss_pu=0.14, rdss_ohm=0.002, cos_phi=0.85, sn_mva=150, pg_percent=0,
                  power_station_trafo=t1[0])

    t2 = pp.create_transformer_from_parameters(net, b3, HG2, sn_mva=100,
        pfe_kw=0, i0_percent=0, vn_hv_kv=120., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5,
        oltc=False)
    pp.create_gen(net, HG2, p_mw=0.9 * 100, vn_kv=10.5,
                  xdss_pu=0.16, rdss_ohm=0.005, cos_phi=0.9, sn_mva=100, pg_percent=7.5,
                  slack=True, power_station_trafo=t2)

    # # Add gen 3
    # pp.create_gen(net, b6, p_mw=0.9 * 100, vn_kv=10.5,
    #               xdss_pu=0.1, rdss_ohm=0.018, cos_phi=0.8, sn_mva=10, pg_percent=5)
    # Add gen 3
    pp.create_gen(net, b6, p_mw=0.9 * 100, vn_kv=10.5,
                  xdss_pu=0.1, rdss_ohm=0.018, cos_phi=0.8, sn_mva=10, pg_percent=0)


    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=H,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
        pfe_kw=0, i0_percent=0,
        vk_hv_percent=21, vkr_hv_percent=.26,
        vk_mv_percent=7, vkr_mv_percent=.16,
        vk_lv_percent=10., vkr_lv_percent=.16)
    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=b8,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
        pfe_kw=0, i0_percent=0,
        vk_hv_percent=21, vkr_hv_percent=.26,
        vk_mv_percent=7, vkr_mv_percent=.16,
        vk_lv_percent=10., vkr_lv_percent=.16)

    pp.create_transformer3w_from_parameters(net,
        hv_bus=b5, mv_bus=b6, lv_bus=T_T5,
        vn_hv_kv=115., vn_mv_kv=10.5, vn_lv_kv=10.5,
        sn_hv_mva=31.5, sn_mv_mva=31.5, sn_lv_mva=31.5,
        pfe_kw=0, i0_percent=0,
        vk_hv_percent=12, vkr_hv_percent=.5,
        vk_mv_percent=12, vkr_mv_percent=.5,
        vk_lv_percent=12, vkr_lv_percent=.5)
    pp.create_transformer3w_from_parameters(net,
        hv_bus=b5, mv_bus=b6, lv_bus=T_T6,
        vn_hv_kv=115., vn_mv_kv=10.5, vn_lv_kv=10.5,
        sn_hv_mva=31.5, sn_mv_mva=31.5, sn_lv_mva=31.5,
        pfe_kw=0, i0_percent=0,
        vk_hv_percent=12, vkr_hv_percent=.5,
        vk_mv_percent=12, vkr_mv_percent=.5,
        vk_lv_percent=12, vkr_lv_percent=.5)

    pp.create_motor(net, b7, pn_mech_mw=5.0, cos_phi=0.88, cos_phi_n=0.88,
                    efficiency_n_percent=97.5,
                    vn_kv=10, rx=0.1, lrc_pu=5)
    for _ in range(2):
        pp.create_motor(net, b7, pn_mech_mw=2.0, cos_phi=0.89, cos_phi_n=0.89,
                        efficiency_n_percent=96.8,
                        vn_kv=10, rx=0.1, lrc_pu=5.2)

    pp.create_line_from_parameters(net, b2, b3, name="L1",
        c_nf_per_km=0, max_i_ka=0,  # FIXME: Optional for SC
        length_km=20, r_ohm_per_km=0.12, x_ohm_per_km=0.39,)
    pp.create_line_from_parameters(net, b3, b4, name="L2",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b2, b5, name="L3a",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b2, b5, name="L3b",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39)
    pp.create_line_from_parameters(net, b5, b3, name="L4",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.096, x_ohm_per_km=0.388)
    pp.create_line_from_parameters(net, b5, b4, name="L5",
        c_nf_per_km=0, max_i_ka=0,
        length_km=15, r_ohm_per_km=0.12, x_ohm_per_km=0.386)
    pp.create_line_from_parameters(net, b6, b7, name="L6",
        c_nf_per_km=0, max_i_ka=0,
        length_km=1, r_ohm_per_km=0.082, x_ohm_per_km=0.086)

    return net

def iec_60909_4_small(with_xward=False):
    net = pp.create_empty_network()
    net.sn_mva = 23

    b1 = pp.create_bus(net, vn_kv=380.)
    b2 = pp.create_bus(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=110.)
    b5 = pp.create_bus(net, vn_kv=110.)
    b8 = pp.create_bus(net, vn_kv=30.)
    H = pp.create_bus(net, vn_kv=30.)
    HG2 = pp.create_bus(net, vn_kv=10)

    pp.create_ext_grid(net, b1, s_sc_max_mva=38 * 380 * np.sqrt(3), rx_max=0.1, x0x_max=3, r0x0_max=0.15)
    pp.create_ext_grid(net, b5, s_sc_max_mva=16 * 110 * np.sqrt(3), rx_max=0.1, x0x_max=3.3, r0x0_max=0.2)

    t1 = pp.create_transformer_from_parameters(net, b3, HG2, sn_mva=100,
        pfe_kw=0, i0_percent=0, vn_hv_kv=120., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5,
        vk0_percent=12, vkr0_percent=0.5, mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5,
        shift_degree=5, vector_group="YNd")
    pp.create_gen(net, HG2, p_mw=0.9 * 100, vn_kv=10.5,
                  xdss_pu=0.16, rdss_ohm=0.005, cos_phi=0.9, sn_mva=100, pg_percent=7.5,
                  slack=True, power_station_trafo=t1)

    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=H,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
        pfe_kw=0, i0_percent=0,  # FIXME: Optional for SC
        vk_hv_percent=21, vkr_hv_percent=.26,
        vk_mv_percent=7, vkr_mv_percent=.16,
        vk_lv_percent=10., vkr_lv_percent=.16)
    pp.create_transformer3w_from_parameters(net,
        hv_bus=b1, mv_bus=b2, lv_bus=b8,
        vn_hv_kv=400, vn_mv_kv=120, vn_lv_kv=30,
        sn_hv_mva=350, sn_mv_mva=350, sn_lv_mva=50,
        pfe_kw=0, i0_percent=0,
        vk_hv_percent=21, vkr_hv_percent=.26,
        vk_mv_percent=7, vkr_mv_percent=.16,
        vk_lv_percent=10., vkr_lv_percent=.16)

    pp.create_line_from_parameters(net, b2, b3, name="L1",
        c_nf_per_km=0, max_i_ka=0,  # FIXME: Optional for SC
        length_km=20, r_ohm_per_km=0.12, x_ohm_per_km=0.39,
        r0_ohm_per_km=0.32, x0_ohm_per_km=1.26, c0_nf_per_km=0, g0_us_per_km=0)
    pp.create_line_from_parameters(net, b2, b5, name="L3a",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39,
        r0_ohm_per_km=0.52, x0_ohm_per_km=1.86, c0_nf_per_km=0, g0_us_per_km=0)
    pp.create_line_from_parameters(net, b2, b5, name="L3b",
        c_nf_per_km=0, max_i_ka=0,
        length_km=5, r_ohm_per_km=0.12, x_ohm_per_km=0.39,
        r0_ohm_per_km=0.52, x0_ohm_per_km=1.86, c0_nf_per_km=0, g0_us_per_km=0)
    pp.create_line_from_parameters(net, b5, b3, name="L4",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.096, x_ohm_per_km=0.388,
        r0_ohm_per_km=0.22, x0_ohm_per_km=1.1, c0_nf_per_km=0, g0_us_per_km=0)

    if with_xward:
        pp.create_xward(net, b5, 1, 0, 0, 0, 10, 20, 1)

    return net

def iec_60909_4_small_gen_only():
    net = pp.create_empty_network()
    net.sn_mva = 56

    b3 = pp.create_bus(net, vn_kv=110.)
    HG2 = pp.create_bus(net, vn_kv=10)

    t1 = pp.create_transformer_from_parameters(net, b3, HG2, sn_mva=100,
        pfe_kw=0, i0_percent=0, vn_hv_kv=120., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5,
        vk0_percent=12, vkr0_percent=0.5, mag0_percent=100, mag0_rx=0, si0_hv_partial=0.5, vector_group="YNd")
    pp.create_gen(net, HG2, p_mw=0.9 * 100, vn_kv=10.5,
                  xdss_pu=0.16, rdss_ohm=0.005, cos_phi=0.9, sn_mva=100, pg_percent=7.5,
                  slack=True, power_station_trafo=t1)

    return net

def iec_60909_4_2gen():
    net = pp.create_empty_network()
    net.sn_mva = 12

    b3 = pp.create_bus(net, vn_kv=110.)
    b4 = pp.create_bus(net, vn_kv=110.)
    HG1 = pp.create_bus(net, vn_kv=20.)
    HG2 = pp.create_bus(net, vn_kv=10.)

    t1 = pp.create_transformer_from_parameters(net, b4, HG1, sn_mva=150,
        pfe_kw=0, i0_percent=0,
        vn_hv_kv=115., vn_lv_kv=21, vk_percent=16, vkr_percent=0.5,
        pt_percent=12, oltc=True)
    pp.create_gen(net, HG1, p_mw=0.85 * 150, vn_kv=21,
                  xdss_pu=0.14, rdss_ohm=0.002, cos_phi=0.85, sn_mva=150, pg_percent=0,
                  power_station_trafo=t1)

    t2 = pp.create_transformer_from_parameters(net, b3, HG2, sn_mva=100,
        pfe_kw=0, i0_percent=0, vn_hv_kv=120., vn_lv_kv=10.5, vk_percent=12, vkr_percent=0.5, oltc=False)
    pp.create_gen(net, HG2, p_mw=0.9 * 100, vn_kv=10.5,
                  xdss_pu=0.16, rdss_ohm=0.005, cos_phi=0.9, sn_mva=100, pg_percent=7.5,
                  slack=True, power_station_trafo=t2)

    pp.create_line_from_parameters(net, b3, b4, name="L2",
        c_nf_per_km=0, max_i_ka=0,
        length_km=10, r_ohm_per_km=0.12, x_ohm_per_km=0.39)

    return net

def test_iec_60909_4_3ph_small_without_gen():
    net = iec_60909_4_small()
    # Deactivate all gens
    net.gen = net.gen.iloc[0:0, :]

    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    ikss_pf = [40.3390,	28.4130, 14.2095, 28.7195, 13.4191]
    ip_pf = [99.7374, 72.6580, 32.1954, 72.1443, 36.5036]

    assert np.allclose(net.res_bus_sc.ikss_ka.values[:5], np.array(ikss_pf), atol=1e-3)
    assert np.allclose(net.res_bus_sc.ip_ka.values[:5], np.array(ip_pf), atol=1e-3)

def test_iec_60909_4_3ph_small_with_gen():
    net = iec_60909_4_small()

    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    ikss_pf = [40.4754, 29.8334, 16.1684, 30.3573]
    ip_pf = [100.1164, 76.1134, 37.3576, 76.2689]
    ib_pf = [40.4754, 29.7337, 15.9593, 30.2245]
    kappa_pf = [1.7490, 1.8040, 1.6338 , 1.7765]

    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf), atol=1e-3)
    assert np.allclose(net.res_bus_sc.ip_ka.values[:4], np.array(ip_pf), atol=1e-3)

def test_iec_60909_4_3ph_small_with_gen_xward():
    net = iec_60909_4_small(with_xward=True)
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    
    ikss_pf = [40.6422, 31.6394, 16.7409, 33.2808]
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:4], np.array(ikss_pf), atol=1e-3)
    

def test_iec_60909_4_3ph_small_gen_only():
    net = iec_60909_4_small_gen_only()

    sc.calc_sc(net, fault="3ph", case="max", ip=True, ith=True, tk_s=0.1, kappa_method="C")
    ikss_pf = [1.9755, 39.5042]
    ip_pf = [5.2316, 104.1085]
    ib_pf = [1.6071, 27.3470]
    kappa = [1.8726, 1.8635]

    assert np.allclose(net.res_bus_sc.ikss_ka[:2].values, np.array(ikss_pf), atol=1e-3)
    assert np.allclose(net.res_bus_sc.ip_ka[:2].values, np.array(ip_pf), atol=1e-3)

def test_iec_60909_4_3ph_2gen():
    net = iec_60909_4_2gen()

    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    ikss_pf = [4.2821, 4.4280, 39.1090, 57.8129]
    ip_pf = [11.1157, 11.6306, 102.7821, 151.5569]
    ib_pf = [3.6605, 3.7571, 28.3801, 45.3742]

    assert np.allclose(net.res_bus_sc.ikss_ka[:4].values, np.array(ikss_pf), atol=1e-3)
    # TODO: Check this
    assert np.allclose(net.res_bus_sc.ip_ka[:4].values, np.array(ip_pf), atol=1e-1)


def test_iec_60909_4_3ph_2gen_no_ps_detection():
    net = iec_60909_4_2gen()
    net.gen.power_station_trafo = np.nan
    net.gen.at[0, "in_service"] = False
    net.gen = net.gen.query("in_service")
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")
    
    ikss_pf = [1.8460, 1.6715, 6.8953, 39.5042]
    assert np.allclose(net.res_bus_sc.ikss_ka[:4].values, np.array(ikss_pf), atol=1e-3)


def test_iec_60909_4_3ph_without_motor():
    # Generator connected to normal bus does not need voltage correction
    net = iec_60909_4()
    net.motor = net.motor.iloc[0:0, :]
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss_pf = [40.6347, 31.6635, 19.6231, 16.1956, 32.9971, 34.3559, 22.2762, 13.5726]
    ip_pf = [100.5427, 80.3509, 45.7157, 36.7855, 82.9406, 90.6143, 43.3826, 36.9103]

    assert np.allclose(net.res_bus_sc.ikss_ka.values[:8], np.array(ikss_pf), atol=1e-3)
    assert np.allclose(net.res_bus_sc.ip_ka[:8].values, np.array(ip_pf), atol=1e-3)

def test_iec_60909_4_3ph():
    net = iec_60909_4()
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss = [40.6447, 31.7831, 19.6730, 16.2277, 33.1894,
            37.5629, 25.5895, 13.5778, 52.4438, 80.5720]
    # Ip for kappa B
    ip_pf = [100.5766, 80.8249, 45.8249, 36.8041, 83.6266,
             99.1910, 51.3864, 36.9201, 136.2801, 210.3159]
    ip_standard_kappa_c = [100.5677, 80.6079, 45.8111, 36.8427,
                           83.4033, 98.1434, 51.6899, 36.9227]
    ib = [40.645, 31.570, 19.388, 16.017, 32.795, 34.028,
          23.212, 13.578, 42.3867, 68.4172]
    skss = [26751.51, 6055.49, 3748.20, 3091.78, 6323.43,
            650.61, 443.22, 705.52, 1816.71, 1395.55]

    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10], np.array(ikss), atol=1e-3)
    assert np.allclose(net.res_bus_sc.ip_ka.values[:8], np.array(ip_standard_kappa_c ), atol=1e-3)
    assert np.allclose(net.res_bus_sc.skss_mw.values[:10], np.array(skss), atol=1e-2)

def test_iec_60909_4_3ph_min():
    net = iec_60909_4()
    net.line["endtemp_degree"] = 80.0
    net.ext_grid["s_sc_min_mva"] = net.ext_grid["s_sc_max_mva"]/10
    net.ext_grid["rx_min"] = net.ext_grid["rx_max"]
    sc.calc_sc(net, fault="3ph", case="min", ip=True, tk_s=0.1, kappa_method="C")

    ikss_min = [5.0501, 12.2915, 10.3292, 9.4708, 11.8604,
                28.3052, 18.6148, 10.9005, 44.5098, 67.9578]

    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10], np.array(ikss_min), atol=1e-3)


def test_iec_60909_4_3ph_ps_trafo_flag():
    net = iec_60909_4()
    net.trafo["power_station_unit"] = False
    ps_trafo = net.gen.power_station_trafo.values
    ps_trafo = ps_trafo[~np.isnan(ps_trafo)].astype(int)
    net.trafo.loc[ps_trafo, "power_station_unit"] = True
    net.gen.power_station_trafo.values[:] = np.nan

    detect_power_station_unit(net, mode="trafo")
    sc.calc_sc(net, fault="3ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss = [40.6447, 31.7831, 19.6730, 16.2277, 33.1894,
            37.5629, 25.5895, 13.5778, 52.4438, 80.5720]
    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10], np.array(ikss), atol=1e-3)

def test_iec_60909_4_2ph():
    net = iec_60909_4()
    sc.calc_sc(net, fault="2ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss = [35.1994, 27.5249, 17.0373, 14.0536, 28.7429,
            32.5304, 22.1611, 11.7586, 45.4177, 69.7774]
    ip = [87.0941, 69.8085, 39.6736, 31.9067, 72.2294,
          84.9946, 44.7648, 31.9760, 118.0221, 182.1389]
    # No ib for 2ph sc calculation
    skss = [7722.50, 1748.07, 1082.01, 892.52, 1825.42,
            187.81, 127.95, 203.67, 524.44, 402.86]

    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10], np.array(ikss), atol=1e-3)
    assert np.allclose(net.res_bus_sc.ip_ka.values[:10], np.array(ip), atol=1e-3)
    assert np.allclose(net.res_bus_sc.skss_mw.values[:10], np.array(skss), atol=1e-1)


@pytest.mark.skip("1ph gen-close sc calculation still under develop")
def test_iec_60909_4_1ph():
    net = iec_60909_4()
    sc.calc_sc(net, fault="1ph", case="max", ip=True, tk_s=0.1, kappa_method="C")

    ikss = [24.6526, 15.9722, 10.4106, 9.0498, 17.0452,
            0.06337, 0.0633, 0, 0.0001, 0.0001]
    ip = [60.9982, 40.5086, 24.2424, 20.5464, 42.8337,
          0.1656, 0.1279, 0.0, 0.00025, 0.00033]
    # No ib for 1ph sc calculation

    assert np.allclose(net.res_bus_sc.ikss_ka.values[:10], np.array(ikss), atol=1e-4)
    # assert np.allclose(net.res_bus_sc.ip.values[:8], np.array(ip), rtol=1e-4)


def test_detect_power_station_units():
    net = iec_60909_4()
    net.gen.power_station_trafo[:] = None

    detect_power_station_unit(net)
    assert np.all(net.gen.power_station_trafo.values[[0,1]] == np.array([0,1]))
    net.gen.power_station_trafo[:] = None

    detect_power_station_unit(net, mode="trafo")
    assert np.all(net.gen.power_station_trafo.values[[0,1]] == np.array([0,1]))


def test_sc_on_line():
    net = iec_60909_4()
    calc_sc_on_line(net, 2, 0.3)
    

if __name__ == '__main__':
    pytest.main(["test_iec60909_4.py"])
