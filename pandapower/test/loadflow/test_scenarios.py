# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.result_test_network_generator import add_test_bus_bus_switch, \
                                                                   add_test_trafo
from pandapower.test.toolbox import create_test_network2, add_grid_connection

#TODO: 2 gen 2 ext_grid missing

def test_2gen_1ext_grid():
    net = create_test_network2()
    net.shunt.q_mvar *= -1
    pp.create_gen(net, 2, p_mw=0.100)
    net.trafo.shift_degree = 150
    pp.runpp(net, init='dc', calculate_voltage_angles=True)

    assert np.allclose(net.res_gen.p_mw.values, [0.100, 0.100])
    assert np.allclose(net.res_gen.q_mvar.values, [-0.447397232056, -0.0518152713776], atol=1e-2)
    assert np.allclose(net.res_gen.va_degree.values, [0.242527288986,-143.558157703])
    assert np.allclose(net.res_gen.vm_pu.values, [1.0, 1.0])

    assert np.allclose(net.res_bus.vm_pu, [1.000000, 0.956422, 1.000000, 1.000000])
    assert np.allclose(net.res_bus.va_degree, [0.000000, -145.536429154, -143.558157703,
                                               0.242527288986])
    assert np.allclose(net.res_bus.p_mw, [0.06187173, 0.03000000, -0.100, 0.00000])
    assert np.allclose(net.res_bus.q_mvar, [-.470929980278, 0.002000000, 0.0218152713776, 0.447397232056], atol=1e-2)
    assert np.allclose(net.res_ext_grid.p_mw.values, [-0.06187173])
    assert np.allclose(net.res_ext_grid.q_mvar, [0.470927898])


def test_0gen_2ext_grid():
    # testing 2 ext grid and 0 gen, both EG on same trafo side
    net = create_test_network2()
    net.shunt.q_mvar *= -1
    pp.create_ext_grid(net, 1)
    net.gen = net.gen.drop(0)
    net.trafo.shift_degree = 150
    net.ext_grid.in_service.at[1] = False
    pp.create_ext_grid(net, 3)

    pp.runpp(net, init='dc', calculate_voltage_angles=True)
    assert np.allclose(net.res_bus.p_mw.values, [-0.000000, 0.03000000, 0.000000, -0.032993015])
    assert np.allclose(net.res_bus.q_mvar.values, [0.00408411026001, 0.002000000,
                                                   -0.0286340014753, 0.027437210083])
    assert np.allclose(net.res_bus.va_degree.values, [0.000000, -155.719283,
                                                      -153.641832, 0.000000])
    assert np.allclose(net.res_bus.vm_pu.values,  [1.000000, 0.932225,
                                                   0.976965, 1.000000])

    assert np.allclose(net.res_ext_grid.p_mw.values, [-0.000000, 0.000000, 0.132993015])
    assert np.allclose(net.res_ext_grid.q_mvar, [-0.00408411026001, 0.000000, -0.027437210083])


def test_0gen_2ext_grid_decoupled():
    net = create_test_network2()
    net.gen = net.gen.drop(0)
    net.shunt.q_mvar *= -1
    pp.create_ext_grid(net, 1)
    net.ext_grid.in_service.at[1] = False
    pp.create_ext_grid(net, 3)
    net.ext_grid.in_service.at[2] = False
    auxbus = pp.create_bus(net, name="bus1", vn_kv=10.)
    net.trafo.shift_degree = 150
    pp.create_std_type(net, {"type": "cs", "r_ohm_per_km": 0.876,  "q_mm2": 35.0,
                             "endtmp_deg": 160.0, "c_nf_per_km": 260.0,
                             "max_i_ka": 0.123, "x_ohm_per_km": 0.1159876},
                             name="NAYSEY 3x35rm/16 6/10kV" , element="line")
    pp.create_line(net, 0, auxbus, 1, name="line_to_decoupled_grid",
                   std_type="NAYSEY 3x35rm/16 6/10kV") #NAYSEY 3x35rm/16 6/10kV
    pp.create_ext_grid(net, auxbus)
    pp.create_switch(net, auxbus, 2, et="l", closed=0, type="LS")
    pp.runpp(net, init='dc', calculate_voltage_angles=True)

    assert np.allclose(net.res_bus.p_mw.values*1e3, [-133.158732, 30.000000,
                                             0.000000, 100.000000, 0.000000])
    assert np.allclose(net.res_bus.q_mvar.values*1e3, [39.5843982697, 2.000000,
                                           -28.5636406913, 0.000000, 0.000000])
    assert np.allclose(net.res_bus.va_degree.values, [0.000000, -155.752225311,
                                                      -153.669395244,
                                                      -0.0225931152895, 0.0])
    assert np.allclose(net.res_bus.vm_pu.values,  [1.000000, 0.930961,
                                                   0.975764, 0.998865, 1.0])

    assert np.allclose(net.res_ext_grid.p_mw.values*1e3, [133.158732, 0.000000, 0.000000, -0.000000])
    assert np.allclose(net.res_ext_grid.q_mvar*1e3, [-39.5843982697, 0.000000, 0.000000, -0.000000])


def test_bus_bus_switch_at_eg():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, name="bus1", vn_kv=.4)
    b2 = pp.create_bus(net, name="bus2", vn_kv=.4)
    b3 = pp.create_bus(net, name="bus3", vn_kv=.4)

    pp.create_ext_grid(net, b1)

    pp.create_switch(net, b1, et="b", element=1)
    pp.create_line(net, b2, b3, 1, name="line1",
                   std_type="NAYY 4x150 SE")

    pp.create_load(net, b3, p_mw=0.01, q_mvar=0, name="load1")

    runpp_with_consistency_checks(net)


def test_bb_switch():
    net = pp.create_empty_network()
    net = add_test_bus_bus_switch(net)
    runpp_with_consistency_checks(net)

def test_two_gens_at_one_bus():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, 380)
    b2 = pp.create_bus(net, 380)
    b3 = pp.create_bus(net, 380)

    pp.create_ext_grid(net, b1, 1.02, max_p_mw=0.)
    p1 = 800
    p2 = 500

    g1 = pp.create_gen(net, b3, vm_pu=1.018, p_mw=p1)
    g2 = pp.create_gen(net, b3, vm_pu=1.018, p_mw=p2)
    pp.create_line(net, b1, b2, 30, "490-AL1/64-ST1A 380.0")
    pp.create_line(net, b2, b3, 20, "490-AL1/64-ST1A 380.0")

    pp.runpp(net)
    assert net.res_gen.p_mw.at[g1] == p1
    assert net.res_gen.p_mw.at[g2] == p2

def test_ext_grid_gen_order_in_ppc():
    net=pp.create_empty_network()

    for b in range(6):
        pp.create_bus(net,vn_kv=1., name=b)

    for l_bus in range(0,5,2):
        pp.create_line(net, from_bus=l_bus, to_bus=l_bus+1, length_km=1, std_type="48-AL1/8-ST1A 10.0")

    for slack_bus in [0,2,5]:
        pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.)

    for gen_bus in [ 0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5]:
        pp.create_gen(net, bus=gen_bus, p_mw=1, vm_pu=1.)

    pp.rundcpp(net)
    assert all(net.res_gen.p_mw==net.gen.p_mw)
    assert all(net.res_ext_grid.p_mw<0)

    pp.runpp(net)
    assert all(net.res_gen.p_mw==net.gen.p_mw)
    assert all(net.res_ext_grid.p_mw<0)


def test_isolated_gen_lookup():
    net=pp.create_empty_network()

    gen_bus=pp.create_bus(net,vn_kv=1., name='gen_bus')
    slack_bus=pp.create_bus(net,vn_kv=1., name='slack_bus')
    gen_iso_bus=pp.create_bus(net,vn_kv=1., name='iso_bus')

    pp.create_line(net, from_bus=slack_bus, to_bus=gen_bus, length_km=1, std_type="48-AL1/8-ST1A 10.0")

    pp.create_ext_grid(net, bus=slack_bus, vm_pu=1.)

    pp.create_gen(net, bus=gen_iso_bus, p_mw=1, vm_pu=1., name='iso_gen')
    pp.create_gen(net, bus=gen_bus, p_mw=1, vm_pu=1., name='oos_gen', in_service=False)
    pp.create_gen(net, bus=gen_bus, p_mw=2, vm_pu=1., name='gen')

    pp.rundcpp(net)
    assert np.allclose(net.res_gen.p_mw.values, [0, 0, 2])

    pp.runpp(net)
    assert np.allclose(net.res_gen.p_mw.values, [0, 0, 2])

    pp.create_xward(net, bus=gen_iso_bus, pz_mw=1., qz_mvar=1., ps_mw=1., qs_mvar=1.,
                    vm_pu=1., x_ohm=1., r_ohm=.1)
    pp.create_xward(net, bus=gen_bus, pz_mw=1., qz_mvar=1., ps_mw=1., qs_mvar=1.,
                    vm_pu=1., x_ohm=1., r_ohm=.1)
    pp.create_xward(net, bus=gen_iso_bus, pz_mw=1., qz_mvar=1., ps_mw=1., qs_mvar=1.,
                    vm_pu=1., x_ohm=1., r_ohm=.1, in_service=False)

    pp.rundcpp(net)
    assert np.allclose(net.res_gen.p_mw.values, [0, 0, 2])
    assert np.allclose(net.res_xward.p_mw.values, [0, 2, 0])

    pp.runpp(net)
    assert np.allclose(net.res_gen.p_mw.values, [0, 0, 2])
    assert np.allclose(net.res_xward.p_mw.values, [0, 2, 0])

def test_transformer_phase_shift():
    net = pp.create_empty_network()
    for side in ["hv", "lv"]:
        b1 = pp.create_bus(net, vn_kv=110.)
        b2 = pp.create_bus(net, vn_kv=20.)
        b3 = pp.create_bus(net, vn_kv=0.4)
        pp.create_ext_grid(net, b1)
        pp.create_transformer_from_parameters(net, b1, b2, 40000, 110, 20, 0.1, 5, 0, 0.1, 30, side,
                                              # 0, 2, -2, 1.25, 10, 0)
                                              0, 2, -2, 0, 10, 0, True)
        pp.create_transformer_from_parameters(net, b2, b3, 630, 20, 0.4, 0.1, 5, 0, 0.1, 20, tap_phase_shifter=True)
    pp.runpp(net, init="dc", calculate_voltage_angles=True)
    b2a_angle = net.res_bus.va_degree.at[1]
    b3a_angle = net.res_bus.va_degree.at[2]
    b2b_angle = net.res_bus.va_degree.at[4]
    b3b_angle = net.res_bus.va_degree.at[5]

    net.trafo.tap_pos.at[0] = 1
    net.trafo.tap_pos.at[2] = 1
    pp.runpp(net, init="dc", calculate_voltage_angles=True)
    assert np.isclose(b2a_angle - net.res_bus.va_degree.at[1], 10)
    assert np.isclose(b3a_angle - net.res_bus.va_degree.at[2], 10)
    assert np.isclose(b2b_angle - net.res_bus.va_degree.at[4], -10)
    assert np.isclose(b3b_angle - net.res_bus.va_degree.at[5], -10)


def test_transformer_phase_shift_complex():
    test_ref = (0.99967, -30.7163)
    test_tap_pos = {
        'hv': (0.9617, -31.1568),
        'lv': (1.0391, -30.3334)
    }
    test_tap_neg = {
        'hv': (1.0407, -30.2467),
        'lv': (0.9603, -31.1306)
    }
    for side in ["hv", "lv"]:
        net = pp.create_empty_network()
        b1 = pp.create_bus(net, vn_kv=110.)
        pp.create_ext_grid(net, b1)
        b2 = pp.create_bus(net, vn_kv=20.)
        pp.create_load(net, b2, p_mw=10)
        pp.create_transformer_from_parameters(net, hv_bus=b1, lv_bus=b2, sn_mva=40, vn_hv_kv=110,
                                              vn_lv_kv=20, vkr_percent=0.1, vk_percent=5,
                                              pfe_kw=0, i0_percent=0.1, shift_degree=30,
                                              tap_side=side, tap_neutral=0, tap_max=2, tap_min=-2,
                                              tap_step_percent=2, tap_step_degree=10, tap_pos=0)
        pp.runpp(net, init="dc", calculate_voltage_angles=True)
        assert np.isclose(net.res_bus.vm_pu.at[b2], test_ref[0], rtol=1e-4)
        assert np.isclose(net.res_bus.va_degree.at[b2], test_ref[1], rtol=1e-4)

        net.trafo.tap_pos.at[0] = 2
        pp.runpp(net, init="dc", calculate_voltage_angles=True)
        assert np.isclose(net.res_bus.vm_pu.at[b2], test_tap_pos[side][0], rtol=1e-4)
        assert np.isclose(net.res_bus.va_degree.at[b2], test_tap_pos[side][1], rtol=1e-4)

        net.trafo.tap_pos.at[0] = -2
        pp.runpp(net, init="dc", calculate_voltage_angles=True)
        assert np.isclose(net.res_bus.vm_pu.at[b2], test_tap_neg[side][0], rtol=1e-4)
        assert np.isclose(net.res_bus.va_degree.at[b2], test_tap_neg[side][1], rtol=1e-4)


def test_transformer3w_phase_shift():
    test_ref = ((0.9995, -31.003), (0.9996, -60.764))
    test_tap_pos = {
        'hv': ((0.9615, -31.466), (0.9617, -61.209)),
        'mv': ((1.0389, -30.620), (0.9996, -60.764)),
        'lv': ((0.9995, -31.003), (1.039, -60.381))
    }
    test_tap_neg = {
        'hv': ((1.0405, -30.511), (1.0406, -60.291)),
        'mv': ((0.9602, -31.417), (0.9996, -60.764)),
        'lv': ((0.9995, -31.003), (0.9603, -61.178))
    }
    for side in ["hv", "mv", "lv"]:
        net = pp.create_empty_network()
        b1 = pp.create_bus(net, vn_kv=110.)
        pp.create_ext_grid(net, b1)
        b2 = pp.create_bus(net, vn_kv=20.)
        pp.create_load(net, b2, p_mw=10)
        b3 = pp.create_bus(net, vn_kv=0.4)
        pp.create_load(net, b3, p_mw=1)
        pp.create_transformer3w_from_parameters(net, hv_bus=b1, mv_bus=b2, lv_bus=b3, vn_hv_kv=110,
                                                vn_mv_kv=20, vn_lv_kv=0.4, sn_hv_mva=40,
                                                sn_mv_mva=30, sn_lv_mva=10,
                                                vk_hv_percent=5, vk_mv_percent=5,
                                                vk_lv_percent=5, vkr_hv_percent=0.1,
                                                vkr_mv_percent=0.1, vkr_lv_percent=0.1, pfe_kw=0,
                                                i0_percent=0.1, shift_mv_degree=30,
                                                shift_lv_degree=60, tap_side=side, tap_step_percent=2,
                                                tap_step_degree=10, tap_pos=0, tap_neutral=0, tap_min=-2,
                                                tap_max=2)
        pp.runpp(net, init="dc", calculate_voltage_angles=True)
        assert np.isclose(net.res_bus.vm_pu.at[b2], test_ref[0][0], rtol=1e-4)
        assert np.isclose(net.res_bus.va_degree.at[b2], test_ref[0][1], rtol=1e-4)
        assert np.isclose(net.res_bus.vm_pu.at[b3], test_ref[1][0], rtol=1e-4)
        assert np.isclose(net.res_bus.va_degree.at[b3], test_ref[1][1], rtol=1e-4)

        net.trafo3w.tap_pos.at[0] = 2
        pp.runpp(net, init="dc", calculate_voltage_angles=True)
        assert np.isclose(net.res_bus.vm_pu.at[b2], test_tap_pos[side][0][0], rtol=1e-4)
        assert np.isclose(net.res_bus.va_degree.at[b2], test_tap_pos[side][0][1], rtol=1e-4)
        assert np.isclose(net.res_bus.vm_pu.at[b3], test_tap_pos[side][1][0], rtol=1e-4)
        assert np.isclose(net.res_bus.va_degree.at[b3], test_tap_pos[side][1][1], rtol=1e-4)

        net.trafo3w.tap_pos.at[0] = -2
        pp.runpp(net, init="dc", calculate_voltage_angles=True)
        assert np.isclose(net.res_bus.vm_pu.at[b2], test_tap_neg[side][0][0], rtol=1e-4)
        assert np.isclose(net.res_bus.va_degree.at[b2], test_tap_neg[side][0][1], rtol=1e-4)
        assert np.isclose(net.res_bus.vm_pu.at[b3], test_tap_neg[side][1][0], rtol=1e-4)
        assert np.isclose(net.res_bus.va_degree.at[b3], test_tap_neg[side][1][1], rtol=1e-4)


def test_volt_dep_load_at_inactive_bus():
    # create empty net
    net = pp.create_empty_network()

    # create buses
    bus1 = pp.create_bus(net, index=0, vn_kv=20., name="Bus 1")
    bus2 = pp.create_bus(net, index=1, vn_kv=0.4, name="Bus 2")
    bus3 = pp.create_bus(net, index=3, in_service=False, vn_kv=0.4, name="Bus 3")
    bus4 = pp.create_bus(net, index=4, vn_kv=0.4, name="Bus 4")
    bus4 = pp.create_bus(net, index=5, vn_kv=0.4, name="Bus 4")

    # create bus elements
    pp.create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=4, p_mw=0.1, q_mvar=0.05, name="Load3", const_i_percent=100)
    pp.create_load(net, bus=5, p_mw=0.1, q_mvar=0.05, name="Load4")

    # create branch elements
    trafo = pp.create_transformer(net, hv_bus=bus1, lv_bus=bus2, std_type="0.4 MVA 20/0.4 kV",
                                  name="Trafo")
    line1 = pp.create_line(net, from_bus=1, to_bus=3, length_km=0.1, std_type="NAYY 4x50 SE",
                           name="Line")
    line2 = pp.create_line(net, from_bus=1, to_bus=4, length_km=0.1, std_type="NAYY 4x50 SE",
                           name="Line")
    line3 = pp.create_line(net, from_bus=1, to_bus=5, length_km=0.1, std_type="NAYY 4x50 SE",
                           name="Line")

    pp.runpp(net)
    assert not np.isnan(net.res_load.p_mw.at[1])
    assert not np.isnan(net.res_bus.p_mw.at[5])
    assert net.res_bus.p_mw.at[3] == 0

def test_two_oos_buses():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)
    b3 = pp.create_bus(net, vn_kv=0.4, in_service=False)
    b4 = pp.create_bus(net, vn_kv=0.4, in_service=False)

    pp.create_ext_grid(net, b1)
    l1 = pp.create_line(net, b1, b2, 0.5, std_type="NAYY 4x50 SE", index=4)
    l2 = pp.create_line(net, b2, b3, 0.5, std_type="NAYY 4x50 SE", index=2)
    l3 = pp.create_line(net, b3, b4, 0.5, std_type="NAYY 4x50 SE", index=7)

    pp.runpp(net)
    assert net.res_line.loading_percent.at[l1] > 0
    assert net.res_line.loading_percent.at[l2] > 0
    assert np.isnan(net.res_line.loading_percent.at[l3])

    net.line.drop(l2, inplace=True)
    pp.runpp(net)
    assert net.res_line.loading_percent.at[l1] > 0
    assert np.isnan(net.res_line.loading_percent.at[l3])

def test_oos_buses_at_trafo3w():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=110.)
    b2 = pp.create_bus(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=110., in_service=False)
    b4 = pp.create_bus(net, vn_kv=20., in_service=False)
    b5 = pp.create_bus(net, vn_kv=10., in_service=False)

    pp.create_ext_grid(net, b1)
    l1 = pp.create_line(net, b1, b2, 0.5, std_type="NAYY 4x50 SE", in_service=True)
    l2 = pp.create_line(net, b2, b3, 0.5, std_type="NAYY 4x50 SE", in_service=False)

    tidx = pp.create_transformer3w(net, b3, b4, b5, std_type='63/25/38 MVA 110/20/10 kV', in_service=True)

    pp.runpp(net, trafo3w_losses = 'star', trafo_model= 'pi', init='flat')

    assert net.res_line.loading_percent.at[l1] > 0
    assert np.isnan(net.res_trafo3w.i_hv_ka.at[tidx])


@pytest.fixture
def network_with_trafo3ws():
    net = pp.create_empty_network()
    add_test_trafo(net)
    slack, hv, ln = add_grid_connection(net, zone="test_trafo3w")
    for _ in range(2):
        mv = pp.create_bus(net, vn_kv=0.6, zone="test_trafo3w")
        pp.create_load(net, mv, p_mw=0.8, q_mvar=0)
        lv = pp.create_bus(net, vn_kv=0.4, zone="test_trafo3w")
        pp.create_load(net, lv, p_mw=0.5, q_mvar=0)
        t3 = pp.create_transformer3w_from_parameters(net, hv_bus=hv, mv_bus=mv, lv_bus=lv, vn_hv_kv=22,
                                                vn_mv_kv=.64, vn_lv_kv=.42, sn_hv_mva=1,
                                                sn_mv_mva=0.7, sn_lv_mva=0.3, vk_hv_percent=1.,
                                                vkr_hv_percent=.03, vk_mv_percent=.5,
                                                vkr_mv_percent=.02, vk_lv_percent=.25,
                                                vkr_lv_percent=.01, pfe_kw=.5, i0_percent=0.1,
                                                name="test", index=pp.get_free_id(net.trafo3w) + 1,
                                                tap_side="hv", tap_pos=2, tap_step_percent=1.25,
                                                tap_min=-5, tap_neutral=0, tap_max=5)
    return (net, t3, hv, mv, lv)
        
    
def test_trafo3w_switches(network_with_trafo3ws):
    net, t3, hv, mv, lv = network_with_trafo3ws

    # open switch at hv side - t3 is disconnected
    s1 = pp.create_switch(net, bus=hv, element=t3, et="t3", closed=False)
    runpp_with_consistency_checks(net)
    assert np.isnan(net.res_bus.vm_pu.at[mv])
    assert np.isnan(net.res_bus.vm_pu.at[lv])
    assert np.isnan(net.res_trafo3w.p_hv_mw.at[t3]) == 0

    # open switch at mv side - mv is disconnected, lv is connected
    net.switch.bus.at[s1] = mv
    runpp_with_consistency_checks(net)

    assert np.isnan(net.res_bus.vm_pu.at[mv])
    assert not np.isnan(net.res_bus.vm_pu.at[lv])
    assert net.res_trafo3w.i_lv_ka.at[t3] > 1e-5
    assert net.res_trafo3w.i_mv_ka.at[t3] < 1e-5
    assert 0.490 < net.res_trafo3w.p_hv_mw.at[t3] < 0.510

    # open switch at lv side - lv is disconnected, mv is connected
    net.switch.bus.at[s1] = lv
    runpp_with_consistency_checks(net)

    assert np.isnan(net.res_bus.vm_pu.at[lv])
    assert not np.isnan(net.res_bus.vm_pu.at[mv])
    assert net.res_trafo3w.i_lv_ka.at[t3] < 1e-5
    assert net.res_trafo3w.i_mv_ka.at[t3] > 1e-5
    assert 0.790 < net.res_trafo3w.p_hv_mw.at[t3] < 0.810

    # open switch at lv and mv side - lv and mv is disconnected, t3 in open loop
    pp.create_switch(net, bus=mv, element=t3, et="t3", closed=False)
    runpp_with_consistency_checks(net)

    assert np.isnan(net.res_bus.vm_pu.at[lv])
    assert np.isnan(net.res_bus.vm_pu.at[mv])
    assert net.res_trafo3w.i_lv_ka.at[t3] < 1e-5
    assert net.res_trafo3w.i_mv_ka.at[t3] < 1e-5
    assert 0 < net.res_trafo3w.p_hv_mw.at[t3] < 1

def test_generator_as_slack():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110.)
    pp.create_ext_grid(net, b1, vm_pu=1.02)
    b2 = pp.create_bus(net, 110.)
    pp.create_line(net, b1, b2, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_load(net, b2, p_mw=2)
    pp.runpp(net)
    res_bus = net.res_bus.vm_pu.values

    pp.create_gen(net, b1, p_mw=0.1, vm_pu=1.02, slack=True)
    net.ext_grid.in_service.iloc[0] = False
    pp.runpp(net)
    assert np.allclose(res_bus, net.res_bus.vm_pu.values)

    net.gen.slack.iloc[0] = False
    with pytest.raises(UserWarning):
        pp.runpp(net)
        
def test_transformer_with_two_open_switches():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110.)
    pp.create_ext_grid(net, b1, vm_pu=1.02)
    b2 = pp.create_bus(net, 20.)
    t = pp.create_transformer(net, b1, b2, std_type='63 MVA 110/20 kV')
    b3 = pp.create_bus(net, 20.)
    pp.create_line(net, b2, b3, length_km=7., std_type='149-AL1/24-ST1A 110.0')
    pp.create_load(net, b2, p_mw=2)
    pp.runpp(net)

    assert net.res_trafo.vm_hv_pu.at[t] == net.res_bus.vm_pu.at[b1]
    assert net.res_trafo.vm_lv_pu.at[t] == net.res_bus.vm_pu.at[b2]

    pp.create_switch(net, b2, element=t, et="t", closed=False)
    pp.runpp(net)
    assert net.res_trafo.vm_hv_pu.at[t] == net.res_bus.vm_pu.at[b1]
    assert net.res_trafo.vm_lv_pu.at[t] != net.res_bus.vm_pu.at[b2]


    pp.create_switch(net, b1, element=t, et="t", closed=False)
    pp.runpp(net)
    assert net.res_trafo.vm_hv_pu.at[t] != net.res_bus.vm_pu.at[b1]
    assert net.res_trafo.vm_lv_pu.at[t] != net.res_bus.vm_pu.at[b2]
    
if __name__ == "__main__":
    pytest.main(["-xs", __file__])
