# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandapower as pp
import pytest

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def test_convenience_create_functions():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110.)
    b2 = pp.create_bus(net, 110.)
    b3 = pp.create_bus(net, 20)
    pp.create_ext_grid(net, b1)
    pp.create_line_from_parameters(net, b1, b2, length_km=20., r_ohm_per_km=0.0487,
                                   x_ohm_per_km=0.1382301, c_nf_per_km=160., max_i_ka=0.664)

    l0 = pp.create_load_from_cosphi(net, b2, 10, 0.95, "ind", name="load")
    pp.runpp(net, init="flat")
    assert net.load.p_mw.at[l0] == 9.5
    assert net.load.q_mvar.at[l0] > 0
    assert np.sqrt(net.load.p_mw.at[l0] ** 2 + net.load.q_mvar.at[l0] ** 2) == 10
    assert np.isclose(net.res_bus.vm_pu.at[b2], 0.99990833838)
    assert net.load.name.at[l0] == "load"

    sh0 = pp.create_shunt_as_capacitor(net, b2, 10, loss_factor=0.01, name="shunt")
    pp.runpp(net, init="flat")
    assert np.isclose(net.res_shunt.q_mvar.at[sh0], -10.043934174)
    assert np.isclose(net.res_shunt.p_mw.at[sh0], 0.10043933665)
    assert np.isclose(net.res_bus.vm_pu.at[b2], 1.0021942964)
    assert net.shunt.name.at[sh0] == "shunt"

    sg0 = pp.create_sgen_from_cosphi(net, b2, 5, 0.95, "cap", name="sgen")
    pp.runpp(net, init="flat")
    assert np.sqrt(net.sgen.p_mw.at[sg0] ** 2 + net.sgen.q_mvar.at[sg0] ** 2) == 5
    assert net.sgen.p_mw.at[sg0] == 4.75
    assert net.sgen.q_mvar.at[sg0] > 0
    assert np.isclose(net.res_bus.vm_pu.at[b2], 1.0029376578)
    assert net.sgen.name.at[sg0] == "sgen"

    tol = 1e-6
    base_z = 110 ** 2 / 100
    sind = pp.create_series_reactor_as_impedance(net, b1, b2, r_ohm=100, x_ohm=200, sn_mva=100)
    assert net.impedance.at[sind, 'rft_pu'] - 100 / base_z < tol
    assert net.impedance.at[sind, 'xft_pu'] - 200 / base_z < tol

    tid = pp.create_transformer_from_parameters(net, hv_bus=b2, lv_bus=b3, sn_mva=0.1, vn_hv_kv=110,
                                                vn_lv_kv=20, vkr_percent=5, vk_percent=20,
                                                pfe_kw=1, i0_percent=1)
    pp.create_load(net, b3, 0.1)
    assert net.trafo.at[tid, 'df'] == 1
    pp.runpp(net)
    tr_l = net.res_trafo.at[tid, 'loading_percent']
    net.trafo.at[tid, 'df'] = 2
    pp.runpp(net)
    tr_l_2 = net.res_trafo.at[tid, 'loading_percent']
    assert tr_l == tr_l_2 * 2
    net.trafo.at[tid, 'df'] = 0
    with pytest.raises(UserWarning):
        pp.runpp(net)



def test_nonexistent_bus():
    from functools import partial
    net = pp.create_empty_network()
    create_functions = [partial(pp.create_load, net=net, p_mw=0, q_mvar=0, bus=0, index=0),
                        partial(pp.create_sgen, net=net, p_mw=0, q_mvar=0, bus=0, index=0),
                        partial(pp.create_dcline, net, from_bus=0, to_bus=1, p_mw=0.1,
                                loss_percent=0, loss_mw=0.01, vm_from_pu=1., vm_to_pu=1., index=0),
                        partial(pp.create_gen, net=net, p_mw=0, bus=0, index=0),
                        partial(pp.create_ward, net, 0, 0, 0, 0, 0, index=0),
                        partial(pp.create_xward, net, 0, 0, 0, 0, 0, 1, 1, 1, index=0),
                        partial(pp.create_shunt, net=net, q_mvar=0, bus=0, index=0),
                        partial(pp.create_ext_grid, net=net, bus=1, index=0),
                        partial(pp.create_line, net=net, from_bus=0, to_bus=1, length_km=1.,
                                std_type="NAYY 4x50 SE", index=0),
                        partial(pp.create_line_from_parameters, net=net, from_bus=0, to_bus=1,
                                length_km=1., r_ohm_per_km=0.1, x_ohm_per_km=0.1, max_i_ka=0.4,
                                c_nf_per_km=10, index=1),
                        partial(pp.create_transformer, net=net, hv_bus=0, lv_bus=1,
                                std_type="63 MVA 110/20 kV", index=0),
                        partial(pp.create_transformer3w, net=net, hv_bus=0, lv_bus=1, mv_bus=2,
                                std_type="63/25/38 MVA 110/20/10 kV", index=0),
                        partial(pp.create_transformer3w_from_parameters, net=net, hv_bus=0,
                                lv_bus=1, mv_bus=2, i0_percent=0.89, pfe_kw=3.5,
                                vn_hv_kv=110, vn_lv_kv=10, vn_mv_kv=20, sn_hv_mva=63,
                                sn_lv_mva=38, sn_mv_mva=25, vk_hv_percent=10.4,
                                vk_lv_percent=10.4, vk_mv_percent=10.4, vkr_hv_percent=0.28,
                                vkr_lv_percent=0.35, vkr_mv_percent=0.32, index=1),
                        partial(pp.create_transformer_from_parameters, net=net, hv_bus=0, lv_bus=1,
                                sn_mva=60, vn_hv_kv=20., vn_lv_kv=0.4, vk_percent=10,
                                vkr_percent=0.1, pfe_kw=0, i0_percent=0, index=1),
                        partial(pp.create_impedance, net=net, from_bus=0, to_bus=1,
                                rft_pu=0.1, xft_pu=0.1, sn_mva=0.6, index=0),
                        partial(pp.create_switch, net, bus=0, element=1, et="b", index=0)]
    for func in create_functions:
        with pytest.raises(Exception):  # exception has to be raised since bus doesn't exist
            func()
    pp.create_bus(net, 0.4)
    pp.create_bus(net, 0.4)
    pp.create_bus(net, 0.4)
    for func in create_functions:
        func()  # buses exist, element can be created
        with pytest.raises(Exception):  # exception is raised because index already exists
            func()


def test_tap_phase_shifter_default():
    expected_default = False
    net = pp.create_empty_network()
    pp.create_bus(net, 110)
    pp.create_bus(net, 20)
    data = pp.load_std_type(net, "25 MVA 110/20 kV", "trafo")
    if "tap_phase_shifter" in data:
        del data["tap_phase_shifter"]
    pp.create_std_type(net, data, "without_tap_shifter_info", "trafo")
    pp.create_transformer_from_parameters(net, 0, 1, 25e3, 110, 20, 0.4, 12, 20, 0.07)
    pp.create_transformer(net, 0, 1, "without_tap_shifter_info")
    assert (net.trafo.tap_phase_shifter == expected_default).all()


def test_create_line_conductance():
    net = pp.create_empty_network()
    pp.create_bus(net, 20)
    pp.create_bus(net, 20)
    pp.create_std_type(net, {'c_nf_per_km': 210, 'max_i_ka': 0.142, 'q_mm2': 50,
                             'r_ohm_per_km': 0.642, 'type': 'cs', 'x_ohm_per_km': 0.083,
                             "g_us_per_km": 1}, "test_conductance")

    l = pp.create_line(net, 0, 1, 1., "test_conductance")
    assert net.line.g_us_per_km.at[l] == 1


def test_create_buses():
    net = pp.create_empty_network()
    # standard
    b1 = pp.create_buses(net, 3, 110)
    # with geodata
    b2 = pp.create_buses(net, 3, 110, geodata=(10, 20))
    # with geodata as array
    geodata = np.array([[10, 20], [20, 30], [30, 40]])
    b3 = pp.create_buses(net, 3, 110, geodata=geodata)

    assert len(net.bus) == 9
    assert len(net.bus_geodata) == 6

    for i in b2:
        assert net.bus_geodata.at[i, 'x'] == 10
        assert net.bus_geodata.at[i, 'y'] == 20

    assert (net.bus_geodata.loc[b3, ['x', 'y']].values == geodata).all()

    # no way of creating buses with not matching shape
    with pytest.raises(ValueError):
        pp.create_buses(net, 2, 110, geodata=geodata)


def test_create_lines():
    # standard
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines(net, [b1, b1], [b2, b2], 4, std_type="48-AL1/8-ST1A 10.0")
    assert len(net.line) == 2
    assert len(net.line_geodata) == 0
    assert sum(net.line.std_type == "48-AL1/8-ST1A 10.0") == 2
    assert len(set(net.line.r_ohm_per_km)) == 1

    # with geodata
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines(net, [b1, b1], [b2, b2], [1.5, 3], std_type="48-AL1/8-ST1A 10.0",
                         geodata=[[(1,1),(2,2),(3,3)], [(1,1),(1,2)]])

    assert len(net.line) == 2
    assert len(net.line_geodata) == 2
    assert net.line_geodata.at[l[0], "coords"] == [(1,1),(2,2),(3,3)]
    assert net.line_geodata.at[l[1], "coords"] == [(1,1),(1,2)]

    # setting params as single value
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines(net, [b1, b1], [b2, b2], length_km=5, df=0.8, in_service=False,
                        geodata=[(10, 10), (20, 20)], parallel=1, max_loading_percent=90,
                        name="test", std_type="48-AL1/8-ST1A 10.0")

    assert len(net.line) == 2
    assert len(net.line_geodata) == 2
    assert net.line.length_km.at[l[0]] == 5
    assert net.line.length_km.at[l[1]] == 5
    assert net.line.at[l[0], "in_service"] == False  # is actually <class 'numpy.bool_'>
    assert net.line.at[l[1], "in_service"] == False  # is actually <class 'numpy.bool_'>
    assert net.line_geodata.at[l[0], "coords"] == [(10,10), (20,20)]
    assert net.line_geodata.at[l[1], "coords"] == [(10,10), (20,20)]
    assert net.line.at[l[0], "name"] == "test"
    assert net.line.at[l[1], "name"] == "test"
    assert net.line.at[l[0], "max_loading_percent"] == 90
    assert net.line.at[l[1], "max_loading_percent"] == 90
    assert net.line.at[l[0], "parallel"] == 1
    assert net.line.at[l[1], "parallel"] == 1

    # setting params as array
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 10)
    b2 = pp.create_bus(net, 10)
    l = pp.create_lines(net, [b1, b1], [b2, b2], length_km=[1, 5], df=[0.8, 0.7],
                        in_service=[True, False],
                        geodata=[[(10, 10), (20, 20)], [(100, 10), (200, 20)]], parallel=[2, 1],
                        max_loading_percent=[80, 90], name=["test1", "test2"],
                        std_type="48-AL1/8-ST1A 10.0")

    assert len(net.line) == 2
    assert len(net.line_geodata) == 2
    assert net.line.at[l[0], "length_km"] == 1
    assert net.line.at[l[1], "length_km"] == 5
    assert net.line.at[l[0], "in_service"] == True  # is actually <class 'numpy.bool_'>
    assert net.line.at[l[1], "in_service"] == False  # is actually <class 'numpy.bool_'>
    assert net.line_geodata.at[l[0], "coords"] == [(10,10), (20,20)]
    assert net.line_geodata.at[l[1], "coords"] == [(100,10), (200,20)]
    assert net.line.at[l[0], "name"] == "test1"
    assert net.line.at[l[1], "name"] == "test2"
    assert net.line.at[l[0], "max_loading_percent"] == 80
    assert net.line.at[l[1], "max_loading_percent"] == 90
    assert net.line.at[l[0], "parallel"] == 2
    assert net.line.at[l[1], "parallel"] == 1


def test_create_line_alpha_temperature():
    net=pp.create_empty_network()
    b = pp.create_buses(net, 5, 110)

    l1=pp.create_line(net,0,1, 10, "48-AL1/8-ST1A 10.0")
    l2=pp.create_line(net,1,2, 10, "48-AL1/8-ST1A 10.0", alpha=4.03e-3, temperature_degree_celsius=80)
    l3=pp.create_line(net,2,3, 10, "48-AL1/8-ST1A 10.0")
    l4=pp.create_line_from_parameters(net, 3,4,10, 1,1,1,100)
    l5=pp.create_line_from_parameters(net, 3,4,10, 1,1,1,100, alpha=4.03e-3)

    assert 'alpha' in net.line.columns
    assert all(net.line.loc[[l2,l3,l5], 'alpha'] == 4.03e-3)
    assert all(net.line.loc[[l1,l4], 'alpha'].isnull())
    assert net.line.loc[l2, 'temperature_degree_celsius'] == 80
    assert all(net.line.loc[[l1,l3,l4,l5], 'temperature_degree_celsius'].isnull())



if __name__ == '__main__':
    test_create_lines()
    # pytest.main(["test_create.py"])
