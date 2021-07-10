# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc


def wind_park_example():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110., index=1)
    b2 = pp.create_bus(net, vn_kv=110., index=2)
    b3 = pp.create_bus(net, vn_kv=110., index=3)
    b4 = pp.create_bus(net, vn_kv=110., index=4)
    pp.create_ext_grid(net, b1, s_sc_max_mva=20*110*np.sqrt(3), rx_max=0.1)

    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=100, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b3, length_km=50, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3, length_km=50, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b3, to_bus=b4, length_km=25, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)

    pp.create_sgen(net, b2, p_mw=0.1e3, sn_mva=100)
    pp.create_sgen(net, b3, p_mw=0.050e3, sn_mva=50)
    pp.create_sgen(net, b4, p_mw=0.050e3, sn_mva=50)
    net.sgen["k"] = 1.2
    return net


def three_bus_example():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4, x0x_max=0.2, x0x_min=0.1,
                       r0x0_max=0.3, r0x0_min=0.2)
    net.ext_grid['x0x_min'] = 0.1
    net.ext_grid['r0x0_min'] = 0.1
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20.)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
    net.line['r0_ohm_per_km'] = 0.1
    net.line['x0_ohm_per_km'] = 0.1
    net.line['c0_nf_per_km'] = 0.1
    net.line["endtemp_degree"] = 80

    pp.create_sgen(net, b2, sn_mva=2, p_mw=0, k=1.2)
    return net


def big_sgen_three_bus_example():
    # ext_grid-bus1--line0--bus2--line1--bus3
    #                        |
    #                       sgen0
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    net.ext_grid['x0x_min'] = 0.1
    net.ext_grid['r0x0_min'] = 0.1
    net.ext_grid['x0x_max'] = 0.1
    net.ext_grid['r0x0_max'] = 0.1

    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20.)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
    net.line['r0_ohm_per_km'] = 0.1
    net.line['x0_ohm_per_km'] = 0.1
    net.line['c0_nf_per_km'] = 0.1
    net.line["endtemp_degree"] = 80

    pp.create_sgen(net, b2, sn_mva=200., p_mw=0, k=1.2)
    return net


def test_max_3ph_branch_small_sgen():
    net = three_bus_example()
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, np.array([0.53746061, 0.50852707, 0.4988896]))
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([ 0.49593034,  0.4988896]))
    assert np.allclose(net.res_line_sc.ip_ka.values, np.array([ 0.92787443,  0.9251165]))
    assert np.allclose(net.res_line_sc.ith_ka.values, np.array([ 0.49811957,  0.50106881]))


def test_max_3ph_branch_big_sgen():
    net = big_sgen_three_bus_example()
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, np.array([1.78453722, 1.75560368, 1.72233192]))
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([1.25967331, 1.72233192]))
    assert np.allclose(net.res_line_sc.ip_ka.values, np.array([1.78144709, 2.65532524]))
    assert np.allclose(net.res_line_sc.ith_ka.values, np.array([1.26511638, 1.7298553]))


def test_min_3ph_branch_results_small_sgen():
    net = three_bus_example()
    sc.calc_sc(net, case="min", ip=True, ith=True, branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, np.array([ 0.43248784,  0.41156533,  0.40431286]))
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([0.01259673,  0.40431286]))
    assert np.allclose(net.res_line_sc.ip_ka.values, np.array([0.01781447, 0.74576565]))
    assert np.allclose(net.res_line_sc.ith_ka.values, np.array([0.01265116, 0.40605375]))


def test_min_3ph_branch_results_big_sgen():
    net = big_sgen_three_bus_example()
    sc.calc_sc(net, case="min", ip=True, ith=True, branch_results=True)
    assert np.allclose(net.res_bus_sc.ikss_ka.values, np.array([1.67956442, 1.65864191, 1.62941387]))
    assert np.allclose(net.res_line_sc.ikss_ka.values, np.array([0.36974055, 1.62941387]))
    assert np.allclose(net.res_line_sc.ip_ka.values, np.array([0.69687302,   2.47832011]))
    assert np.allclose(net.res_line_sc.ith_ka.values, np.array([0.37133258,  1.63642978]))


def test_max_1ph_branch_small_sgen():
    # This test just check coherence between branch ikss_ka results and bus ikss_ka results

    # With generator
    net = three_bus_example()
    sc.calc_sc(net, case="max", fault='1ph', branch_results=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_gen = net.res_line_sc.copy()

    # Without generator
    net = three_bus_example()
    net.sgen.in_service = False
    sc.calc_sc(net, case="max", fault='1ph', branch_results=True)
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Check coherence between bus result and branch results
    assert np.isclose(i_line_with_gen.ikss_ka.at[0], i_bus_without_sgen.ikss_ka.at[1], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.at[1], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)



def test_max_1ph_branch_big_sgen():
    # This test just check coherence between branch ikss_ka results and bus ikss_ka results

    # With generator
    net = big_sgen_three_bus_example()
    sc.calc_sc(net, case="max", fault='1ph', branch_results=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_gen = net.res_line_sc.copy()

    # Without generator
    net = big_sgen_three_bus_example()
    net.sgen.in_service = False
    sc.calc_sc(net, case="max", fault='1ph', branch_results=True)
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Isolate sgen contribution
    i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

    # Check coherence between bus result and branch results
    # Since here sgen ikss_ka > ext_grid ikss_ka
    assert np.isclose(i_line_with_gen.ikss_ka.at[0], i_bus_only_sgen.ikss_ka.at[0], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.at[1], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)


def test_min_1ph_branch_small_sgen():
    # This test just check coherence between branch ikss_ka results and bus ikss_ka results

    # With generator
    net = three_bus_example()
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_gen = net.res_line_sc.copy()

    # Without generator
    net = three_bus_example()
    net.sgen.in_service = False
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True)
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Isolate sgen contribution
    i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

    # Check coherence between bus result and branch results
    assert np.isclose(i_line_with_gen.ikss_ka.at[0], i_bus_only_sgen.ikss_ka.at[0], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.at[1], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)


def test_min_1ph_branch_big_sgen():
    # This test just check coherence between branch ikss_ka results and bus ikss_ka results

    # With generator
    net = big_sgen_three_bus_example()
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_sgen = net.res_line_sc.copy()

    # Without generator
    net = big_sgen_three_bus_example()
    net.sgen.in_service = False
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True)
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Isolate sgen contribution
    i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

    # Applying current divider: when sc is on bus 2 a small portion sgen ikss_ka flows along line_0 in the opposite
    # direction with respect of ext_grid ikss_ka
    i_line_1 = i_bus_without_sgen.ikss_ka.at[2] - (i_bus_only_sgen.ikss_ka.at[1] - i_bus_only_sgen.ikss_ka.at[2])

    # Check coherence between bus result and branch results
    assert np.isclose(i_line_with_sgen.ikss_ka.at[0], i_line_1, atol=1e-4)
    assert np.isclose(i_line_with_sgen.ikss_ka.at[1], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)


def test_wind_park():
    net = wind_park_example()
    sc.calc_sc(net, ip=True)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[2], 3.9034, rtol=1e-4)
    assert np.isclose(net.res_bus_sc.ip_ka.at[2], 7.3746, rtol=1e-4)

if __name__ == '__main__':
    pytest.main([__file__])
