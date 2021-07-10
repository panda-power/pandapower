# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc


def three_bus_example():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20.)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
    net.line["endtemp_degree"] = 80

    pp.create_sgen(net, b2, sn_mva=2, p_mw=0, k=1.2)

    net.ext_grid['x0x_min'] = 0.1
    net.ext_grid['r0x0_min'] = 0.1
    net.ext_grid['x0x_max'] = 0.1
    net.ext_grid['r0x0_max'] = 0.1

    net.line['r0_ohm_per_km'] = 0.1
    net.line['x0_ohm_per_km'] = 0.1
    net.line['c0_nf_per_km'] = 0.1
    net.line["endtemp_degree"] = 80
    return net


def three_bus_permuted_index():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110, index=4)
    b2 = pp.create_bus(net, 110, index=3)
    b3 = pp.create_bus(net, 110, index=0)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20., index=1)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15., index=0)
    net.line["endtemp_degree"] = 80

    pp.create_sgen(net, b2, sn_mva=2, p_mw=0, k=1.2)
    return net


# def gen_three_bus_example():
#     net = pp.create_empty_network(sn_mva=2)
#     b1 = pp.create_bus(net, vn_kv=10.)
#     b2 = pp.create_bus(net, vn_kv=10.)
#     b3 = pp.create_bus(net, vn_kv=10.)
#     #pp.create_bus(net, vn_kv=0.4, in_service=False)
#     pp.create_gen(net, b2, vn_kv=10.5, xdss_pu=0.2, rdss_pu=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5)
#     pp.create_line_from_parameters(net, b1, b2, length_km=1.0, max_i_ka=0.29,
#                                    r_ohm_per_km=0.1548, x_ohm_per_km=0.0816814, c_nf_per_km=165)
#     pp.create_line_from_parameters(net, b2, b3, length_km=1.0, max_i_ka=0.29,
#                                    r_ohm_per_km=0.1548, x_ohm_per_km=0.0816814, c_nf_per_km=165)
#     net.line["endtemp_degree"] = 165
#     pp.create_ext_grid(net, b1, s_sc_max_mva=10., s_sc_min_mva=8., rx_min=0.4, rx_max=0.4)
#     #pp.create_switch(net, b3, b1, et="b")
#     return net


def net_transformer():
    net = pp.create_empty_network(sn_mva=2)
    b1a = pp.create_bus(net, vn_kv=10.)
    b1b = pp.create_bus(net, vn_kv=10.)
    b2 = pp.create_bus(net, vn_kv=.4)
    pp.create_bus(net, vn_kv=0.4, in_service=False) #add out of service bus to test oos indexing
    pp.create_ext_grid(net, b1a, s_sc_max_mva=100., s_sc_min_mva=40., rx_min=0.1, rx_max=0.1)
    pp.create_switch(net, b1a, b1b, et="b")
    pp.create_transformer_from_parameters(net, b1b, b2, vn_hv_kv=11., vn_lv_kv=0.42, vk_percent=6.,
                                          vkr_percent=0.5, pfe_kw=14, shift_degree=0.0,
                                          tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2, tap_pos=2,
                                          tap_step_percent=2.5, parallel=2, sn_mva=0.4, i0_percent=0.5)
    pp.create_shunt(net, b2, q_mvar=0.050, p_mw=0.0500) #adding a shunt shouldn't change the result
    return net


def test_all_currents_sgen():
    #
    # eg--0---l0---1---l1---2
    #              |
    #              g
    #
    net = three_bus_example()
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True, return_all_currents=True)
    assert np.allclose(net.res_line_sc.ikss_ka.values,
                       np.array([0.01259673, 0.49593036, 0.48628848, 0., 0., 0.49888962]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ip_ka.values,
                       np.array([0.01781447, 0.92787447, 0.90729584, 0., 0., 0.92511655]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ith_ka.values,
                       np.array([0.01265116, 0.4981196, 0.48841266, 0., 0., 0.50106884]), atol=1e-5)

    sc.calc_sc(net, case="min", ip=True, ith=True, branch_results=True, return_all_currents=True)
    assert np.allclose(net.res_line_sc.ikss_ka.values,
                       np.array([0.01259673, 0.3989686, 0.39170662, 0., 0., 0.40431286]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ip_ka.values,
                       np.array([0.01781447, 0.74438751, 0.72793774, 0., 0., 0.74576565]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ith_ka.values,
                       np.array([0.01265116, 0.40071219, 0.39339323, 0., 0., 0.40605375]), atol=1e-5)


def test_all_currents_1ph_max():
    # Only check coherence between branch currents and bus currents
    #
    # eg--0---l0---1---l1---2
    #              |
    #              g
    #
    # With generator
    net = three_bus_example()
    sc.calc_sc(net, case="max", fault='1ph', branch_results=True, return_all_currents=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_gen = net.res_line_sc.copy()

    # Without generator
    net = three_bus_example()
    net.sgen.in_service = False
    sc.calc_sc(net, case="max", fault='1ph')
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Isolate sgen contrib
    i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 0)], i_bus_only_sgen.ikss_ka.at[0], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 1)], i_bus_without_sgen.ikss_ka.at[1], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 2)], i_bus_without_sgen.ikss_ka.at[2] -
                      (i_bus_only_sgen.ikss_ka.at[1] - i_bus_only_sgen.ikss_ka.at[2]) , atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 0)], 0., atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 1)], 0., atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 2)], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)


def test_all_currents_1ph_min():
    # Only check coherence between branch currents and bus currents
    #
    # eg--0---l0---1---l1---2
    #              |
    #              g
    #
    # With generator
    net = three_bus_example()
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True, return_all_currents=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_gen = net.res_line_sc.copy()

    # Without generator
    net.sgen.in_service = False
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True)
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Isolate sgen contrib
    i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 0)], i_bus_only_sgen.ikss_ka.at[0], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 1)], i_bus_without_sgen.ikss_ka.at[1], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 2)], i_bus_without_sgen.ikss_ka.at[2] -
                      (i_bus_only_sgen.ikss_ka.at[1] - i_bus_only_sgen.ikss_ka.at[2]) , atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 0)], 0., atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 1)], 0., atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 2)], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)


def test_with_permuted_index():
    # Check that if element's index are permuted the results are still consistent
    #
    # eg--4---l1---3---l0---1
    #              |
    #              g
    #
    net = three_bus_permuted_index()
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True, return_all_currents=True)
    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(1, 4), (1, 3), (1, 0), (0, 4), (0, 3), (0, 0)]].values,
                       np.array([0.01259673, 0.49593036, 0.48628848, 0., 0., 0.49888962]), atol=1e-5)
    sc.calc_sc(net, case="min", ip=True, ith=True, branch_results=True, return_all_currents=True)
    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(1, 4), (1, 3), (1, 0), (0, 4), (0, 3), (0, 0)]].values,
                       np.array([0.01259673, 0.3989686, 0.39170662, 0., 0., 0.40431286]), atol=1e-5)


def test_all_currents_with_oos_elements():

    net = three_bus_example()
    net.bus.in_service.loc[2] = False
    net.line.in_service.loc[1] = False
    sc.calc_sc(net, case="max", branch_results=True, return_all_currents=True)

    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(0, 0), (0, 1)]].values,
                       np.array([0.01259673, 0.49593036]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(0, 2), (1, 0), (1, 1), (1, 2)]].values,
                       0, atol=1e-10)

    sc.calc_sc(net, case="min", branch_results=True, return_all_currents=True)
    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(0, 0), (0, 1)]].values,
                       np.array([0.01259673, 0.3989686]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(0, 2), (1, 0), (1, 1), (1, 2)]].values,
                       0, atol=1e-10)


# TODO: This example should not work anymore
# def test_branch_all_currents_gen():
#     net = gen_three_bus_example()
#     sc.calc_sc(net, case="max", branch_results=True, return_all_currents=True)
#     assert np.allclose(net.res_line_sc.ikss_ka.values,
#                        np.array([0.76204252, 0.57040645, 0.55786693, 0., 0., 1.28698045]))

#     sc.calc_sc(net, case="min", branch_results=True, return_all_currents=True)
#     assert np.allclose(net.res_line_sc.ikss_ka.values,
#                        np.array([0.69255026, 0.45574755, 0.44487882, 0., 0., 1.10747517]))


def test_branch_all_currents_trafo():
    net = net_transformer()
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent=10., branch_results=True, return_all_currents=True)

    assert (abs(net.res_trafo_sc.ikss_lv_ka.loc[(0,0)] - 0.) <1e-5)
    assert (abs(net.res_trafo_sc.ikss_lv_ka.loc[(0,1)] - 0.) <1e-5)
    assert (abs(net.res_trafo_sc.ikss_lv_ka.loc[(0,2)] - 16.992258758) <1e-5)

    assert (abs(net.res_trafo_sc.ikss_hv_ka.loc[(0,0)] - 0.) <1e-5)
    assert (abs(net.res_trafo_sc.ikss_hv_ka.loc[(0,1)] - 0.) <1e-5)
    assert (abs(net.res_trafo_sc.ikss_hv_ka.loc[(0,2)] - 0.648795) <1e-5)


def test_against_single_sc_results_line():
    net = three_bus_permuted_index()

    sc.calc_sc(net, case="max", branch_results=True, return_all_currents=True)
    multi_results = net.res_line_sc.copy()

    for bus in net.bus.index:
        sc.calc_sc(net, bus=bus, case="max", branch_results=True, return_all_currents=True)
        line_bus_indices = [(line, bus) for line in net.line.index]
        single_result = net.res_line_sc.ikss_ka.values
        multi_result = multi_results.ikss_ka.loc[line_bus_indices].values
        assert np.allclose(single_result, multi_result)


def test_against_single_sc_results_trafo():
    net = net_transformer()
    sc.calc_sc(net, case="max", branch_results=True, return_all_currents=True, inverse_y=False)
    multi_results = net.res_trafo_sc.copy()

    for bus in net.bus.index[net.bus.in_service]:
        sc.calc_sc(net, bus=bus, case="max", branch_results=True, return_all_currents=True, inverse_y=False)
        trafo_bus_indices = [(trafo, bus) for trafo in net.trafo.index]
        single_result_lv = net.res_trafo_sc.ikss_lv_ka.values
        multi_result_lv = multi_results.ikss_lv_ka.loc[trafo_bus_indices].values
        assert np.allclose(single_result_lv, multi_result_lv)

        single_result_hv = net.res_trafo_sc.ikss_hv_ka.values
        multi_result_hv = multi_results.ikss_hv_ka.loc[trafo_bus_indices].values
        assert np.allclose(single_result_hv, multi_result_hv)


if __name__ == '__main__':
    pytest.main([__file__])
