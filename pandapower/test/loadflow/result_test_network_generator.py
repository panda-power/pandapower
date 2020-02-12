# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandapower as pp
from pandapower.test.toolbox import add_grid_connection, create_test_line
from pandapower.toolbox import nets_equal

def result_test_network_generator2(net, sn_mva=1, skip_test_impedance=False):
    """ This is a generator for the result_test_network
        It is structured like this so it can be tested for consistency at
        different stages of adding elements
    """
    yield add_test_trafo(net)
#    yield add_test_line(net)
    yield add_test_load_sgen(net)
    yield add_test_load_sgen_split(net)
    yield add_test_ext_grid(net)
    yield add_test_trafo(net)
    yield add_test_single_load_single_eg(net)
    yield add_test_ward(net)
    yield add_test_ward_split(net)
    yield add_test_xward(net)
    yield add_test_xward_combination(net)
    yield add_test_gen(net)
    yield add_test_ext_grid_gen_switch(net)
    yield add_test_enforce_qlims(net)
    yield add_test_trafo3w(net)
    if not skip_test_impedance:
        yield add_test_impedance(net)
    yield add_test_bus_bus_switch(net)
    yield add_test_oos_bus_with_is_element(net)
    yield add_test_shunt(net)
    yield add_test_shunt_split(net)
    yield add_test_two_open_switches_on_deactive_line(net)


def result_test_network_generator(sn_mva=1, skip_test_impedance=False):
    """ This is a generator for the result_test_network
        It is structured like this so it can be tested for consistency at
        different stages of adding elements
    """
    net = pp.create_empty_network(sn_mva=sn_mva)
    yield add_test_line(net)
    yield add_test_load_sgen(net)
    yield add_test_load_sgen_split(net)
    yield add_test_ext_grid(net)
    yield add_test_trafo(net)
    yield add_test_single_load_single_eg(net)
    yield add_test_ward(net)
    yield add_test_ward_split(net)
    yield add_test_xward(net)
    yield add_test_xward_combination(net)
    yield add_test_gen(net)
    yield add_test_ext_grid_gen_switch(net)
    yield add_test_enforce_qlims(net)
    yield add_test_trafo3w(net)
    if not skip_test_impedance:
        yield add_test_impedance(net)
    yield add_test_bus_bus_switch(net)
    yield add_test_oos_bus_with_is_element(net)
    yield add_test_shunt(net)
    yield add_test_shunt_split(net)
    yield add_test_two_open_switches_on_deactive_line(net)


def result_test_network_generator_dcpp(sn_mva=1):
    """ This is a generator for the result_test_network
        It is structured like this so it can be tested for consistency at
        different stages of adding elements
    """
    # ToDo: Uncommented tests fail in rundcpp -> Check why and correct it

    net = pp.create_empty_network(sn_mva=sn_mva)
    yield add_test_line(net)
    yield add_test_load_sgen(net)
    yield add_test_load_sgen_split(net)
    # yield add_test_ext_grid(net)
    # yield add_test_trafo(net)
    yield add_test_single_load_single_eg(net)
    yield add_test_ward(net)
    yield add_test_ward_split(net)
    yield add_test_xward(net)
    yield add_test_xward_combination(net)
    # yield add_test_gen(net)
    # yield add_test_ext_grid_gen_switch(net)
    # yield add_test_enforce_qlims(net)
    # yield add_test_trafo3w(net)
    # yield add_test_impedance(net)
    yield add_test_bus_bus_switch(net)
    # yield add_test_oos_bus_with_is_element(net)
    yield add_test_shunt(net)
    yield add_test_shunt_split(net)
    # yield add_test_two_open_switches_on_deactive_line(net)


def add_test_line(net):
    b1, b2, l1 = add_grid_connection(net, zone="test_line")
    net.line.parallel.at[l1] = 2
    net.line.g_us_per_km.at[l1] = 1
    pp.create_load(net, b2, p_mw=1.2, q_mvar=1.1)
    l2 = create_test_line(net, b1, b2)
    net.line.g_us_per_km.at[l2] = 1
    pp.create_switch(net, b2, l2, et="l", closed=False)
    create_test_line(net, b1, b2, in_service=False)
    net.last_added_case = "test_line"
    return net


def add_test_ext_grid(net):
    b1, b2, ln = add_grid_connection(net, zone="test_ext_grid")
    b3 = pp.create_bus(net, vn_kv=20., zone="test_ext_grid")
    create_test_line(net, b2, b3)
    pp.create_ext_grid(net, b3, vm_pu=1.02, va_degree=3.)
    return net


def add_test_ext_grid_gen_switch(net):
    zone = "test_ext_grid_gen_switch"
    vn_kv = 20.
    b1 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)
    b2 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)
    b3 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)
    b4 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)
    b5 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)

    pp.create_switch(net, bus=b1, element=b2, et="b")
    create_test_line(net, b2, b3)
    create_test_line(net, b3, b4)
    pp.create_switch(net, bus=b4, element=b5, et="b")

    pp.create_ext_grid(net, bus=b1, vm_pu=1.01)
    pp.create_gen(net, bus=b5, vm_pu=1.015, p_mw=0.3)
    return net


def add_test_load_sgen(net):
    b1, b2, ln = add_grid_connection(net, zone="test_load_sgen")
    pl = 1.2
    ql = 1.1
    ps = 0.50
    qs = -0.1
    # load and sgen at one bus
    pp.create_load(net, b2, p_mw=pl, q_mvar=ql)
    pp.create_sgen(net, b2, p_mw=ps, q_mvar=qs)
    # adding out of serivce loads and sgens should not change the result
    pp.create_load(net, b2, p_mw=pl, q_mvar=ql, in_service=False,
                   index=pp.get_free_id(net.load) + 1)
    pp.create_sgen(net, b2, p_mw=ps, q_mvar=qs, in_service=False,
                   index=pp.get_free_id(net.sgen) + 1)
    net.last_added_case = "test_load_sgen"
    return net


def add_test_load_sgen_split(net):
    b1, b2, ln = add_grid_connection(net, zone="test_load_sgen_split")
    nr = 2
    pl = 1.2
    ql = 1.1
    ps = 0.5
    qs = -0.1
    for _ in list(range(nr)):
        pp.create_load(net, b2, p_mw=pl, q_mvar=ql, scaling=1. / nr)
        pp.create_sgen(net, b2, p_mw=ps, q_mvar=qs, scaling=1. / nr)
    net.last_added_case = "test_load_sgen_split"
    return net


def add_test_trafo(net):
    b1, b2, ln = add_grid_connection(net, zone="test_trafo")
    b3 = pp.create_bus(net, vn_kv=0.4, zone="test_trafo")
    pp.create_transformer_from_parameters(net, b2, b3, vk_percent=5., vkr_percent=2.,
                                          i0_percent=.4, pfe_kw=2., sn_mva=0.4, vn_hv_kv=22,
                                          vn_lv_kv=0.42, tap_max=10, tap_neutral=5, tap_min=0,
                                          tap_step_percent=1.25, tap_pos=3, shift_degree=150,
                                          tap_side="hv", parallel=2)
    t2 = pp.create_transformer_from_parameters(net, b2, b3, vk_percent=5., vkr_percent=2.,
                                               i0_percent=.4, pfe_kw=2, sn_mva=0.4, vn_hv_kv=22,
                                               vn_lv_kv=0.42, tap_max=10, tap_neutral=5, tap_min=0,
                                               tap_step_percent=1.25, tap_pos=3, tap_side="hv",
                                               shift_degree=150, index=pp.get_free_id(net.trafo) + 1)
    pp.create_switch(net, b3, t2, et="t", closed=False)
    pp.create_transformer_from_parameters(net, b2, b3, vk_percent=5., vkr_percent=2.,
                                          i0_percent=1., pfe_kw=20, sn_mva=0.4, vn_hv_kv=20,
                                          vn_lv_kv=0.4, in_service=False)
    pp.create_load(net, b3, p_mw=0.2, q_mvar=0.05)
    net.last_added_case = "test_trafo"
    return net


def add_test_single_load_single_eg(net):
    b1 = pp.create_bus(net, vn_kv=20., zone="test_single_load_single_eg")
    pp.create_ext_grid(net, b1)
    pp.create_load(net, b1, p_mw=0.1, q_mvar=0.1)
    net.last_added_case = "test_single_load_single_eg"
    return net


def add_test_ward(net):
    b1, b2, ln = add_grid_connection(net, zone="test_ward")

    pz = 1.2
    qz = 1.1
    ps = 0.5
    qs = 0.2
    # one shunt at a bus
    pp.create_ward(net, b2, pz_mw=pz, qz_mvar=qz, ps_mw=ps, qs_mvar=qs)
    # add out of service ward shuold not change the result
    pp.create_ward(net, b2, pz_mw=pz, qz_mvar=qz, ps_mw=ps, qs_mvar=qs, in_service=False,
                   index=pp.get_free_id(net.ward) + 1)
    net.last_added_case = "test_ward"
    return net


def add_test_ward_split(net):
    # splitting up the wards should not change results
    pz = 1.2
    qz = 1.1
    ps = 0.5
    qs = 0.2
    b1, b2, ln = add_grid_connection(net, zone="test_ward_split")
    pp.create_ward(net, b2, pz_mw=pz / 2, qz_mvar=qz / 2, ps_mw=ps / 2, qs_mvar=qs / 2)
    pp.create_ward(net, b2, pz_mw=pz / 2, qz_mvar=qz / 2, ps_mw=ps / 2, qs_mvar=qs / 2)
    net.last_added_case = "test_ward_split"
    return net


def add_test_xward(net):
    b1, b2, ln = add_grid_connection(net, zone="test_xward")

    pz = 1.200
    qz = 1.100
    ps = 0.500
    qs = 0.200
    vm_pu = 1.06
    r_ohm = 50
    x_ohm = 70
    # one xward at a bus
    pp.create_xward(net, b2, pz_mw=pz, qz_mvar=qz, ps_mw=ps, qs_mvar=qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    # add out of service xward should not change the result
    pp.create_xward(net, b2, pz_mw=pz, qz_mvar=qz, ps_mw=ps, qs_mvar=qs, vm_pu=vm_pu,
                    x_ohm=x_ohm, r_ohm=r_ohm, in_service=False,
                    index=pp.get_free_id(net.xward) + 1)
    net.last_added_case = "test_xward"
    return net


def add_test_xward_combination(net):
    b1, b2, ln = add_grid_connection(net, zone="test_xward_combination")

    pz = 1.200
    qz = 1.100
    ps = 0.500
    qs = 0.200
    vm_pu = 1.06
    r_ohm = 50
    x_ohm = 70
    # one xward at a bus
    pp.create_xward(net, b2, pz_mw=pz, qz_mvar=qz, ps_mw=ps, qs_mvar=qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    # add out of service xward should not change the result
    pp.create_xward(net, b2, pz_mw=pz, qz_mvar=qz, ps_mw=ps, qs_mvar=qs, vm_pu=vm_pu,
                    x_ohm=x_ohm, r_ohm=r_ohm, in_service=False)
    # add second xward at the bus
    pp.create_xward(net, b2, pz_mw=pz, qz_mvar=qz, ps_mw=ps, qs_mvar=qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    net.last_added_case = "test_xward_combination"
    return net


def add_test_gen(net):
    b1, b2, ln = add_grid_connection(net, zone="test_gen")
    pl = 1.200
    ql = 1.100
    ps = 0.500
    vm_set_pu = 1.0

    b3 = pp.create_bus(net, zone="test_gen", vn_kv=.4)
    pp.create_line_from_parameters(net, b2, b3, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)

    pp.create_load(net, b3, p_mw=pl, q_mvar=ql)
    pp.create_gen(net, b3, p_mw=ps, vm_pu=vm_set_pu)
    # adding out of serivce gens should not change the result
    pp.create_gen(net, b2, p_mw=ps, vm_pu=vm_set_pu, in_service=False, index=pp.get_free_id(net.gen) + 1)

    net.last_added_case = "test_gen"
    return net


def add_test_enforce_qlims(net):
    b1, b2, ln = add_grid_connection(net, zone="test_enforce_qlims")
    pl = 1.200
    ql = 1.100
    ps = 0.500
    qmin = -0.200
    vm_set_pu = 1.0

    b3 = pp.create_bus(net, zone="test_enforce_qlims", vn_kv=.4)
    pp.create_line_from_parameters(net, b2, b3, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)

    pp.create_load(net, b3, p_mw=pl, q_mvar=ql)
    pp.create_gen(net, b3, p_mw=ps, vm_pu=vm_set_pu, min_q_mvar=qmin)

    net.last_added_case = "test_enforce_qlims"
    return net


def add_test_trafo3w(net):
    b1, b2, ln = add_grid_connection(net, zone="test_trafo3w")
    b3 = pp.create_bus(net, vn_kv=0.6, zone="test_trafo3w")
    pp.create_load(net, b3, p_mw=0.2, q_mvar=0)
    b4 = pp.create_bus(net, vn_kv=0.4, zone="test_trafo3w")
    pp.create_load(net, b4, p_mw=0.1, q_mvar=0)

    pp.create_transformer3w_from_parameters(net, hv_bus=b2, mv_bus=b3, lv_bus=b4, vn_hv_kv=22,
                                            vn_mv_kv=.64, vn_lv_kv=.42, sn_hv_mva=1,
                                            sn_mv_mva=0.7, sn_lv_mva=0.3, vk_hv_percent=1.,
                                            vkr_hv_percent=.03, vk_mv_percent=.5,
                                            vkr_mv_percent=.02, vk_lv_percent=.25,
                                            vkr_lv_percent=.01, pfe_kw=0.5, i0_percent=0.1,
                                            name="test", index=pp.get_free_id(net.trafo3w) + 1,
                                            tap_side="hv", tap_pos=2, tap_step_percent=1.25,
                                            tap_min=-5, tap_neutral=0, tap_max=5)
    # adding out of service 3w trafo should not change results
    pp.create_transformer3w_from_parameters(net, hv_bus=b2, mv_bus=b3, lv_bus=b4, vn_hv_kv=20,
                                            vn_mv_kv=.6, vn_lv_kv=.4, sn_hv_mva=1, sn_mv_mva=0.7,
                                            sn_lv_mva=0.3, vk_hv_percent=2., vkr_hv_percent=.3,
                                            vk_mv_percent=1., vkr_mv_percent=.2,
                                            vk_lv_percent=.5, vkr_lv_percent=.1, pfe_kw=50,
                                            i0_percent=1., name="test", in_service=False,
                                            index=pp.get_free_id(net.trafo3w) + 1)
    net.last_added_case = "test_trafo3w"
    return net


def add_test_impedance(net):
    b1, b2, ln = add_grid_connection(net, zone="test_impedance")
    b3 = pp.create_bus(net, vn_kv=220., zone="test_impedance")
    rij = 0.02
    xij = 0.01
    rji = 0.03
    xji = 0.005
    s = 2.

    pl = 1
    ql = 0.5

    pp.create_impedance(net, b2, b3, rft_pu=rij, xft_pu=xij, rtf_pu=rji, xtf_pu=xji,
                        sn_mva=s, index=pp.get_free_id(net.impedance) + 1)
    pp.create_impedance(net, b2, b3, rft_pu=rij, xft_pu=xij, rtf_pu=rji, xtf_pu=xji,
                        sn_mva=s, index=pp.get_free_id(net.impedance) + 1, in_service=False)
    pp.create_load(net, b3, p_mw=pl, q_mvar=ql)
    net.last_added_case = "test_impedance"
    return net


def add_test_bus_bus_switch(net):
    b1, b2, ln = add_grid_connection(net, zone="test_bus_bus_switch")
    b3 = pp.create_bus(net, vn_kv=20., zone="test_bus_bus_switch")
    pp.create_switch(net, b2, b3, et="b")

    pl = 1
    ql = 0.5

    psg = 0.500
    qsg = -0.100

    pz = 1.200
    qz = 1.100
    ps = 0.500
    qs = 0.200

    vm_pu = 1.06
    r_ohm = 50
    x_ohm = 70

    pp.create_load(net, b2, p_mw=pl, q_mvar=ql)
    pp.create_load(net, b3, p_mw=pl, q_mvar=ql, scaling=0.5)

    pp.create_sgen(net, b2, p_mw=psg, q_mvar=qsg)
    pp.create_sgen(net, b3, p_mw=psg, q_mvar=qsg, scaling=0.5)

    pp.create_ward(net, b2, pz_mw=pz, qz_mvar=qz, ps_mw=ps, qs_mvar=qs)
    pp.create_ward(net, b3, pz_mw=0.5 * pz, qz_mvar=0.5 * qz, ps_mw=0.5 * ps, qs_mvar=0.5 * qs)

    pp.create_xward(net, b3, pz_mw=0.5 * pz, qz_mvar=0.5 * qz, ps_mw=0.5 * ps, qs_mvar=0.5 * qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    pp.create_xward(net, b2, pz_mw=pz, qz_mvar=qz, ps_mw=ps, qs_mvar=qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    net.last_added_case = "test_bus_bus_switch"
    return net


def add_test_oos_bus_with_is_element(net):
    b1, b2, ln = add_grid_connection(net, zone="test_oos_bus_with_is_element")

    pl = 1.200
    ql = 1.100
    ps = -0.500
    vm_set_pu = 1.0

    pz = 1.200
    qz = 1.100
    qs = 0.200

    vm_pu = 1.06
    r_ohm = 50
    x_ohm = 70

    # OOS buses
    b3 = pp.create_bus(net, zone="test_oos_bus_with_is_element", vn_kv=0.4, in_service=False)
    b4 = pp.create_bus(net, zone="test_oos_bus_with_is_element", vn_kv=0.4, in_service=False)
    b5 = pp.create_bus(net, zone="test_oos_bus_with_is_element", vn_kv=0.4, in_service=False)

    pp.create_line_from_parameters(net, b2, b3, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)
    pp.create_line_from_parameters(net, b2, b4, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)
    pp.create_line_from_parameters(net, b2, b5, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)

    # in service elements
    pp.create_load(net, b3, p_mw=pl, q_mvar=ql)
    pp.create_gen(net, b4, p_mw=ps, vm_pu=vm_set_pu)
    pp.create_sgen(net, b5, p_mw=ps, q_mvar=ql)
    pp.create_ward(net, b3, pz_mw=pz, qz_mvar=qz, ps_mw=ps, qs_mvar=qs)
    pp.create_xward(net, b4, pz_mw=0.5 * pz, qz_mvar=0.5 * qz, ps_mw=0.5 * ps, qs_mvar=0.5 * qs,
                    vm_pu=vm_pu, x_ohm=x_ohm, r_ohm=r_ohm)
    pp.create_shunt(net, b5, q_mvar=-800, p_mw=0)

    net.last_added_case = "test_oos_bus_with_is_element"
    return net


def add_test_shunt(net):
    b1, b2, ln = add_grid_connection(net, zone="test_shunt")
    pz = 0.12
    qz = -1.2
    # one shunt at a bus
    pp.create_shunt_as_capacitor(net, b2, q_mvar=1.2, loss_factor=0.1, vn_kv=22., step=2)
    # add out of service shunt shuold not change the result
    pp.create_shunt(net, b2, p_mw=pz, q_mvar=qz, in_service=False)
    net.last_added_case = "test_shunt"
    return net


def add_test_shunt_split(net):
    b1, b2, ln = add_grid_connection(net, zone="test_shunt_split")
    pz = 0.120
    qz = -1.200
    # one shunt at a bus
    pp.create_shunt(net, b2, p_mw=pz / 2, q_mvar=qz / 2)
    pp.create_shunt(net, b2, p_mw=pz / 2, q_mvar=qz / 2)
    net.last_added_case = "test_shunt_split"
    return net


def add_test_two_open_switches_on_deactive_line(net):
    b1, b2, l1 = add_grid_connection(net, zone="two_open_switches_on_deactive_line")
    b3 = pp.create_bus(net, vn_kv=20.)
    l2 = create_test_line(net, b2, b3, in_service=False)
    create_test_line(net, b3, b1)
    pp.create_switch(net, b2, l2, et="l", closed=False)
    pp.create_switch(net, b3, l2, et="l", closed=False)
    net.last_added_case = "test_two_open_switches_on_deactive_line"
    return net


if __name__ == '__main__':
    from pandapower.test.consistency_checks import runpp_with_consistency_checks
    from pandapower import LoadflowNotConverged
    for net in result_test_network_generator():
        try:
            runpp_with_consistency_checks(net, enforce_q_lims=True, numba=True)
        except (AssertionError):
            raise UserWarning("Consistency Error after adding %s" % net.last_added_case)
        except(LoadflowNotConverged):
            raise UserWarning("Power flow did not converge after adding %s" % net.last_added_case)