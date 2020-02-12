# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.networks as nw


def test_create_and_load_std_type_line():
    net = pp.create_empty_network()
    c = 40
    r = 0.01
    x = 0.02
    i = 0.2
    name = "test_line"

    typdata = {}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="line")

    typdata = {"c_nf_per_km": c}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="line")

    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="line")

    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="line")

    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x, "max_i_ka": i}
    pp.create_std_type(net, name=name, data=typdata, element="line")
    assert net.std_types["line"][name] == typdata

    loaded_type = pp.load_std_type(net, name)
    assert loaded_type == typdata

def test_create_std_types_line():
    net = pp.create_empty_network()
    c = 40
    r = 0.01
    x = 0.02
    i = 0.2

    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x, "max_i_ka": i}

    typdatas = {"typ1": typdata, "typ2": typdata}
    pp.create_std_types(net, data=typdatas, element="line")
    assert net.std_types["line"]["typ1"] == typdata
    assert net.std_types["line"]["typ1"] == typdata

def test_create_std_types_from_net_line():
    net1 = pp.create_empty_network()
    net2 = pp.create_empty_network()

    c = 40
    r = 0.01
    x = 0.02
    i = 0.2

    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x, "max_i_ka": i,
               "additional": 8}
    pp.create_std_type(net1, typdata, "test_copy")
    pp.copy_std_types(net2, net1, element="line")
    assert pp.std_type_exists(net2, "test_copy")

def test_create_and_load_std_type_trafo():
    net = pp.create_empty_network()
    sn_mva = 40
    vn_hv_kv = 110
    vn_lv_kv =  20
    vk_percent = 5.
    vkr_percent = 2.
    pfe_kw=50
    i0_percent = 0.1
    shift_degree = 30
    name = "test_trafo"

    typdata = {"sn_mva": sn_mva}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
               "vkr_percent": vkr_percent}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
               "vkr_percent": vkr_percent, "pfe_kw": pfe_kw}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo")

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
               "vkr_percent": vkr_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo")
    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
               "vkr_percent": vkr_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
               "shift_degree": shift_degree}
    pp.create_std_type(net, name=name, data=typdata, element="trafo")
    assert net.std_types["trafo"][name] == typdata

    loaded_type = pp.load_std_type(net, name, element="trafo")
    assert loaded_type == typdata

def test_create_and_load_std_type_trafo3w():
    net = pp.create_empty_network()
    sn_hv_mva = 40; sn_mv_mva = 20; sn_lv_mva = 20
    vn_hv_kv = 110; vn_mv_kv = 50; vn_lv_kv = 20
    vk_hv_percent = 5.; vk_mv_percent = 5.; vk_lv_percent = 5.
    vkr_hv_percent = 2.; vkr_mv_percent = 2.; vkr_lv_percent = 2.
    pfe_kw=50
    i0_percent = 0.1
    shift_mv_degree = 30; shift_lv_degree = 30
    name = "test_trafo3w"

    typdata = {"sn_hv_mva": sn_hv_mva}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_mv_mva": sn_mv_mva, "vn_hv_kv": vn_hv_kv}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_lv_mva": sn_lv_mva, "vn_mv_kv": vn_mv_kv, "vn_lv_kv": vn_lv_kv}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_mv_mva": sn_mv_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_hv_percent": vk_hv_percent}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_hv_mva": sn_hv_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_mv_percent": vk_mv_percent,
               "vkr_hv_percent": vkr_hv_percent}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_hv_mva": sn_hv_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_lv_percent": vk_lv_percent,
               "vkr_mv_percent": vkr_mv_percent, "pfe_kw": pfe_kw}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo3w")

    typdata = {"sn_hv_mva": sn_hv_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_hv_percent": vk_hv_percent,
               "vkr_lv_percent": vkr_lv_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo3w")
    typdata = {"sn_hv_mva": sn_hv_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_hv_percent": vk_hv_percent,
               "vkr_hv_percent": vkr_hv_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
               "shift_mv_degree": shift_mv_degree}
    with pytest.raises(UserWarning):
        pp.create_std_type(net, name=name, data=typdata, element="trafo3w")
    typdata = {"vn_hv_kv": vn_hv_kv, "vn_mv_kv": vn_mv_kv, "vn_lv_kv": vn_lv_kv, "sn_hv_mva": sn_hv_mva,
          "sn_mv_mva": sn_mv_mva, "sn_lv_mva": sn_lv_mva, "vk_hv_percent": vk_hv_percent, "vk_mv_percent": vk_mv_percent,
          "vk_lv_percent": vk_lv_percent, "vkr_hv_percent": vkr_hv_percent, "vkr_mv_percent": vkr_mv_percent,
          "vkr_lv_percent": vkr_lv_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
          "shift_mv_degree":shift_mv_degree, "shift_lv_degree": shift_lv_degree}
    pp.create_std_type(net, name=name, data=typdata, element="trafo3w")
    assert net.std_types["trafo3w"][name] == typdata

    loaded_type = pp.load_std_type(net, name, element="trafo3w")
    assert loaded_type == typdata

def test_create_std_types_trafo():
    net = pp.create_empty_network()
    sn_mva = 40
    vn_hv_kv = 110
    vn_lv_kv =  20
    vk_percent = 5.
    vkr_percent = 2.
    pfe_kw=50
    i0_percent = 0.1
    shift_degree = 30

    typdata = {"sn_mva": sn_mva, "vn_hv_kv": vn_hv_kv, "vn_lv_kv": vn_lv_kv, "vk_percent": vk_percent,
               "vkr_percent": vkr_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
               "shift_degree": shift_degree}
    typdatas = {"typ1": typdata, "typ2": typdata}
    pp.create_std_types(net, data=typdatas, element="trafo")
    assert net.std_types["trafo"]["typ1"] == typdata
    assert net.std_types["trafo"]["typ2"] == typdata

def test_create_std_types_trafo3w():
    net = pp.create_empty_network()
    sn_hv_mva = 40; sn_mv_mva = 20; sn_lv_mva = 20
    vn_hv_kv = 110; vn_mv_kv = 50; vn_lv_kv = 20
    vk_hv_percent = 5.; vk_mv_percent = 5.; vk_lv_percent = 5.
    vkr_hv_percent = 2.; vkr_mv_percent = 2.; vkr_lv_percent = 2.
    pfe_kw=50
    i0_percent = 0.1
    shift_mv_degree = 30; shift_lv_degree = 30

    typdata = {"vn_hv_kv": vn_hv_kv, "vn_mv_kv": vn_mv_kv, "vn_lv_kv": vn_lv_kv, "sn_hv_mva": sn_hv_mva,
          "sn_mv_mva": sn_mv_mva, "sn_lv_mva": sn_lv_mva, "vk_hv_percent": vk_hv_percent, "vk_mv_percent": vk_mv_percent,
          "vk_lv_percent": vk_lv_percent, "vkr_hv_percent": vkr_hv_percent, "vkr_mv_percent": vkr_mv_percent,
          "vkr_lv_percent": vkr_lv_percent, "pfe_kw": pfe_kw, "i0_percent": i0_percent,
          "shift_mv_degree":shift_mv_degree, "shift_lv_degree": shift_lv_degree}

    typdatas = {"typ1": typdata, "typ2": typdata}
    pp.create_std_types(net, data=typdatas, element="trafo3w")
    assert net.std_types["trafo3w"]["typ1"] == typdata
    assert net.std_types["trafo3w"]["typ2"] == typdata

def test_find_line_type():
    net = pp.create_empty_network()
    c = 40000
    r = 1.5
    x = 2.0
    i = 10
    name = "test_line1"
    typdata = {"c_nf_per_km": c, "r_ohm_per_km": r, "x_ohm_per_km": x, "max_i_ka": i}
    pp.create_std_type(net, data=typdata, name=name, element="line")
    fitting_type = pp.find_std_type_by_parameter(net, typdata)
    assert len(fitting_type) == 1
    assert fitting_type[0] == name

    fitting_type = pp.find_std_type_by_parameter(net, {"r_ohm_per_km":r+0.05}, epsilon=.06)
    assert len(fitting_type) == 1
    assert fitting_type[0] == name

    fitting_type = pp.find_std_type_by_parameter(net, {"r_ohm_per_km":r+0.07}, epsilon=.06)
    assert len(fitting_type) == 0

def test_change_type_line():
    net = pp.create_empty_network()
    r1 = 0.01
    x1 = 0.02
    c1 = 40
    i1 = 0.2
    name1 = "test_line1"
    typ1 = {"c_nf_per_km": c1, "r_ohm_per_km": r1, "x_ohm_per_km": x1, "max_i_ka": i1}
    pp.create_std_type(net, data=typ1, name=name1, element="line")

    r2 = 0.02
    x2 = 0.04
    c2 = 20
    i2 = 0.4
    name2 = "test_line2"
    typ2 = {"c_nf_per_km": c2, "r_ohm_per_km": r2, "x_ohm_per_km": x2, "max_i_ka": i2}
    pp.create_std_type(net, data=typ2, name=name2, element="line")

    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)
    lid = pp.create_line(net, b1, b2, 1., std_type=name1)
    assert net.line.r_ohm_per_km.at[lid] == r1
    assert net.line.x_ohm_per_km.at[lid] == x1
    assert net.line.c_nf_per_km.at[lid] == c1
    assert net.line.max_i_ka.at[lid] == i1
    assert net.line.std_type.at[lid] == name1

    pp.change_std_type(net, lid, name2)

    assert net.line.r_ohm_per_km.at[lid] == r2
    assert net.line.x_ohm_per_km.at[lid] == x2
    assert net.line.c_nf_per_km.at[lid] == c2
    assert net.line.max_i_ka.at[lid] == i2
    assert net.line.std_type.at[lid] == name2


def test_parameter_from_std_type_line():
    net = pp.create_empty_network()
    r1 = 0.01
    x1 = 0.02
    c1 = 40
    i1 = 0.2
    name1 = "test_line1"
    typ1 = {"c_nf_per_km": c1, "r_ohm_per_km": r1, "x_ohm_per_km": x1, "max_i_ka": i1}
    pp.create_std_type(net, data=typ1, name=name1, element="line")

    r2 = 0.02
    x2 = 0.04
    c2 = 20
    i2 = 0.4
    endtemp2 = 40

    endtemp_fill = 20

    name2 = "test_line2"
    typ2 = {"c_nf_per_km": c2, "r_ohm_per_km": r2, "x_ohm_per_km": x2, "max_i_ka": i2,
            "endtemp_degree": endtemp2}
    pp.create_std_type(net, data=typ2, name=name2, element="line")

    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)
    lid1 = pp.create_line(net, b1, b2, 1., std_type=name1)
    lid2 = pp.create_line(net, b1, b2, 1., std_type=name2)
    lid3 = pp.create_line_from_parameters(net, b1, b2, 1., r_ohm_per_km=0.03, x_ohm_per_km=0.04,
                                          c_nf_per_km=20, max_i_ka=0.3)

    pp.parameter_from_std_type(net, "endtemp_degree", fill=endtemp_fill)
    assert net.line.endtemp_degree.at[lid1] == endtemp_fill #type1 one has not specified an endtemp
    assert net.line.endtemp_degree.at[lid2] == endtemp2 #type2 has specified endtemp
    assert net.line.endtemp_degree.at[lid3] == endtemp_fill #line3 has no standard type

    net.line.endtemp_degree.at[lid3] = 10
    pp.parameter_from_std_type(net, "endtemp_degree", fill=endtemp_fill)
    assert net.line.endtemp_degree.at[lid3] == 10 #check that existing values arent overwritten


def test_add_temperature_coefficient():
    net = nw.simple_mv_open_ring_net()
    pp.add_temperature_coefficient(net)
    assert "alpha" in net.line.columns
    assert all(net.line.alpha == 4.03e-3)


if __name__ == "__main__":
    pytest.main([__file__])
