# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import random as rd

import pandapower as pp


# --- support functions

def _create_empty_network_with_transformer(trafotype, V_OS=10., V_US=0.4):
    """
    Creates a Network with transformer and infeeder. The reference bus on the \
    high-voltage side is called "Trafostation_OS". The bus on the low-voltage \
    side is called "main_busbar".
    The voltage levels can be set manually and the transformer parameter can \
    be set with "ti"
    """
    pd_net = pp.create_empty_network()
    NFA2X4x70 = {"c_nf_per_km": 0, "r_ohm_per_km": 0.443, "x_ohm_per_km": 0.069, "max_i_ka": 0.270,
                 "type": "ol", "q_mm2": 70}
    NAYY4x50 = {"c_nf_per_km": 670, "r_ohm_per_km": 0.6417, "x_ohm_per_km": 0.084823,
                "max_i_ka": 0.141, "type": "cs", "q_mm2": 50}
    NAYY4x150 = {"c_nf_per_km": 830, "r_ohm_per_km": 0.2067, "x_ohm_per_km": 0.08042478,
                 "max_i_ka": 0.275, "type": "cs", "q_mm2": 150}
    NAYY4x185 = {"c_nf_per_km": 830, "r_ohm_per_km": 0.165, "x_ohm_per_km": 0.08042478,
                 "max_i_ka": 0.313, "type": "cs", "q_mm2": 185}
    NYY4x35 = {"c_nf_per_km": 0, "r_ohm_per_km": 0.5240284, "x_ohm_per_km": 0.08513716,
               "max_i_ka": 0.156, "type": "cs", "q_mm2": 35}
    pp.create_std_type(net=pd_net, data=NFA2X4x70, name="NFA2X 4x70", element="line")
    pp.create_std_type(net=pd_net, data=NAYY4x50, name="NAYY 4x50", element="line")
    pp.create_std_type(net=pd_net, data=NAYY4x150, name="NAYY 4x150", element="line")
    pp.create_std_type(net=pd_net, data=NAYY4x185, name="NAYY 4x185", element="line")
    pp.create_std_type(net=pd_net, data=NYY4x35, name="NYY 4x35", element="line")
    T100kVA = {"sn_mva": 0.100, "vn_hv_kv": 10, "vn_lv_kv": 0.4, "vk_percent": 4,
               "vkr_percent": 1.2, "pfe_kw": 0.45, "i0_percent": 0.25, "shift_degree": 150,
               "vector_group": "Dyn5"}
    T160kVA = {"sn_mva": 0.160, "vn_hv_kv": 10, "vn_lv_kv": 0.4, "vk_percent": 4,
               "vkr_percent": 1.2, "pfe_kw": 0.38, "i0_percent": 0.26, "shift_degree": 150,
               "vector_group": "Dyn5"}
    pp.create_std_type(net=pd_net, data=T100kVA, name="0.1 MVA 10/0.4 kV", element="trafo")
    pp.create_std_type(net=pd_net, data=T160kVA, name="0.16 MVA 10/0.4 kV", element="trafo")

    busnr1 = pp.create_bus(pd_net, name="Trafostation_OS", vn_kv=V_OS)
    pp.create_ext_grid(pd_net, bus=busnr1)
    main_busbar_nr = pp.create_bus(pd_net, name="main_busbar", vn_kv=V_US, type="b")
    pp.create_transformer(pd_net, hv_bus=busnr1, lv_bus=main_busbar_nr, std_type=trafotype,
                          name="trafo 1")
    return pd_net, main_busbar_nr


def _add_lines_and_loads(pd_net, n_lines, startbusnr, length_per_line,
                         std_type="NAYY 4x150 SE", p_load_mw=0,
                         q_load_mvar=0, branchnr=1,
                         l_para_per_km=None):
    """
    Creates a single unsplitted branch on the startbus of n lines. It \
    sequencely adds lines, buses and loads.

    Loads will only be added if p_load_mw or q_load_mvar \
    is assigned

    The branch number could be assigned with branchnr. It will be added to \
    the name ti keep track on the node position
    """

    startpoint_bus = 1
    startpoint_line = 1
    bus_before = startbusnr
    for i in list(range(n_lines)):
        buscounter = startpoint_bus + i
        linecounter = startpoint_line + i
        created_bus_nr = pp.create_bus(pd_net, name="bus_%d_%d" % (branchnr, buscounter), vn_kv=.4)

        pp.create_line(pd_net, bus_before, created_bus_nr, length_km=length_per_line,
                       name="line_%d_%d" % (branchnr, linecounter), std_type=std_type)

        if p_load_mw or q_load_mvar:
            pp.create_load(pd_net, created_bus_nr, p_mw=p_load_mw, q_mvar=q_load_mvar)

        bus_before = created_bus_nr  # rueckgefuehrter Wert in der Schleife

    return pd_net


def _add_lines_with_branched_loads(net, n_lines, startbus, length_per_line,
                                   std_type="NAYY 4x150 SE", p_load_mw=0,
                                   q_load_mvar=0,
                                   length_branchout_line_1=0.022,
                                   length_branchout_line_2=0,
                                   std_type_branchout_line_1="NAYY 4x50 SE",
                                   std_type_branchout_line_2="NAYY 4x50 SE",
                                   prob_branchout_line_1=0.5, branchnr=1):
    """
    Creates a single unsplitted branch on the startbus. each bus on the main \
    line is connected to a branch out line which connects \
    the loadbus (households).

    If there are two std_types given for the branch_out_lin. The cable_types \
    interchange with the given probability

    If there are two lengths of branchoutlines are given, the \
    lengths interchange.
    It begins with length 1 and switches to length 2. The cable with length 1 \
    is named as "MUF_" and length 2 becomes "KV_".

    Loads will only be added if p_load_mw or q_load_mvar \
    is assigned

    The branch number could be assigned with branchnr. It will be added to the\
     name ti keep track on the node position
    """

    # support function
    startpoint_bus = 1
    startpoint_line = 1
    bus_before = startbus
    length_branchout_line = length_branchout_line_1
    # destinct between Muffe und Kabelverteiler
    if length_branchout_line_2:
        bustype = "MUF"
    else:
        bustype = "bus"
    std_type_branchout_line = std_type_branchout_line_1
    for i in range(n_lines):
        buscounter = startpoint_bus + i
        linecounter = startpoint_line + i
        created_bus_nr = pp.create_bus(net, name="%s_%d_%d" % (bustype, branchnr, buscounter),
                                       type="b" if bustype == "KV" else "n", vn_kv=.4)
        pp.create_line(net, bus_before, created_bus_nr,
                       length_km=length_per_line,
                       name="line_%d_%d" % (branchnr, linecounter),
                       std_type=std_type)

        loadbusnr = pp.create_bus(net, name="loadbus_%d_%d" % (branchnr, buscounter), vn_kv=.4)

        pp.create_line(net, created_bus_nr, loadbusnr,
                       length_km=length_branchout_line,
                       name="branchout_line_%d_%d" % (branchnr, linecounter),
                       std_type=std_type_branchout_line)

        if p_load_mw or q_load_mvar:
            pp.create_load(net, loadbusnr,
                           p_mw=p_load_mw, q_mvar=q_load_mvar)

        bus_before = created_bus_nr  # rueckgefuehrter Wert in der Schleife

        # alternates the length of the branch out lines if needed
        if length_branchout_line_2:
            if length_branchout_line == length_branchout_line_1:
                length_branchout_line = length_branchout_line_2
                bustype = "KV"
            else:
                length_branchout_line = length_branchout_line_1
                bustype = "MUF"
        #  changes branch out lines according to the probabillity if needed
        if std_type_branchout_line_2:
            if rd.random() > prob_branchout_line_1:
                std_type_branchout_line = std_type_branchout_line_2
            else:
                std_type_branchout_line = std_type_branchout_line_1
    return net

def _create_branch_network(trafotype, v_os, num_lines, len_lines, **kwargs):
    pd_net, main_busbar_nr = _create_empty_network_with_transformer(trafotype, V_OS=v_os)
    for i, (n_line, len_line) in enumerate(zip(num_lines, len_lines), start=1):
        _add_lines_and_loads(pd_net, n_line, startbusnr=main_busbar_nr,
                             length_per_line=len_line, branchnr=i, **kwargs)
    return pd_net


def _create_branched_loads_network(trafotype, v_os, num_lines, len_lines, **kwargs):
    pd_net, main_busbar_nr = _create_empty_network_with_transformer(trafotype, V_OS=v_os)
    for i, (n_line, len_line) in enumerate(zip(num_lines, len_lines), start=1):
        _add_lines_with_branched_loads(pd_net, n_line, startbus=main_busbar_nr,
                                       length_per_line=len_line, branchnr=i, **kwargs)
    return pd_net

# --- main functions

def create_kerber_landnetz_freileitung_1(n_lines=13,
                                         l_lines_in_km=0.021, std_type="NFA2X 4x70",
                                         trafotype="0.16 MVA 10/0.4 kV",
                                         p_load_mw=0.008, q_load_mvar=0, v_os=10.):

    pd_net, main_busbar_nr = _create_empty_network_with_transformer(trafotype, V_OS=v_os)
    _add_lines_and_loads(pd_net, n_lines, startbusnr=main_busbar_nr,
                         length_per_line=l_lines_in_km, std_type=std_type,
                         p_load_mw=p_load_mw,
                         q_load_mvar=q_load_mvar)
    return pd_net


def create_kerber_landnetz_freileitung_2(n_branch_1=6, n_branch_2=2,
                                         l_lines_1_in_km=0.038,
                                         l_lines_2_in_km=0.081,
                                         std_type="NFA2X 4x70",
                                         trafotype="0.1 MVA 10/0.4 kV",
                                         p_load_mw=0.008, q_load_mvar=0,
                                         v_os=10.):
    num_lines = [n_branch_1, n_branch_2]
    len_lines = [l_lines_1_in_km, l_lines_2_in_km]
    pd_net = _create_branch_network(trafotype=trafotype, v_os=v_os, num_lines=num_lines,
                                    len_lines=len_lines, std_type=std_type,
                                    p_load_mw=p_load_mw, q_load_mvar=q_load_mvar)
    return pd_net


def create_kerber_landnetz_kabel_1(n_branch_1=6, n_branch_2=2, l_lines_1_in_km=0.082,
                                   l_lines_2_in_km=0.175, std_type="NAYY 4x150",
                                   std_type_branchout_line="NAYY 4x50",
                                   trafotype="0.1 MVA 10/0.4 kV", p_load_mw=0.008,
                                   q_load_mvar=0., length_branchout_line_1=0.018,
                                   length_branchout_line_2=0.033, v_os=10.):
    """
    .. note:: It is assumed that every second bus in a branch is a "KV".
    """
    num_lines = [n_branch_1, n_branch_2]
    len_lines = [l_lines_1_in_km, l_lines_2_in_km]
    pd_net = _create_branched_loads_network(trafotype=trafotype, v_os=v_os, num_lines=num_lines,
                                            len_lines=len_lines, std_type=std_type,
                                            p_load_mw=p_load_mw,
                                            q_load_mvar=q_load_mvar,
                                            length_branchout_line_1=length_branchout_line_1,
                                            length_branchout_line_2=length_branchout_line_2,
                                            std_type_branchout_line_1=std_type_branchout_line)
    return pd_net


def create_kerber_landnetz_kabel_2(n_branch_1=12, n_branch_2=2, l_lines_1_in_km=0.053,
                                   l_lines_2_in_km=0.175, std_type="NAYY 4x150",
                                   trafotype="0.16 MVA 10/0.4 kV", p_load_mw=0.008,
                                   q_load_mvar=0., length_branchout_line_1=0.018,
                                   length_branchout_line_2=0.033,
                                   std_type_branchout_line="NAYY 4x50", v_os=10.):
    """
    .. note:: It is assumed that every second bus in a branch is a "KV".
    """
    num_lines = [n_branch_1, n_branch_2]
    len_lines = [l_lines_1_in_km, l_lines_2_in_km]
    pd_net = _create_branched_loads_network(trafotype=trafotype, v_os=v_os, num_lines=num_lines,
                                            len_lines=len_lines, std_type=std_type,
                                            p_load_mw=p_load_mw,
                                            q_load_mvar=q_load_mvar,
                                            length_branchout_line_1=length_branchout_line_1,
                                            length_branchout_line_2=length_branchout_line_2,
                                            std_type_branchout_line_1=std_type_branchout_line)
    return pd_net


def create_kerber_dorfnetz(std_type="NAYY 4x150", trafotype="0.4 MVA 10/0.4 kV",
                           p_load_mw=0.006,
                           q_load_mvar=0., length_branchout_line_1=0.015,
                           length_branchout_line_2=0.031,
                           std_type_branchout_line="NAYY 4x50", v_os=10.):
    """
    .. note:: It is assumed that every second bus in a branch is a "KV".
    """
    num_lines = [9, 9, 16, 12, 7, 4]
    len_lines = [0.04, 0.04, 0.029, 0.032, 0.043, 0.064]
    pd_net = _create_branched_loads_network(trafotype=trafotype, v_os=v_os, num_lines=num_lines,
                                            len_lines=len_lines, std_type=std_type,
                                            p_load_mw=p_load_mw,
                                            q_load_mvar=q_load_mvar,
                                            length_branchout_line_1=length_branchout_line_1,
                                            length_branchout_line_2=length_branchout_line_2,
                                            std_type_branchout_line_1=std_type_branchout_line)
    return pd_net


def create_kerber_vorstadtnetz_kabel_1(std_type="NAYY 4x150", p_load_mw=0.002, q_load_mvar=0.,
                                       trafotype="0.63 MVA 10/0.4 kV", v_os=10.):
    """
    .. note:: Please pay attention, that the linetypes of the branch out house connections are \
    randomly distributed according to the probability 50% between "NAYY 4x50" and "NYY 4x35"
    """
    num_lines = [14, 14, 14, 19, 19, 10, 10, 10, 32, 4]
    len_lines = [0.021, 0.021, 0.021, 0.017, 0.017, 0.025, 0.025, 0.025, 0.011, 0.060]
    pd_net = _create_branched_loads_network(trafotype=trafotype, v_os=v_os, num_lines=num_lines,
                                            len_lines=len_lines, std_type=std_type,
                                            p_load_mw=p_load_mw,
                                            q_load_mvar=q_load_mvar,
                                            length_branchout_line_1=0.011,
                                            std_type_branchout_line_1="NAYY 4x50",
                                            std_type_branchout_line_2="NYY 4x35",
                                            prob_branchout_line_1=0.5)
    return pd_net


def create_kerber_vorstadtnetz_kabel_2(std_type="NAYY 4x185", p_load_mw=.002, q_load_mvar=0.,
                                       trafotype="0.63 MVA 10/0.4 kV", v_os=10.):
    """
    .. note:: Please pay attention, that the linetypes of the branch out house \
    connections are randomlydistributed according to the probability 50% between \
    "NAYY 50" and "NYY 35"
    """

    num_lines = [15, 15, 15, 20, 20, 11, 11, 32, 5]
    len_lines = [0.023, 0.023, 0.023, 0.020, 0.020, 0.026, 0.026, 0.014, 0.050]
    pd_net = _create_branched_loads_network(trafotype=trafotype, v_os=v_os, num_lines=num_lines,
                                            len_lines=len_lines, std_type=std_type,
                                            p_load_mw=p_load_mw,
                                            q_load_mvar=q_load_mvar,
                                            length_branchout_line_1=0.011,
                                            std_type_branchout_line_1="NAYY 4x50",
                                            std_type_branchout_line_2="NYY 4x35",
                                            prob_branchout_line_1=0.5)
    return pd_net


