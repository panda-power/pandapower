# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from pandapower.networks.kerber_networks import _create_branched_loads_network, _create_branch_network

# The Kerber extreme networks complement the typical Kerber networks.
# Categories:
# I. Kerber networks with extreme lines
# II. Kerber networks with extreme lines and high loaded transformer


# --- Extreme Landnetze

# I.  Kerber Landnetz mit extremen Netzstrahlen (Freileitung):
def kb_extrem_landnetz_freileitung(n_lines=26, l_lines_in_km=0.012, std_type="NFA2X 4x70",
                                   # trafotype="0.4 MVA 10/0.4 kV Yyn6 4 ASEA"
                                   trafotype="0.25 MVA 10/0.4 kV", p_load_mw=0.008,
                                   q_load_mvar=0, v_os=10.):
    pd_net = _create_branch_network(num_lines=[n_lines], len_lines=[l_lines_in_km],
                                    trafotype=trafotype, v_os=v_os,
                                    std_type=std_type, p_load_mw=p_load_mw, q_load_mvar=q_load_mvar)
    return pd_net


# I.  Kerber Landnetz mit extremen Netzstrahlen (Kabel):
def kb_extrem_landnetz_kabel(n_branch_1=26, l_lines_1_in_km=0.026, std_type="NAYY 4x150",
                             # trafotype="0.4 MVA 10/0.4 kV Yyn6 4 ASEA"
                             trafotype="0.25 MVA 10/0.4 kV", p_load_mw=.008,
                             q_load_mvar=0., length_branchout_line_1=0.018,
                             length_branchout_line_2=0.033, std_type_branchout_line="NAYY 4x50",
                             v_os=10.):
    num_lines = [n_branch_1]
    len_lines = [l_lines_1_in_km]
    pd_net = _create_branched_loads_network(trafotype=trafotype, v_os=v_os, num_lines=num_lines,
                                    len_lines=len_lines, std_type=std_type,
                                    p_load_mw=p_load_mw,
                                    q_load_mvar=q_load_mvar,
                                    length_branchout_line_1=length_branchout_line_1,
                                    length_branchout_line_2=length_branchout_line_2,
                                    std_type_branchout_line_1=std_type_branchout_line)
    return pd_net


# II.  Kerber Landnetz mit extremen Netzstrahlen (Freileitung) und hoch ausgelastetem Transformator:
def kb_extrem_landnetz_freileitung_trafo(n_branch_1=26, n_branch_2=1, l_lines_1_in_km=0.012,
                                         l_lines_2_in_km=0.036, std_type="NFA2X 4x70",
                                         trafotype="0.1 MVA 10/0.4 kV",
                                         p_load_mw=.008, q_load_mvar=0,
                                         v_os=10.):
    # num_lines = [26, 1] <- Solution with _add_lines_with_branched_loads, if branches = 0
    # len_lines = [0.012, 0.036]
    num_lines = [n_branch_1, n_branch_2]
    len_lines = [l_lines_1_in_km, l_lines_2_in_km]
    pd_net = _create_branch_network(num_lines=num_lines, len_lines=len_lines, trafotype=trafotype, v_os=v_os,
                             std_type=std_type, p_load_mw=p_load_mw,
                             q_load_mvar=q_load_mvar)

    return pd_net


# II. Kerber Landnetz mit extremen Netzstrahlen (Kabel) und hoch ausgelastetem Transformator:
def kb_extrem_landnetz_kabel_trafo(n_branch_1=26, n_branch_2=1, l_lines_1_in_km=0.026,
                                   l_lines_2_in_km=0.078, std_type="NAYY 4x150",
                                   trafotype="0.1 MVA 10/0.4 kV", p_load_mw=.008,
                                   q_load_mvar=0., length_branchout_line_1=0.018,
                                   length_branchout_line_2=0.033,
                                   std_type_branchout_line="NAYY 4x50",
                                   v_os=10.):
    num_lines = [n_branch_1, n_branch_2]
    len_lines = [l_lines_1_in_km, l_lines_2_in_km]
    pd_net = _create_branched_loads_network(len_lines=len_lines, num_lines=num_lines,
                                   std_type=std_type, p_load_mw=p_load_mw, v_os=v_os,
                                   q_load_mvar=q_load_mvar, trafotype=trafotype,
                                   length_branchout_line_1=length_branchout_line_1,
                                   length_branchout_line_2=length_branchout_line_2,
                                   std_type_branchout_line_1=std_type_branchout_line)
    return pd_net


# --- Extreme Dorfnetze

# I. Kerber Dorfnetz mit extremen Netzstrahlen:
def kb_extrem_dorfnetz(std_type="NAYY 4x150", trafotype="0.4 MVA 10/0.4 kV",
                       p_load_mw=.006, q_load_mvar=0., length_branchout_line_1=0.015,
                       length_branchout_line_2=0.031, std_type_branchout_line="NAYY 4x50",
                       v_os=10.):
    num_lines = [28, 16, 9, 4, 1]
    len_lines = [0.021, 0.029, 0.040, 0.064, 0.102]
    pd_net = _create_branched_loads_network(len_lines=len_lines, num_lines=num_lines,
                                   std_type=std_type, p_load_mw=p_load_mw, v_os=v_os,
                                   q_load_mvar=q_load_mvar, trafotype=trafotype,
                                   length_branchout_line_1=length_branchout_line_1,
                                   length_branchout_line_2=length_branchout_line_2,
                                   std_type_branchout_line_1=std_type_branchout_line)
    return pd_net


# II. Kerber Dorfnetz mit extremen Netzstrahlen und hoch ausgelastetem Transformator:
def kb_extrem_dorfnetz_trafo(std_type="NAYY 4x150", trafotype="0.25 MVA 10/0.4 kV",
                             # trafotype="0.4 MVA 10/0.4 kV Yyn6 4 ASEA",
                             p_load_mw=0.006, q_load_mvar=0., length_branchout_line_1=0.015,
                             length_branchout_line_2=0.031, std_type_branchout_line="NAYY 4x50",
                             v_os=10.):
    num_lines = [28, 28, 16, 12, 12, 9, 7, 4, 1]
    len_lines = [0.021, 0.021, 0.029, 0.032, 0.032, 0.040, 0.043, 0.064, 0.102]
    pd_net = _create_branched_loads_network(len_lines=len_lines, num_lines=num_lines,
                                   std_type=std_type, p_load_mw=p_load_mw, v_os=v_os,
                                   q_load_mvar=q_load_mvar, trafotype=trafotype,
                                   length_branchout_line_1=length_branchout_line_1,
                                   length_branchout_line_2=length_branchout_line_2,
                                   std_type_branchout_line_1=std_type_branchout_line)
    return pd_net


# --- Extreme Vorstadtnetze

# I. Kerber Vorstadtnetz mit extremen Netzstrahlen (Typ 1):
def kb_extrem_vorstadtnetz_1(std_type="NAYY 4x150", p_load_mw=0.002, q_load_mvar=0.,
                             trafotype="0.63 MVA 10/0.4 kV", v_os=10.):

    num_lines = [69, 32, 19, 14, 10, 1]
    len_lines = [0.006, 0.011, 0.017, 0.021, 0.025, 0.068]
    pd_net = _create_branched_loads_network(len_lines=len_lines, num_lines=num_lines,
                                   std_type=std_type, p_load_mw=p_load_mw, v_os=v_os,
                                   q_load_mvar=q_load_mvar, trafotype=trafotype,
                                   length_branchout_line_1=0.011,
                                   std_type_branchout_line_1="NAYY 4x50",
                                   std_type_branchout_line_2="NYY 4x35",
                                   prob_branchout_line_1=0.5)
    return pd_net


# I. Kerber Vorstadtnetz mit extremen Netzstrahlen (Typ 2):
def kb_extrem_vorstadtnetz_2(std_type="NAYY 4x185", p_load_mw=0.002, q_load_mvar=0.,
                             trafotype="0.63 MVA 10/0.4 kV", v_os=10.):
    num_lines = [61, 32, 20, 15, 11, 5, 1]
    len_lines = [0.01, 0.014, 0.02, 0.023, 0.026, 0.05, 0.085]
    pd_net = _create_branched_loads_network(len_lines=len_lines, num_lines=num_lines,
                                   std_type=std_type, p_load_mw=p_load_mw, v_os=v_os,
                                   q_load_mvar=q_load_mvar, trafotype=trafotype,
                                   length_branchout_line_1=0.011,
                                   std_type_branchout_line_1="NAYY 4x50",
                                   std_type_branchout_line_2="NYY 4x35",
                                   prob_branchout_line_1=0.5)
    return pd_net


# II. Kerber Vorstadtnetz mit extremen Netzstrahlen und hoch ausgeslatetem Transformator:
def kb_extrem_vorstadtnetz_trafo_1(std_type="NAYY 4x150", p_load_mw=0.002, q_load_mvar=0.,
                                   trafotype="0.25 MVA 10/0.4 kV", v_os=10.):
    num_lines = [69, 32, 32, 19, 14, 10, 10, 4, 1]
    len_lines = [0.006, 0.011, 0.011, 0.017, 0.021, 0.025, 0.025, 0.06, 0.68]
    pd_net = _create_branched_loads_network(len_lines=len_lines, num_lines=num_lines,
                                   std_type=std_type, p_load_mw=p_load_mw, v_os=v_os,
                                   q_load_mvar=q_load_mvar, trafotype=trafotype,
                                   length_branchout_line_1=0.011,
                                   std_type_branchout_line_1="NAYY 4x50",
                                   std_type_branchout_line_2="NYY 4x35",
                                   prob_branchout_line_1=0.5)
    return pd_net


# II. Kerber Vorstadtnetz mit extremen Netzstrahlen und hoch ausgeslatetem Transformator:
def kb_extrem_vorstadtnetz_trafo_2(std_type="NAYY 4x185", p_load_mw=0.002, q_load_mvar=0.,
                                   trafotype="0.25 MVA 10/0.4 kV", v_os=10.):
    num_lines = [61, 32, 32, 20, 15, 15, 11, 5, 1]
    len_lines = [0.01, 0.014, 0.014, 0.02, 0.023, 0.023, 0.026, 0.05, 0.085]
    pd_net = _create_branched_loads_network(len_lines=len_lines, num_lines=num_lines,
                                   std_type=std_type, p_load_mw=p_load_mw, v_os=v_os,
                                   q_load_mvar=q_load_mvar, trafotype=trafotype,
                                   length_branchout_line_1=0.011,
                                   std_type_branchout_line_1="NAYY 4x50",
                                   std_type_branchout_line_2="NYY 4x35",
                                   prob_branchout_line_1=0.5)
    return pd_net