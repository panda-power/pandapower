# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy
import math
from functools import partial

import numpy as np
import pandas as pd

from pandapower.auxiliary import get_values
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, RATE_A, \
    BR_R_ASYM, BR_X_ASYM, branch_cols
from pandapower.pypower.idx_bus import BASE_KV, VM, VA


def _build_branch_ppc(net, ppc):
    """
    Takes the empty ppc network and fills it with the branch values. The branch
    datatype will be np.complex 128 afterwards.

    .. note:: The order of branches in the ppc is:
            1. Lines
            2. Transformers
            3. 3W Transformers (each 3W Transformer takes up three branches)
            4. Impedances
            5. Internal branch for extended ward

    **INPUT**:
        **net** -The pandapower format network

        **ppc** - The PYPOWER format network to fill in values

    """
    length = _initialize_branch_lookup(net)
    lookup = net._pd2ppc_lookups["branch"]
    mode = net._options["mode"]
    ppc["branch"] = np.zeros(shape=(length, branch_cols), dtype=np.complex128)
    # Check if this should be moved to somewhere else
    if mode == "sc":
        from pandapower.shortcircuit.idx_brch import branch_cols_sc
        branch_sc = np.empty(shape=(length, branch_cols_sc), dtype=float)
        branch_sc.fill(np.nan)
        ppc["branch"] = np.hstack((ppc["branch"], branch_sc))
    ppc["branch"][:, :13] = np.array([0, 0, 0, 0, 0, 250, 250, 250, 1, 0, 1, -360, 360])
    if "line" in lookup:
        _calc_line_parameter(net, ppc)
    if "trafo" in lookup:
        _calc_trafo_parameter(net, ppc)
    if "trafo3w" in lookup:
        _calc_trafo3w_parameter(net, ppc)
    if "impedance" in lookup:
        _calc_impedance_parameter(net, ppc)
    if "xward" in lookup:
        _calc_xward_parameter(net, ppc)
    if "switch" in lookup:
        _calc_switch_parameter(net, ppc)


def _initialize_branch_lookup(net):
    start = 0
    end = 0
    net._pd2ppc_lookups["branch"] = {}
    for element in ["line", "trafo", "trafo3w", "impedance", "xward"]:
        if len(net[element]) > 0:
            if element == "trafo3w":
                end = start + len(net[element]) * 3
            else:
                end = start + len(net[element])
            net._pd2ppc_lookups["branch"][element] = (start, end)
            start = end
    if "_impedance_bb_switches" in net and net._impedance_bb_switches.any():
        end = start + net._impedance_bb_switches.sum()
        net._pd2ppc_lookups["branch"]["switch"] = (start, end)
    return end


def _calc_trafo3w_parameter(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    branch = ppc["branch"]
    f, t = net["_pd2ppc_lookups"]["branch"]["trafo3w"]
    trafo_df = _trafo_df_from_trafo3w(net)
    hv_bus = get_trafo_values(trafo_df, "hv_bus").astype(int)
    lv_bus = get_trafo_values(trafo_df, "lv_bus").astype(int)
    in_service = get_trafo_values(trafo_df, "in_service").astype(int)
    branch[f:t, F_BUS] = bus_lookup[hv_bus]
    branch[f:t, T_BUS] = bus_lookup[lv_bus]
    r, x, y, ratio, shift = _calc_branch_values_from_trafo_df(net, ppc, trafo_df)
    branch[f:t, BR_R] = r
    branch[f:t, BR_X] = x
    branch[f:t, BR_B] = y
    branch[f:t, TAP] = ratio
    branch[f:t, SHIFT] = shift
    branch[f:t, BR_STATUS] = in_service
    if net["_options"]["mode"] == "opf":
        if "max_loading_percent" in trafo_df:
            max_load = get_trafo_values(trafo_df, "max_loading_percent")
            sn_mva = get_trafo_values(trafo_df, "sn_mva")
            branch[f:t, RATE_A] = max_load / 100. * sn_mva
        else:
            branch[f:t, RATE_A] = np.nan


def _calc_line_parameter(net, ppc, elm="line", ppc_elm="branch"):
    """
    calculates the line parameter in per unit.

    **INPUT**:
        **net** - The pandapower format network

        **ppc** - the ppc array

    **OPTIONAL**:
        **elm** - The pandapower element (normally "line")

        **ppc_elm** - The ppc element (normally "branch")

    **RETURN**:
        **t** - Temporary line parameter. Which is a complex128
                Nunmpy array. with the following order:
                0:bus_a; 1:bus_b; 2:r_pu; 3:x_pu; 4:b_pu
    """
    f, t = net._pd2ppc_lookups[ppc_elm][elm]
    branch = ppc[ppc_elm]
    mode = net["_options"]["mode"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    line = net[elm]
    from_bus = bus_lookup[line["from_bus"].values]
    to_bus = bus_lookup[line["to_bus"].values]
    length_km = line["length_km"].values
    parallel = line["parallel"].values
    base_kv = ppc["bus"][from_bus, BASE_KV]
    baseR = np.square(base_kv) / (3 * net.sn_mva) if mode == "pf_3ph" else np.square(base_kv) / net.sn_mva 

    branch[f:t, F_BUS] = from_bus
    branch[f:t, T_BUS] = to_bus
    branch[f:t, BR_R] = line["r_ohm_per_km"].values * length_km / baseR / parallel
    branch[f:t, BR_X] = line["x_ohm_per_km"].values * length_km / baseR / parallel

    if mode == "sc":
        # temperature correction
        if net["_options"]["case"] == "min":
            branch[f:t, BR_R] *= _end_temperature_correction_factor(net, short_circuit=True)
    else:
        # temperature correction
        if net["_options"]["consider_line_temperature"]:
            branch[f:t, BR_R] *= _end_temperature_correction_factor(net)

        b = 2 * net.f_hz * math.pi * line["c_nf_per_km"].values * 1e-9 * baseR * length_km * parallel
        g = line["g_us_per_km"].values * 1e-6 * baseR * length_km * parallel
        branch[f:t, BR_B] = b - g * 1j
    # in service of lines
    branch[f:t, BR_STATUS] = line["in_service"].values
    if net._options["mode"] == "opf":
        # RATE_A is conisdered by the (PowerModels) OPF. If zero -> unlimited
        max_load = line.max_loading_percent.values if "max_loading_percent" in line else 0.
        vr = net.bus.loc[line["from_bus"].values, "vn_kv"].values * np.sqrt(3.)
        max_i_ka = line.max_i_ka.values
        df = line.df.values
        # This calculates the maximum apparent power at 1.0 p.u.
        branch[f:t, RATE_A] = max_load / 100. * max_i_ka * df * parallel * vr


def _calc_trafo_parameter(net, ppc):
    '''
    Calculates the transformer parameter in per unit.

    **INPUT**:
        **net** - The pandapower format network

    **RETURN**:
        **temp_para** -
        Temporary transformer parameter. Which is a np.complex128
        Numpy array. with the following order:
        0:hv_bus; 1:lv_bus; 2:r_pu; 3:x_pu; 4:b_pu; 5:tab, 6:shift
    '''

    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    f, t = net["_pd2ppc_lookups"]["branch"]["trafo"]
    branch = ppc["branch"]
    trafo = net["trafo"]
    parallel = trafo["parallel"].values
    branch[f:t, F_BUS] = bus_lookup[trafo["hv_bus"].values]
    branch[f:t, T_BUS] = bus_lookup[trafo["lv_bus"].values]
    r, x, y, ratio, shift = _calc_branch_values_from_trafo_df(net, ppc)
    branch[f:t, BR_R] = r
    branch[f:t, BR_X] = x
    branch[f:t, BR_B] = y
    branch[f:t, TAP] = ratio
    branch[f:t, SHIFT] = shift
    branch[f:t, BR_STATUS] = trafo["in_service"].values
    if any(trafo.df.values <= 0):
        raise UserWarning("Rating factor df must be positive. Transformers with false "
                          "rating factors: %s" % trafo.query('df<=0').index.tolist())
    if net._options["mode"] == "opf":
        max_load = trafo.max_loading_percent.values if "max_loading_percent" in trafo else 0
        sn_mva = trafo.sn_mva.values
        df = trafo.df.values
        branch[f:t, RATE_A] = max_load / 100. * sn_mva * df * parallel


def get_trafo_values(trafo_df, par):
    if isinstance(trafo_df, dict):
        return trafo_df[par]
    else:
        return trafo_df[par].values


def _calc_branch_values_from_trafo_df(net, ppc, trafo_df=None, sequence=1):
    """
    Calculates the MAT/PYPOWER-branch-attributes from the pandapower trafo dataframe.

    PYPOWER and MATPOWER uses the PI-model to model transformers.
    This function calculates the resistance r, reactance x, complex susceptance c and the tap ratio
    according to the given parameters.

    .. warning:: This function returns the subsceptance b as a complex number
        **(-img + -re*i)**. MAT/PYPOWER is only intended to calculate the
        imaginary part of the subceptance. However, internally c is
        multiplied by i. By using subsceptance in this way, it is possible
        to consider the ferromagnetic loss of the coil. Which would
        otherwise be neglected.


    .. warning:: Tab switches effect calculation as following:
        On **high-voltage** side(=1) -> only **tab** gets adapted.
        On **low-voltage** side(=2) -> **tab, x, r** get adapted.
        This is consistent with Sincal.
        The Sincal method in this case is questionable.


    **INPUT**:
        **pd_trafo** - The pandapower format Transformer Dataframe.
                        The Transformer modell will only readfrom pd_net

    **RETURN**:
        **temp_para** - Temporary transformer parameter. Which is a complex128
                        Nunmpy array. with the following order:
                        0:r_pu; 1:x_pu; 2:b_pu; 3:tab;

    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    if trafo_df is None:
        trafo_df = net["trafo"]
    lv_bus = get_trafo_values(trafo_df, "lv_bus")
    vn_lv = ppc["bus"][bus_lookup[lv_bus], BASE_KV]
    ### Construct np.array to parse results in ###
    # 0:r_pu; 1:x_pu; 2:b_pu; 3:tab;
    vn_trafo_hv, vn_trafo_lv, shift = _calc_tap_from_dataframe(net, trafo_df)
    ratio = _calc_nominal_ratio_from_dataframe(ppc, trafo_df, vn_trafo_hv, vn_trafo_lv,
                                               bus_lookup)
    r, x, y = _calc_r_x_y_from_dataframe(net, trafo_df, vn_trafo_lv, vn_lv, ppc, sequence=sequence)
    return r, x, y, ratio, shift


def _calc_r_x_y_from_dataframe(net, trafo_df, vn_trafo_lv, vn_lv, ppc, sequence=1):
    mode = net["_options"]["mode"]
    trafo_model = net["_options"]["trafo_model"]

    r, x = _calc_r_x_from_dataframe(mode, trafo_df, vn_lv, vn_trafo_lv, net.sn_mva, sequence=sequence)

    if mode == "sc":
        y = 0
        if isinstance(trafo_df, pd.DataFrame):  # 2w trafo is dataframe, 3w trafo is dict
            from pandapower.shortcircuit.idx_bus import C_MAX
            bus_lookup = net._pd2ppc_lookups["bus"]
            cmax = ppc["bus"][bus_lookup[net.trafo.lv_bus.values], C_MAX]
            kt = _transformer_correction_factor(trafo_df.vk_percent, trafo_df.vkr_percent,
                                                trafo_df.sn_mva, cmax)
            r *= kt
            x *= kt
    else:
        y = _calc_y_from_dataframe(mode, trafo_df, vn_lv, vn_trafo_lv, net.sn_mva)

    if trafo_model == "pi":
        return r, x, y
    elif trafo_model == "t":
        return _wye_delta(r, x, y)
    else:
        raise ValueError("Unkonwn Transformer Model %s - valid values ar 'pi' or 't'" % trafo_model)


def _wye_delta(r, x, y):
    """
    20.05.2016 added by Lothar Löwer

    Calculate transformer Pi-Data based on T-Data

    """
    tidx = np.where(y != 0)
    za_star = (r[tidx] + x[tidx] * 1j) / 2
    zc_star = -1j / y[tidx]
    zSum_triangle = za_star * za_star + 2 * za_star * zc_star
    zab_triangle = zSum_triangle / zc_star
    zbc_triangle = zSum_triangle / za_star
    r[tidx] = zab_triangle.real
    x[tidx] = zab_triangle.imag
    y[tidx] = -2j / zbc_triangle
    return r, x, y


def _calc_y_from_dataframe(mode, trafo_df, vn_lv, vn_trafo_lv, sn_mva):
    """
    Calculate the subsceptance y from the transformer dataframe.

    INPUT:

        **trafo** (Dataframe) - The dataframe in net.trafo
        which contains transformer calculation values.

    OUTPUT:
        **subsceptance** (1d array, np.complex128) - The subsceptance in pu in
        the form (-b_img, -b_real)
    """

    baseR = np.square(vn_lv) / (3*sn_mva) if mode == 'pf_3ph' else np.square(vn_lv) / sn_mva
    vn_lv_kv = get_trafo_values(trafo_df, "vn_lv_kv")
    pfe = (get_trafo_values(trafo_df, "pfe_kw") * 1e-3) / 3 if mode == 'pf_3ph'\
        else get_trafo_values(trafo_df, "pfe_kw") * 1e-3
    parallel = get_trafo_values(trafo_df, "parallel")

    ### Calculate subsceptance ###
    vnl_squared = (vn_lv_kv ** 2)/3 if mode == 'pf_3ph' else vn_lv_kv ** 2
    b_real = pfe / vnl_squared * baseR
    i0 = get_trafo_values(trafo_df, "i0_percent") / 3 if mode == 'pf_3ph'\
        else get_trafo_values(trafo_df, "i0_percent")
    sn = get_trafo_values(trafo_df, "sn_mva")
    b_img = (i0 / 100. * sn) ** 2 - pfe ** 2

    b_img[b_img < 0] = 0
    b_img = np.sqrt(b_img) * baseR / vnl_squared
    y = - b_real * 1j - b_img * np.sign(i0)
    return y / np.square(vn_trafo_lv / vn_lv_kv) * parallel


def _calc_tap_from_dataframe(net, trafo_df):
    """
    Adjust the nominal voltage vnh and vnl to the active tab position "tap_pos".
    If "side" is 1 (high-voltage side) the high voltage vnh is adjusted.
    If "side" is 2 (low-voltage side) the low voltage vnl is adjusted

    INPUT:
        **net** - The pandapower format network

        **trafo** (Dataframe) - The dataframe in pd_net["structure"]["trafo"]
        which contains transformer calculation values.

    OUTPUT:
        **vn_hv_kv** (1d array, float) - The adusted high voltages

        **vn_lv_kv** (1d array, float) - The adjusted low voltages

        **trafo_shift** (1d array, float) - phase shift angle

    """
    calculate_voltage_angles = net["_options"]["calculate_voltage_angles"]
    mode = net["_options"]["mode"]
    vnh = copy.copy(get_trafo_values(trafo_df, "vn_hv_kv").astype(float))
    vnl = copy.copy(get_trafo_values(trafo_df, "vn_lv_kv").astype(float))
    trafo_shift = get_trafo_values(trafo_df, "shift_degree").astype(float) if calculate_voltage_angles else \
        np.zeros(len(vnh))
    if mode == "sc":
        return vnh, vnl, trafo_shift

    tap_pos = get_trafo_values(trafo_df, "tap_pos")
    tap_neutral = get_trafo_values(trafo_df, "tap_neutral")
    tap_diff = tap_pos - tap_neutral
    tap_phase_shifter = get_trafo_values(trafo_df, "tap_phase_shifter")
    tap_side = get_trafo_values(trafo_df, "tap_side")
    tap_step_percent = get_trafo_values(trafo_df, "tap_step_percent")
    tap_step_degree = get_trafo_values(trafo_df, "tap_step_degree")

    cos = lambda x: np.cos(np.deg2rad(x))
    sin = lambda x: np.sin(np.deg2rad(x))
    arctan = lambda x: np.rad2deg(np.arctan(x))

    for side, vn, direction in [("hv", vnh, 1), ("lv", vnl, -1)]:
        phase_shifters = tap_phase_shifter & (tap_side == side)
        tap_complex = np.isfinite(tap_step_percent) & np.isfinite(tap_pos) & (tap_side == side) & \
            ~phase_shifters
        if tap_complex.any():
            tap_steps = tap_step_percent[tap_complex] * tap_diff[tap_complex] / 100
            tap_angles = _replace_nan(tap_step_degree[tap_complex])
            u1 = vn[tap_complex]
            du = u1 * _replace_nan(tap_steps)
            vn[tap_complex] = np.sqrt((u1 + du * cos(tap_angles)) ** 2 + (du * sin(tap_angles)) ** 2)
            trafo_shift[tap_complex] += (arctan(direction * du * sin(tap_angles) /
                                                (u1 + du * cos(tap_angles))))
        if phase_shifters.any():
            degree_is_set = _replace_nan(tap_step_degree[phase_shifters]) != 0
            percent_is_set = _replace_nan(tap_step_percent[phase_shifters]) != 0
            if (degree_is_set & percent_is_set).any():
                raise UserWarning("Both tap_step_degree and tap_step_percent set for ideal phase shifter")
            trafo_shift[phase_shifters] += np.where(
                (degree_is_set),
                (direction * tap_diff[phase_shifters] * tap_step_degree[phase_shifters]),
                (direction * 2 * np.rad2deg(np.arcsin(tap_diff[phase_shifters] * \
                                                      tap_step_percent[phase_shifters] / 100 / 2)))
            )
    return vnh, vnl, trafo_shift


def _replace_nan(array, value=0):
    mask = np.isnan(array)
    array[mask] = value
    return array

def _calc_r_x_from_dataframe(mode, trafo_df, vn_lv, vn_trafo_lv, sn_mva, sequence=1):
    """
    Calculates (Vectorized) the resitance and reactance according to the
    transformer values

    """
    parallel = get_trafo_values(trafo_df, "parallel")
    if sequence == 1:
        vk_percent = get_trafo_values(trafo_df, "vk_percent")
        vkr_percent = get_trafo_values(trafo_df, "vkr_percent")
    elif sequence == 0:
        vk_percent = get_trafo_values(trafo_df, "vk0_percent")
        vkr_percent = get_trafo_values(trafo_df, "vkr0_percent")
    else:
        raise UserWarning("Unsupported sequence")
    tap_lv = np.square(vn_trafo_lv / vn_lv) * (3* sn_mva)  if mode == 'pf_3ph' else\
    np.square(vn_trafo_lv / vn_lv) * sn_mva  # adjust for low voltage side voltage converter

    sn_trafo_mva = get_trafo_values(trafo_df, "sn_mva")
    z_sc = vk_percent / 100. / sn_trafo_mva * tap_lv
    r_sc = vkr_percent / 100. / sn_trafo_mva * tap_lv
    x_sc = np.sign(z_sc) * np.sqrt((z_sc ** 2 - r_sc ** 2).astype(float))
    return r_sc / parallel, x_sc / parallel


def _calc_nominal_ratio_from_dataframe(ppc, trafo_df, vn_hv_kv, vn_lv_kv, bus_lookup):
    """
    Calculates (Vectorized) the off nominal tap ratio::

                  (vn_hv_kv / vn_lv_kv) / (ub1_in_kv / ub2_in_kv)

    INPUT:
        **net** (Dataframe) - The net for which to calc the tap ratio.

        **vn_hv_kv** (1d array, float) - The adjusted nominal high voltages

        **vn_lv_kv** (1d array, float) - The adjusted nominal low voltages

    OUTPUT:
        **tab** (1d array, float) - The off-nominal tap ratio
    """
    # Calculating tab (trasformer off nominal turns ratio)
    tap_rat = vn_hv_kv / vn_lv_kv
    hv_bus = get_trafo_values(trafo_df, "hv_bus")
    lv_bus = get_trafo_values(trafo_df, "lv_bus")
    nom_rat = get_values(ppc["bus"][:, BASE_KV], hv_bus, bus_lookup) / \
        get_values(ppc["bus"][:, BASE_KV], lv_bus, bus_lookup)
    return tap_rat / nom_rat


def _calc_impedance_parameter(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    f, t = net["_pd2ppc_lookups"]["branch"]["impedance"]
    branch = ppc["branch"]
    rij, xij, r_asym, x_asym = _calc_impedance_parameters_from_dataframe(net)
    branch[f:t, BR_R] = rij
    branch[f:t, BR_X] = xij
    branch[f:t, BR_R_ASYM] = r_asym
    branch[f:t, BR_X_ASYM] = x_asym
    branch[f:t, F_BUS] = bus_lookup[net.impedance["from_bus"].values]
    branch[f:t, T_BUS] = bus_lookup[net.impedance["to_bus"].values]
    branch[f:t, BR_STATUS] = net["impedance"]["in_service"].values


def _calc_impedance_parameters_from_dataframe(net):
    impedance = net.impedance
    sn_impedance = impedance["sn_mva"].values
    sn_net = net.sn_mva
    rij = impedance["rft_pu"].values
    xij = impedance["xft_pu"].values
    rji = impedance["rtf_pu"].values
    xji = impedance["xtf_pu"].values

    r = rij / sn_impedance * sn_net
    x = xij / sn_impedance * sn_net
    r_asym = (rji - rij) / sn_impedance * sn_net
    x_asym = (xji - xij) / sn_impedance * sn_net
    return r, x, r_asym, x_asym


def _calc_xward_parameter(net, ppc):
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    f, t = net["_pd2ppc_lookups"]["branch"]["xward"]
    branch = ppc["branch"]
    baseR = np.square(get_values(ppc["bus"][:, BASE_KV], net["xward"]["bus"].values, bus_lookup)) / \
        net.sn_mva
    xw_is = net["_is_elements"]["xward"]
    branch[f:t, F_BUS] = bus_lookup[net["xward"]["bus"].values]
    branch[f:t, T_BUS] = bus_lookup[net._pd2ppc_lookups["aux"]["xward"]]
    branch[f:t, BR_R] = net["xward"]["r_ohm"] / baseR
    branch[f:t, BR_X] = net["xward"]["x_ohm"] / baseR
    branch[f:t, BR_STATUS] = xw_is


def _gather_branch_switch_info(bus, branch_id, branch_type, net):
    # determine at which end the switch is located
    # 1 = to-bus/lv-bus; 0 = from-bus/hv-bus
    branch_id = int(branch_id)
    lookup = net._pd2ppc_lookups["branch"]
    if branch_type == "l":
        side = "to" if net["line"]["to_bus"].at[branch_id] == bus else "from"
        branch_idx = net["line"].index.get_loc(branch_id)
        return side, int(bus), int(branch_idx)
    elif branch_type == "t":
        side = "hv" if net["trafo"]["hv_bus"].at[branch_id] == bus else "lv"
        branch_idx = lookup["trafo"][0] + net["trafo"].index.get_loc(branch_id)
        return side, int(bus), int(branch_idx)
    elif branch_type == "t3":
        f, t = lookup["trafo3w"]
        if net["trafo3w"]["hv_bus"].at[branch_id] == bus:
            side = "hv"
            offset = 0
        elif net["trafo3w"]["mv_bus"].at[branch_id] == bus:
            side = "mv"
            offset = (t - f) / 3
        elif net["trafo3w"]["lv_bus"].at[branch_id] == bus:
            side = "lv"
            offset = (t - f) / 3 * 2
        branch_idx = lookup["trafo3w"][0] + net["trafo3w"].index.get_loc(branch_id) + offset
        return side, int(bus), int(branch_idx)


def _switch_branches(net, ppc):
    from pandapower.shortcircuit.idx_bus import C_MIN, C_MAX
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    calculate_voltage_angles = net._options["calculate_voltage_angles"]
    neglect_open_switch_branches = net._options["neglect_open_switch_branches"]
    mode = net._options["mode"]
    open_switches = (net.switch.closed.values == False)
    n_bus = ppc["bus"].shape[0]
    for et, element in [("l", "line"), ("t", "trafo"), ("t3", "trafo3w")]:
        switch_mask = open_switches & (net.switch.et.values == et)
        if not switch_mask.any():
            continue
        nr_open_switches = np.count_nonzero(switch_mask)
        mapfunc = partial(_gather_branch_switch_info, branch_type=et, net=net)
        switch_element = net["switch"]["element"].values[switch_mask]
        switch_buses = net["switch"]["bus"].values[switch_mask]
        switch_info = np.array(list(map(mapfunc, switch_buses, switch_element)))
        sw_sides = switch_info[:, 0]
        sw_bus_index = bus_lookup[switch_info[:, 1].astype(int)]
        sw_branch_index = switch_info[:, 2].astype(int)
        if neglect_open_switch_branches:
            # deactivate switches which have an open switch instead of creating aux buses
            ppc["branch"][sw_branch_index, BR_STATUS] = 0
            continue

        new_buses = np.zeros(shape=(nr_open_switches, ppc["bus"].shape[1]), dtype=float)
        new_buses[:, :15] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9, 0, 0])
        new_indices = np.arange(n_bus, n_bus + nr_open_switches)
        new_buses[:, 0] = new_indices
        new_buses[:, BASE_KV] = ppc["bus"][sw_bus_index, BASE_KV]
        ppc["bus"] = np.vstack([ppc["bus"], new_buses])
        n_bus += new_buses.shape[0]
        init_vm = net._options["init_vm_pu"]
        init_va = net._options["init_va_degree"]
        for location in np.unique(sw_sides):
            mask = sw_sides == location
            buses = new_indices[mask]
            side = F_BUS if location == "hv" or location == "from" else T_BUS
            for init, col in [(init_vm, VM), (init_va, VA)]:
                if isinstance(init, str) and init == "results":
                    if col == VM:
                        res_column = net["res_%s" % element]["vm_%s_pu" % location]
                    else:
                        res_column = net["res_%s" % element]["va_%s_degree" % location]
                    init_values = res_column.loc[switch_element].values[mask]
                else:
                    if element == "line":
                        opposite_buses = ppc["branch"][sw_branch_index[mask], side].real.astype(int)
                        init_values = ppc["bus"][opposite_buses, col]
                    else:
                        opposite_side = T_BUS if side == F_BUS else F_BUS
                        opposite_buses = ppc["branch"][sw_branch_index[mask], opposite_side].real.astype(int)
                        if col == VM:
                            taps = ppc["branch"][sw_branch_index[mask], TAP].real
                            init_values = ppc["bus"][opposite_buses, col] * taps
                        else:
                            if calculate_voltage_angles:
                                shift = ppc["branch"][sw_branch_index[mask], SHIFT].real.astype(int)
                                init_values = ppc["bus"][opposite_buses, col] + shift
                            else:
                                init_values = ppc["bus"][opposite_buses, col]
                ppc["bus"][buses, col] = init_values
            if mode == "sc":
                ppc["bus"][buses, C_MAX] = ppc["bus"][opposite_buses, C_MAX]
                ppc["bus"][buses, C_MIN] = ppc["bus"][opposite_buses, C_MIN]
            ppc["branch"][sw_branch_index[mask], side] = new_indices[mask]


def _branches_with_oos_buses(net, ppc):
    """
    Updates the ppc["branch"] matrix with the changed from or to values
    if the branch is connected to an out of service bus

    Adds auxiliary buses if branch is connected to an out of service bus
    Sets branch out of service if connected to two out of service buses

    **INPUT**:
        **n** - The pandapower format network

        **ppc** - The PYPOWER format network to fill in values
        **bus_is** - The in service buses
    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    # get in service elements
    _is_elements = net["_is_elements"]
    bus_is_idx = _is_elements['bus_is_idx']
    line_is_idx = _is_elements['line_is_idx']

    n_oos_buses = len(net['bus']) - len(bus_is_idx)

    # only filter lines at oos buses if oos buses exists
    if n_oos_buses > 0:
        n_bus = len(ppc["bus"])
        future_buses = [ppc["bus"]]
        # out of service buses
        bus_oos = np.setdiff1d(net['bus'].index.values, bus_is_idx)
        # from buses of line
        line_buses = net["line"][["from_bus", "to_bus"]].loc[line_is_idx].values
        f_bus = line_buses[:, 0]
        t_bus = line_buses[:, 1]

        # determine on which side of the line the oos bus is located
        mask_from = np.in1d(f_bus, bus_oos)
        mask_to = np.in1d(t_bus, bus_oos)

        mask_and = mask_to & mask_from
        if np.any(mask_and):
            mask_from[mask_and] = False
            mask_to[mask_and] = False

        # get lines that are connected to oos bus at exactly one side
        # buses that are connected to two oos buses will be removed by ext2int
        mask_or = mask_to | mask_from
        # check whether buses are connected to line
        oos_buses_at_lines = np.hstack([f_bus[mask_from], t_bus[mask_to]])
        n_oos_buses_at_lines = len(oos_buses_at_lines)

        # only if oos_buses are at lines (they could be isolated as well)
        if n_oos_buses_at_lines > 0:
            ls_info = np.zeros((n_oos_buses_at_lines, 3), dtype=int)
            ls_info[:, 0] = mask_to[mask_or] & ~mask_from[mask_or]
            ls_info[:, 1] = oos_buses_at_lines
            ls_info[:, 2] = np.nonzero(np.in1d(net['line'].index, line_is_idx[mask_or]))[0]

            # ls_info = list(map(mapfunc,
            #               line_switches["bus"].values,
            #               line_switches["element"].values))
            # we now have the following matrix
            # 0: 1 if switch is at to_bus, 0 else
            # 1: bus of the switch
            # 2: position of the line a switch is connected to
            # ls_info = np.array(ls_info, dtype=int)

            # build new buses
            new_ls_buses = np.zeros(shape=(n_oos_buses_at_lines, ppc["bus"].shape[1]), dtype=float)
            new_indices = np.arange(n_bus, n_bus + n_oos_buses_at_lines)
            # the newly created buses
            new_ls_buses[:, :15] = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1.1, 0.9, 0, 0])
            new_ls_buses[:, 0] = new_indices
            new_ls_buses[:, BASE_KV] = get_values(ppc["bus"][:, BASE_KV], ls_info[:, 1], bus_lookup)

            future_buses.append(new_ls_buses)

            # re-route the end of lines to a new bus
            ppc["branch"][ls_info[ls_info[:, 0].astype(bool), 2], 1] = \
                new_indices[ls_info[:, 0].astype(bool)]
            ppc["branch"][ls_info[np.logical_not(ls_info[:, 0]), 2], 0] = \
                new_indices[np.logical_not(ls_info[:, 0])]

            ppc["bus"] = np.vstack(future_buses)


def _calc_switch_parameter(net, ppc):
    """
    calculates the line parameter in per unit.

    **INPUT**:
        **net** -The pandapower format network

    **RETURN**:
        **t** - Temporary line parameter. Which is a complex128
                Nunmpy array. with the following order:
                0:bus_a; 1:bus_b; 2:r_pu; 3:x_pu; 4:b_pu
    """
    rx_ratio = net["_options"]["switch_rx_ratio"]
    rz_ratio = rx_ratio / np.sqrt(1 + rx_ratio ** 2)
    xz_ratio = 1 / np.sqrt(1 + rx_ratio ** 2)

    f, t = net["_pd2ppc_lookups"]["branch"]["switch"]
    branch = ppc["branch"]
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    switch = net.switch[net._impedance_bb_switches]
    fb = bus_lookup[switch["bus"].values]
    tb = bus_lookup[switch["element"].values]
    baseR = np.square(ppc["bus"][fb, BASE_KV]) / net.sn_mva
    branch[f:t, F_BUS] = fb
    branch[f:t, T_BUS] = tb

    z_switch = switch['z_ohm'].values
    # x_switch will have the same value of r_switch to avoid zero dividence
    branch[f:t, BR_R] = z_switch / baseR * rz_ratio
    branch[f:t, BR_X] = z_switch / baseR * xz_ratio


def _end_temperature_correction_factor(net, short_circuit=False):
    """
    Function to calculate resistance correction factor for the given temperature ("endtemp_degree").
    When multiplied by the factor, the value of r_ohm_per_km will correspond to the resistance at
    the given temperature.

    In case of short circuit calculation, the relevant value for the temperature is
    "endtemp_degree", which stands for the final temperature of a line after the short circuit.
    The temperature coefficient "alpha" is a constant value of 0.004 in the short circuit
    calculation standard IEC 60909-0:2016.

    In case of a load flow calculation, the relelvant parameter is "temperature_degree_celsius",
    which is specified by the user and allows calculating load flow for a given operating
    temperature.

    The alpha value can be provided according to the used material for the purpose of load flow
    calculation, e.g. 0.0039 for copper or 0.00403 for aluminum. If alpha is not provided in the
    net.line table, the default value of 0.004 is used.

    The calculation of the electrical resistance is based on the formula R = R20(1+alpha*(T-20°C)),
    where R is the calculated resistance, R20 is the resistance at 20 °C, alpha is the temperature
    coefficient of resistance of the conducting material and T is the line temperature in °C.
    Accordingly, the resulting correction factor is (1+alpha*(T-20°C)).

    Args:
        net: pandapowerNet
        short_circuit: whether the factor is calculated in the scope of a short circuit calculation

    Returns:
        correction factor for line R, by which the line parameter should be multiplied to
                obtain the value of resistance at line temperature "endtemp_degree"

    """

    if short_circuit:
        # endtemp_degree is line temperature that is reached as the result of a short circuit
        # this value is the property of the lines
        if "endtemp_degree" not in net.line.columns:
            raise UserWarning("Specify end temperature for lines in net.line.endtemp_degree")

        delta_t_degree_celsius = net.line.endtemp_degree.values.astype(np.float64) - 20
        # alpha is the temperature correction factor for the electric resistance of the material
        # formula from standard, used this way in short-circuit calculation
        alpha = 4e-3
    else:
        # temperature_degree_celsius is line temperature for load flow calculation
        if "temperature_degree_celsius" not in net.line.columns:
            raise UserWarning("Specify line temperature in net.line.temperature_degree_celsius")

        delta_t_degree_celsius = net.line.temperature_degree_celsius.values.astype(np.float64) - 20

        if 'alpha' in net.line.columns:
            alpha = net.line.alpha.values.astype(np.float64)
        else:
            alpha = 4e-3

    r_correction_for_temperature = 1 + alpha * delta_t_degree_celsius

    return r_correction_for_temperature


def _transformer_correction_factor(vk, vkr, sn, cmax):
    """
        2W-Transformer impedance correction factor in short circuit calculations,
        based on the IEC 60909-0:2016 standard.
        Args:
            vk: transformer short-circuit voltage, percent
            vkr: real-part of transformer short-circuit voltage, percent
            sn: transformer rating, kVA
            cmax: voltage factor to account for maximum worst-case currents, based on the lv side

        Returns:
            kt: transformer impedance correction factor for short-circuit calculations
        """
    zt = vk / 100 / sn
    rt = vkr / 100 / sn
    xt = np.sqrt(zt ** 2 - rt ** 2)
    kt = 0.95 * cmax / (1 + .6 * xt * sn)
    return kt


def get_is_lines(net):
    """
    get indices of lines that are in service and save that information in net
    """
    _is_elements = net["_is_elements"]
    _is_elements["line"] = net["line"][net["line"]["in_service"].values.astype(bool)]


def _trafo_df_from_trafo3w(net, sequence=1):
    nr_trafos = len(net["trafo3w"])
    trafo2 = dict()
    sides = ["hv", "mv", "lv"]
    mode = net._options["mode"]
    loss_side = net._options["trafo3w_losses"].lower()
    nr_trafos = len(net["trafo3w"])
    t3 = net["trafo3w"]
    if sequence==1:
        _calculate_sc_voltages_of_equivalent_transformers(t3, trafo2, mode)
    elif sequence==0:
        if mode != "sc":
            raise NotImplementedError("0 seq impedance calculation only implemented for short-circuit calculation!")
        _calculate_sc_voltages_of_equivalent_transformers_zero_sequence(t3, trafo2,)
    else:
        raise UserWarning("Unsupported sequence for trafo3w convertion")
    _calculate_3w_tap_changers(t3, trafo2, sides)
    zeros = np.zeros(len(net.trafo3w))
    aux_buses = net._pd2ppc_lookups["aux"]["trafo3w"]
    trafo2["hv_bus"] = {"hv": t3.hv_bus.values, "mv": aux_buses, "lv": aux_buses}
    trafo2["lv_bus"] = {"hv": aux_buses, "mv": t3.mv_bus.values, "lv": t3.lv_bus.values}
    trafo2["in_service"] = {side: t3.in_service.values for side in sides}
    trafo2["i0_percent"] = {side: t3.i0_percent.values if loss_side == side else zeros for side in sides}
    trafo2["pfe_kw"] = {side: t3.pfe_kw.values if loss_side == side else zeros for side in sides}
    trafo2["vn_hv_kv"] = {side: t3.vn_hv_kv.values for side in sides}
    trafo2["vn_lv_kv"] = {side: t3["vn_%s_kv" % side].values for side in sides}
    trafo2["shift_degree"] = {"hv": np.zeros(nr_trafos), "mv": t3.shift_mv_degree.values,
                              "lv": t3.shift_lv_degree.values}
    trafo2["tap_phase_shifter"] = {side: np.zeros(nr_trafos).astype(bool) for side in sides}
    trafo2["parallel"] = {side: np.ones(nr_trafos) for side in sides}
    trafo2["df"] = {side: np.ones(nr_trafos) for side in sides}
    if net._options["mode"] == "opf" and "max_loading_percent" in net.trafo3w:
        trafo2["max_loading_percent"] = {side: net.trafo3w.max_loading_percent.values for side in sides}
    return {var: np.concatenate([trafo2[var][side] for side in sides]) for var in trafo2.keys()}


def _calculate_sc_voltages_of_equivalent_transformers(t3, t2, mode):
    vk_3w = np.stack([t3.vk_hv_percent.values, t3.vk_mv_percent.values, t3.vk_lv_percent.values])
    vkr_3w = np.stack([t3.vkr_hv_percent.values, t3.vkr_mv_percent.values, t3.vkr_lv_percent.values])
    sn = np.stack([t3.sn_hv_mva.values, t3.sn_mv_mva.values, t3.sn_lv_mva.values])

    vk_2w_delta = z_br_to_bus_vector(vk_3w, sn)
    vkr_2w_delta = z_br_to_bus_vector(vkr_3w, sn)
    if mode == "sc":
        kt = _transformer_correction_factor(vk_3w, vkr_3w, sn, 1.1)
        vk_2w_delta *= kt
        vkr_2w_delta *= kt
    vki_2w_delta = np.sqrt(vk_2w_delta ** 2 - vkr_2w_delta ** 2)
    vkr_2w = wye_delta_vector(vkr_2w_delta, sn)
    vki_2w = wye_delta_vector(vki_2w_delta, sn)
    vk_2w = np.sign(vki_2w) * np.sqrt(vki_2w ** 2 + vkr_2w ** 2)
    if np.any(vk_2w == 0):
        raise UserWarning("Equivalent transformer with zero impedance!")
    t2["vk_percent"] = {"hv": vk_2w[0, :], "mv": vk_2w[1, :], "lv": vk_2w[2, :]}
    t2["vkr_percent"] = {"hv": vkr_2w[0, :], "mv": vkr_2w[1, :], "lv": vkr_2w[2, :]}
    t2["sn_mva"] = {"hv": sn[0, :], "mv": sn[1, :], "lv": sn[2, :]}


def _calculate_sc_voltages_of_equivalent_transformers_zero_sequence(t3, t2):
    vk_3w = np.stack([t3.vk_hv_percent.values, t3.vk_mv_percent.values, t3.vk_lv_percent.values])
    vkr_3w = np.stack([t3.vkr_hv_percent.values, t3.vkr_mv_percent.values, t3.vkr_lv_percent.values])
    vk0_3w = np.stack([t3.vk0_hv_percent.values, t3.vk0_mv_percent.values, t3.vk0_lv_percent.values])
    vkr0_3w = np.stack([t3.vkr0_hv_percent.values, t3.vkr0_mv_percent.values, t3.vkr0_lv_percent.values])
    sn = np.stack([t3.sn_hv_mva.values, t3.sn_mv_mva.values, t3.sn_lv_mva.values])

    vk0_2w_delta = z_br_to_bus_vector(vk0_3w, sn)
    vkr0_2w_delta = z_br_to_bus_vector(vkr0_3w, sn)

    # Only for "sc", calculated with positive sequence value
    kt = _transformer_correction_factor(vk_3w, vkr_3w, sn, 1.1)
    vk0_2w_delta *= kt
    vkr0_2w_delta *= kt

    vki0_2w_delta = np.sqrt(vk0_2w_delta ** 2 - vkr0_2w_delta ** 2)
    vkr0_2w = wye_delta_vector(vkr0_2w_delta, sn)
    vki0_2w = wye_delta_vector(vki0_2w_delta, sn)
    vk0_2w = np.sign(vki0_2w) * np.sqrt(vki0_2w ** 2 + vkr0_2w ** 2)
    if np.any(vk0_2w == 0):
        raise UserWarning("Equivalent transformer with zero impedance!")
    t2["vk0_percent"] = {"hv": vk0_2w[0, :], "mv": vk0_2w[1, :], "lv": vk0_2w[2, :]}
    t2["vkr0_percent"] = {"hv": vkr0_2w[0, :], "mv": vkr0_2w[1, :], "lv": vkr0_2w[2, :]}
    t2["sn_mva"] = {"hv": sn[0, :], "mv": sn[1, :], "lv": sn[2, :]}


def z_br_to_bus_vector(z, sn):
    return sn[0, :] * np.array([z[0, :] / sn[[0, 1], :].min(axis=0), z[1, :] /
                                sn[[1, 2], :].min(axis=0), z[2, :] / sn[[0, 2], :].min(axis=0)])


def wye_delta(zbr_n, s):
    return .5 * s / s[0] * np.array([(zbr_n[0] + zbr_n[2] - zbr_n[1]),
                                     (zbr_n[1] + zbr_n[0] - zbr_n[2]),
                                     (zbr_n[2] + zbr_n[1] - zbr_n[0])])


def wye_delta_vector(zbr_n, s):
    return .5 * s / s[0, :] * np.array([(zbr_n[0, :] + zbr_n[2, :] - zbr_n[1, :]),
                                        (zbr_n[1, :] + zbr_n[0, :] - zbr_n[2, :]),
                                        (zbr_n[2, :] + zbr_n[1, :] - zbr_n[0, :])])


def _calculate_3w_tap_changers(t3, t2, sides):
    tap_variables = ["tap_side", "tap_pos", "tap_neutral", "tap_max", "tap_min", "tap_step_percent",
                     "tap_step_degree"]
    sides = ["hv", "mv", "lv"]
    nr_trafos = len(t3)
    empty = np.zeros(nr_trafos)
    empty.fill(np.nan)
    tap_arrays = {var: {side: empty.copy() for side in sides} for var in tap_variables}
    tap_arrays["tap_side"] = {side: np.array([None] * nr_trafos) for side in sides}
    at_star_point = t3.tap_at_star_point.values
    any_at_star_point = at_star_point.any()
    for side in sides:
        tap_mask = t3.tap_side.values == side
        for var in tap_variables:
            tap_arrays[var][side][tap_mask] = t3[var].values[tap_mask]

        # t3 trafos with tap changer at terminals
        tap_arrays["tap_side"][side][tap_mask] = "hv" if side == "hv" else "lv"

        # t3 trafos with tap changer at star points
        if any_at_star_point:
            mask_star_point = tap_mask & at_star_point
            tap_arrays["tap_side"][side][mask_star_point] = "lv" if side == "hv" else "hv"
            tap_arrays["tap_step_degree"][side][mask_star_point] += 180
    t2.update(tap_arrays)
