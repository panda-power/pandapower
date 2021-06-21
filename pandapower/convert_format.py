# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pandas as pd
from packaging import version

from pandapower import __version__
from pandapower.create import create_empty_network, create_poly_cost
from pandapower.results import reset_results


def convert_format(net, table_selection=None):
    """
    Converts old nets to new format to ensure consistency. The converted net is returned.
    """
    from pandapower.toolbox import set_data_type_of_columns_to_default
    if isinstance(net.version, str) and version.parse(net.version) >= version.parse(__version__):
        return net
    _add_nominal_power(net)
    _add_missing_tables(net)
    _rename_columns(net, table_selection)
    _add_missing_columns(net, table_selection)
    _create_seperate_cost_tables(net, table_selection)
    if version.parse(str(net.version)) < version.parse("2.4.0"):
        _convert_bus_pq_meas_to_load_reference(net, table_selection)
    if isinstance(net.version, float) and net.version < 2:
        _convert_to_generation_system(net, table_selection)
        _convert_costs(net)
        _convert_to_mw(net)
        _update_trafo_parameter_names(net, table_selection)
        reset_results(net)
    if isinstance(net.version, float) and net.version < 1.6:
        set_data_type_of_columns_to_default(net)
    _convert_objects(net, table_selection)
    net.version = __version__
    return net


def _convert_bus_pq_meas_to_load_reference(net, table_selection):
    if _check_table_selection('measurement', table_selection):
        bus_pq_meas_mask = net.measurement.measurement_type.isin(["p", "q"])&\
            (net.measurement.element_type=="bus")
        net.measurement.loc[bus_pq_meas_mask, "value"] *= -1


def _convert_to_generation_system(net, table_selection):
    update_elements = []
    if _check_table_selection('sgen', table_selection):
        net.sgen.p_kw *= -1
        net.sgen.q_kvar *= -1
        update_elements += ['sgen']
    if _check_table_selection('gen', table_selection):
        net.gen.p_kw *= -1
        update_elements += ['gen']
    if _check_table_selection('ext_grid', table_selection):
        update_elements += ['ext_grid']
    for element in update_elements:
        for suffix in ["p_kw", "q_kvar"]:
            constraints = {}
            if "min_%s" % suffix in net[element]:
                constraints["max_%s" % suffix] = net[element]["min_%s" % suffix] * -1
                del net[element]["min_%s" % suffix]
            if "max_%s" % suffix in net[element]:
                constraints["min_%s" % suffix] = net[element]["max_%s" % suffix] * -1
                del net[element]["max_%s" % suffix]
            for column, values in constraints.items():
                net[element][column] = values
    pq_measurements = net.measurement[net.measurement.measurement_type.isin(["p", "q"])].index
    net.measurement.loc[pq_measurements, ["value", "std_dev"]] *= 1e-3


def _convert_costs(net):
    if "polynomial_cost" in net:
        for cost in net.polynomial_cost.itertuples():
            values = cost.c[0]
            if len(values) == 2:
                cp0 = values[1]
                cp1 = values[0]
                cp2 = 0
            elif len(values) == 3:
                cp0 = values[2]
                cp1 = values[1]
                cp2 = values[0]
            create_poly_cost(net, et=cost.element_type, element=cost.element, cp0_eur=cp0,
                             cp1_eur_per_mw=cp1 * 1e3, cp2_eur_per_mw2=cp2 * 1e6)
        del net.polynomial_cost
    if "piecewise_linear_cost" in net:
        if len(net.piecewise_linear_cost) > 0:
            raise NotImplementedError
        del net.piecewise_linear_cost


def _add_nominal_power(net):
    if "sn_kva" in net:
        net.sn_mva = net.pop("sn_kva") * 1e-3

    # Reset sn_mva only if sn_mva not available
    if "sn_mva" not in net:
        net.sn_mva = 1.0


def _add_missing_tables(net):
    net_new = create_empty_network()
    for key in net_new.keys():
        if key.startswith("_empty_res"):
            net[key] = net_new[key]
        elif key not in net.keys():
            net[key] = net_new[key]


def _create_seperate_cost_tables(net, table_selection):
    if _check_table_selection('gen', table_selection) and "cost_per_kw" in net.gen:
        for index, cost in net.gen.cost_per_kw.iteritems():
            if not np.isnan(cost):
                create_poly_cost(net, index, "gen", cp1_eur_per_mw=cost * 1e3)

    if _check_table_selection('sgen', table_selection) and "cost_per_kw" in net.sgen:
        for index, cost in net.sgen.cost_per_kw.iteritems():
            if not np.isnan(cost):
                create_poly_cost(net, index, "sgen", cp1_eur_per_kw=cost)

    if _check_table_selection('ext_grid', table_selection) and "cost_per_kw" in net.ext_grid:
        for index, cost in net.ext_grid.cost_per_kw.iteritems():
            if not np.isnan(cost):
                create_poly_cost(net, index, "ext_grid", cp1_eur_per_kw=cost)

    if _check_table_selection('gen', table_selection) and "cost_per_kvar" in net.gen:
        for index, cost in net.gen.cost_per_kvar.iteritems():
            if not np.isnan(cost):
                create_poly_cost(net, index, "ext_grid", cp1_eur_per_mw=0,
                                 cq1_eur_per_mvar=cost * 1e3)

    if _check_table_selection('sgen', table_selection) and "cost_per_kvar" in net.sgen:
        for index, cost in net.sgen.cost_per_kvar.iteritems():
            if not np.isnan(cost):
                create_poly_cost(net, index, "sgen", cp1_eur_per_mw=0,
                                 cq1_eur_per_mvar=cost * 1e3)

    if _check_table_selection('ext_grid', table_selection) and "cost_per_kvar" in net.ext_grid:
        for index, cost in net.ext_grid.cost_per_kvar.iteritems():
            if not np.isnan(cost):
                create_poly_cost(net, index, "ext_grid", cp1_eur_per_mw=0,
                                 cq1_eur_per_mvar=cost * 1e3)


def _rename_columns(net, table_selection):
    if _check_table_selection('line', table_selection):
        net.line.rename(columns={'imax_ka': 'max_i_ka'}, inplace=True)
    for typ, data in net.std_types["line"].items():
        if "imax_ka" in data:
            net.std_types["line"][typ]["max_i_ka"] = net.std_types["line"][typ].pop("imax_ka")
    _update_trafo_parameter_names(net, table_selection)
    # initialize measurement dataframe
    if _check_table_selection('measurement', table_selection):
        if "measurement" in net and "type" in net.measurement and "measurement":
            if net.measurement.empty:
                net["measurement"] = create_empty_network()["measurement"]
            else:
                net.measurement["side"] = None
                bus_measurements = net.measurement.element_type == "bus"
                net.measurement.loc[bus_measurements, "element"] = \
                    net.measurement.loc[bus_measurements, "bus"].values
                net.measurement.loc[~bus_measurements, "side"] = \
                    net.measurement.loc[~bus_measurements, "bus"].values
                net.measurement.rename(columns={'type': 'measurement_type'}, inplace=True)
                net.measurement.drop(["bus"], axis=1, inplace=True)
    if _check_table_selection('controller', table_selection):
        if "controller" in net:
            net["controller"].rename(columns={"controller": "object"}, inplace=True)
    if "options" in net:
        if "recycle" in net["options"]:
            if "Ybus" in net["options"]["recycle"]:
                if net["options"]["recycle"]["Ybus"]:
                    net["options"]["recycle"]["trafo"] = False
                del net["options"]["recycle"]["Ybus"]
            else:
                net["options"]["recycle"]["trafo"] = True
            if "ppc" in net["options"]["recycle"]:
                if net["options"]["recycle"]["ppc"]:
                    net["options"]["recycle"]["bus_pq"] = False
                del net["options"]["recycle"]["ppc"]
            else:
                net["options"]["recycle"]["bus_pq"] = True


def _add_missing_columns(net, table_selection):
    update_elements = []
    if _check_table_selection('trafo', table_selection):
        update_elements += ['trafo']
    if _check_table_selection('line', table_selection):
        update_elements += ['line']
    for element in update_elements:
        if "df" not in net[element]:
            net[element]["df"] = 1.0

    if _check_table_selection('bus_geodata', table_selection) and "coords" not in net.bus_geodata:
        net.bus_geodata["coords"] = None
    if _check_table_selection('trafo3w', table_selection) and not "tap_at_star_point" in net.trafo3w:
        net.trafo3w["tap_at_star_point"] = False
    if _check_table_selection('trafo3w', table_selection) and not "tap_step_degree" in net.trafo3w:
        net.trafo3w["tap_step_degree"] = 0
    if _check_table_selection('load', table_selection) and "const_z_percent" not in net.load or "const_i_percent" not in net.load:
        net.load["const_z_percent"] = np.zeros(net.load.shape[0])
        net.load["const_i_percent"] = np.zeros(net.load.shape[0])

    if _check_table_selection('shunt', table_selection) and "vn_kv" not in net["shunt"]:
        net.shunt["vn_kv"] = net.bus.vn_kv.loc[net.shunt.bus.values].values
    if _check_table_selection('shunt', table_selection) and "step" not in net["shunt"]:
        net.shunt["step"] = 1
    if _check_table_selection('shunt', table_selection) and "max_step" not in net["shunt"]:
        net.shunt["max_step"] = 1
    if _check_table_selection('trafo3w', table_selection) and "std_type" not in net.trafo3w:
        net.trafo3w["std_type"] = None

    if _check_table_selection('sgen', table_selection) and "current_source" not in net.sgen:
        net.sgen["current_source"] = net.sgen["type"].apply(
            func=lambda x: False if x == "motor" else True)

    if _check_table_selection('line', table_selection) and "g_us_per_km" not in net.line:
        net.line["g_us_per_km"] = 0.

    if _check_table_selection('gen', table_selection) and "slack" not in net.gen:
        net.gen["slack"] = False

    if _check_table_selection('trafo', table_selection) and "tap_phase_shifter" not in net.trafo and "tp_phase_shifter" not in net.trafo:
        net.trafo["tap_phase_shifter"] = False

    # unsymmetric impedance
    if _check_table_selection('impedance', table_selection) and "r_pu" in net.impedance:
        net.impedance["rft_pu"] = net.impedance["rtf_pu"] = net.impedance["r_pu"]
        net.impedance["xft_pu"] = net.impedance["xtf_pu"] = net.impedance["x_pu"]

    # Update the switch table with 'z_ohm'
    if _check_table_selection('switch', table_selection) and 'z_ohm' not in net.switch:
        net.switch['z_ohm'] = 0

    if _check_table_selection('measurement', table_selection) and "name" not in net.measurement:
        net.measurement.insert(0, "name", None)

    if _check_table_selection('controller', table_selection) and "initial_run" not in net.controller:
        net.controller.insert(4, 'initial_run', False)
        for _, ctrl in net.controller.iterrows():
            if hasattr(ctrl['object'], 'initial_run'):
                net.controller.at[ctrl.name, 'initial_run'] = ctrl['object'].initial_run
            else:
                net.controller.at[ctrl.name, 'initial_run'] = ctrl['object'].initial_powerflow

def _update_trafo_type_parameter_names(net):
    for element in ('trafo', 'trafo3w'):
        for type in net.std_types[element].keys():
            keys = {col: _update_column(col) for col in net.std_types[element][type].keys() if
                    col.startswith("tp") or col.startswith("vsc")}
            for old_key, new_key in keys.items():
                net.std_types[element][type][new_key] = net.std_types[element][type].pop(old_key)


def _update_trafo_parameter_names(net, table_selection):
    update_data = []
    if _check_table_selection('trafo', table_selection):
        update_data += ['trafo']
    if _check_table_selection('trafo3w', table_selection):
        update_data += ['trafo3w']
    for element in update_data:
        replace_cols = {col: _update_column(col) for col in net[element].columns if
                        col.startswith("tp") or col.startswith("vsc")}
        net[element].rename(columns=replace_cols, inplace=True)
    _update_trafo_type_parameter_names(net)


def _update_column(column):
    column = column.replace("tp_", "tap_")
    column = column.replace("_st_", "_step_")
    column = column.replace("_mid", "_neutral")
    column = column.replace("vsc", "vk")
    return column


def _set_data_type_of_columns(net):
    new_net = create_empty_network()
    for key, item in net.items():
        if isinstance(item, pd.DataFrame):
            for col in item.columns:
                if col == "tap_pos":
                    continue
                if key in new_net and col in new_net[key].columns:
                    if set(item.columns) == set(new_net[key]):
                        if version.parse(pd.__version__) < version.parse("0.21"):
                            net[key] = net[key].reindex_axis(new_net[key].columns, axis=1)
                        else:
                            net[key] = net[key].reindex(new_net[key].columns, axis=1)
                    if version.parse(pd.__version__) < version.parse("0.20.0"):
                        net[key][col] = net[key][col].astype(new_net[key][col].dtype,
                                                             raise_on_error=False)
                    else:
                        net[key][col] = net[key][col].astype(new_net[key][col].dtype,
                                                             errors="ignore")


def _convert_to_mw(net):
    replace = [("kw", "mw"), ("kvar", "mvar"), ("kva", "mva")]
    for element, tab in net.items():
        if isinstance(tab, pd.DataFrame):
            for old, new in replace:
                diff = {column: column.replace(old, new) for column in tab.columns if old in column
                        and column != "pfe_kw"}
                tab.rename(columns=diff, inplace=True)
                if len(tab) == 0:
                    continue
                for old, new in diff.items():
                    tab[new] *= 1e-3

    for element, std_types in net.std_types.items():
        for std_type, parameters in std_types.items():
            for parameter in set(parameters.keys()):
                for old, new in replace:
                    if old in parameter and parameter != "pfe_kw":
                        parameters[parameter.replace(old, new)] = parameters[parameter] * 1e-3
                        del parameters[parameter]


def _update_object_attributes(obj):
    """
    Rename attributes of a given object. A new attribute is added and the old one is removed.
    """
    to_rename = {"u_set": "vm_set_pu",
                 "u_lower": "vm_lower_pu",
                 "u_upper": "vm_upper_pu"}

    for key, val in to_rename.items():
        if key in obj.__dict__:
            obj.__dict__[val] = obj.__dict__.pop(key)


def _convert_objects(net, table_selection):
    """
    The function updates attribute names in pandapower objects. For now, it affects TrafoController.
    Should be expanded for other objects if necessary.
    """
    _check_table_selection('controller', table_selection)
    if _check_table_selection('controller', table_selection) and "controller" in net.keys():
        for obj in net["controller"].object.values:
            _update_object_attributes(obj)


def _check_table_selection(element, table_selection):
    if table_selection is None:
        return True
    elif element in table_selection:
        return True
    else:
        return False