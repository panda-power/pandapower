# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
from pandapower.toolbox import get_connected_buses_at_element

# separator between log messages
log_message_sep = ("\n --------\n")


def diagnostic_report(net, diag_results, diag_errors, diag_params, compact_report, warnings_only):
    diag_report = DiagnosticReports(net, diag_results, diag_errors, diag_params, compact_report)
    if warnings_only:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(logging.INFO)

    report_methods = {
        "missing_bus_indices": diag_report.report_missing_bus_indices,
        "disconnected_elements": diag_report.report_disconnected_elements,
        "different_voltage_levels_connected": diag_report.report_different_voltage_levels_connected,
        "impedance_values_close_to_zero": diag_report.report_impedance_values_close_to_zero,
        "nominal_voltages_dont_match": diag_report.report_nominal_voltages_dont_match,
        "invalid_values": diag_report.report_invalid_values,
        "overload": diag_report.report_overload,
        "multiple_voltage_controlling_elements_per_bus": diag_report.report_multiple_voltage_controlling_elements_per_bus,
        "wrong_switch_configuration": diag_report.report_wrong_switch_configuration,
        "no_ext_grid": diag_report.report_no_ext_grid,
        "wrong_reference_system": diag_report.report_wrong_reference_system,
        "deviation_from_std_type": diag_report.report_deviation_from_std_type,
        "numba_comparison": diag_report.report_numba_comparison,
        "parallel_switches": diag_report.report_parallel_switches
    }

    logger.warning("\n\n_____________ PANDAPOWER DIAGNOSTIC TOOL _____________ \n")
    for key in report_methods:
        if (key in diag_results) or not warnings_only:
            report_methods[key]()
            logger.warning(log_message_sep)

    logger.warning("_____________ END OF PANDAPOWER DIAGNOSTIC _____________ ")


class DiagnosticReports:
    def __init__(self, net, diag_results, diag_errors, diag_params, compact_report):
        self.net = net
        self.diag_results = diag_results
        self.diag_errors = diag_errors
        self.diag_params = diag_params
        self.compact_report = compact_report

    def report_disconnected_elements(self):
        if "disconnected_elements" in self.diag_results:
            # message header
            if self.compact_report:
                logger.warning("disconnected_elements:")
            else:
                logger.warning("Checking for elements without a connection to an external grid...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["disconnected_elements"]
            element_counter = 0
            for disc_section in diag_result:
                if self.compact_report:
                    logger.warning("disonnected_section: %s" % (disc_section))

                else:
                    logger.warning("Disconnected section found,"
                                   " consisting of the following elements:")
                    for key in disc_section:
                        element_counter += len(disc_section[key])
                        logger.warning("%s: %s" % (key, disc_section[key]))

            # message summary
            if not self.compact_report:
                logger.warning("")
                logger.warning("SUMMARY: %s disconnected element(s) found." % (element_counter))
        elif "disconnected_elements" in self.diag_errors:
            logger.warning("Check for disconnected elements failed due to the following error:")
            logger.warning(self.diag_errors["disconnected_elements"])
        else:
            logger.info("PASSED: No problematic switches found")

    def report_different_voltage_levels_connected(self):

        if "different_voltage_levels_connected" in self.diag_results:

            # message header
            if self.compact_report:
                logger.warning("different_voltage_levels_connected:")

            else:
                logger.warning("Checking for connections of different voltage levels...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["different_voltage_levels_connected"]
            element_counter = 0
            for key in diag_result:
                element_counter += len(diag_result[key])
                if key == "lines":
                    element_type = "line"
                elif key == "switches":
                    element_type = "switch"
                if self.compact_report:
                    logger.warning("%s:" % (key))
                for element in diag_result[key]:
                    buses = list(get_connected_buses_at_element(self.net, element,
                                                                key[0]))
                    if self.compact_report:
                        logger.warning("%s %s: buses %s" % (element_type, element, buses))
                    else:
                        logger.warning("%s %s connects bus %s: %s (vn_kv = %s) "
                                       "and bus %s: %s (vn_kv = %s)"
                                       % (element_type, element, buses[0],
                                          self.net.bus.name.at[buses[0]],
                                          self.net.bus.vn_kv.at[buses[0]],
                                          buses[1], self.net.bus.name.at[buses[1]],
                                          self.net.bus.vn_kv.at[buses[1]]))
                        # message summary
            if not self.compact_report:
                logger.warning("")
                logger.warning("SUMMARY: %s element(s) that connect different voltage "
                               "levels found." % (element_counter))
        elif "different_voltage_levels_connected" in self.diag_errors:
            logger.warning("Check for connection of different voltage levels failed due to the following error:")
            logger.warning(self.diag_errors["different_voltage_levels_connected"])
        else:
            logger.info("PASSED: No connection of different voltage levels found")


    def report_impedance_values_close_to_zero(self):

        if "impedance_values_close_to_zero" in self.diag_results:

            # message header
            if self.compact_report:
                logger.warning("impedance_values_close_to_zero:")

            else:
                logger.warning("Checking for impedance values close to zero...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["impedance_values_close_to_zero"][0]
            element_counter = 0
            for key in diag_result:
                element_counter += len(diag_result[key])
                for element in diag_result[key]:
                    if key == "line":
                        min_r_type = "r_ohm"
                        min_x_type = "x_ohm"
                    elif key == "xward":
                        min_r_type = "r_ohm"
                        min_x_type = "x_ohm"
                    elif key == "impedance":
                        min_r_type = "r_pu"
                        min_x_type = "x_pu"

                    logger.warning("%s %s: %s <= %s or %s <= %s"
                                   % (key, element, min_r_type, self.diag_params["min_"+min_r_type],
                                      min_x_type, self.diag_params["min_"+min_x_type]))

            if len(self.diag_results["impedance_values_close_to_zero"]) > 1:
                switch_replacement = self.diag_results["impedance_values_close_to_zero"][1]
                if switch_replacement["loadflow_converges_with_switch_replacement"]:
                    logger.warning("Switch replacement successful: Power flow converges after "
                                   "replacing implausible elements with switches.")
                else:
                    logger.warning("Power flow still does not converge after replacing implausible "
                                   "elements with switches.")

            # message summary
            if not self.compact_report:
                logger.warning("")
                logger.warning("SUMMARY: %s element(s) with impedance values close to zero found."
                               % (element_counter))
        elif "impedance_values_close_to_zero" in self.diag_errors:
            logger.warning("Check for elements with impedance values close to zero failed due "
                           "to the following error:")
            logger.warning(self.diag_errors["impedance_values_close_to_zero"])
        else:
            logger.info("PASSED: No elements with impedance values close to zero found...")


    def report_nominal_voltages_dont_match(self):

        if "nominal_voltages_dont_match" in self.diag_results:

            # message header
            if self.compact_report:
                logger.warning("nominal_voltages_dont_match:")

            else:
                logger.warning("Checking for components with deviating nominal voltages...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["nominal_voltages_dont_match"]
            nom_voltage_tolerance = self.diag_params["nom_voltage_tolerance"]
            element_counter = 0
            for element in diag_result:
                if self.compact_report:
                    logger.warning("%s:" % (element))
                for key in diag_result[element]:
                    element_counter += len(diag_result[element][key])
                    if element == "trafo":
                        if self.compact_report:
                            logger.warning("%s: %s" % (key, diag_result[element][key]))
                        else:
                            if key == "hv_lv_swapped":
                                logger.warning("Trafo(s) %s: hv and lv connectors seem to "
                                               "be swapped" % (diag_result[element][key]))
                            elif key == "hv_bus":
                                for trafo in diag_result[element][key]:
                                    logger.warning("Trafo %s: Nominal voltage on hv_side "
                                                   "(%s kV) and voltage_level of hv_bus "
                                                   "(bus %s with voltage_level %s kV) "
                                                   "deviate more than +/- %s percent."
                                                   % (trafo, self.net.trafo.vn_hv_kv.at[trafo],
                                                      self.net.trafo.hv_bus.at[trafo],
                                                      self.net.bus.vn_kv.at[self.net.trafo.hv_bus.at[trafo]],
                                                      nom_voltage_tolerance * 100))
                            elif key == "lv_bus":
                                for trafo in diag_result[element][key]:
                                    logger.warning("Trafo %s: Nominal voltage on lv_side "
                                                   "(%s kV) and voltage_level of lv_bus "
                                                   "(bus %s with voltage_level %s kV) "
                                                   "deviate more than +/- %s percent."
                                                   % (trafo, self.net.trafo.vn_lv_kv.at[trafo],
                                                      self.net.trafo.lv_bus.at[trafo],
                                                      self.net.bus.vn_kv.at[self.net.trafo.lv_bus.at[trafo]],
                                                      nom_voltage_tolerance * 100))
                    if element == "trafo3w":
                        if self.compact_report:
                            logger.warning("%s: %s" % (key, diag_result[element][key]))
                        else:
                            if key == "connectors_swapped_3w":
                                logger.warning("Trafo3w %s: connectors seem to "
                                               "be swapped" % (diag_result[element][key]))
                            elif key == "hv_bus":
                                for trafo3w in diag_result[element][key]:
                                    logger.warning("Trafo3w %s: Nominal voltage on hv_side "
                                                   "(%s kV) and voltage_level of hv_bus "
                                                   "(bus %s with voltage_level %s kV) "
                                                   "deviate more than +/- %s percent."
                                                   % (trafo3w, self.net.trafo3w.vn_hv_kv.at[trafo3w],
                                                      self.net.trafo3w.hv_bus.at[trafo3w],
                                                      self.net.bus.vn_kv.at[self.net.trafo3w.hv_bus.at[trafo3w]],
                                                      nom_voltage_tolerance * 100))
                            elif key == "mv_bus":
                                for trafo3w in diag_result[element][key]:
                                    logger.warning("Trafo3w %s: Nominal voltage on mv_side "
                                                   "(%s kV) and voltage_level of mv_bus "
                                                   "(bus %s with voltage_level %s kV) "
                                                   "deviate more than +/- %s percent."
                                                   % (trafo3w, self.net.trafo3w.vn_mv_kv.at[trafo3w],
                                                      self.net.trafo3w.mv_bus.at[trafo3w],
                                                      self.net.bus.vn_kv.at[self.net.trafo3w.mv_bus.at[trafo3w]],
                                                      nom_voltage_tolerance * 100))
                            elif key == "lv_bus":
                                for trafo3w in diag_result[element][key]:
                                    logger.warning("Trafo3w %s: Nominal voltage on lv_side "
                                                   "(%s kV) and voltage_level of lv_bus "
                                                   "(bus %s with voltage_level %s kV) "
                                                   "deviate more than +/- %s percent."
                                                   % (trafo3w, self.net.trafo3w.vn_lv_kv.at[trafo3w],
                                                      self.net.trafo3w.lv_bus.at[trafo3w],
                                                      self.net.bus.vn_kv.at[self.net.trafo3w.lv_bus.at[trafo3w]],
                                                      nom_voltage_tolerance * 100))

                                    # message summary
            if not self.compact_report:
                logger.warning("")
                logger.warning("SUMMARY: %s component(s) with deviating nominal voltages found"
                               % (element_counter))
        elif "nominal_voltages_dont_match" in self.diag_errors:
            logger.warning("Check for components with deviating nominal voltages failed due "
                           "to the following error:")
            logger.warning(self.diag_errors["nominal_voltages_dont_match"])
        else:
            logger.info("PASSED: No components with deviating nominal voltages found")

    def report_invalid_values(self):

        if "invalid_values" in self.diag_results:

            # message header
            if self.compact_report:
                logger.warning("invalid_values:")
            else:
                logger.warning("Checking for invalid_values...")
            logger.warning("")

        # message body
        if "invalid_values" in self.diag_results:
            diag_result = self.diag_results["invalid_values"]
            element_counter = 0
            for element_type in diag_result:
                element_counter += len(diag_result[element_type])
                logger.warning("%s:" % (element_type))
                for inv_value in diag_result[element_type]:
                    if self.compact_report:
                        logger.warning("%s %s: '%s' = %s (restriction: %s)"
                                       % (element_type, inv_value[0], inv_value[1], inv_value[2],
                                          inv_value[3]))
                    else:
                        logger.warning("Invalid value found: '%s %s' with attribute '%s' = %s "
                                       "(data type: %s). Valid input needs to be %s."
                                       % (element_type, inv_value[0], inv_value[1], inv_value[2],
                                          type(inv_value[2]), inv_value[3]))

                        # message summary
            if not self.compact_report:
                logger.warning("")
                logger.warning("SUMMARY: %s invalid values found." % element_counter)

        elif "invalid_values" in self.diag_errors:
            logger.warning("Check for invalid values failed due to the following error:")
            logger.warning(self.diag_errors["invalid_values"])
        else:
            logger.info("PASSED: No invalid values found")


    def report_overload(self):

        if "overload" in self.diag_results:
            # message header
            if self.compact_report:
                logger.warning("overload:")
            else:
                logger.warning("Checking for overload...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["overload"]
            overload_scaling_factor = self.diag_params["overload_scaling_factor"]
            if not diag_result["load"] and not diag_result["generation"]:
                logger.warning("Overload check failed: Power flow still does not "
                               "converge with load and generation scaled down to %s percent."
                               % (overload_scaling_factor * 100))
            elif (diag_result["load"] and diag_result["generation"]):
                logger.warning("overload found: Power flow converges "
                               "with load and generation scaled down to %s percent."
                               % (overload_scaling_factor * 100))
            else:
                if diag_result["load"]:
                    logger.warning("overload found: Power flow converges "
                                   "with load scaled down to %s percent."
                                   % (overload_scaling_factor * 100))
                elif diag_result["generation"]:
                    logger.warning("overload found: Power flow converges "
                                   "with generation scaled down to %s percent."
                                   % (overload_scaling_factor * 100))
        # message summary
        elif "overload" in self.diag_errors:
            logger.warning("Check for overload failed due to the following error:")
            logger.warning(self.diag_errors["overload"])
        else:
            logger.info("PASSED: Power flow converges. No overload found.")


    def report_wrong_switch_configuration(self):

        if "wrong_switch_configuration" in self.diag_results:

            # message header
            if self.compact_report:
                logger.warning("wrong_switch_configuration:")
            else:
                logger.warning("Checking switch configuration...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["wrong_switch_configuration"]
            if diag_result:
                logger.warning("Possibly wrong switch configuration found: power flow "
                               "converges with all switches closed.")
            else:
                logger.warning("Power flow still does not converge with all switches closed.")

        # message summary
        elif "wrong_switch_configuration" in self.diag_errors:
            logger.warning("Check for wrong switch configuration failed due to the following error:")
            logger.warning(self.diag_errors["wrong_switch_configuration"])
        else:
            logger.info("PASSED: Power flow converges. Switch configuration seems ok.")


    def report_no_ext_grid(self):

        if "no_ext_grid" in self.diag_results:
            # message header
            if self.compact_report:
                logger.warning("no_external_grid:")
            else:
                logger.warning("Checking if there is at least one external grid...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["no_ext_grid"]
            if diag_result is True:
                logger.warning("No ext_grid found. There has to be at least one ext_grid!")

        # message summary
        elif "no_ext_grid" in self.diag_errors:
            logger.warning("Check for external grid failed due to the following error:")
            logger.warning(self.diag_errors["no_ext_grid"])
        else:
            logger.info("PASSED: External grid found.")


    def report_multiple_voltage_controlling_elements_per_bus(self):

        if "multiple_voltage_controlling_elements_per_bus" in self.diag_results:
            # message header
            if self.compact_report:
                logger.warning("multiple_voltage_controlling_elements_per_bus:")
            else:
                logger.warning("Checking for multiple gens and/or external grids per bus...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["multiple_voltage_controlling_elements_per_bus"]
            element_counter = 0
            for feeder_type in diag_result:
                element_counter += len(diag_result[feeder_type])
                if self.compact_report:
                    logger.warning("%s: %s" % (feeder_type, diag_result[feeder_type]))

                else:
                    for bus in diag_result[feeder_type]:
                        if feeder_type is "buses_with_mult_ext_grids":
                            logger.warning("External grids %s are connected to bus %s. Only one "
                                           "external grid per bus is allowed."
                                           % (list(self.net.ext_grid[self.net.ext_grid.bus
                                                                     == bus].index), bus))
                        elif feeder_type is "buses_with_gens_and_ext_grids":
                            logger.warning("Generator(s) %s and external grid(s) %s are connected "
                                           "to bus %s. Only one generator OR one external grid "
                                           "per bus is allowed."
                                           % (list(self.net.gen[self.net.gen.bus == bus].index),
                                              list(self.net.ext_grid[self.net.ext_grid.bus
                                                                     == bus].index), bus))

                            # message summary
            if not self.compact_report:
                logger.warning("")
                logger.warning("SUMMARY: %s bus(ses) with multiple gens and/or ext_grids "
                               "found." % (element_counter))
        elif "multiple_voltage_controlling_elements_per_bus" in self.diag_errors:
            logger.warning("Check for multiple voltage controlling elements per bus failed due "
                           "to the following error:")
            logger.warning(self.diag_errors["multiple_voltage_controlling_elements_per_bus"])
        else:
            logger.info("PASSED: No buses with multiple gens and/or ext_grids found.")


    def report_wrong_reference_system(self):

        if "wrong_reference_system" in self.diag_results:

            # message header
            if self.compact_report:
                logger.warning("wrong_reference_system:")
            else:
                logger.warning("Checking for usage of wrong reference system...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["wrong_reference_system"]
            for element_type in diag_result:
                if element_type is "loads":
                    if self.compact_report:
                        logger.warning("loads %s: wrong reference system."
                                       % (diag_result[element_type]))
                    else:
                        for load in diag_result[element_type]:
                            logger.warning("Found load %s: '%s' with p_mw = %s. In load reference "
                                           "system p_mw should be positive."
                                           % (load, self.net.load.name.at[load],
                                              self.net.load.p_mw.at[load]))

                elif element_type is "gens":
                    if self.compact_report:
                        logger.warning("gens %s: wrong reference system."
                                       % (diag_result[element_type]))
                    else:
                        for gen in diag_result[element_type]:
                            logger.warning("Found gen %s: '%s' with p_mw = %s. In load reference "
                                           "system p_mw should be negative."
                                           % (gen, self.net.gen.name.at[gen], self.net.gen.p_mw.at[gen]))

                elif element_type is "sgens":
                    if self.compact_report:
                        logger.warning("sgens %s: wrong reference system."
                                       % (diag_result[element_type]))
                    else:
                        for sgen in diag_result[element_type]:
                            logger.warning("Found sgen %s: '%s' with p_mw = %s. In load reference "
                                           "system p_mw should be negative."
                                           % (sgen, self.net.sgen.name.at[sgen], self.net.sgen.p_mw.at[sgen]))

                            # message summary
            if not self.compact_report:
                logger.warning("")
                if 'loads' in diag_result:
                    logger.warning("SUMMARY: Found %s load(s) with negative p_mw. In load "
                                   "reference system, p_mw should be positive. If the intention "
                                   "was to model a constant generation, please use an sgen instead."
                                   % (len(diag_result['loads'])))
                if 'gens' in diag_result:
                    logger.warning("SUMMARY: Found %s gen(s) with positive p_mw. In load "
                                   "reference system, p_mw should be negative. If the intention "
                                   "was to model a load, please use a load instead."
                                   % (len(diag_result['gens'])))
                if 'sgens' in diag_result:
                    logger.warning("SUMMARY: Found %s sgen(s) with positive p_mw. In load "
                                   "reference system, p_mw should be negative. If the intention "
                                   "was to model a load, please use a load instead."
                                   % (len(diag_result['sgens'])))

        elif "wrong_reference_system" in self.diag_errors:
            logger.warning("Check for wrong reference system failed due to the following error:")
            logger.warning(self.diag_errors["wrong_reference_system"])
        else:
            logger.info("PASSED: correct reference system")


    def report_deviation_from_std_type(self):

        if "deviation_from_std_type" in self.diag_results:
            # message header
            if self.compact_report:
                logger.warning("deviation_from_std_type:")
            else:
                logger.warning("Checking for deviation from std type...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["deviation_from_std_type"]
            element_counter = 0
            for et in diag_result:
                for eid in diag_result[et]:
                    element_counter += 1
                    values = diag_result[et][eid]
                    if values['std_type_in_lib']:
                        logger.warning("%s %s: %s = %s, std_type_value = %s"
                                       % (et, eid, values['param'], values['e_value'],
                                          values['std_type_value']))
                    else:
                        logger.warning("%s %s: No valid std_type or std_type not in net.std_types"
                                       % (et, eid))

                        # message summary
            if not self.compact_report:
                logger.warning("")
                logger.warning("SUMMARY: %s elements with deviations from std_type found."
                               % (element_counter))

        elif "deviation_from_std_type" in self.diag_errors:
            logger.warning("Check for deviation from std_type failed due to the following error:")
            logger.warning(self.diag_errors["deviation_from_std_type"])
        else:
            logger.info("PASSED: No elements with deviations from std_type found.")

    def report_numba_comparison(self):

        if "numba_comparison" in self.diag_results:
            # message header
            if self.compact_report:
                logger.warning("numba_comparison:")
            else:
                logger.warning("Checking for deviations between numba = True vs. False...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["numba_comparison"]
            element_counter = 0
            for element_type in diag_result:
                for res_type in diag_result[element_type]:
                    if self.compact_report:
                        logger.warning("%s.%s absolute deviations:\n%s"
                                       % (element_type, res_type,
                                          diag_result[element_type][res_type]))
                    else:
                        for idx in diag_result[element_type][res_type].index:
                            element_counter += 1
                            dev = diag_result[element_type][res_type].loc[idx]
                            logger.warning("%s.%s at index %s: absolute deviation = %s"
                                           % (element_type, res_type, idx, dev))

                                # message summary
                if not self.compact_report:
                    logger.warning("")
                    logger.warning("SUMMARY: %s results with deviations between numba = True vs. \
                                    False found." % (element_counter))

        elif "numba_comparison" in self.diag_errors:
            logger.warning("numba_comparison failed due to the following error:")
            logger.warning(self.diag_errors["numba_comparison"])
        else:
            logger.info("PASSED: No results with deviations between numba = True vs. False found.")


    def report_parallel_switches(self):

        if "parallel_switches" in self.diag_results:
            # message header
            if self.compact_report:
                logger.warning("parallel_switches:")
            else:
                logger.warning("Checking for parallel switches...")
            logger.warning("")

            # message body
            diag_result = self.diag_results["parallel_switches"]
            for switch_tuple in diag_result:
                logger.warning("switches %s are parallel." % switch_tuple)

                # message summary
            if not self.compact_report:
                logger.warning("")
                logger.warning("SUMMARY: %s occurences of parallel switches found."
                               % len(diag_result))
        elif "parallel_switches" in self.diag_errors:
            logger.warning("Check for parallel_switches failed due to the following error:")
            logger.warning(self.diag_errors["parallel_switches"])
        else:
            logger.info("PASSED: No parallel switches found.")


    def report_missing_bus_indices(self):
        # message header
        if self.compact_report:
            logger.info("missing_bus_indices:")
        else:
            logger.info("Checking for missing bus indices...")
        logger.info("")
        if "missing_bus_indices" in self.diag_results:
            # message body
            diag_result = self.diag_results["missing_bus_indices"]
            element_counter = 0
            for element_type in diag_result:
                for element in diag_result[element_type]:
                    element_counter += 1
                    logger.warning("%s %s: %s (%s) not in net.bus.index" % (element_type,
                                                                            element[0], element[1], element[2]))

                    # message summary
            if not self.compact_report:
                logger.warning("")
                logger.warning("SUMMARY: %s missing bus indices found." % element_counter)

        elif "missing_bus_indices" in self.diag_errors:
            logger.warning("Check for missing bus indices failed due to the following error:")
            logger.warning(self.diag_errors["missing_bus_indices"])
        else:
            logger.info("PASSED: No missing bus indices found.")