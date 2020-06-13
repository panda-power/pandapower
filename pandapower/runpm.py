# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

from pandapower import pp_dir
from pandapower.auxiliary import _add_ppc_options, _add_opf_options
from pandapower.converter.powermodels.from_pm import read_ots_results, read_tnep_results
from pandapower.opf.pm_storage import add_storage_opf_settings, read_pm_storage_results
from pandapower.opf.run_powermodels import _runpm


def runpm(net, julia_file=None, pp_to_pm_callback=None, calculate_voltage_angles=True,
          trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
          correct_pm_network_data=True, pm_model="ACPPowerModel", pm_solver="ipopt",
          pm_mip_solver="cbc", pm_nl_solver="ipopt", pm_time_limits=None, pm_log_level=0):  # pragma: no cover
    """
    Runs a power system optimization using PowerModels.jl. with a custom julia file.
    
    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities can be defined in net.sgen / net.gen /net.load
    net.sgen.controllable if a static generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If True, the following
    flexibilities apply:
        - net.sgen.min_p_mw / net.sgen.max_p_mw
        - net.sgen.min_q_mvar / net.sgen.max_q_mvar
        - net.load.min_p_mw / net.load.max_p_mw
        - net.load.min_q_mvar / net.load.max_q_mvar
        - net.gen.min_p_mw / net.gen.max_p_mw
        - net.gen.min_q_mvar / net.gen.max_q_mvar
        - net.ext_grid.min_p_mw / net.ext_grid.max_p_mw
        - net.ext_grid.min_q_mvar / net.ext_grid.max_q_mvar
        - net.dcline.min_q_to_mvar / net.dcline.max_q_to_mvar / net.dcline.min_q_from_mvar / net.dcline.max_q_from_mvar

    Controllable loads behave just like controllable static generators. It must be stated if they are controllable.
    Otherwise, they are not respected as flexibilities.
    Dc lines are controllable per default

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.bus.min_vm_pu / net.bus.max_vm_pu
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent
        - net.trafo3w.max_loading_percent

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **julia_file** (str, None) - path to a custom julia optimization file

        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

        **correct_pm_network_data** (bool, True) - checks if network data is correct. If not tries to correct it

        **pm_model** (str, "ACPPowerModel") - The PowerModels.jl model to use

        **pm_solver** (str, "ipopt") - The "main" power models solver

        **pm_mip_solver** (str, "cbc") - The mixed integer solver (when "main" solver == juniper)

        **pm_nl_solver** (str, "ipopt") - The nonlinear solver (when "main" solver == juniper)

        **pm_time_limits** (Dict, None) - Time limits in seconds for power models interface. To be set as a dict like
                                          {"pm_time_limit": 300., "pm_nl_time_limit": 300., "pm_mip_time_limit": 300.}
                                          
        **pm_log_level** (int, 0) - solver log level in power models

     """
    net._options = {}
    ac = True if "DC" not in pm_model else False
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels.jl') if julia_file is None else julia_file
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file, pm_solver=pm_solver, pm_model=pm_model,
                     correct_pm_network_data=correct_pm_network_data, pm_mip_solver=pm_mip_solver,
                     pm_nl_solver=pm_nl_solver, pm_time_limits=pm_time_limits, pm_log_level=pm_log_level)
    _runpm(net)


def runpm_dc_opf(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                 trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                 correct_pm_network_data=True, pm_model="DCPPowerModel", pm_solver="ipopt",
                 pm_time_limits=None, pm_log_level=0):  # pragma: no cover
    """
    Runs a linearized power system optimization using PowerModels.jl.

    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities can be defined in net.sgen / net.gen /net.load
    net.sgen.controllable if a static generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If True, the following
    flexibilities apply:
        - net.sgen.min_p_mw / net.sgen.max_p_mw
        - net.sgen.min_q_mvar / net.sgen.max_q_mvar
        - net.load.min_p_mw / net.load.max_p_mw
        - net.load.min_q_mvar / net.load.max_q_mvar
        - net.gen.min_p_mw / net.gen.max_p_mw
        - net.gen.min_q_mvar / net.gen.max_q_mvar
        - net.ext_grid.min_p_mw / net.ext_grid.max_p_mw
        - net.ext_grid.min_q_mvar / net.ext_grid.max_q_mvar
        - net.dcline.min_q_to_mvar / net.dcline.max_q_to_mvar / net.dcline.min_q_from_mvar / net.dcline.max_q_from_mvar

    Controllable loads behave just like controllable static generators. It must be stated if they are controllable.
    Otherwise, they are not respected as flexibilities.
    Dc lines are controllable per default

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.bus.min_vm_pu / net.bus.max_vm_pu
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent
        - net.trafo3w.max_loading_percent

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

        **pm_model** (str, "DCPPowerModel") - model to use. Default is DC model

        **pm_solver** (str, "ipopt") - The "main" power models solver

        **correct_pm_network_data** (bool, True) - checks if network data is correct. If not tries to correct it

        **pm_time_limits** (Dict, None) - Time limits in seconds for power models interface. To be set as a dict like
                                          {"pm_time_limit": 300.}
        
        **pm_log_level** (int, 0) - solver log level in power models
     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels.jl')
    ac = True if "DC" not in pm_model else False

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file,
                     correct_pm_network_data=correct_pm_network_data, pm_model=pm_model, pm_solver=pm_solver,
                     pm_time_limits=pm_time_limits, pm_log_level=pm_log_level)
    _runpm(net)


def runpm_ac_opf(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
                 trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                 pm_model="ACPPowerModel", pm_solver="ipopt", correct_pm_network_data=True,
                 pm_time_limits=None, pm_log_level=0):  # pragma: no cover
    """
    Runs a non-linear power system optimization using PowerModels.jl.

    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities can be defined in net.sgen / net.gen /net.load
    net.sgen.controllable if a static generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If True, the following
    flexibilities apply:
        - net.sgen.min_p_mw / net.sgen.max_p_mw
        - net.sgen.min_q_mvar / net.sgen.max_q_mvar
        - net.load.min_p_mw / net.load.max_p_mw
        - net.load.min_q_mvar / net.load.max_q_mvar
        - net.gen.min_p_mw / net.gen.max_p_mw
        - net.gen.min_q_mvar / net.gen.max_q_mvar
        - net.ext_grid.min_p_mw / net.ext_grid.max_p_mw
        - net.ext_grid.min_q_mvar / net.ext_grid.max_q_mvar
        - net.dcline.min_q_to_mvar / net.dcline.max_q_to_mvar / net.dcline.min_q_from_mvar / net.dcline.max_q_from_mvar

    Controllable loads behave just like controllable static generators. It must be stated if they are controllable.
    Otherwise, they are not respected as flexibilities.
    Dc lines are controllable per default

    Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.bus.min_vm_pu / net.bus.max_vm_pu
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent
        - net.trafo3w.max_loading_percent

    How these costs are combined into a cost function depends on the cost_function parameter.

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:

        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

        **pm_model** (str, "ACPPowerModel") - model to use. Default is AC model

        **pm_solver** (str, "ipopt") - default solver to use. If ipopt is not available use Ipopt

        **correct_pm_network_data** (bool, True) - checks if network data is correct. If not tries to correct it

        **pm_time_limits** (Dict, None) - Time limits in seconds for power models interface. To be set as a dict like
                                          {"pm_time_limit": 300.}

        **pm_log_level** (int, 0) - solver log level in power models                                        
     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels.jl')
    ac = True if "DC" not in pm_model else False

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file, pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level)
    _runpm(net)


def runpm_tnep(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
               trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
               pm_model="DCPPowerModel", pm_solver=None, correct_pm_network_data=True,
               pm_nl_solver="ipopt", pm_mip_solver="cbc", pm_time_limits=None, pm_log_level=0):  # pragma: no cover
    """
    Runs a non-linear transmission network extension planning (tnep) optimization using PowerModels.jl.

    OPTIONAL:
        **julia_file** (str, None) - path to a custom julia optimization file

        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

        **correct_pm_network_data** (bool, True) - checks if network data is correct. If not tries to correct it

        **pm_model** (str, "ACPPowerModel") - The PowerModels.jl model to use

        **pm_solver** (str, "juniper") - The "main" power models solver

        **pm_mip_solver** (str, "cbc") - The mixed integer solver (when "main" solver == juniper)

        **pm_nl_solver** (str, "ipopt") - The nonlinear solver (when "main" solver == juniper)

        **pm_time_limits** (Dict, None) - Time limits in seconds for power models interface. To be set as a dict like
                                          {"pm_time_limit": 300., "pm_nl_time_limit": 300., "pm_mip_time_limit": 300.}

        **pm_log_level** (int, 0) - solver log level in power models
     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_tnep.jl')
    ac = True if "DC" not in pm_model else False
    if pm_solver is None:
        if pm_model == "DCPPowerModel":
            pm_solver = "gurobi"
        else:
            pm_solver = "juniper"

    if "ne_line" not in net:
        raise ValueError("ne_line DataFrame missing in net. Please define to run tnep")
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file, pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, pm_nl_solver=pm_nl_solver,
                     pm_mip_solver=pm_mip_solver, pm_time_limits=pm_time_limits, pm_log_level=pm_log_level)
    _runpm(net)
    read_tnep_results(net)


def runpm_ots(net, pp_to_pm_callback=None, calculate_voltage_angles=True,
              trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
              pm_model="DCPPowerModel", pm_solver="juniper", pm_nl_solver="ipopt", pm_mip_solver="cbc",
              correct_pm_network_data=True, pm_time_limits=None, pm_log_level=0):  # pragma: no cover
    """
    Runs a non-linear optimal transmission switching (OTS) optimization using PowerModels.jl.

    OPTIONAL:
        **julia_file** (str, None) - path to a custom julia optimization file

        **pp_to_pm_callback** (function, None) - callback function to add data to the PowerModels data structure

        **correct_pm_network_data** (bool, True) - checks if network data is correct. If not tries to correct it

        **pm_model** (str, "ACPPowerModel") - The PowerModels.jl model to use

        **pm_solver** (str, "juniper") - The "main" power models solver

        **pm_mip_solver** (str, "cbc") - The mixed integer solver (when "main" solver == juniper)

        **pm_nl_solver** (str, "ipopt") - The nonlinear solver (when "main" solver == juniper)

        **pm_time_limits** (Dict, None) - Time limits in seconds for power models interface. To be set as a dict like
                                          {"pm_time_limit": 300., "pm_nl_time_limit": 300., "pm_mip_time_limit": 300.}

        **pm_log_level** (int, 0) - solver log level in power models
     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_ots.jl')
    ac = True if "DC" not in pm_model else False
    if pm_solver is None:
        pm_solver = "juniper"

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=pp_to_pm_callback, julia_file=julia_file, pm_model=pm_model, pm_solver=pm_solver,
                     correct_pm_network_data=correct_pm_network_data, pm_mip_solver=pm_mip_solver,
                     pm_nl_solver=pm_nl_solver, pm_time_limits=pm_time_limits, pm_log_level=pm_log_level)
    _runpm(net)
    read_ots_results(net)


def runpm_storage_opf(net, calculate_voltage_angles=True,
                      trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                      n_timesteps=24, time_elapsed=1.0, correct_pm_network_data=True,
                      pm_model="ACPPowerModel", pm_time_limits=None, pm_log_level=0):  # pragma: no cover
    """
    Runs a non-linear power system optimization with storages and time series using PowerModels.jl.


    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **n_timesteps** (int, 24) - number of time steps to optimize

        **time_elapsed** (float, 1.0) - time elapsed between time steps (1.0 = 1 hour)

        **pm_time_limits** (Dict, None) - Time limits in seconds for power models interface. To be set as a dict like
                                          {"pm_time_limit": 300., "pm_nl_time_limit": 300., "pm_mip_time_limit": 300.}

        **pm_log_level** (int, 0) - solver log level in power models
     """
    julia_file = os.path.join(pp_dir, "opf", 'run_powermodels_mn_storage.jl')
    ac = True if "DC" not in pm_model else False
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="opf", switch_rx_ratio=2, init_vm_pu="flat", init_va_degree="flat",
                     enforce_q_lims=True, recycle=dict(_is_elements=False, ppc=False, Ybus=False),
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading='power', ac=ac, init="flat", numba=True,
                     pp_to_pm_callback=add_storage_opf_settings, julia_file=julia_file,
                     correct_pm_network_data=correct_pm_network_data, pm_model=pm_model, pm_time_limits=pm_time_limits,
                     pm_log_level=pm_log_level)

    net._options["n_time_steps"] = n_timesteps
    net._options["time_elapsed"] = time_elapsed

    _runpm(net)
    storage_results = read_pm_storage_results(net)
    return storage_results
