# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import inspect

from pandapower.auxiliary import _check_bus_index_and_print_warning_if_high, \
    _check_gen_index_and_print_warning_if_high, _init_runpp_options, _init_rundcopp_options, \
    _init_rundcpp_options, _init_runopp_options, _internal_stored
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters
from pandapower.optimal_powerflow import _optimal_powerflow
from pandapower.powerflow import _powerflow, _recycled_powerflow

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def set_user_pf_options(net, overwrite=False, **kwargs):
    """
    This function sets the 'user_pf_options' dict for net. These options overrule
    net.__internal_options once they are added to net. These options are used in configuration of
    load flow calculation.
    At the same time, user-defined arguments for pandapower.runpp() always have a higher priority.
    To remove user_pf_options, set overwrite=True and provide no additional arguments

    :param net: pandaPower network
    :param overwrite: specifies whether the user_pf_options is removed before setting new options
    :param kwargs: load flow options, e. g. tolerance_mva = 1e-3
    :return: None
    """
    standard_parameters = ['calculate_voltage_angles', 'trafo_model', 'check_connectivity', 'mode',
                           'copy_constraints_to_ppc', 'switch_rx_ratio', 'enforce_q_lims',
                           'recycle', 'voltage_depend_loads', 'consider_line_temperature', 'delta',
                           'trafo3w_losses', 'init_vm_pu', 'init_va_degree', 'init_results',
                           'tolerance_mva', 'trafo_loading', 'numba', 'ac', 'algorithm',
                           'max_iteration', 'v_debug', 'run_control']

    if overwrite or 'user_pf_options' not in net.keys():
        net['user_pf_options'] = dict()

    net.user_pf_options.update({key: val for key, val in kwargs.items()
                                if key in standard_parameters})

    additional_kwargs = {key: val for key, val in kwargs.items()
                         if key not in standard_parameters}

    # this part is to inform user and to make typos in parameters visible
    if len(additional_kwargs) > 0:
        logger.info('parameters %s are not in the list of standard options' % list(
            additional_kwargs.keys()))

        net.user_pf_options.update(additional_kwargs)


def runpp(net, algorithm='nr', calculate_voltage_angles="auto", init="auto",
          max_iteration="auto", tolerance_mva=1e-8, trafo_model="t",
          trafo_loading="current", enforce_q_lims=False, check_connectivity=True,
          voltage_depend_loads=True, consider_line_temperature=False,
          run_control=False, **kwargs):
    """
    Runs a power flow

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **algorithm** (str, "nr") - algorithm that is used to solve the power flow problem.

            The following algorithms are available:

                - "nr" Newton-Raphson (pypower implementation with numba accelerations)
                - "iwamoto_nr" Newton-Raphson with Iwamoto multiplier (maybe slower than NR but more robust)
                - "bfsw" backward/forward sweep (specially suited for radial and weakly-meshed networks)
                - "gs" gauss-seidel (pypower implementation)
                - "fdbx" fast-decoupled (pypower implementation)
                - "fdxb" fast-decoupled (pypower implementation)

        **calculate_voltage_angles** (str or bool, "auto") - consider voltage angles in loadflow calculation

            If True, voltage angles of ext_grids and transformer shifts are considered in the
            loadflow calculation. Considering the voltage angles is only necessary in meshed
            networks that are usually found in higher voltage levels. calculate_voltage_angles
            in "auto" mode defaults to:

                - True, if the network voltage level is above 70 kV
                - False otherwise

            The network voltage level is defined as the maximum rated voltage of any bus in the network that
            is connected to a line.

        **init** (str, "auto") - initialization method of the loadflow
        pandapower supports four methods for initializing the loadflow:

            - "auto" - init defaults to "dc" if calculate_voltage_angles is True or "flat" otherwise
            - "flat"- flat start with voltage of 1.0pu and angle of 0° at all PQ-buses and 0° for PV buses as initial solution, the slack bus is initialized with the values provided in net["ext_grid"]
            - "dc" - initial DC loadflow before the AC loadflow. The results of the DC loadflow are used as initial solution for the AC loadflow. Note that the DC loadflow only calculates voltage angles at PQ and PV buses, voltage magnitudes are still flat started.
            - "results" - voltage vector of last loadflow from net.res_bus is used as initial solution. This can be useful to accelerate convergence in iterative loadflows like time series calculations.

        Considering the voltage angles might lead to non-convergence of the power flow in flat start.
        That is why in "auto" mode, init defaults to "dc" if calculate_voltage_angles is True or "flat" otherwise

        **max_iteration** (int, "auto") - maximum number of iterations carried out in the power flow algorithm.

            In "auto" mode, the default value depends on the power flow solver:

                - 10 for "nr"
                - 100 for "bfsw"
                - 1000 for "gs"
                - 30 for "fdbx"
                - 30 for "fdxb"

        **tolerance_mva** (float, 1e-8) - loadflow termination condition referring to P / Q mismatch of node power in MVA

        **trafo_model** (str, "t")  - transformer equivalent circuit model
        pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modeled as equivalent with the T-model.
            - "pi" - transformer is modeled as equivalent PI-model. This is not recommended, since it is less exact than the T-model. It is only recommended for valdiation with other software that uses the pi-model.

        **trafo_loading** (str, "current") - mode of calculation for transformer loading

            Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer.

            - "current"- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current.
            - "power" - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer.

        **enforce_q_lims** (bool, False) - respect generator reactive power limits

            If True, the reactive power limits in net.gen.max_q_mvar/min_q_mvar are respected in the
            loadflow. This is done by running a second loadflow if reactive power limits are
            violated at any generator, so that the runtime for the loadflow will increase if reactive
            power has to be curtailed.

            Note: enforce_q_lims only works if algorithm="nr"!


        **check_connectivity** (bool, True) - Perform an extra connectivity test after the conversion from pandapower to PYPOWER

            If True, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is perfomed.
            If check finds unsupplied buses, they are set out of service in the ppc

        **voltage_depend_loads** (bool, True) - consideration of voltage-dependent loads. If False, net.load.const_z_percent and net.load.const_i_percent are not considered, i.e. net.load.p_mw and net.load.q_mvar are considered as constant-power loads.

        **consider_line_temperature** (bool, False) - adjustment of line impedance based on provided
            line temperature. If True, net.line must contain a column "temperature_degree_celsius".
            The temperature dependency coefficient alpha must be provided in the net.line.alpha
            column, otherwise the default value of 0.004 is used


        **KWARGS:

        **numba** (bool, True) - Activation of numba JIT compiler in the newton solver

            If set to True, the numba JIT compiler is used to generate matrices for the powerflow,
            which leads to significant speed improvements.

        **switch_rx_ratio** (float, 2) - rx_ratio of bus-bus-switches. If impedance is zero, buses connected by a closed bus-bus switch are fused to model an ideal bus. Otherwise, they are modelled as branches with resistance defined as z_ohm column in switch table and this parameter

        **delta_q** - Reactive power tolerance for option "enforce_q_lims" in kvar - helps convergence in some cases.

        **trafo3w_losses** - defines where open loop losses of three-winding transformers are considered. Valid options are "hv", "mv", "lv" for HV/MV/LV side or "star" for the star point.

        **v_debug** (bool, False) - if True, voltage values in each newton-raphson iteration are logged in the ppc

        **init_vm_pu** (string/float/array/Series, None) - Allows to define initialization specifically for voltage magnitudes. Only works with init == "auto"!

            - "auto": all buses are initialized with the mean value of all voltage controlled elements in the grid
            - "flat" for flat start from 1.0
            - "results": voltage magnitude vector is taken from result table
            - a float with which all voltage magnitudes are initialized
            - an iterable with a voltage magnitude value for each bus (length and order has to match with the buses in net.bus)
            - a pandas Series with a voltage magnitude value for each bus (indexes have to match the indexes in net.bus)

        **init_va_degree** (string/float/array/Series, None) - Allows to define initialization specifically for voltage angles. Only works with init == "auto"!

            - "auto": voltage angles are initialized from DC power flow if angles are calculated or as 0 otherwise
            - "dc": voltage angles are initialized from DC power flow
            - "flat" for flat start from 0
            - "results": voltage angle vector is taken from result table
            - a float with which all voltage angles are initialized
            - an iterable with a voltage angle value for each bus (length and order has to match with the buses in net.bus)
            - a pandas Series with a voltage angle value for each bus (indexes have to match the indexes in net.bus)

        **recycle** (dict, none) - Reuse of internal powerflow variables for time series calculation

            Contains a dict with the following parameters:
            bus_pq: If True PQ values of buses are updated
            trafo: If True trafo relevant variables, e.g., the Ybus matrix, is recalculated
            gen: If True Sbus and the gen table in the ppc are recalculated

        **neglect_open_switch_branches** (bool, False) - If True no auxiliary buses are created for branches when switches are opened at the branch. Instead branches are set out of service

    """

    # if dict 'user_pf_options' is present in net, these options overrule the net.__internal_options
    # except for parameters that are passed by user
    recycle = kwargs.get("recycle", None)
    if isinstance(recycle, dict) and _internal_stored(net):
        _recycled_powerflow(net, **kwargs)
        return

    if run_control and net.controller.in_service.any():
        from pandapower.control import run_control
        parameters = {**locals(), **kwargs}
        # disable run control for inner loop to avoid infinite loop
        parameters["run_control"] = False
        run_control(**parameters)
    else:
        passed_parameters = _passed_runpp_parameters(locals())
        _init_runpp_options(net, algorithm=algorithm, calculate_voltage_angles=calculate_voltage_angles,
                            init=init, max_iteration=max_iteration, tolerance_mva=tolerance_mva,
                            trafo_model=trafo_model, trafo_loading=trafo_loading,
                            enforce_q_lims=enforce_q_lims, check_connectivity=check_connectivity,
                            voltage_depend_loads=voltage_depend_loads,
                            consider_line_temperature=consider_line_temperature,
                            passed_parameters=passed_parameters, **kwargs)
        _check_bus_index_and_print_warning_if_high(net)
        _check_gen_index_and_print_warning_if_high(net)
        _powerflow(net, **kwargs)


def rundcpp(net, trafo_model="t", trafo_loading="current", recycle=None, check_connectivity=True,
            switch_rx_ratio=2, trafo3w_losses="hv", **kwargs):
    """
    Runs PANDAPOWER DC Flow

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **trafo_model** (str, "t")  - transformer equivalent circuit model
        pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modeled as equivalent with the T-model. This is consistent with PowerFactory and is also more accurate than the PI-model. We recommend using this transformer model.
            - "pi" - transformer is modeled as equivalent PI-model. This is consistent with Sincal, but the method is questionable since the transformer is physically T-shaped. We therefore recommend the use of the T-model.

        **trafo_loading** (str, "current") - mode of calculation for transformer loading

            Transformer loading can be calculated relative to the rated current or the rated power. In both cases the overall transformer loading is defined as the maximum loading on the two sides of the transformer.

            - "current"- transformer loading is given as ratio of current flow and rated current of the transformer. This is the recommended setting, since thermal as well as magnetic effects in the transformer depend on the current.
            - "power" - transformer loading is given as ratio of apparent power flow to the rated apparent power of the transformer.

        **check_connectivity** (bool, False) - Perform an extra connectivity test after the conversion from pandapower to PYPOWER

            If true, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is perfomed.
            If check finds unsupplied buses, they are put out of service in the PYPOWER matrix

        **switch_rx_ratio** (float, 2) - rx_ratio of bus-bus-switches. If impedance is zero, buses connected by a closed bus-bus switch are fused to model an ideal bus. Otherwise, they are modelled as branches with resistance defined as z_ohm column in switch table and this parameter

        **trafo3w_losses** (str, "hv") - defines where open loop losses of three-winding transformers are considered. Valid options are "hv", "mv", "lv" for HV/MV/LV side or "star" for the star point.

        ****kwargs** - options to use for PYPOWER.runpf
    """
    _init_rundcpp_options(net, trafo_model=trafo_model, trafo_loading=trafo_loading,
                          recycle=recycle, check_connectivity=check_connectivity,
                          switch_rx_ratio=switch_rx_ratio, trafo3w_losses=trafo3w_losses, **kwargs)

    _check_bus_index_and_print_warning_if_high(net)
    _check_gen_index_and_print_warning_if_high(net)
    _powerflow(net, **kwargs)


def runopp(net, verbose=False, calculate_voltage_angles=True, check_connectivity=True,
           suppress_warnings=True, switch_rx_ratio=2, delta=1e-10, init="flat", numba=True,
           trafo3w_losses="hv", consider_line_temperature=False, **kwargs):
    """
    Runs the  pandapower Optimal Power Flow.
    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities can be defined in net.sgen / net.gen /net.load / net.storage
    net.sgen.controllable if a static generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If True, the following
    flexibilities apply:
        - net.gen.min_p_mw / net.gen.max_p_mw
        - net.gen.min_q_mvar / net.gen.max_q_mvar
        - net.sgen.min_p_mw / net.sgen.max_p_mw
        - net.sgen.min_q_mvar / net.sgen.max_q_mvar
        - net.dcline.max_p_mw
        - net.dcline.min_q_to_mvar / net.dcline.max_q_to_mvar / net.dcline.min_q_from_mvar / net.dcline.max_q_from_mvar
        - net.ext_grid.min_p_mw / net.ext_grid.max_p_mw
        - net.ext_grid.min_q_mvar / net.ext_grid.max_q_mvar
        - net.load.min_p_mw / net.load.max_p_mw
        - net.load.min_q_mvar / net.load.max_q_mvar
        - net.storage.min_p_mw / net.storage.max_p_mw
        - net.storage.min_q_mvar / net.storage.max_q_mvar

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
        **verbose** (bool, False) - If True, some basic information is printed

        **suppress_warnings** (bool, True) - suppress warnings in pypower

            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow.
            These warnings are suppressed by this option, however keep in mind all other pypower
            warnings are suppressed, too.

        **init** (str, "flat") - init of starting opf vector. Options are "flat" or "pf"

            Starting solution vector (x0) for opf calculations is determined by this flag. Options are:
            "flat" (default): starting vector is (upper bound - lower bound) / 2
            "pf": a power flow is executed prior to the opf and the pf solution is the starting vector. This may improve
            convergence, but takes a longer runtime (which are probably neglectible for opf calculations)

        **delta** (float, 1e-10) - power tolerance

        **trafo3w_losses** (str, "hv") - defines where open loop losses of three-winding transformers are considered. Valid options are "hv", "mv", "lv" for HV/MV/LV side or "star" for the star point.

        **consider_line_temperature** (bool, False) - adjustment of line impedance based on provided\
            line temperature. If True, net.line must contain a column "temperature_degree_celsius".\
            The temperature dependency coefficient alpha must be provided in the net.line.alpha\
            column, otherwise the default value of 0.004 is used

         **kwargs** - Pypower / Matpower keyword arguments:

         - OPF_VIOLATION (5e-6) constraint violation tolerance
         - PDIPM_COSTTOL (1e-6) optimality tolerance
         - PDIPM_GRADTOL (1e-6) gradient tolerance
         - PDIPM_COMPTOL (1e-6) complementarity condition (inequality) tolerance
         - PDIPM_FEASTOL (set to OPF_VIOLATION if not specified) feasibiliy (equality) tolerance
         - PDIPM_MAX_IT  (150) maximum number of iterations
         - SCPDIPM_RED_IT(20) maximum number of step size reductions per iteration

    """
    _check_necessary_opf_parameters(net, logger)
    _init_runopp_options(net, calculate_voltage_angles=calculate_voltage_angles,
                         check_connectivity=check_connectivity,
                         switch_rx_ratio=switch_rx_ratio, delta=delta, init=init, numba=numba,
                         trafo3w_losses=trafo3w_losses,
                         consider_line_temperature=consider_line_temperature, **kwargs)
    _check_bus_index_and_print_warning_if_high(net)
    _check_gen_index_and_print_warning_if_high(net)
    _optimal_powerflow(net, verbose, suppress_warnings, **kwargs)


def rundcopp(net, verbose=False, check_connectivity=True, suppress_warnings=True,
             switch_rx_ratio=0.5, delta=1e-10, trafo3w_losses="hv", **kwargs):
    """
    Runs the  pandapower Optimal Power Flow.
    Flexibilities, constraints and cost parameters are defined in the pandapower element tables.

    Flexibilities for generators can be defined in net.sgen / net.gen.
    net.sgen.controllable / net.gen.controllable signals if a generator is controllable. If False,
    the active and reactive power are assigned as in a normal power flow. If yes, the following
    flexibilities apply:
        - net.sgen.min_p_mw / net.sgen.max_p_mw
        - net.gen.min_p_mw / net.gen.max_p_mw
        - net.load.min_p_mw / net.load.max_p_mw

        Network constraints can be defined for buses, lines and transformers the elements in the following columns:
        - net.line.max_loading_percent
        - net.trafo.max_loading_percent
        - net.trafo3w.max_loading_percent

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **verbose** (bool, False) - If True, some basic information is printed

        **suppress_warnings** (bool, True) - suppress warnings in pypower

            If set to True, warnings are disabled during the loadflow. Because of the way data is
            processed in pypower, ComplexWarnings are raised during the loadflow.
            These warnings are suppressed by this option, however keep in mind all other pypower
            warnings are suppressed, too.

        **delta** (float, 1e-10) - power tolerance

        **trafo3w_losses** (str, "hv") - defines where open loop losses of three-winding transformers are considered. Valid options are "hv", "mv", "lv" for HV/MV/LV side or "star" for the star point.
    """
    if (not net.sgen.empty) & ("controllable" not in net.sgen.columns):
        logger.warning('Warning: Please specify sgen["controllable"]\n')

    if (not net.load.empty) & ("controllable" not in net.load.columns):
        logger.warning('Warning: Please specify load["controllable"]\n')

    _init_rundcopp_options(net, check_connectivity=check_connectivity,
                           switch_rx_ratio=switch_rx_ratio, delta=delta,
                           trafo3w_losses=trafo3w_losses, **kwargs)
    _check_bus_index_and_print_warning_if_high(net)
    _check_gen_index_and_print_warning_if_high(net)
    _optimal_powerflow(net, verbose, suppress_warnings, **kwargs)


def _passed_runpp_parameters(local_parameters):
    """
    Internal function to distinguish arguments for pandapower.runpp() that are explicitly passed by
    the user.
    :param local_parameters: locals() in the runpp() function
    :return: dictionary of explicitly passed parameters
    """
    net = local_parameters.pop("net")
    if not ("user_pf_options" in net.keys() and len(net.user_pf_options) > 0):
        return None
    try:
        default_parameters = {k: v.default for k, v in inspect.signature(runpp).parameters.items()}
    except:
        args, varargs, keywords, defaults = inspect.getfullargspec(runpp)
        default_parameters = dict(zip(args[-len(defaults):], defaults))
    default_parameters.update({"init": "auto"})

    passed_parameters = {
        key: val for key, val in local_parameters.items()
        if key in default_parameters.keys() and val != default_parameters.get(key, None)}

    return passed_parameters
