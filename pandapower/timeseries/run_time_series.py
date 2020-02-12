# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import tempfile

import pandapower as pp
from pandapower import LoadflowNotConverged, OPFNotConverged
from pandapower.control.run_control import ControllerNotConverged, get_controller_order, \
    check_for_initial_powerflow, run_control
from pandapower.control.util.diagnostic import control_diagnostic
from pandapower.timeseries.output_writer import OutputWriter
from collections.abc import Iterable

try:
    import pplog
except ImportError:
    import logging as pplog

logger = pplog.getLogger(__name__)
logger.setLevel(level=pplog.WARNING)


def init_default_outputwriter(net, time_steps, **kwargs):
    """
    Initializes the output writer. If output_writer is None, default output_writer is created

    INPUT:
        **net** - The pandapower format network

        **time_steps** (list) - time steps to be calculated

    """
    output_writer = kwargs.get("output_writer", None)
    if output_writer is not None:
        # write the output_writer to net
        logger.warning("deprecated: output_writer should not be given to run_timeseries(). "
                       "This overwrites the stored one in net.output_writer.")
        net.output_writer.iat[0, 0] = output_writer
    if "output_writer" not in net or net.output_writer.iat[0, 0] is None:
        # create a default output writer for this net
        ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir())
        logger.info("No output writer specified. Using default:")
        logger.info(ow)


def init_output_writer(net, time_steps):
    # init output writer before time series calculation
    output_writer = net.output_writer.iat[0, 0]
    output_writer.time_steps = time_steps
    output_writer.init_all()


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar.
    the code is mentioned in : https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    # logger.info('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print("\n")


def controller_not_converged(time_step, ts_variables):
    logger.error('ControllerNotConverged at time step %s' % time_step)
    if not ts_variables["continue_on_divergence"]:
        raise ControllerNotConverged


def pf_not_converged(time_step, ts_variables):
    logger.error('LoadflowNotConverged at time step %s' % time_step)
    if not ts_variables["continue_on_divergence"]:
        raise LoadflowNotConverged


def run_time_step(net, time_step, ts_variables, **kwargs):
    """
    Time Series step function
    Is called to run the PANDAPOWER AC power flows with the timeseries module

    INPUT:
        **net** - The pandapower format network

        **time_step** (int) - time_step to be calculated

        **ts_variables** (dict) - contains settings for controller and time series simulation. See init_time_series()
    """
    ctrl_converged = True
    pf_converged = True
    output_writer = net["output_writer"].iat[0, 0]
    # update time step for output writer
    output_writer.time_step = time_step
    # run time step function for each controller
    for levelorder in ts_variables["controller_order"]:
        for ctrl in levelorder:
            ctrl.time_step(time_step)

    try:
        # calls controller init, control steps and run function (runpp usually is called in here)
        run_control(net, run_control=False, ctrl_variables=ts_variables, **kwargs)
    except ControllerNotConverged:
        ctrl_converged = False
        # If controller did not converge do some stuff
        controller_not_converged(time_step, ts_variables)
    except (LoadflowNotConverged, OPFNotConverged):
        # If power flow did not converge simulation aborts or continues if continue_on_divergence is True
        pf_converged = False
        pf_not_converged(time_step, ts_variables)

    # save
    output_writer.save_results(time_step, pf_converged=pf_converged, ctrl_converged=ctrl_converged,
                               recycle_options=ts_variables["recycle_options"])


def _check_controller_recyclability(net):
    # if a parameter is set to True here, it will be recalculated during the time series simulation
    recycle = dict(trafo=False, gen=False, bus_pq=False)
    if "controller" not in net:
        # everything can be recycled since no controller is in net. But the time series simulation makes no sense
        # then anyway...
        return recycle

    for idx in net.controller.index:
        # todo: write to controller data frame recycle column instead of using self.recycle of controller instance
        ctrl_recycle = net.controller.at[idx, "object"].recycle
        if not isinstance(ctrl_recycle, dict):
            # if one controller has a wrong recycle configuration it is deactived
            recycle = False
            break
        # else check which recycle parameter are set to True
        for rp in ["trafo", "bus_pq", "gen"]:
            recycle[rp] = recycle[rp] or ctrl_recycle[rp]

    return recycle


def _check_output_writer_recyclability(net, recycle):
    if "output_writer" not in net:
        raise ValueError("OutputWriter not defined")
    ow = net.output_writer.at[0, "object"]
    # results which are read with a faster batch function after the time series simulation
    recycle["batch_read"] = list()
    recycle["only_v_results"] = False
    new_log_variables = list()

    for output in ow.log_variables:
        table, variable = output[0], output[1]
        if table not in ["res_bus", "res_line", "res_trafo", "res_trafo3w"] or recycle["trafo"] or len(output) > 2:
            # no fast read of outputs possible if other elements are required as these or tap changer is active
            recycle["only_v_results"] = False
            recycle["batch_read"] = False
            return recycle
        else:
            # fast read is possible
            if variable in ["vm_pu", "va_degree"] and recycle["only_v_results"] is False:
                recycle["only_v_results"] = True
                new_log_variables.append(('ppc_bus', 'vm'))
                new_log_variables.append(('ppc_bus', 'va'))
                recycle["batch_read"].append((table, variable))
            if variable in ["loading_percent", "i_ka", "i_to_ka", "i_from_ka", "i_hv_ka", "i_mv_ka", "i_lv_ka"]:
                recycle["only_v_results"] = True
                recycle["batch_read"].append((table, variable))

    ow.log_variables = new_log_variables
    ow.log_variable('ppc_bus', 'vm')
    ow.log_variable('ppc_bus', 'va')
    return recycle


def get_recycle_settings(net, **kwargs):
    """
    checks if "run" is specified in kwargs and calls this function in time series loop.
    if "recycle" is in kwargs we use the TimeSeriesRunpp class (not implemented yet)

    INPUT:
        **net** - The pandapower format network

    RETURN:
        **recycle** - a dict with recycle options to be used by runpp
    """

    recycle = kwargs.get("recycle", None)
    if recycle is not False:
        # check if every controller can be recycled and what can be recycled
        recycle = _check_controller_recyclability(net)
        # if still recycle is not None, also check for fast output_writer features
        if recycle is not False:
            recycle = _check_output_writer_recyclability(net, recycle)

    return recycle


def init_time_steps(net, time_steps, **kwargs):
    # initializes time steps if as a range
    if not isinstance(time_steps, Iterable):
        if time_steps is None and ("start_step" in kwargs and "stop_step" in kwargs):
            logger.warning("start_step and stop_step are depricated. "
                           "Please use a tuple like time_steps = (start_step, stop_step) instead or a list")
            time_steps = range(kwargs["start_step"], kwargs["stop_step"] + 1)
        elif isinstance(time_steps, tuple):
            time_steps = range(time_steps[0], time_steps[1])
        else:
            logger.warning("No time steps to calculate are specified. "
                           "I'll check the datasource of the first controller for avaiable time steps")
            max_timestep = net.controller.object.at[0].data_source.get_time_steps_len()
            time_steps = range(max_timestep)
    return time_steps


def init_time_series(net, time_steps, continue_on_divergence=False, verbose=True,
                     **kwargs):
    """
    inits the time series calculation
    creates the dict ts_variables, which includes necessary variables for the time series / control function

    INPUT:
        **net** - The pandapower format network

        **time_steps** (list or tuple, None) - time_steps to calculate as list or tuple (start, stop)
        if None, all time steps from provided data source are simulated

    OPTIONAL:

        **continue_on_divergence** (bool, False) - If True time series calculation continues in case of errors.

        **verbose** (bool, True) - prints progress bar or logger debug messages
    """

    time_steps = init_time_steps(net, time_steps, **kwargs)

    ts_variables = dict()

    init_default_outputwriter(net, time_steps, **kwargs)
    level, order = get_controller_order(net)
    # get run function
    run = kwargs.get("run", pp.runpp)
    recycle_options = False
    if hasattr(run, "__name__") and run.__name__ == "runpp":
        # use faster runpp options if possible
        recycle_options = get_recycle_settings(net, **kwargs)

    init_output_writer(net, time_steps)
    # True at default. Initial power flow is calculated before each control step (some controllers need inits)
    ts_variables["initial_powerflow"] = check_for_initial_powerflow(order)
    # order of controller (controllers are called in a for loop.)
    ts_variables["controller_order"] = order
    # run function to be called in run_control - default is pp.runpp, but can be runopf or whatever you like
    ts_variables["run"] = run
    # recycle options, which define what can be recycled
    ts_variables["recycle_options"] = recycle_options
    # time steps to be calculated (list or range)
    ts_variables["time_steps"] = time_steps
    # If True, a diverged power flow is ignored and the next step is calculated
    ts_variables["continue_on_divergence"] = continue_on_divergence
    # print settings
    ts_variables["verbose"] = verbose

    if logger.level is not 10 and verbose:
        # simple progress bar
        print_progress_bar(0, len(time_steps), prefix='Progress:', suffix='Complete', length=50)

    return ts_variables


def cleanup(ts_variables):
    if isinstance(ts_variables["recycle_options"], dict):
        # Todo: delete internal variables and dumped results which are not needed
        pass


def print_progress(i, time_step, time_steps, verbose, **kwargs):
    # simple status print in each time step.
    if logger.level is not 10 and verbose:
        len_timesteps = len(time_steps)
        print_progress_bar(i + 1, len_timesteps, prefix='Progress:', suffix='Complete', length=50)

    # print debug info
    if logger.level == pplog.DEBUG and verbose:
        logger.debug("run time step %i" % time_step)

    # call a custom progress function
    if "progress_function" in kwargs:
        func = kwargs["progress_function"]
        func(i, time_step, time_steps, **kwargs)

def run_loop(net, ts_variables, **kwargs):
    """
    runs the time series loop which calls pp.runpp (or another run function) in each iteration

    Parameters
    ----------
    net - pandapower net
    ts_variables - settings for time series

    """
    for i, time_step in enumerate(ts_variables["time_steps"]):
        print_progress(i, time_step, ts_variables["time_steps"], ts_variables["verbose"], **kwargs)
        run_time_step(net, time_step, ts_variables, **kwargs)


def run_timeseries(net, time_steps=None, continue_on_divergence=False, verbose=True, **kwargs):
    """
    Time Series main function

    Runs multiple PANDAPOWER AC power flows based on time series which are stored in a **DataSource** inside
    **Controllers**. Optionally other functions than the pp power flow can be called by setting the run function in kwargs

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **time_steps** (list or tuple, None) - time_steps to calculate as list or tuple (start, stop)
        if None, all time steps from provided data source are simulated

        **continue_on_divergence** (bool, False) - If True time series calculation continues in case of errors.

        **verbose** (bool, True) - prints progress bar or if logger.level == Debug it prints debug messages

        **kwargs** - Keyword arguments for run_control and runpp. If "run" is in kwargs the default call to runpp()
        is replaced by the function kwargs["run"]
    """

    ts_variables = init_time_series(net, time_steps, continue_on_divergence, verbose, **kwargs)

    control_diagnostic(net)
    run_loop(net, ts_variables, **kwargs)

    # cleanup functions after the last time step was calculated
    cleanup(ts_variables)
