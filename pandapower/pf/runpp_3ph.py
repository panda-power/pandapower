# -*- coding: utf-8 -*-
"""
# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

@author: Shankho Ghosh (sghosh) (started Feb 2018)
@author: Alexander Prostejovsky (alepros), Technical University of Denmark
"""
from time import time
import numpy as np
from pandapower import LoadflowNotConverged
from pandapower.pypower.pfsoln import pfsoln
from pandapower.pypower.idx_gen import PG, QG
from pandapower.pd2ppc import _pd2ppc_recycle
from pandapower.pypower.makeYbus import makeYbus
from pandapower.pypower.idx_bus import GS, BS, PD , QD
from pandapower.auxiliary import _sum_by_group, _check_if_numba_is_installed,\
    _check_bus_index_and_print_warning_if_high,\
    _check_gen_index_and_print_warning_if_high, \
    _add_pf_options, _add_ppc_options, _clean_up, sequence_to_phase, \
    phase_to_sequence, X012_to_X0, X012_to_X2, \
    I1_from_V012, S_from_VI_elementwise, V1_from_ppc, V_from_I,\
    combine_X012, I0_from_V012, I2_from_V012, ppException
from pandapower.pf.run_newton_raphson_pf import _run_newton_raphson_pf
from pandapower.build_bus import _add_ext_grid_sc_impedance
from pandapower.pypower.bustypes import bustypes
from pandapower.run import _passed_runpp_parameters
from pandapower.pypower.idx_bus import VM, VA
from pandapower.pypower.idx_gen import GEN_BUS
from pandapower.results import _copy_results_ppci_to_ppc, _extract_results_3ph,\
    init_results
try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


class Not_implemented(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass


def _get_pf_variables_from_ppci(ppci):
    """
    Used for getting values for pfsoln function in one convinient function
    """
    # default arguments
    if ppci is None:
        ValueError('ppci is empty')
    # get data for calc
    base_mva, bus, gen, branch = \
        ppci["baseMVA"], ppci["bus"], ppci["gen"], ppci["branch"]
    # get bus index lists of each type of bus
    ref, pv, pq = bustypes(bus, gen)
    return base_mva, bus, gen, branch, ref, pv, pq


def _store_results_from_pf_in_ppci(ppci, bus, gen, branch):
    ppci["bus"], ppci["gen"], ppci["branch"] = bus, gen, branch
    return ppci


def _get_elements(params,net,element,phase,typ):
    sign = -1 if element.endswith("sgen") else 1
    elm = net[element].values
#   # Trying to find the column no for using numpy filters for active loads
    scaling = net[element].columns.get_loc("scaling")
    typ_col = net[element].columns.get_loc("type") # Type = Delta or Wye load
    # active wye or active delta row selection
    active = (net["_is_elements"][element]) & (elm[:,typ_col] == typ)
    bus = [net[element].columns.get_loc("bus")]
    if len(elm):
        if element == 'load' or element == 'sgen':
            vl = elm[active,scaling].ravel()
            p_mw = [net[element].columns.get_loc("p_mw")]
            q_mvar = [net[element].columns.get_loc("q_mvar")]
            params['p'+phase+typ] = np.hstack([params['p'+phase+typ],
                                               elm[active,p_mw]/3 * vl * sign])
            params['q'+phase+typ] = np.hstack([params['q'+phase+typ],
                                               (elm[active,q_mvar]/3) * vl * sign])
            params['b'+typ] = np.hstack([params['b'+typ],
                                         elm[active,bus].astype(int)])

        elif element.startswith('asymmetric'):
            vl = elm[active,scaling].ravel()
            p = {'a': net[element].columns.get_loc("p_a_mw")
         ,'b': net[element].columns.get_loc("p_b_mw")
         ,'c': net[element].columns.get_loc("p_c_mw")
            }
            q = {'a' : net[element].columns.get_loc("q_a_mvar")
         ,'b' : net[element].columns.get_loc("q_b_mvar")
         ,'c' : net[element].columns.get_loc("q_c_mvar")
            }

            params['p'+phase+typ] = np.hstack([params['p'+phase+typ],
                                             elm[active,p[phase]] * vl * sign])
            params['q'+phase+typ] = np.hstack([params['q'+phase+typ],
                                             elm[active,q[phase]] * vl * sign])
            params['b'+typ] = np.hstack([params['b'+typ],
                                         elm[active,bus].astype(int)])
    return params


def _load_mapping(net, ppci1):
    """
    Takes three phase P, Q values from PQ elements
    sums them up for each bus
    maps them in ppc bus order and forms s_abc matrix
    """
    bus_lookup = net["_pd2ppc_lookups"]["bus"]
    params = dict()
    phases = ['a', 'b', 'c']
    load_types = ['wye', 'delta']
    load_elements = ['load', 'asymmetric_load', 'sgen', 'asymmetric_sgen']
# =============================================================================
#        Loop to initialize and feed s_abc wye and delta values
# =============================================================================
    for phase in phases:
        for typ in load_types:
            params['S'+phase+typ] = (ppci1["bus"][:, PD] +
                                     ppci1["bus"][:, QD]*1j)*0
            params['p'+phase+typ] = np.array([])  # p values from loads/sgens
            params['q'+phase+typ] = np.array([])  # q values from loads/sgens
            params['P'+phase+typ] = np.array([])  # Aggregated Active Power
            params['Q'+phase+typ] = np.array([])  # Aggregated reactive Power
            params['b'+phase+typ] = np.array([], dtype=int)  # bus map for phases
            params['b'+typ] = np.array([], dtype=int)  # aggregated bus map(s_abc)
            for element in load_elements:
                _get_elements(params,net,element,phase,typ)
            # Mapping constant power loads to buses
            if params['b'+typ].size:
                params['b'+phase+typ] = bus_lookup[params['b'+typ]]
                params['b'+phase+typ], params['P'+phase+typ],\
                    params['Q'+phase+typ] = _sum_by_group(params['b'+phase+typ],
                                                          params['p'+phase+typ],
                                                          params['q'+phase+typ] * 1j)
                params['S'+phase+typ][params['b'+phase+typ]] = \
                    (params['P'+phase+typ] + params['Q'+phase+typ])
    Sabc_del = np.vstack((params['Sadelta'],params['Sbdelta'],params['Scdelta']))
    Sabc_wye = np.vstack((params['Sawye'],params['Sbwye'],params['Scwye']))
    # last return varaible left for constant impedance loads
    return Sabc_del, Sabc_wye


# =============================================================================
# 3 phase algorithm function
# =============================================================================
def runpp_3ph(net, calculate_voltage_angles=True, init="auto",
              max_iteration="auto", tolerance_mva=1e-8, trafo_model='t',
              trafo_loading="current", enforce_q_lims=False, numba=True,
              recycle=None, check_connectivity=True, switch_rx_ratio=2.0,
              delta_q=0, v_debug=False, **kwargs):
    """
 runpp_3ph: Performs Unbalanced/Asymmetric/Three Phase Load flow

    INPUT:
        **net** - The pandapower format network

    OPTIONAL:
        **algorithm** (str, "nr") - algorithm that is used to solve the power
        flow problem.

            The following algorithms are available:

                - "nr" Newton-Raphson (pypower implementation with numba
                accelerations)

                Used only for positive sequence network

                Zero and Negative sequence networks use Current Injection method

                Vnew = Y.inv * Ispecified ( from s_abc/v_abc old)

                Icalculated = Y * Vnew


        **calculate_voltage_angles** (bool, "auto") - consider voltage angles
        in loadflow calculation

            If True, voltage angles of ext_grids and transformer shifts are
            considered in the loadflow calculation. Considering the voltage
            angles is only necessary in meshed networks that are usually
            found in higher voltage levels. calculate_voltage_angles
            in "auto" mode defaults to:

                - True, if the network voltage level is above 70 kV
                - False otherwise

            The network voltage level is defined as the maximum rated voltage
            of any bus in the network that is connected to a line.


        **max_iteration** (int, "auto") - maximum number of iterations carried
        out in the power flow algorithm.

            In "auto" mode, the default value depends on the power flow solver:

                - 10 for "nr"

            For three phase calculations, its extended to 3 * max_iteration

        **tolerance_mva** (float, 1e-8) - loadflow termination condition
        referring to P / Q mismatch of node power in MVA

        **trafo_model**
        - transformer equivalent models

            - "t" - transformer is modeled as equivalent with the T-model.
            - "pi" - This is not recommended, since it is less exact than the T-model.
             So, for three phase load flow, its not
            implemented

        **trafo_loading** (str, "current") - mode of calculation for
        transformer loading

            Transformer loading can be calculated relative to the rated
            current or the rated power. In both cases the overall transformer
            loading is defined as the maximum loading on the two sides of
            the transformer.

            - "current"- transformer loading is given as ratio of current
            flow and rated current of the transformer. This is the recommended
            setting, since thermal as well as magnetic effects in the
            transformer depend on the current.
            - "power" - transformer loading is given as ratio of apparent
            power flow to the rated apparent power of the transformer.

        **enforce_q_lims** (bool, False)

        (Not tested with 3 Phase load flow) - respect generator reactive power
        limits

            If True, the reactive power limits in net.gen.max_q_mvar/min_q_mvar
            are respected in the loadflow. This is done by running a second
            loadflow if reactive power limits are violated at any generator,
            so that the runtime for the loadflow will increase if reactive
            power has to be curtailed.

            Note: enforce_q_lims only works if algorithm="nr"!


        **check_connectivity** (bool, True) - Perform an extra connectivity
        test after the conversion from pandapower to PYPOWER

            If True, an extra connectivity test based on SciPy Compressed
            Sparse Graph Routines is perfomed. If check finds unsupplied buses,
            they are set out of service in the ppc

        **voltage_depend_loads** (bool, True)

        (Not tested with 3 Phase load flow)  - consideration of
        voltage-dependent loads. If False, net.load.const_z_percent and
        net.load.const_i_percent are not considered, i.e. net.load.p_mw and
        net.load.q_mvar are considered as constant-power loads.

        **consider_line_temperature** (bool, False)

        (Not tested with 3 Phase load flow) - adjustment of line
        impedance based on provided line temperature. If True, net.line must
        contain a column "temperature_degree_celsius". The temperature
        dependency coefficient alpha must be provided in the net.line.alpha
            column, otherwise the default value of 0.004 is used


        **KWARGS:

        **numba** (bool, True) - Activation of numba JIT compiler in the
        newton solver

            If set to True, the numba JIT compiler is used to generate
            matrices for the powerflow, which leads to significant speed
            improvements.

        **switch_rx_ratio** (float, 2)

        (Not tested with 3 Phase load flow)  - rx_ratio of bus-bus-switches.
        If impedance is zero, buses connected by a closed bus-bus switch
        are fused to model an ideal bus. Otherwise, they are modelled
        as branches with resistance defined as z_ohm column in switch
        table and this parameter

        **delta_q**

        (Not tested with 3 Phase load flow) - Reactive power tolerance for option "enforce_q_lims"
        in kvar - helps convergence in some cases.

        **trafo3w_losses**

        (Not tested with 3 Phase load flow) - defines where open loop losses of three-winding
        transformers are considered. Valid options are "hv", "mv", "lv"
        for HV/MV/LV side or "star" for the star point.

        **v_debug** (bool, False)

        (Not tested with 3 Phase load flow) - if True, voltage values in each
        newton-raphson iteration are logged in the ppc

        **init_vm_pu** (string/float/array/Series, None)

        (Not tested with 3 Phase load flow) - Allows to define
        initialization specifically for voltage magnitudes.
        Only works with init == "auto"!

            - "auto": all buses are initialized with the mean value of all
            voltage controlled elements in the grid
            - "flat" for flat start from 1.0
            - "results": voltage magnitude vector is taken from result table
            - a float with which all voltage magnitudes are initialized
            - an iterable with a voltage magnitude value for each bus
            (length and order has to match with the buses in net.bus)
            - a pandas Series with a voltage magnitude value for each bus
            (indexes have to match the indexes in net.bus)

         **init_va_degree** (string/float/array/Series, None)

         (Not tested with 3 Phase load flow)-
        Allows to define initialization specifically for voltage angles.
        Only works with init == "auto"!

            - "auto": voltage angles are initialized from DC power flow
            if angles are calculated or as 0 otherwise
            - "dc": voltage angles are initialized from DC power flow
            - "flat" for flat start from 0
            - "results": voltage angle vector is taken from result table
            - a float with which all voltage angles are initialized
            - an iterable with a voltage angle value for each bus (length
            and order has to match with the buses in net.bus)
            - a pandas Series with a voltage angle value for each bus (indexes
            have to match the indexes in net.bus)

        **recycle** (dict, none)

        - Reuse of internal powerflow variables for
        time series calculation

            Contains a dict with the following parameters:
            bus_pq: If True PQ values of buses are updated
            gen: If True Sbus and the gen table in the ppc are recalculated
            Ybus: If True the admittance matrix (Ybus, Yf, Yt) is taken from
            ppc["internal"] and not reconstructed

        **neglect_open_switch_branches** (bool, False)

        (Not tested with 3 Phase load flow) - If True no auxiliary
        buses are created for branches when switches are opened at the branch.
        Instead branches are set out of service

    SEE ALSO:
         pp.add_zero_impedance_parameters(net):
         To add zero sequence parameters into network from the standard type

    EXAMPLES:
        >>> from pandapower.pf.runpp_3ph import runpp_3ph

        >>> runpp_3ph(net)

    NOTES:
        - Three phase load flow uses Sequence Frame for power flow solution.
        - Three phase system is modelled with earth return.
        - PH-E load type is called as wye since Neutral and Earth are considered same
        - This solver has proved successful only for Earthed transformers (i.e Dyn,Yyn,YNyn & Yzn vector groups)
    """
    # =============================================================================
    # pandapower settings
    # =============================================================================
    overrule_options = {}
    if "user_pf_options" in net.keys() and len(net.user_pf_options) > 0:
        passed_parameters = _passed_runpp_parameters(locals())
        overrule_options = {key: val for key, val in net.user_pf_options.items()
                            if key not in passed_parameters.keys()}
    if numba:
        numba = _check_if_numba_is_installed(numba)

    ac = True
    mode = "pf_3ph"  # TODO: Make valid modes (pf, pf_3ph, se, etc.) available in seperate file (similar to idx_bus.py)
#    v_debug = kwargs.get("v_debug", False)
    copy_constraints_to_ppc = False
    if trafo_model == 'pi':
        raise Not_implemented("Three phase Power Flow doesnot support pi model\
                                because of lack of accuracy")
#    if calculate_voltage_angles == "auto":
#        calculate_voltage_angles = False
#        hv_buses = np.where(net.bus.vn_kv.values > 70)[0]  # Todo: Where does that number come from?
#        if len(hv_buses) > 0:
#            line_buses = net.line[["from_bus", "to_bus"]].values.flatten()
#            if len(set(net.bus.index[hv_buses]) & set(line_buses)) > 0:
    # scipy spsolve options in NR power flow
    use_umfpack = kwargs.get("use_umfpack", True)
    permc_spec = kwargs.get("permc_spec", None)
    calculate_voltage_angles = True
    if init == "results" and net.res_bus_3ph.empty:
        init = "auto"
    if init == "auto":
        init = "dc" if calculate_voltage_angles else "flat"
    default_max_iteration = {"nr": 10, "bfsw": 10, "gs": 10000, "fdxb": 30,
                             "fdbx": 30}
    if max_iteration == "auto":
        max_iteration = default_max_iteration["nr"]

    neglect_open_switch_branches = kwargs.get("neglect_open_switch_branches", False)
    only_v_results = kwargs.get("only_v_results", False)
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, switch_rx_ratio=switch_rx_ratio,
                     init_vm_pu=init, init_va_degree=init,
                     enforce_q_lims=enforce_q_lims, recycle=None,
                     voltage_depend_loads=False, delta=delta_q,\
                     neglect_open_switch_branches=neglect_open_switch_branches
                     )
    _add_pf_options(net, tolerance_mva=tolerance_mva, trafo_loading=trafo_loading,
                    numba=numba, ac=ac, algorithm="nr", max_iteration=max_iteration,\
                    only_v_results=only_v_results,v_debug=v_debug, use_umfpack=use_umfpack,
                    permc_spec=permc_spec)
    net._options.update(overrule_options)
    _check_bus_index_and_print_warning_if_high(net)
    _check_gen_index_and_print_warning_if_high(net)
    # =========================================================================
    # pd2ppc conversion
    # =========================================================================
    _, ppci1 = _pd2ppc_recycle(net, 1, recycle=recycle)

    _, ppci2 = _pd2ppc_recycle(net, 2, recycle=recycle)
    gs_eg, bs_eg = _add_ext_grid_sc_impedance(net, ppci2)

    _, ppci0 = _pd2ppc_recycle(net, 0, recycle=recycle)

    _,        bus0, gen0, branch0,      _,      _,      _ = _get_pf_variables_from_ppci(ppci0)
    base_mva, bus1, gen1, branch1, sl_bus,      _, pq_bus = _get_pf_variables_from_ppci(ppci1)
    _,        bus2, gen2, branch2,      _,      _,      _ = _get_pf_variables_from_ppci(ppci2)

    # initialize the results after the conversion to ppc is done, otherwise init=results does not work
    init_results(net, "pf_3ph")

# =============================================================================
#     P Q values aggragated and summed up for each bus to make s_abc matrix
#     s_abc for wye connections ; s_abc_delta for delta connection
# =============================================================================
    s_abc_delta, s_abc = _load_mapping(net, ppci1)
    # =========================================================================
    # Construct Sequence Frame Bus admittance matrices Ybus
    # =========================================================================

    ppci0, ppci1, ppci2, y_0_pu, y_1_pu, y_2_pu, y_0_f, y_1_f, _,\
        y_0_t, y_1_t, _ = _get_y_bus(ppci0, ppci1, ppci2, recycle)
    # =========================================================================
    # Initial voltage values
    # =========================================================================
    nb = ppci1["bus"].shape[0]

    # make sure flat start is always respected, even with other voltage data in recycled ppc
    if init == "flat":
        v_012_it = np.zeros((3, nb), dtype=np.complex128)
        v_012_it[1, :] = 1.0
    else:
        v_012_it = np.concatenate(
            [np.array(ppc["bus"][:, VM] * np.exp(1j * np.deg2rad(ppc["bus"][:, VA]))).reshape((1, nb))
             for ppc in (ppci0, ppci1, ppci2)], axis=0).astype(np.complex128)

    # For Delta transformation:
    # Voltage changed from line-earth to line-line using V_T
    # s_abc/v_abc will now give line-line currents. This is converted to line-earth
    # current using I-T
    v_del_xfmn = np.array([[1, -1, 0],
                           [0, 1, -1],
                           [-1, 0, 1]])
    i_del_xfmn = np.array([[1, 0, -1],
                           [-1, 1, 0],
                           [0, -1, 1]])
    v_abc_it = sequence_to_phase(v_012_it)

    # =========================================================================
    #             Iteration using Power mismatch criterion
    # =========================================================================
    outer_tolerance_mva = 3e-8
    count = 0
    s_mismatch = np.array([[True], [True]], dtype=bool)
    t0 = time()
    while (s_mismatch > outer_tolerance_mva).any() and count < 30*max_iteration:
        # =====================================================================
        #     Voltages and Current transformation for PQ and Slack bus
        # =====================================================================
        s_abc_pu = -np.divide(s_abc, ppci1["baseMVA"])
        s_abc_delta_pu = -np.divide(s_abc_delta, ppci1["baseMVA"])

        i_abc_it_wye = (np.divide(s_abc_pu, v_abc_it)).conjugate()
        i_abc_it_delta = np.matmul(i_del_xfmn, (np.divide(s_abc_delta_pu, np.matmul
                                                          (v_del_xfmn, v_abc_it))).conjugate())

        # For buses with both delta and wye loads we need to sum of their currents
        # to sum up the currents
        i_abc_it = i_abc_it_wye + i_abc_it_delta
        i012_it = phase_to_sequence(i_abc_it)
        v1_for_s1 = v_012_it[1, :]
        i1_for_s1 = -i012_it[1, :]
        v0_pu_it = X012_to_X0(v_012_it)
        v2_pu_it = X012_to_X2(v_012_it)
        i0_pu_it = X012_to_X0(i012_it)
        i2_pu_it = X012_to_X2(i012_it)
        s1 = np.multiply(v1_for_s1, i1_for_s1.conjugate())
        # =============================================================================
        # Current used to find S1 Positive sequence power
        # =============================================================================

        ppci1["bus"][pq_bus, PD] = np.real(s1[pq_bus]) * ppci1["baseMVA"]
        ppci1["bus"][pq_bus, QD] = np.imag(s1[pq_bus]) * ppci1["baseMVA"]
        # =============================================================================
        # Conduct Positive sequence power flow
        # =============================================================================
        _run_newton_raphson_pf(ppci1, net._options)
        # =============================================================================
        # Conduct Negative and Zero sequence power flow
        # =============================================================================
        v0_pu_it = V_from_I(y_0_pu, i0_pu_it)
        v2_pu_it = V_from_I(y_2_pu, i2_pu_it)
        # =============================================================================
        #    Evaluate Positive Sequence Power Mismatch
        # =============================================================================
        i1_from_v_it = I1_from_V012(v_012_it, y_1_pu).flatten()
        s_from_voltage = S_from_VI_elementwise(v1_for_s1, i1_from_v_it)
        v1_pu_it = V1_from_ppc(ppci1)

        v_012_new = combine_X012(v0_pu_it, v1_pu_it, v2_pu_it)

        s_mismatch = np.abs(np.abs(s1[pq_bus]) - np.abs(s_from_voltage[pq_bus]))
        v_012_it = v_012_new
        v_abc_it = sequence_to_phase(v_012_it)
        count += 1
    et = time() - t0
    success = (count < 30 * max_iteration)
    for ppc in [ppci0, ppci1, ppci2]:
        ppc["et"] = et
        ppc["success"] = success
    # TODO: Add reference to paper to explain the following steps
    # This is required since the ext_grid power results are not correct if its
    # not done
    ref, pv, pq = bustypes(ppci0["bus"], ppci0["gen"])
    ref_gens = ppci0["internal"]["ref_gens"]
    ppci0["bus"][ref, GS] -= gs_eg
    ppci0["bus"][ref, BS] -= bs_eg
    y_0_pu, y_0_f, y_0_t = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
    # revert the change, otherwise repeated calculation with recycled elements will fail
    ppci0["bus"][ref, GS] += gs_eg
    ppci0["bus"][ref, BS] += bs_eg
    # Bus, Branch, and Gen  power values
    bus0, gen0, branch0 = pfsoln(base_mva, bus0, gen0, branch0, y_0_pu, y_0_f, y_0_t, v_012_it[0, :].flatten(),
                                 sl_bus, ref_gens)
    bus1, gen1, branch1 = pfsoln(base_mva, bus1, gen1, branch1, y_1_pu, y_1_f, y_1_t, v_012_it[1, :].flatten(),
                                 sl_bus, ref_gens)
    bus2, gen2, branch2 = pfsoln(base_mva, bus2, gen2, branch2, y_1_pu, y_1_f, y_1_t, v_012_it[2, :].flatten(),
                                 sl_bus, ref_gens)
    ppci0 = _store_results_from_pf_in_ppci(ppci0, bus0, gen0, branch0)
    ppci1 = _store_results_from_pf_in_ppci(ppci1, bus1, gen1, branch1)
    ppci2 = _store_results_from_pf_in_ppci(ppci2, bus2, gen2, branch2)
    i_012_res = _current_from_voltage_results(y_0_pu, y_1_pu, v_012_it)
    s_012_res = S_from_VI_elementwise(v_012_it, i_012_res) * ppci1["baseMVA"]
    eg_is_mask = net["_is_elements"]['ext_grid']
    ext_grid_lookup = net["_pd2ppc_lookups"]["ext_grid"]
    eg_is_idx = net["ext_grid"].index.values[eg_is_mask]
    eg_idx_ppc = ext_grid_lookup[eg_is_idx]
    """ # 2 ext_grids Fix: Instead of the generator index, bus indices of the generators are used"""
    eg_bus_idx_ppc = np.real(ppci1["gen"][eg_idx_ppc, GEN_BUS]).astype(int)

    ppci0["gen"][eg_idx_ppc, PG] = s_012_res[0, eg_bus_idx_ppc].real
    ppci1["gen"][eg_idx_ppc, PG] = s_012_res[1, eg_bus_idx_ppc].real
    ppci2["gen"][eg_idx_ppc, PG] = s_012_res[2, eg_bus_idx_ppc].real
    ppci0["gen"][eg_idx_ppc, QG] = s_012_res[0, eg_bus_idx_ppc].imag
    ppci1["gen"][eg_idx_ppc, QG] = s_012_res[1, eg_bus_idx_ppc].imag
    ppci2["gen"][eg_idx_ppc, QG] = s_012_res[2, eg_bus_idx_ppc].imag

    ppc0 = net["_ppc0"]
    ppc1 = net["_ppc1"]
    ppc2 = net["_ppc2"]

    # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
    ppc0 = _copy_results_ppci_to_ppc(ppci0, ppc0, mode=mode)
    ppc1 = _copy_results_ppci_to_ppc(ppci1, ppc1, mode=mode)
    ppc2 = _copy_results_ppci_to_ppc(ppci2, ppc2, mode=mode)

    _extract_results_3ph(net, ppc0, ppc1, ppc2)

    #    Raise error if PF was not successful. If DC -> success is always 1

    if not ppci0["success"]:
        net["converged"] = False
        _clean_up(net, res=False)
        raise LoadflowNotConverged("Power Flow {0} did not converge after\
                                {1} iterations!".format("nr", count))
    else:
        net["converged"] = True

    _clean_up(net)


def _current_from_voltage_results(y_0_pu, y_1_pu, v_012_pu):
    I012_pu = combine_X012(I0_from_V012(v_012_pu, y_0_pu),
                            I1_from_V012(v_012_pu, y_1_pu),
                            I2_from_V012(v_012_pu, y_1_pu))
    return I012_pu


def _get_y_bus(ppci0, ppci1, ppci2, recycle):
    if recycle and recycle["Ybus"] and ppci0["internal"]["Ybus"].size and \
            ppci1["internal"]["Ybus"].size and ppci2["internal"]["Ybus"].size:
        y_0_bus, y_0_f, y_0_t = ppci0["internal"]['Ybus'], ppci0["internal"]['Yf'], ppci0["internal"]['Yt']
        y_1_bus, y_1_f, y_1_t = ppci1["internal"]['Ybus'], ppci1["internal"]['Yf'], ppci1["internal"]['Yt']
        y_2_bus, y_2_f, y_2_t = ppci2["internal"]['Ybus'], ppci2["internal"]['Yf'], ppci2["internal"]['Yt']
    else:
        # build admittance matrices
        y_0_bus, y_0_f, y_0_t = makeYbus(ppci0["baseMVA"], ppci0["bus"], ppci0["branch"])
        y_1_bus, y_1_f, y_1_t = makeYbus(ppci1["baseMVA"], ppci1["bus"], ppci1["branch"])
        y_2_bus, y_2_f, y_2_t = makeYbus(ppci2["baseMVA"], ppci2["bus"], ppci2["branch"])
        if recycle and recycle["Ybus"]:
            ppci0["internal"]['Ybus'], ppci0["internal"]['Yf'], ppci0["internal"]['Yt'] = y_0_bus, y_0_f, y_0_t
            ppci1["internal"]['Ybus'], ppci1["internal"]['Yf'], ppci1["internal"]['Yt'] = y_1_bus, y_1_f, y_1_t
            ppci2["internal"]['Ybus'], ppci2["internal"]['Yf'], ppci2["internal"]['Yt'] = y_2_bus, y_2_f, y_2_t

    return ppci0, ppci1, ppci2, y_0_bus, y_1_bus, y_2_bus, y_0_f, y_1_f, y_2_f, y_0_t, y_1_t, y_2_t
