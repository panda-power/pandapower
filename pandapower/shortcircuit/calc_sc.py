# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

import numpy as np
from scipy.sparse.linalg import factorized

from pandapower.auxiliary import _clean_up, _add_ppc_options, _add_sc_options, _add_auxiliary_elements
from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.results import _copy_results_ppci_to_ppc

from pandapower.shortcircuit.currents import _calc_ikss,\
    _calc_ikss_1ph, _calc_ip, _calc_ith, _calc_branch_currents
from pandapower.shortcircuit.impedance import _calc_zbus, _calc_ybus, _calc_rx
from pandapower.shortcircuit.ppc_conversion import _init_ppc, _create_k_updated_ppci, _get_is_ppci_bus
from pandapower.shortcircuit.kappa import _add_kappa_to_ppc
from pandapower.shortcircuit.results import _extract_results, _copy_result_to_ppci_orig
from pandapower.results import init_results


def calc_sc(net, bus=None,
            fault="3ph", case='max', lv_tol_percent=10, topology="auto", ip=False,
            ith=False, tk_s=1., kappa_method="C", r_fault_ohm=0., x_fault_ohm=0.,
            branch_results=False, check_connectivity=True, return_all_currents=False,
            inverse_y=True):

    """
    Calculates minimal or maximal symmetrical short-circuit currents.
    The calculation is based on the method of the equivalent voltage source
    according to DIN/IEC EN 60909.
    The initial short-circuit alternating current *ikss* is the basis of the short-circuit
    calculation and is therefore always calculated.
    Other short-circuit currents can be calculated from *ikss* with the conversion factors defined
    in DIN/IEC EN 60909.

    The output is stored in the net.res_bus_sc table as a short_circuit current
    for each bus.

    INPUT:
        **net** (pandapowerNet) pandapower Network

        **bus** (int, list, np.array, None) defines if short-circuit calculations should only be calculated for defined bus

        ***fault** (str, 3ph) type of fault

            - "3ph" for three-phase

            - "2ph" for two-phase short-circuits

            - "1ph" for single-phase ground faults

        **case** (str, "max")

            - "max" for maximal current calculation

            - "min" for minimal current calculation

        **lv_tol_percent** (int, 10) voltage tolerance in low voltage grids

            - 6 for 6% voltage tolerance

            - 10 for 10% voltage olerance

        **ip** (bool, False) if True, calculate aperiodic short-circuit current

        **ith** (bool, False) if True, calculate equivalent thermical short-circuit current Ith

        **topology** (str, "auto") define option for meshing (only relevant for ip and ith)

            - "meshed" - it is assumed all buses are supplied over multiple paths

            - "radial" - it is assumed all buses are supplied over exactly one path

            - "auto" - topology check for each bus is performed to see if it is supplied over multiple paths

        **tk_s** (float, 1) failure clearing time in seconds (only relevant for ith)

        **r_fault_ohm** (float, 0) fault resistance in Ohm

        **x_fault_ohm** (float, 0) fault reactance in Ohm

        **branch_results** (bool, False) defines if short-circuit results should also be generated for branches

        **return_all_currents** (bool, False) applies only if branch_results=True, if True short-circuit currents for
        each (branch, bus) tuple is returned otherwise only the max/min is returned

        **inverse_y** (bool, True) defines if complete inverse should be used instead of LU factorization, factorization version is in experiment which should be faster and memory efficienter


    OUTPUT:

    EXAMPLE:
        calc_sc(net)

        print(net.res_bus_sc)
    """
    if fault not in ["3ph", "2ph", "1ph"]:
        raise NotImplementedError(
            "Only 3ph, 2ph and 1ph short-circuit currents implemented")

    if len(net.gen) and (ip or ith):
        logger.warning("aperiodic, thermal short-circuit currents are only implemented for "
                       "faults far from generators!")

    if case not in ['max', 'min']:
        raise ValueError('case can only be "min" or "max" for minimal or maximal short "\
                                "circuit current')

    if topology not in ["meshed", "radial", "auto"]:
        raise ValueError(
            'specify network structure as "meshed", "radial" or "auto"')

    if branch_results:
        logger.warning("Branch results are in beta mode and might not always be reliable, "
                       "especially for transformers")

    # Convert bus to numpy array
    if bus is None:
        bus = net.bus.index.values
    else:
        bus = np.array([bus]).ravel()

    kappa = ith or ip
    net["_options"] = {}
    _add_ppc_options(net, calculate_voltage_angles=False, trafo_model="pi",
                      check_connectivity=check_connectivity, mode="sc", switch_rx_ratio=2,
                      init_vm_pu="flat", init_va_degree="flat", enforce_q_lims=False,
                      recycle=None)
    _add_sc_options(net, fault=fault, case=case, lv_tol_percent=lv_tol_percent, tk_s=tk_s,
                    topology=topology, r_fault_ohm=r_fault_ohm, kappa_method=kappa_method,
                    x_fault_ohm=x_fault_ohm, kappa=kappa, ip=ip, ith=ith,
                    branch_results=branch_results, return_all_currents=return_all_currents,
                    inverse_y=inverse_y)
    init_results(net, "sc")

    if fault in ("2ph", "3ph"):
        _calc_sc(net, bus)
    elif fault == "1ph":
        _calc_sc_1ph(net, bus)
    else:
        raise ValueError("Invalid fault %s" % fault)


def _calc_current(net, ppci_orig, bus):
    # Select required ppci bus
    ppci_bus = _get_is_ppci_bus(net, bus)

    # update ppci
    non_ps_gen_ppci_bus, non_ps_gen_ppci, ps_gen_bus_ppci_dict =\
        _create_k_updated_ppci(net, ppci_orig, ppci_bus=ppci_bus)

    # For each ps_gen_bus one unique ppci is required
    ps_gen_ppci_bus = list(ps_gen_bus_ppci_dict.keys())

    for calc_bus in ps_gen_ppci_bus+[non_ps_gen_ppci_bus]:
        if isinstance(calc_bus, np.ndarray):
            # Use ppci for general bus
            this_ppci, this_ppci_bus = non_ps_gen_ppci, calc_bus
        else:
            # Use specific ps_gen_bus ppci
            this_ppci, this_ppci_bus = ps_gen_bus_ppci_dict[calc_bus], np.array([calc_bus])

        _calc_ybus(this_ppci)
        if net["_options"]["inverse_y"]:
            _calc_zbus(net, this_ppci)
        else:
            # Factorization Ybus once
            this_ppci["internal"]["ybus_fact"] = factorized(this_ppci["internal"]["Ybus"])

        _calc_rx(net, this_ppci, this_ppci_bus)
        _calc_ikss(net, this_ppci, this_ppci_bus)
        _add_kappa_to_ppc(net, this_ppci)
        if net["_options"]["ip"]:
            _calc_ip(net, this_ppci)
        if net["_options"]["ith"]:
            _calc_ith(net, this_ppci)

        if net._options["branch_results"]:
            _calc_branch_currents(net, this_ppci, this_ppci_bus)

        _copy_result_to_ppci_orig(ppci_orig, this_ppci, this_ppci_bus,
                                  calc_options=net._options)


def _calc_sc(net, bus):
    ppc, ppci = _init_ppc(net)

    _calc_current(net, ppci, bus)

    ppc = _copy_results_ppci_to_ppc(ppci, ppc, "sc")
    _extract_results(net, ppc, ppc_0=None, bus=bus)
    _clean_up(net)

    if "ybus_fact" in ppci["internal"]:
        # Delete factorization object
        ppci["internal"].pop("ybus_fact")


def _calc_sc_1ph(net, bus):
    """
    calculation method for single phase to ground short-circuit currents
    """
    _add_auxiliary_elements(net)
    # pos. seq bus impedance
    ppc, ppci = _init_ppc(net)
    # Create k updated ppci
    ppci_bus = _get_is_ppci_bus(net, bus)
    _, ppci, _ =\
        _create_k_updated_ppci(net, ppci, ppci_bus=ppci_bus)
    _calc_ybus(ppci)

    # zero seq bus impedance
    ppc_0, ppci_0 = _pd2ppc_zero(net)
    _calc_ybus(ppci_0)

    if net["_options"]["inverse_y"]:
        _calc_zbus(net, ppci)
        _calc_zbus(net, ppci_0)
    else:
        # Factorization Ybus once
        ppci["internal"]["ybus_fact"] = factorized(ppci["internal"]["Ybus"])
        ppci_0["internal"]["ybus_fact"] = factorized(ppci_0["internal"]["Ybus"])

    ppci_bus = _get_is_ppci_bus(net, bus)
    _calc_rx(net, ppci, ppci_bus)
    _add_kappa_to_ppc(net, ppci)

    _calc_rx(net, ppci_0, ppci_bus)
    _calc_ikss_1ph(net, ppci, ppci_0, ppci_bus)

    if net._options["branch_results"]:
        _calc_branch_currents(net, ppci, ppci_bus)

    ppc_0 = _copy_results_ppci_to_ppc(ppci_0, ppc_0, "sc")
    ppc = _copy_results_ppci_to_ppc(ppci, ppc, "sc")
    _extract_results(net, ppc, ppc_0, bus=bus)
    _clean_up(net)
