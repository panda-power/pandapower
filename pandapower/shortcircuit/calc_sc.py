# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
#import time

from pandapower.auxiliary import _clean_up, _add_ppc_options, _add_sc_options, _add_auxiliary_elements
from pandapower.pd2ppc import _pd2ppc
from pandapower.pd2ppc_zero import _pd2ppc_zero
from pandapower.results import _copy_results_ppci_to_ppc
from pandapower.shortcircuit.currents import _calc_ikss, _calc_ikss_1ph, _calc_ip, _calc_ith, _calc_branch_currents, _calc_single_bus_sc
from pandapower.shortcircuit.impedance import _calc_zbus, _calc_ybus, _calc_rx
from pandapower.shortcircuit.kappa import _add_kappa_to_ppc
from pandapower.shortcircuit.results import _extract_results, _extract_single_results


def calc_sc(net, fault="3ph", case='max', lv_tol_percent=10, topology="auto", ip=False,
            ith=False, tk_s=1., kappa_method="C", r_fault_ohm=0., x_fault_ohm=0.,
            branch_results=False, check_connectivity=True):
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

        **Ith** (bool, False) if True, calculate equivalent thermical short-circuit current Ith

        **topology** (str, "auto") define option for meshing (only relevant for ip and ith)

            - "meshed" - it is assumed all buses are supplied over multiple paths

            - "radial" - it is assumed all buses are supplied over exactly one path

            - "auto" - topology check for each bus is performed to see if it is supplied over multiple paths

        **tk_s** (float, 1) failure clearing time in seconds (only relevant for ith)

        **r_fault_ohm** (float, 0) fault resistance in Ohm

        **x_fault_ohm** (float, 0) fault reactance in Ohm

        **branch_results** (bool, False) defines if short-circuit results should also be generated for branches


    OUTPUT:

    EXAMPLE:
        calc_sc(net)

        print(net.res_bus_sc)
    """
    if fault not in ["3ph", "2ph", "1ph"]:
        raise NotImplementedError(
            "Only 3ph, 2ph and 1ph short-circuit currents implemented")

    if len(net.gen) and (ip or ith):
        logger.warning("aperiodic and thermal short-circuit currents are only implemented for "
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

    kappa = ith or ip
    net["_options"] = {}
    _add_ppc_options(net, calculate_voltage_angles=False, trafo_model="pi",
                     check_connectivity=check_connectivity, mode="sc", switch_rx_ratio=2, 
                     init_vm_pu="flat", init_va_degree="flat", enforce_q_lims=False,
                     recycle=None)
    _add_sc_options(net, fault=fault, case=case, lv_tol_percent=lv_tol_percent, tk_s=tk_s,
                    topology=topology, r_fault_ohm=r_fault_ohm, kappa_method=kappa_method,
                    x_fault_ohm=x_fault_ohm, kappa=kappa, ip=ip, ith=ith,
                    branch_results=branch_results)
    if fault == "3ph":
        _calc_sc(net)
    if fault == "2ph":
        _calc_sc(net)
    if fault == "1ph":
        if case == "min":
            raise NotImplementedError("Minimum 1ph short-circuits are not yet implemented")
        _calc_sc_1ph(net)

def calc_single_sc(net, bus, fault="3ph", case='max', lv_tol_percent=10, check_connectivity=True):
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

        ***fault** (str, 3ph) type of fault

            - "3ph" for three-phase

            - "2ph" for two-phase short-circuits

        **case** (str, "max")

            - "max" for maximal current calculation

            - "min" for minimal current calculation

        **lv_tol_percent** (int, 10) voltage tolerance in low voltage grids

            - 6 for 6% voltage tolerance

            - 10 for 10% voltage olerance

        **r_fault_ohm** (float, 0) fault resistance in Ohm

        **x_fault_ohm** (float, 0) fault reactance in Ohm

    OUTPUT:

    EXAMPLE:
        calc_sc(net)

        print(net.res_bus_sc)
    """
    if fault not in ["3ph", "2ph"]:
        raise NotImplementedError("Only 3ph and 2ph short-circuit currents implemented")

    if case not in ['max', 'min']:
        raise ValueError('case can only be "min" or "max" for minimal or maximal short "\
                                "circuit current')
    net["_options"] = {}
    _add_ppc_options(net, calculate_voltage_angles=False, trafo_model="pi",
                     check_connectivity=check_connectivity, mode="sc", switch_rx_ratio=2, 
                     init_vm_pu="flat", init_va_degree="flat", enforce_q_lims=False,
                     recycle=None)
    _add_sc_options(net, fault=fault, case=case, lv_tol_percent=lv_tol_percent, tk_s=1.,
                    topology="auto", r_fault_ohm=0., kappa_method="C",
                    x_fault_ohm=0., kappa=False, ip=False, ith=False,
                    branch_results=True)
    if fault == "3ph" or fault == "2ph":
        _calc_sc_single(net, bus)
    elif fault == "1ph":
        raise NotImplementedError("1ph short-circuits are not yet implemented")
    else:
        raise ValueError("Invalid fault %s"%fault)

def _calc_sc_single(net, bus):
    _add_auxiliary_elements(net)
    ppc, ppci = _pd2ppc(net)
    _calc_ybus(ppci)
    try:
        _calc_zbus(ppci)
    except Exception as e:
        _clean_up(net, res=False)
        raise(e)
    _calc_rx(net, ppci)
    _calc_ikss(net, ppci)
    _calc_single_bus_sc(net, ppci, bus)
    ppc = _copy_results_ppci_to_ppc(ppci, ppc, "sc")
    _extract_single_results(net, ppc)
    _clean_up(net)    


def _calc_sc(net):
    _add_auxiliary_elements(net)
    ppc, ppci = _pd2ppc(net)
    _calc_ybus(ppci)
    try:
        _calc_zbus(ppci)
    except Exception as e:
        _clean_up(net, res=False)
        raise(e)
    _calc_rx(net, ppci)
    _add_kappa_to_ppc(net, ppci)
    _calc_ikss(net, ppci)
    if net["_options"]["ip"]:
        _calc_ip(net, ppci)
    if net["_options"]["ith"]:
        _calc_ith(net, ppci)
    if net._options["branch_results"]:
        _calc_branch_currents(net, ppci)
    ppc = _copy_results_ppci_to_ppc(ppci, ppc, "sc")
    _extract_results(net, ppc, ppc_0=None)
    _clean_up(net)


def _calc_sc_1ph(net):
    """
    calculation method for single phase to ground short-circuit currents
    """
    _add_auxiliary_elements(net)
    # pos. seq bus impedance
    ppc, ppci = _pd2ppc(net)
    _calc_ybus(ppci)
    try:
        _calc_zbus(ppci)
    except Exception as e:
        _clean_up(net, res=False)
        raise(e)
    _calc_rx(net, ppci)
    _add_kappa_to_ppc(net, ppci)
    # zero seq bus impedance
    ppc_0, ppci_0 = _pd2ppc_zero(net)
    _calc_ybus(ppci_0)
    try:
        _calc_zbus(ppci_0)
    except Exception as e:
        _clean_up(net, res=False)
        raise(e)
    _calc_rx(net, ppci_0)
    _calc_ikss_1ph(net, ppci, ppci_0)
    ppc_0 = _copy_results_ppci_to_ppc(ppci_0, ppc_0, "sc")
    ppc = _copy_results_ppci_to_ppc(ppci, ppc, "sc")
    _extract_results(net, ppc, ppc_0)
    _clean_up(net)