# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from pandapower.auxiliary import _add_ppc_options
from pandapower.powerflow import _pd2ppc
from pandapower.opf.validate_opf_input import _check_necessary_opf_parameters

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def to_ppc(net, calculate_voltage_angles=False, trafo_model="t", switch_rx_ratio=2,
           check_connectivity=True, voltage_depend_loads=True, init="results", mode=None):

    """
     This function converts a pandapower net to a pypower case file.

    INPUT:

        **net** - The pandapower net.

    OPTIONAL:

        **calculate_voltage_angles** (bool, False) - consider voltage angles in loadflow calculation

        If True, voltage angles of ext_grids and transformer shifts are considered in the
        loadflow calculation. Considering the voltage angles is only necessary in meshed
        networks that are usually found in higher networks.

        **trafo_model** (str, "t") - transformer equivalent circuit model
        pandapower provides two equivalent circuit models for the transformer:

            - "t" - transformer is modeled as equivalent with the T-model.
            - "pi" - transformer is modeled as equivalent PI-model. This is not recommended, since \
            it is less exact than the T-model. It is only recommended for validation with other \
            software that uses the pi-model.

        **switch_rx_ratio** (float, 2) - rx_ratio of bus-bus-switches. If impedance is zero, \
        buses connected by a closed bus-bus switch are fused to model an ideal bus. \
        Otherwise, they are modelled as branches with resistance defined as z_ohm column in \
        switch table and this parameter 

        **check_connectivity** (bool, True) - Perform an extra connectivity test after the
        conversion from pandapower to PYPOWER

            If True, an extra connectivity test based on SciPy Compressed Sparse Graph Routines is
            perfomed. If check finds unsupplied buses, they are set out of service in the ppc

        **voltage_depend_loads** (bool, True) - consideration of voltage-dependent loads. \
        If False, net.load.const_z_percent and net.load.const_i_percent are not considered, i.e. \
        net.load.p_mw and net.load.q_mvar are considered as constant-power loads.

        **init** (str, "results") - initialization method of the converter
        pandapower ppc converter supports two methods for initializing the converter:

            - "flat"- flat start with voltage of 1.0pu and angle of 0° at all PQ-buses and 0° for \
            PV buses as initial solution
            - "results" - voltage vector from net.res_bus is used as initial solution.

        **mode** (str, None) - mode of power flow calculation type ("pf" - power flow, "opf" - \
        optimal power flow or "sc" - short circuit). "mode" influences for instance whether opf \
        cost data will be converted or which slack bus voltage limits are respected. If "mode" \
        is None, cost data will be respected via mode="opf" if cost data are existing.

    OUTPUT:

        **ppc** - The Pypower casefile for usage with pypower

    EXAMPLE:

        import pandapower.converter as pc

        import pandapower.networks as pn

        net = pn.case9()

        ppc = pc.to_ppc(net)

    """
    if (not (net["poly_cost"].empty and net["pwl_cost"].empty) and
       mode is None) or mode == "opf":
        mode = "opf"
        _check_necessary_opf_parameters(net, logger)
    else:
        mode = "pf"

    # select elements in service
    net["_options"] = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, switch_rx_ratio=switch_rx_ratio, init_vm_pu=init,
                     init_va_degree=init, enforce_q_lims=True,
                     recycle=None, voltage_depend_loads=voltage_depend_loads)
    #  do the conversion
    _, ppci = _pd2ppc(net)
    ppci['branch'] = ppci['branch'].real
#    ppci.pop('internal')
    return ppci
