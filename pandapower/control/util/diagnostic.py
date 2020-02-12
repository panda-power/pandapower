# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
from copy import deepcopy

from pandapower.control.util.auxiliary import get_controller_index
from pandapower.control.controller.trafo_control import TrafoController

try:
    import pplog
except:
    import logging as pplog

logger = pplog.getLogger(__name__)


def control_diagnostic(net, respect_in_service=True):
    """
    Diagnostic function to find obvious mistakes in control data
    """
    # --- find and log same type controllers connected to same elements
    indices = list(net.controller.index)
    for idx in indices:
        current_controller = net.controller.object.loc[idx]
        parameters = deepcopy(current_controller.matching_params) if "matching_params" in \
                                                                     current_controller.__dict__ else {}
        if respect_in_service:
            if not net.controller.in_service.at[idx]:
                continue
            parameters["in_service"] = True
        same_type_existing_ctrl = get_controller_index(net, ctrl_type=type(current_controller),
                                                       parameters=parameters)
        if len(same_type_existing_ctrl) > 1:
            logger.info("Same type and same matching parameters controllers " + str([
                '%i' % i for i in same_type_existing_ctrl]) +
                        " could affect convergence.")
            for val in same_type_existing_ctrl:
                indices.remove(val)

    # --- find trafo controller of the same trafo
    trafo_ctrl = []
    for idx in net.controller.index:
        current_controller = net.controller.object.loc[idx]
        if issubclass(type(current_controller), TrafoController):
            trafo_ctrl += [idx]
    for idx in trafo_ctrl:
        current_controller = net.controller.object.loc[idx]
        parameters = {"tid": current_controller.tid}
        if respect_in_service:
            if not net.controller.in_service.at[idx]:
                continue
            parameters["in_service"] = True
        trafo_ctrl_at_same_trafo = get_controller_index(net, parameters=parameters, idx=trafo_ctrl)
        if len(trafo_ctrl_at_same_trafo) > 1:
            logger.info("Trafo Controllers " + str([
                '%i' % i for i in trafo_ctrl_at_same_trafo]) +
                        " at the transformer probably could affect convergence.")
            for val in trafo_ctrl_at_same_trafo:
                trafo_ctrl.remove(val)
