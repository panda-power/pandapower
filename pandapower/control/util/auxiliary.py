# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import csv
import random
from functools import reduce
from pandapower.auxiliary import ADict

import numpy as np
import pandas

import pandas as pd
from pandas import Int64Index

from pandapower.toolbox import ensure_iterability
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

try:
    import pplog
except ImportError:
    import logging as pplog

logger = pplog.getLogger(__name__)


def asarray(val, dtype=np.float64):
    """
    make sure the value is a numpy array of a certain dtype
    in contrast to np.asarray, it also converts a scalar value to an array,
    e.g. asarray(0) = array([0])
    :param val: scalar or list-like or pandas index input
    :param dtype: np.dtype for the output array
    :return: numpy array of the dtype set
    """
    # val is a list, tuple etc
    if not np.isscalar(val) and np.ndim(val) > 0:
        np_val = np.asarray(val, dtype=dtype)
    else:
        # val is a scalar number
        np_val = np.asarray([val], dtype=dtype)

    return np_val


def get_controller_index_by_type(net, ctrl_type, idx=[]):
    """
    Returns controller indices of a given type as list.
    """
    idx = idx if len(idx) else net.controller.index
    return [i for i in idx if isinstance(net.controller.object.at[i], ctrl_type)]


def get_controller_index_by_typename(net, typename, idx=[], case_sensitive=False):
    """
    Returns controller indices of a given name of type as list.
    """
    idx = idx if len(idx) else net.controller.index
    if case_sensitive:
        return [i for i in idx if str(net.controller.object.at[i]).split(" ")[0] == typename]
    else:
        return [i for i in idx if str(net.controller.object.at[i]).split(" ")[0].lower() ==
                typename.lower()]


def _controller_attributes_query(controller, parameters):
    """
    Returns a boolean if the controller attributes matches given parameter dict data
    """
    complete_match = True
    element_index_match = True
    for key in parameters.keys():
        if key not in controller.__dict__:
            logger.debug(str(key) + " is no attribute of controller object " + str(controller))
            return False
        try:
            match = bool(controller.__getattribute__(key) == parameters[key])
        except ValueError:
            try:
                match = all(controller.__getattribute__(key) == parameters[key])
            except ValueError:
                match = bool(len(set(controller.__getattribute__(key)) & set(parameters[key])))
        if key == "element_index":
            element_index_match = match
        else:
            complete_match &= match

    if complete_match and not element_index_match:
        intersect_elms = set(ensure_iterability(controller.__getattribute__("element_index"))) & \
            set(ensure_iterability(parameters["element_index"]))
        if len(intersect_elms):
            logger.info("'element_index' has an intersection of " + str(intersect_elms) +
                        " with Controller %i" % controller.index)

    return complete_match & element_index_match


def get_controller_index(net, ctrl_type=None, parameters=None, idx=[]):
    """ Returns indices of searched controllers. Parameters can specify the search query.

    INPUT:
        **net** (pandapowerNet) - The pandapower network

    OPTINAL:
        **controller_type** (controller object or string name of controller object)

        **parameters** (None, dict) - Dict of parameter names, which are in the controller object or
            net.controller DataFrame

        **idx** ([], list) - list of indices in net.controller to be searched for. If list is empty
            all indices are considered.

    OUTPUT:
        **idx** (list) - index / indices of controllers in net.controller which are in idx and
            matches given ctrl_type or parameters
    """
    #    logger.debug(ctrl_type, parameters, idx)
    idx = idx if len(idx) else net.controller.index
    if ctrl_type is not None:
        if isinstance(ctrl_type, str):
            idx = get_controller_index_by_typename(net, ctrl_type, idx)
        else:
            idx = get_controller_index_by_type(net, ctrl_type, idx)
    if isinstance(parameters, dict):
        df_keys = [k for k in parameters.keys() if k in net.controller.columns]
        attributes_keys = list(set(parameters.keys()) - set(df_keys))
        attributes_dict = {k: parameters[k] for k in attributes_keys}
        # query of parameters in net.controller dataframe
        idx = Int64Index(idx)
        for df_key in df_keys:
            idx = idx.intersection(net.controller.index[net.controller[df_key] == parameters[df_key]])
        # query of parameters in controller object attributes
        idx = [i for i in idx if _controller_attributes_query(
            net.controller.object.loc[i], attributes_dict)]
    return idx


def log_same_type_existing_controllers(net, this_ctrl_type, index=None, matching_params=None,
                                       **kwargs):
    """
    Logs same type controllers, if a controller is created.
    INPUT:
        **net** - pandapower net

        **this_ctrl_type** (controller object or string name of controller object)

    OPTIONAL:
        **index** (int) - index in net.controller of the controller to be created

        **matching_params** (dict) - parameters, which must be equal if same type controller should
            be logged.

        ****kwargs** - unused arguments, given to avoid unexpected input arguments
    """
    index = str(index)
    if isinstance(matching_params, dict):
        same_type_existing_ctrl = get_controller_index(net, ctrl_type=this_ctrl_type,
                                                       parameters=matching_params)
        if len(same_type_existing_ctrl):
            logger.info("Controller " + index + " has same type and matching parameters like " +
                        "controllers " + str(['%i' % idx for idx in same_type_existing_ctrl]))
    else:
        logger.info("Creating controller " + index + " of type %s " % this_ctrl_type)
        logger.debug("no matching parameters are given to check whether problematic, " +
                     "same type controllers already exist.")


def drop_same_type_existing_controllers(net, this_ctrl_type, index=None, matching_params=None,
                                        **kwargs):
    """
    Drops same type controllers to create a new controller of this type.
    INPUT:
        **net** - pandapower net

        **this_ctrl_type** (controller object or string name of controller object)

    OPTIONAL:
        **index** (int) - index in net.controller of the controller to be created

        **matching_params** (dict) - parameters, which must be equal if same type controller should
            be logged.

        ****kwargs** - unused arguments, given to avoid unexpected input arguments
    """
    index = str(index)
    if isinstance(matching_params, dict):
        same_type_existing_ctrl = get_controller_index(net, ctrl_type=this_ctrl_type,
                                                       parameters=matching_params)
        if len(same_type_existing_ctrl):
            net.controller.drop(same_type_existing_ctrl, inplace=True)
            logger.debug("Controllers " + str(['%i' % idx for idx in same_type_existing_ctrl]) +
                         "got removed because of same type and matching parameters as new " +
                         "controller " + index + ".")
    else:
        logger.info("Creating controller " + index + " of type %s, " % this_ctrl_type +
                    "no matching parameters are given to check which " +
                    "same type controllers should be dropped.")


def plot_characteristic(characteristic, start, stop, num=20):
    x = np.linspace(start, stop, num)
    y = characteristic(x)
    if MATPLOTLIB_INSTALLED:
        plt.plot(x, y, marker='x')
    else:
        logger.info("matplotlib not installed. y-values: %s" % y)
