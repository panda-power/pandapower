# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from pandas import Index
from pandapower.control.basic_controller import Controller

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class ConstControl(Controller):
    """
    Class representing a generic time series controller for a specified element and variable
    Control strategy: "No Control" -> just updates timeseries

    INPUT:

        **net** (attrdict) - The net in which the controller resides

        **element** - element table ('sgen', 'load' etc.)

        **variable** - variable ('p_mw', 'q_mvar', 'vm_pu', 'tap_pos' etc.)

        **element_index** (int[]) - IDs of the controlled elements

        **data_source** (obj) - The data source that provides profile data

        **profile_name** (str[]) - The profile names of the elements in the data source


    OPTIONAL:

        **scale_factor** (real, 1.0) - Scaling factor for time series input values

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **recycle** (bool, True) - Re-use of internal-data in a time series loop.

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped

    .. note:: If multiple elements are represented with one controller, the data source must have integer columns. At the moment, only the DFData format is tested for the multiple const control.
    """

    def __init__(self, net, element, variable, element_index, profile_name=None, data_source=None,
                 scale_factor=1.0, in_service=True, recycle=True, order=0, level=0,
                 drop_same_existing_ctrl=False, set_q_from_cosphi=False, matching_params=None, initial_powerflow=False,
                 **kwargs):
        # just calling init of the parent
        if matching_params is None:
            matching_params = {"element": element, "variable": variable,
                               "element_index": element_index}
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_powerflow = initial_powerflow,
                         **kwargs)
        self.matching_params = {"element": element, "variable": variable,
                                "element_index": element_index}

        # data source for time series values
        self.data_source = data_source
        # ids of sgens or loads
        self.element_index = element_index
        # element type
        self.element = element
        self.variable = variable
        self.values = None
        self.profile_name = profile_name
        self.scale_factor = scale_factor
        if set_q_from_cosphi:
            logger.error("Parameter set_q_from_cosphi deprecated!")
            raise ValueError
        self.applied = False
        self.initial_powerflow = initial_powerflow
        # write functions faster, depending on type of self.element_index
        if isinstance(self.element_index, int):
            # use .at if element_index is integer for speedup
            self.write = "single_index"
        # commenting this out for now, see issue 609
        # elif self.net[self.element].index.equals(Index(self.element_index)):
        #     # use : indexer if all elements are in index
        #     self.write = "all_index"
        else:
            # use common .loc
            self.write = "loc"
        self.set_recycle()

    def set_recycle(self):
        allowed_elements = ["load", "sgen", "storage", "gen", "ext_grid", "trafo", "trafo3w", "line"]
        if self.recycle is False or self.element not in allowed_elements:
            # if recycle is set to False by the user when creating the controller it is deactivated or when
            # const control controls an element which is not able to be recycled
            self.recycle = False
            return
        # these variables determine what is re-calculated during a time series run
        recycle = dict(trafo=False, gen=False, bus_pq=False)
        if self.element in ["sgen", "load", "storage"] and self.variable in ["p_mw", "q_mvar"]:
            recycle["bus_pq"] = True
        if self.element in ["gen"] and self.variable in ["p_mw", "vm_pu"] \
                or self.element in ["ext_grid"] and self.variable in ["vm_pu", "va_degree"]:
            recycle["gen"] = True
        if self.element in ["trafo", "trafo3w", "line"]:
            recycle["trafo"] = True
        self.recycle = recycle

    def write_to_net(self):
        """
        Writes to self.element at index self.element_index in the column self.variable the data
        from self.values
        """
        # write functions faster, depending on type of self.element_index
        if self.write == "single_index":
            self._write_to_single_index()
        elif self.write == "all_index":
            self._write_to_all_index()
        elif self.write == "loc":
            self._write_with_loc()
        else:
            raise NotImplementedError("ConstControl: self.write must be one of "
                                      "['single_index', 'all_index', 'loc']")

    def time_step(self, time):
        """
        Get the values of the element from data source
        """
        self.values = self.data_source.get_time_step_value(time_step=time,
                                                           profile_name=self.profile_name,
                                                           scale_factor=self.scale_factor)
        # self.write_to_net()

    def initialize_control(self):
        """
        At the beginning of each run_control call reset applied-flag
        """
        #
        if self.data_source is None:
            self.values = self.net[self.element][self.variable].loc[self.element_index]
        self.applied = False

    def is_converged(self):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        return self.applied

    def control_step(self):
        """
        Write to pandapower net by calling write_to_net()
        """
        if self.values is not None:
            self.write_to_net()
        self.applied = True

    def _write_to_single_index(self):
        self.net[self.element].at[self.element_index, self.variable] = self.values

    def _write_to_all_index(self):
        self.net[self.element].loc[:, self.variable] = self.values

    def _write_with_loc(self):
        self.net[self.element].loc[self.element_index, self.variable] = self.values
