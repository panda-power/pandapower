# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

try:
    import pplog
except:
    import logging as pplog

from pandapower.io_utils import JSONSerializableClass

logger = pplog.getLogger(__name__)


class DataSource(JSONSerializableClass):
    """
    The DataSource class is a skeleton for data sources such as pandas DataFrames
    Controllers call get_time_step_values(time) in each time step to get values from the data source
    """

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return self.__class__.__name__

    def get_time_step_value(self, time_step, profile_name, scale_factor=1.0):
        """
        This method retrieves values of the data source according to the given parameters.
        For actual parameters look into the DataSource you are actually using.
        """
        raise NotImplementedError("Subclasses should implement this!")
