# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.timeseries.data_source import DataSource

try:
    import pplog
except:
    import logging as pplog

logger = pplog.getLogger(__name__)


class DFData(DataSource):
    """
    Hold a pandas.DataFrame as the data source, that uses index column as time step.
    Take care that the data is numeric and all the time steps are present.

    Please note: The columns should be integers for scalar accessing!

    INPUT:
        **df** - A pandas DataFrame which holds the time series data

    OPTIONAL:

        **multi** (bool, False) - If True casts columns and indexes to integers.
        This might be necessary if you read your data from csv.
    """

    def __init__(self, df, multi=False):
        super().__init__()
        self.df = df
        if multi:
            # casting column and index to int for multi- columns accessing
            self.df.index = self.df.index.astype(int)
            self.df.columns = self.df.columns.astype(int)

    def __repr__(self):
        s = "%s with %d rows and %d columns" % (
            self.__class__.__name__, len(self.df), len(self.df.columns))

        if len(self.df.columns) <= 10:
            s += ": %s" % self.df.columns.values.__str__()
        return s

    def get_time_step_value(self, time_step, profile_name, scale_factor=1.0):
        res = self.df.loc[time_step, profile_name]
        if hasattr(res, 'values'):
            res = res.values
        res *= scale_factor
        return res

    def get_time_steps_len(self):
        return len(self.df)
