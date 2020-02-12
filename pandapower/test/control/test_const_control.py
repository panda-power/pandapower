# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
import pandas as pd

import pandapower as pp
import pandapower.networks as nw
import pandapower.control
import pandapower.timeseries
import logging as log

logger = log.getLogger(__name__)


def test_write():
    net = nw.simple_four_bus_system()
    ds = pp.timeseries.DFData(pd.DataFrame(data=[[0., 1., 2.], [2., 3., 4.]]))
    c1 = pp.control.ConstControl(net, 'sgen', 'p_mw', element_index=[0, 1], profile_name=[0, 1], data_source=ds)
    pp.create_sgen(net, 0, 0)
    c2 = pp.control.ConstControl(net, 'sgen', 'p_mw', element_index=[2], profile_name=[2], data_source=ds)
    c1.time_step(0)
    c1.control_step()
    c2.time_step(0)
    c2.control_step()


if __name__ == '__main__':
    pytest.main([__file__])
