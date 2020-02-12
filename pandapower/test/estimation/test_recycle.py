# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import StateEstimation, estimate
from pandapower.estimation.util import add_virtual_meas_from_loadflow
from copy import deepcopy


def test_recycle_case30():
    net = nw.case30()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)
    se = StateEstimation(net, recycle=True)
    se.estimate()
    assert np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=5e-1)  

    # Run SE again
    net.load.p_mw -= 10
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)
    assert se.estimate()
    assert np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=1)  

if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
