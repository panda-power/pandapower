# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import estimate
from pandapower.estimation.util import add_virtual_meas_from_loadflow
from pandapower.estimation.ppc_conversion import pp2eppci
from copy import deepcopy

from pandapower.estimation.algorithm.estimator import SHGMEstimatorIRWLS


def test_irwls_comp_wls():
    # it should be the same since wls will not update weight matrix
    net = nw.case14()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)

    success = estimate(net, init='flat', algorithm="irwls", estimator='wls')
    assert success

    net_wls = deepcopy(net)
    estimate(net_wls)
    assert np.allclose(net_wls.res_bus_est.vm_pu, net.res_bus_est.vm_pu, 1e-6)
    assert np.allclose(net_wls.res_bus_est.va_degree, net.res_bus_est.va_degree, 1e-6)


def test_shgm_ps():
    # we need an random eppci object to initialize estimator
    net = nw.case14()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)
    _,_,eppci = pp2eppci(net)

    # Using the example from Mili's paper
    H = np.array([[10, -10],
                  [1, 0],
                  [-1, 0],
                  [0, -1],
                  [0, 1],
                  [11, -10],
                  [-1, -1]])
    estm = SHGMEstimatorIRWLS(eppci, a=3)
    ps_estm = estm._ps(H)
    assert np.allclose(ps_estm,
                       np.array([8.39, 0.84, 0.84, 0.84, 0.84, 8.82, 1.68]),
                       atol=0.005)


def test_irwls_shgm():
    net = nw.case14()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, p_std_dev=0.01, q_std_dev=0.01)
    success = estimate(net, algorithm="irwls", estimator="shgm", 
                       a=3, maximum_iterations=50)
    assert success
    assert np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, 1e-2)
    assert np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, 1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
    
