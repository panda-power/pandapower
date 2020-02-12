# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import estimate
from pandapower.estimation.util import add_virtual_meas_from_loadflow

from pandapower.estimation.algorithm.optimization import OptAlgorithm
from pandapower.estimation.algorithm.base import WLSAlgorithm
from pandapower.estimation.ppc_conversion import pp2eppci
from pandapower.estimation.results import eppci2pp
from copy import deepcopy


def test_case30_compare_classical_wls_opt_wls():
    net = nw.case30()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net)
    
    try:
        success = estimate(net, init='flat', algorithm="opt", estimator='wls')
        assert success
    except:
        # if failed give it a warm start
        net, ppc, eppci = pp2eppci(net)
        estimation_wls = WLSAlgorithm(1e-3, 3)
        estimation_opt = OptAlgorithm(1e-6, 1000)

        eppci = estimation_wls.estimate(eppci)
        eppci = estimation_opt.estimate(eppci, estimator="wls")
        assert estimation_opt.successful
        net = eppci2pp(net, ppc, eppci)

    net_wls = deepcopy(net)
    estimate(net_wls)
    assert np.allclose(net_wls.res_bus_est.vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(net_wls.res_bus_est.va_degree, net.res_bus_est.va_degree, atol=1e-2)


def test_lp_lav():
    net = nw.case14()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, p_std_dev=0.01, q_std_dev=0.01)

    estimate(net, algorithm="lp")

    assert np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=5e-2)  


def test_opt_lav():
    net = nw.case9()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, with_random_error=False)

    net, ppc, eppci = pp2eppci(net)
    estimation_wls = WLSAlgorithm(1e-3, 5)
    estimation_opt = OptAlgorithm(1e-6, 1000)

    eppci = estimation_wls.estimate(eppci)
    eppci = estimation_opt.estimate(eppci, estimator="lav")
    assert estimation_opt.successful
    net = eppci2pp(net, ppc, eppci)

    assert np.allclose(net.res_bus.vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(net.res_bus.va_degree, net.res_bus_est.va_degree, atol=5e-2)
    

def test_ql_qc():
    net = nw.case9()
    pp.runpp(net)
    add_virtual_meas_from_loadflow(net, p_std_dev=0.01, q_std_dev=0.01)
    pf_vm_pu, pf_va_degree = net.res_bus.vm_pu, net.res_bus.va_degree

    #  give it a warm start
    net, ppc, eppci = pp2eppci(net)
    estimation_wls = WLSAlgorithm(1e-3, 5)
    estimation_opt = OptAlgorithm(1e-6, 3000)

    eppci = estimation_wls.estimate(eppci)
    try:
        eppci = estimation_opt.estimate(eppci, estimator="ql", a=3)
        assert estimation_opt.successful
    except:
        eppci = estimation_opt.estimate(eppci, estimator="ql", a=3, opt_method="Newton-CG")
        assert estimation_opt.successful
    net = eppci2pp(net, ppc, eppci)

    assert np.allclose(pf_vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(pf_va_degree, net.res_bus_est.va_degree, atol=5e-2)

    # give it a warm start
    net, ppc, eppci = pp2eppci(net)
    estimation_wls = WLSAlgorithm(1e-6, 5)
    estimation_opt = OptAlgorithm(1e-6, 3000)

    eppci = estimation_wls.estimate(eppci)
    try:
        eppci = estimation_opt.estimate(eppci, estimator="qc", a=3)
        assert estimation_opt.successful
    except:
        eppci = estimation_opt.estimate(eppci, estimator="qc", a=3, opt_method="Newton-CG")
        assert estimation_opt.successful
    net = eppci2pp(net, ppc, eppci)

    assert np.allclose(pf_vm_pu, net.res_bus_est.vm_pu, atol=1e-2)
    assert np.allclose(pf_va_degree, net.res_bus_est.va_degree, atol=5e-2)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
