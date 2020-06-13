# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def test_opf_cigre():
    """ Testing a  simple network with transformer for loading
    constraints with OPF using a generator """
    # create net
    net = nw.create_cigre_network_mv(with_der="pv_wind")

    net.bus["max_vm_pu"] = 1.1
    net.bus["min_vm_pu"] = 0.9
    net.line["max_loading_percent"] = 200
    net.trafo["max_loading_percent"] = 100
    net.sgen["max_p_mw"] = net.sgen.sn_mva
    net.sgen["min_p_mw"] = 0
    net.sgen["max_q_mvar"] = 0.01
    net.sgen["min_q_mvar"] = -0.01
    net.sgen["controllable"] = True
    net.load["controllable"] = False
    net.sgen.in_service[net.sgen.bus == 4] = False
    net.sgen.in_service[net.sgen.bus == 6] = False
    net.sgen.in_service[net.sgen.bus == 8] = False
    net.sgen.in_service[net.sgen.bus == 9] = False

    # run OPF
    pp.runopp(net)
    assert net["OPF_converged"]


def test_some_sgens_not_controllable():
    """ Testing a  simple network with transformer for loading
    constraints with OPF using a generator """
    # create net
    net = nw.create_cigre_network_mv(with_der="pv_wind")

    net.bus["max_vm_pu"] = 1.1
    net.bus["min_vm_pu"] = 0.9
    net.line["max_loading_percent"] = 200
    net.trafo["max_loading_percent"] = 100
    net.sgen["max_p_mw"] = net.sgen.sn_mva
    net.sgen["min_p_mw"] = 0
    net.sgen["max_q_mvar"] = 0.01
    net.sgen["min_q_mvar"] = -0.01
    net.sgen["controllable"] = True
    net.load["controllable"] = False
    net.sgen.controllable[net.sgen.bus == 4] = False
    net.sgen.controllable[net.sgen.bus == 6] = False
    net.sgen.controllable[net.sgen.bus == 8] = False
    net.sgen.controllable[net.sgen.bus == 9] = False

    for sgen_idx, row in net["sgen"].iterrows():
        cost_sgen = pp.create_poly_cost(net, sgen_idx, 'sgen', cp1_eur_per_mw=1.)
        net.poly_cost.cp1_eur_per_mw.at[cost_sgen] = 100

    # run OPF
    pp.runopp(net, calculate_voltage_angles=False)
    assert net["OPF_converged"]
    # check if p_mw of non conrollable sgens are unchanged
    assert np.allclose(net.res_sgen.p_mw[net.sgen.controllable == False], net.sgen.p_mw[net.sgen.controllable == False])
    assert not np.allclose(net.res_sgen.p_mw[net.sgen.controllable == True],
                           net.sgen.p_mw[net.sgen.controllable == True])


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
