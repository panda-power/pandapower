# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc

@pytest.fixture
def motor_net():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, b1, s_sc_max_mva=10., rx_max=0.1)
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=1., r_ohm_per_km=0.3211,
                                   x_ohm_per_km=0.06911504, c_nf_per_km=0, max_i_ka=1)
    pp.create_sgen(net, b2, p_mw=0.11, sn_mva=0.5, type="motor", k=7, rx=0.6, current_source=False)
    return net

def test_motor(motor_net):
    net = motor_net
    sc.calc_sc(net)
    np.isclose(net.res_bus_sc.ikss_ka.at[0], 14.724523289, rtol=1e-4)
    np.isclose(net.res_bus_sc.ikss_ka.at[1], 6.1263193169, rtol=1e-4)
    #TODO: ip, ith, Ib, Ik with motors

if __name__ == '__main__':
    pytest.main([__file__])
