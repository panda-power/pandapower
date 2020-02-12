# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
from pandapower.test.loadflow.result_test_network_generator import result_test_network_generator


@pytest.fixture(scope="session")
def simple_network():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, name="bus1", vn_kv=10.)
    pp.create_ext_grid(net, b1)
    b2 = pp.create_bus(net, name="bus2", geodata=(1, 2))
    b3 = pp.create_bus(net, name="bus3", geodata=(1, 3))
    b4 = pp.create_bus(net, name="bus4", vn_kv=10.)
    pp.create_transformer(net, b4, b2,
                          std_type="0.25 MVA 10/0.4 kV",
                          name=None, in_service=True, index=None)
    pp.create_line(net, b2, b3, 1, name="line1",
                   std_type="NAYY 4x150 SE",
                   geodata=np.array([[1, 2], [3, 4]]))
    pp.create_line(net, b1, b4, 1, name="line2",
                   std_type="NAYY 4x150 SE")
    pp.create_load(net, b2, p_mw=0.01, q_mvar=0, name="load1")
    pp.create_load(net, b3, p_mw=0.04, q_mvar=0.002, name="load2")
    pp.create_gen(net, 3, q_mvar=0.020, vm_pu=1.0)
    pp.create_sgen(net, 2, p_mw=0.050, sn_mva=0.1)
    return net



@pytest.fixture(scope="session")
def result_test_network():
    for net in result_test_network_generator():
        pass
    pp.runpp(net, trafo_model="t", trafo_loading="current")
    return net

if __name__ == '__main__':
    net = result_test_network()
    # pp.rundcpp(net)