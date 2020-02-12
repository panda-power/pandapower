# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
#from pandapower.auxiliary import _check_connectivity, _add_ppc_options
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pypower.makeLODF import makeLODF

from pandapower.test.loadflow.result_test_network_generator import result_test_network_generator_dcpp
from pandapower.test.toolbox import add_grid_connection, create_test_line, assert_net_equal

    
def test_PTDF():
    net = nw.case9()
    pp.runpp(net)
    _, ppci = _pd2ppc(net)
    
    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    assert ptdf.shape == (ppci["bus"].shape[0], ppci["branch"].shape[0])
    assert np.all(~np.isnan(ptdf))


def test_LODF():
    net = nw.case9()
    pp.runpp(net)
    _, ppci = _pd2ppc(net)
    
    ptdf = makePTDF(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    lodf = makeLODF(ppci["branch"], ptdf)
    assert lodf.shape == (ppci["branch"].shape[0], ppci["branch"].shape[0])
    assert np.all(~np.isnan(lodf))
    

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
