# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.networks as pn


def test_mv_oberrhein():
    scenarios = ["load", "generation"]
    include_substations = [False, True]
    separation_by_sub = [False, True]
    
    for i in scenarios:
        for j in include_substations:
            for k in separation_by_sub:
                net = pn.mv_oberrhein(scenario=i, include_substations=j)
                pp.runpp(net)
    
                if i == "load":
                    assert net.sgen.scaling.mean() < 0.2
                    assert net.load.scaling.mean() > 0.5
                elif i == "generation":
                    net.sgen.scaling.mean() > 0.6
                    net.load.scaling.mean() < 0.2
    
                if j is False:
                    assert len(net.bus) == 179
                    assert len(net.trafo) == 2
                elif j is True:
                    assert len(net.bus) == 320
                    assert len(net.trafo) == 143
    
                assert len(net.line) == 181
                assert len(net.switch) == 322
                assert net.converged
                
                if k is True:
                    net0, net1 = pn.mv_oberrhein(scenario=i, include_substations=j, separation_by_sub=k)
                    assert len(net1.keys()) == len(net0.keys()) == len(net.keys())
                    assert net1.res_ext_grid.loc[1].all() == net.res_ext_grid.loc[1].all()
                    assert net0.res_ext_grid.loc[0].all() == net.res_ext_grid.loc[0].all() 
                    assert len(net.bus) == len(net0.bus) + len(net1.bus)

if __name__ == '__main__':
    pytest.main(['-x', "test_mv_oberrhein.py"])
