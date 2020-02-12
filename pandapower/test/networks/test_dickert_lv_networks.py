# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.networks as pn

@pytest.mark.slow
def test_dickert_lv_networks():
    net_data = [('short', 'cable', 'single', 'good', 1),
                ('short', 'cable', 'single', 'average', 1),
                ('short', 'cable', 'single', 'bad', 2),
                ('short', 'cable', 'multiple', 'good', 2*3),
                ('short', 'cable', 'multiple', 'average', 2*6),
                ('short', 'cable', 'multiple', 'bad', 2*10),
                ('middle', 'cable', 'multiple', 'good', 3*15),
                ('middle', 'cable', 'multiple', 'average', 3*20),
                ('middle', 'cable', 'multiple', 'bad', 3*25),
                ('middle', 'C&OHL', 'multiple', 'good', 3*10),
                ('middle', 'C&OHL', 'multiple', 'average', 3*13),
                ('middle', 'C&OHL', 'multiple', 'bad', 3*16),
                ('long', 'cable', 'multiple', 'good', 3*30),
                ('long', 'cable', 'multiple', 'average', 3*40),
                ('long', 'cable', 'multiple', 'bad', 3*50),
                ('long', 'C&OHL', 'multiple', 'good', 3*20),
                ('long', 'C&OHL', 'multiple', 'average', 3*30),
                ('long', 'C&OHL', 'multiple', 'bad', 3*40)]
    for i in net_data:
        net = pn.create_dickert_lv_network(i[0], i[1], i[2], i[3])
        assert net.bus.shape[0] == i[4] + 2
        pp.runpp(net)
        assert net.converged


if __name__ == '__main__':
    pytest.main(['-x', "test_dickert_lv_networks.py"])
