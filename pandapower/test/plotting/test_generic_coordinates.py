# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
import copy
from pandapower.test.toolbox import net_in
from pandapower.plotting.generic_geodata import create_generic_coordinates
try:
    import igraph
    IGRAPH_INSTALLED = True
except ImportError:
    IGRAPH_INSTALLED = False


@pytest.mark.skipif(IGRAPH_INSTALLED is False, reason="Requires python-igraph.")
def test_create_generic_coordinates_igraph(net_in):
    net = copy.deepcopy(net_in)
    net.bus_geodata.drop(net.bus_geodata.index, inplace=True)
    create_generic_coordinates(net, library="igraph")
    assert len(net.bus_geodata) == len(net.bus)


@pytest.mark.xfail(reason="The current implementation is not working properly, as multigraph edges "
                          "as AtlasViews are accessed with list logic.")
def test_create_generic_coordinates_nx(net_in):
    net = copy.deepcopy(net_in)
    net.bus_geodata.drop(net.bus_geodata.index, inplace=True)
    create_generic_coordinates(net, library="networkx")
    assert len(net.bus_geodata) == len(net.bus)


if __name__ == "__main__":
    pytest.main(["test_generic_coordinates.py"])
