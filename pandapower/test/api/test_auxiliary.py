# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
import gc
import copy
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    import shapely.geometry

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

from pandapower.auxiliary import get_indices

import pandapower as pp
import pandapower.networks
import pandapower.control
import pandapower.timeseries


class MemoryLeakDemo:
    """
    Dummy class to demonstrate memory leaks
    """

    def __init__(self, net):
        self.net = net
        # it is interesting, that if "self" is just an attribute of net, there are no problems
        # if "self" is saved in a DataFrame, it causes a memory leak
        net['memory_leak_demo'] = pd.DataFrame(data=[self], columns=['object'])


class MemoryLeakDemoDF:
    """
    Dummy class to demonstrate memory leaks
    """

    def __init__(self, df):
        self.df = df
        # if "self" is saved in a DataFrame, it causes a memory leak
        df.loc[0, 'object'] = self


class MemoryLeakDemoDict:
    """
    Dummy class to demonstrate memory leaks
    """

    def __init__(self, d):
        self.d = d
        d['object'] = self


def test_get_indices():
    a = [i + 100 for i in range(10)]
    lookup = {idx: pos for pos, idx in enumerate(a)}
    lookup["before_fuse"] = a

    # First without fused buses no magic here
    # after fuse
    result = get_indices([102, 107], lookup, fused_indices=True)
    assert np.array_equal(result, [2, 7])

    # before fuse
    result = get_indices([2, 7], lookup, fused_indices=False)
    assert np.array_equal(result, [102, 107])

    # Same setup EXCEPT we have fused buses now (bus 102 and 107 are fused)
    lookup[107] = lookup[102]

    # after fuse
    result = get_indices([102, 107], lookup, fused_indices=True)
    assert np.array_equal(result, [2, 2])

    # before fuse
    result = get_indices([2, 7], lookup, fused_indices=False)
    assert np.array_equal(result, [102, 107])


def test_net_deepcopy():
    net = pp.networks.example_simple()
    net.line_geodata.loc[0, 'coords'] = [[0, 1], [1, 2]]
    net.bus_geodata.loc[0, ['x', 'y']] = 0, 1

    pp.control.ContinuousTapControl(net, tid=0, vm_set_pu=1)
    ds = pp.timeseries.DFData(pd.DataFrame(data=[[0, 1, 2], [3, 4, 5]]))
    pp.control.ConstControl(net, element='load', variable='p_mw', element_index=[0], profile_name=[0], data_source=ds)

    net1 = copy.deepcopy(net)
    assert net1.controller.object.at[0].net is net1
    assert net1.controller.object.at[1].net is net1

    assert not net1.controller.object.at[0].net is net
    assert not net1.controller.object.at[1].net is net

    assert not net1.controller.object.at[1].data_source is ds
    assert not net1.controller.object.at[1].data_source.df is ds.df

    assert not net1.line_geodata.coords.at[0] is net.line_geodata.coords.at[0]

    if GEOPANDAS_INSTALLED:
        for tab in ('bus_geodata', 'line_geodata'):
            if tab == 'bus_geodata':
                geometry = net[tab].apply(lambda x: shapely.geometry.Point(x.x, x.y), axis=1)
            else:
                geometry = net[tab].coords.apply(shapely.geometry.LineString)
            net[tab] = gpd.GeoDataFrame(net[tab], geometry=geometry)
        net1 = net.deepcopy()
        assert isinstance(net1.line_geodata, gpd.GeoDataFrame)
        assert isinstance(net1.bus_geodata, gpd.GeoDataFrame)
        assert isinstance(net1.bus_geodata.geometry.iat[0], shapely.geometry.Point)
        assert isinstance(net1.line_geodata.geometry.iat[0], shapely.geometry.LineString)


def test_memory_leaks():
    net = pp.networks.example_simple()

    # first, test to check that there are no memory leaks
    types_dict1 = pp.toolbox.get_gc_objects_dict()
    num = 3
    for _ in range(num):
        net_copy = copy.deepcopy(net)
        # In each net copy it has only one controller
        pp.control.ContinuousTapControl(net_copy, tid=0, vm_set_pu=1)

    gc.collect()

    types_dict2 = pp.toolbox.get_gc_objects_dict()

    assert types_dict2[pandapower.auxiliary.pandapowerNet] - types_dict1[pandapower.auxiliary.pandapowerNet] == 1
    assert types_dict2[pandapower.control.ContinuousTapControl] - types_dict1.get(
        pandapower.control.ContinuousTapControl, 0) == 1


def test_memory_leaks_demo():
    net = pp.networks.example_simple()
    # first, test to check that there are no memory leaks
    types_dict1 = pp.toolbox.get_gc_objects_dict()
    # now, demonstrate how a memory leak occurs
    # emulates the earlier behavior before the fix with weakref
    num = 3
    for _ in range(num):
        net_copy = copy.deepcopy(net)
        MemoryLeakDemo(net_copy)

    # demonstrate how the garbage collector doesn't remove the objects even if called explicitly
    gc.collect()
    types_dict2 = pp.toolbox.get_gc_objects_dict()
    assert types_dict2[pandapower.auxiliary.pandapowerNet] - types_dict1[pandapower.auxiliary.pandapowerNet] == num
    assert types_dict2[MemoryLeakDemo] - types_dict1.get(MemoryLeakDemo, 0) == num


def test_memory_leaks_no_copy():
    types_dict0 = pp.toolbox.get_gc_objects_dict()
    num = 3
    for _ in range(num):
        net = pp.create_empty_network()
        # In each net copy it has only one controller
        pp.control.ConstControl(net, 'sgen', 'p_mw', 0)

    gc.collect()
    types_dict1 = pp.toolbox.get_gc_objects_dict()
    assert types_dict1[pandapower.control.ConstControl] - types_dict0.get(pandapower.control.ConstControl, 0) == 1
    assert types_dict1[pandapower.auxiliary.pandapowerNet] - types_dict0.get(pandapower.auxiliary.pandapowerNet, 0) <= 1


def test_memory_leak_no_copy_demo():
    types_dict1 = pp.toolbox.get_gc_objects_dict()
    # now, demonstrate how a memory leak occurs
    # emulates the earlier behavior before the fix with weakref
    num = 3
    for _ in range(num):
        net = pp.networks.example_simple()
        MemoryLeakDemo(net)

    # demonstrate how the garbage collector doesn't remove the objects even if called explicitly
    gc.collect()
    types_dict2 = pp.toolbox.get_gc_objects_dict()
    assert types_dict2[pandapower.auxiliary.pandapowerNet] - \
           types_dict1.get(pandapower.auxiliary.pandapowerNet, 0) >= num-1
    assert types_dict2[MemoryLeakDemo] - types_dict1.get(MemoryLeakDemo, 0) == num


def test_memory_leak_df():
    types_dict1 = pp.toolbox.get_gc_objects_dict()
    num = 3
    for _ in range(num):
        df = pd.DataFrame()
        MemoryLeakDemoDF(df)

    gc.collect()
    types_dict2 = pp.toolbox.get_gc_objects_dict()
    assert types_dict2[MemoryLeakDemoDF] - types_dict1.get(MemoryLeakDemoDF, 0) == num


def test_memory_leak_dict():
    types_dict1 = pp.toolbox.get_gc_objects_dict()
    num = 3
    for _ in range(num):
        d = dict()
        MemoryLeakDemoDict(d)

    gc.collect()
    types_dict2 = pp.toolbox.get_gc_objects_dict()
    assert types_dict2[MemoryLeakDemoDict] - types_dict1.get(MemoryLeakDemoDict, 0) <= 1


if __name__ == '__main__':
    pytest.main([__file__, "-x"])
