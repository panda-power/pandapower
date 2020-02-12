# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import json
import os

import numpy as np
import pandas as pd
import pytest

import pandapower as pp
import pandapower.control as ct
import pandapower.networks as nw
import pandapower.topology as top
from pandapower.io_utils import PPJSONEncoder, PPJSONDecoder
from pandapower.test.toolbox import assert_net_equal, create_test_network, tempdir, net_in
from pandapower.timeseries import DFData


def test_pickle(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.p")
    pp.to_pickle(net_in, filename)
    net_out = pp.from_pickle(filename)
    assert_net_equal(net_in, net_out)


def test_excel(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.xlsx")
    pp.to_excel(net_in, filename)
    net_out = pp.from_excel(filename)
    assert_net_equal(net_in, net_out)

    # test in user_pf_options are equal
    pp.set_user_pf_options(net_in, tolerance_mva=1e3)
    pp.to_excel(net_in, filename)
    net_out = pp.from_excel(filename)
    assert_net_equal(net_in, net_out)
    assert net_out.user_pf_options == net_in.user_pf_options


def test_json_basic(net_in, tempdir):
    # tests the basic json functionality with the encoder/decoder classes
    filename = os.path.join(tempdir, "testfile.json")
    with open(filename, 'w') as fp:
        json.dump(net_in, fp, cls=PPJSONEncoder)

    with open(filename) as fp:
        net_out = json.load(fp, cls=PPJSONDecoder)
        pp.convert_format(net_out)

    assert_net_equal(net_in, net_out)


def test_json(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.json")
    try:
        net_geo = copy.deepcopy(net_in)
        # make GeodataFrame
        from shapely.geometry import Point, LineString
        from fiona.crs import from_epsg
        import geopandas as gpd

        for tab in ('bus_geodata', 'line_geodata'):
            if tab == 'bus_geodata':
                geometry = net_geo[tab].apply(lambda x: Point(x.x, x.y), axis=1)
            else:
                geometry = net_geo[tab].coords.apply(LineString)
            net_geo[tab] = gpd.GeoDataFrame(net_geo[tab], geometry=geometry, crs=from_epsg(4326))

        pp.to_json(net_geo, filename)
        net_out = pp.from_json(filename)
        assert_net_equal(net_geo, net_out)
    except (NameError, ImportError):
        pass

    # check if restore_all_dtypes works properly:
    net_in.line['test'] = 123
    net_in.res_line['test'] = 123
    pp.to_json(net_in, filename)
    net_out = pp.from_json(filename)
    assert_net_equal(net_in, net_out)


def test_type_casting_json(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.json")
    net_in.sn_kva = 1000
    pp.to_json(net_in, filename)
    net = pp.from_json(filename)
    assert_net_equal(net_in, net)


def test_sqlite(net_in, tempdir):
    filename = os.path.join(tempdir, "testfile.db")
    pp.to_sqlite(net_in, filename)
    net_out = pp.from_sqlite(filename)
    assert_net_equal(net_in, net_out)


def test_convert_format():  # TODO what is this thing testing ?
    net = pp.from_pickle(os.path.join(pp.pp_dir, "test", "api", "old_net.p"))
    pp.runpp(net)
    assert net.converged


def test_to_json_dtypes(tempdir):
    filename = os.path.join(tempdir, "testfile.json")
    net = create_test_network()
    pp.runpp(net)
    net['res_test'] = pd.DataFrame(columns=['test'], data=[1, 2, 3])
    net['test'] = pd.DataFrame(columns=['test'], data=[1, 2, 3])
    net.line['test'] = 123
    net.res_line['test'] = 123
    net.bus['test'] = 123
    net.res_bus['test'] = 123
    net.res_load['test'] = 123
    pp.to_json(net, filename)
    net1 = pp.from_json(filename)
    assert_net_equal(net, net1)


def test_json_encoding_decoding():
    net = nw.mv_oberrhein()
    net.tuple = (1, "4")
    net.mg = top.create_nxgraph(net)
    s = set(['1', 4])
    t = tuple(['2', 3])
    f = frozenset(['12', 3])
    a = np.array([1., 2.])
    d = {"a": net, "b": f}
    json_string = json.dumps([s, t, f, net, a, d], cls=PPJSONEncoder)
    s1, t1, f1, net1, a1, d1 = json.loads(json_string, cls=PPJSONDecoder)

    assert s == s1
    assert t == t1
    assert f == f1
    assert net.tuple == net1.tuple
    assert np.allclose(a, a1)

    # TODO line_geodata isn't the same since tuples inside DataFrames are converted to lists
    #  (see test_json_tuple_in_dataframe)
    assert pp.nets_equal(net, net1, exclude_elms=["line_geodata"])
    assert pp.nets_equal(d["a"], d1["a"], exclude_elms=["line_geodata"])
    assert d["b"] == d1["b"]
    assert_graphs_equal(net.mg, net1.mg)


def test_dataframes_with_integer_columns():
    obj = pd.DataFrame(index=[1, 2, 3], columns=[0, 1])
    json_string = json.dumps(obj, cls=PPJSONEncoder)
    obj_loaded = json.loads(json_string, cls=PPJSONDecoder)
    assert all(obj.columns == obj_loaded.columns)


def assert_graphs_equal(mg1, mg2):
    edge1 = mg1.edges(data=True)
    edge2 = mg2.edges(data=True)
    for (u, v, data), (u1, v1, data1) in zip(sorted(edge1), sorted(edge2)):
        assert u == u1
        assert v == v1
        if "json_id" in data1:
            del data1["json_id"]
        if "json_key" in data1:
            del data1["json_key"]
        assert data == data1


@pytest.mark.xfail
def test_json_tuple_in_pandas():
    s = pd.Series()
    s["test"] = [(1, 2), (3, 4)]
    json_string = json.dumps(s, cls=PPJSONEncoder)
    s1 = json.loads(json_string, cls=PPJSONDecoder)
    assert (type(s["test"][0]) == type(s1["test"][0]))


def test_new_pp_object_io():
    net = nw.mv_oberrhein()
    ds = DFData(pd.DataFrame(data=np.array([[0, 1, 2], [7, 8, 9]])))
    pp.control.ConstControl(net, 'sgen', 'p_mw', 42, profile_name=0, data_source=ds)
    pp.control.ContinuousTapControl(net, 142, 1)

    obj = net.controller.object.at[0]
    obj.run = pp.runpp

    s = json.dumps(net, cls=PPJSONEncoder)

    net1 = json.loads(s, cls=PPJSONDecoder)

    obj1 = net1.controller.object.at[0]
    obj2 = net1.controller.object.at[1]

    assert obj1.net is net1
    assert obj2.net is net1
    assert obj1.run is pp.runpp
    assert isinstance(obj1.data_source, DFData)
    assert isinstance(obj1.data_source.df, pd.DataFrame)


def test_convert_format_for_pp_objects(net_in):
    pp.create_transformer(net_in, net_in.bus.index.values[0], net_in.bus.index.values[1],
                          '0.25 MVA 20/0.4 kV', tap_pos=0)
    c1 = pp.control.ContinuousTapControl(net_in, 0, 1.02)
    c2 = pp.control.DiscreteTapControl(net_in, 0, 1, 1)
    c1.u_set = 0.98
    c2.u_lower = 0.99
    c2.u_upper = 1.1
    # needed to trigger conversion
    net_in.version = "2.1.0"

    net_in.controller.rename(columns={'object': 'controller'}, inplace=True)
    assert 'controller' in net_in.controller.columns

    s = json.dumps(net_in, cls=PPJSONEncoder)
    net1 = pp.from_json_string(s, convert=True)

    assert 'controller' not in net1.controller.columns
    assert 'object' in net1.controller.columns

    obj1 = net1.controller.object.at[0]
    obj2 = net1.controller.object.at[1]

    assert not hasattr(obj1, 'u_set')
    assert not hasattr(obj2, 'u_lower')
    assert not hasattr(obj2, 'u_upper')
    assert obj1.vm_set_pu == 0.98
    assert obj2.vm_lower_pu == 0.99
    assert obj2.vm_upper_pu == 1.1


def test_json_io_same_net(net_in, tempdir):
    pp.control.ConstControl(net_in, 'load', 'p_mw', 0)

    s = pp.to_json(net_in)
    net1 = pp.from_json_string(s)
    assert net1.controller.object.at[0].net is net1

    filename = os.path.join(tempdir, "testfile.json")
    pp.to_json(net_in, filename)
    net2 = pp.from_json(filename)
    assert net2.controller.object.at[0].net is net2


def test_deepcopy_controller():
    net = nw.mv_oberrhein()
    ct.ContinuousTapControl(net, 114, 1.01)
    assert net == net.controller.object.iloc[0].net
    net2 = copy.deepcopy(net)
    assert net2 == net2.controller.object.iloc[0].net
    net3 = copy.copy(net)
    assert net3 == net3.controller.object.iloc[0].net


if __name__ == "__main__":
    pytest.main([__file__, "-x"])
