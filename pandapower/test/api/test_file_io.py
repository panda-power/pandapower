# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import copy
import json
import os
import sys

import numpy as np
import pandapower as pp
import pandapower.control as control
import pandapower.networks as networks
import pandapower.topology as topology
import pandas as pd
import pytest
from pandapower import pp_dir
from pandapower.io_utils import PPJSONEncoder, PPJSONDecoder
from pandapower.test.toolbox import assert_net_equal, create_test_network
from pandapower.timeseries import DFData

try:
    import geopandas as gpd
except ImportError:
    pass


@pytest.fixture(params=[1])
def net_in(request):
    if request.param == 1:
        net = create_test_network()
        net.line_geodata.loc[0, "coords"] = [(1.1, 2.2), (3.3, 4.4)]
        net.line_geodata.loc[11, "coords"] = [(5.5, 5.5), (6.6, 6.6), (7.7, 7.7)]
        return net


#    if request.param == 2:
#        return networks.case145()


def test_pickle(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.p"
    pp.to_pickle(net_in, filename)
    net_out = pp.from_pickle(filename)
    assert_net_equal(net_in, net_out)


def test_excel(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.xlsx"
    pp.to_excel(net_in, filename)
    net_out = pp.from_excel(filename)
    assert_net_equal(net_in, net_out)

    # test in user_pf_options are equal
    pp.set_user_pf_options(net_in, tolerance_mva=1e3)
    pp.to_excel(net_in, filename)
    net_out = pp.from_excel(filename)
    assert_net_equal(net_in, net_out)
    assert net_out.user_pf_options == net_in.user_pf_options


def test_json_basic(net_in, tmp_path):
    # tests the basic json functionality with the encoder/decoder classes
    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    with open(filename, 'w') as fp:
        json.dump(net_in, fp, cls=PPJSONEncoder)

    with open(filename) as fp:
        net_out = json.load(fp, cls=PPJSONDecoder)
        pp.convert_format(net_out)

    assert_net_equal(net_in, net_out)


def test_json_controller_none():
    try:
        pp.from_json(os.path.join(pp_dir, 'test', 'test_files', 'controller_containing_NoneNan.json'), convert=False)
    except:
        raise (UserWarning("empty net with controller containing Nan/None can't be loaded"))


def test_json(net_in, tmp_path):
    filename = os.path.join(os.path.abspath(str(tmp_path)), "testfile.json")
    try:
        net_geo = copy.deepcopy(net_in)
        # make GeodataFrame
        from shapely.geometry import Point, LineString
        import geopandas as gpd

        for tab in ('bus_geodata', 'line_geodata'):
            if tab == 'bus_geodata':
                geometry = net_geo[tab].apply(lambda x: Point(x.x, x.y), axis=1)
            else:
                geometry = net_geo[tab].coords.apply(LineString)
            net_geo[tab] = gpd.GeoDataFrame(net_geo[tab], geometry=geometry, crs=f"epsg:4326")

        pp.to_json(net_geo, filename)
        net_out = pp.from_json(filename)
        assert_net_equal(net_geo, net_out)
        # assert isinstance(net_out.line_geodata, gpd.GeoDataFrame)
        # assert isinstance(net_out.bus_geodata, gpd.GeoDataFrame)
        assert isinstance(net_out.bus_geodata.geometry.iat[0], Point)
        assert isinstance(net_out.line_geodata.geometry.iat[0], LineString)
    except (NameError, ImportError):
        pass

    # check if restore_all_dtypes works properly:
    net_in.line['test'] = 123
    net_in.res_line['test'] = 123
    pp.to_json(net_in, filename)
    net_out = pp.from_json(filename)
    assert_net_equal(net_in, net_out)


def test_encrypted_json(net_in, tmp_path):
    import cryptography.fernet
    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    pp.to_json(net_in, filename, encryption_key="verysecret")
    with pytest.raises(json.JSONDecodeError):
        pp.from_json(filename)
    with pytest.raises(cryptography.fernet.InvalidToken):
        pp.from_json(filename, encryption_key="wrong")
    net_out = pp.from_json(filename, encryption_key="verysecret")
    assert_net_equal(net_in, net_out)


def test_type_casting_json(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    net_in.sn_kva = 1000
    pp.to_json(net_in, filename)
    net = pp.from_json(filename)
    assert_net_equal(net_in, net)


def test_sqlite(net_in, tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.db"
    pp.to_sqlite(net_in, filename)
    net_out = pp.from_sqlite(filename)
    assert_net_equal(net_in, net_out)


def test_convert_format():  # TODO what is this thing testing ?
    net = pp.from_pickle(os.path.join(pp.pp_dir, "test", "api", "old_net.p"))
    pp.runpp(net)
    assert net.converged


def test_to_json_dtypes(tmp_path):
    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
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
    net = networks.mv_oberrhein()
    net.tuple = (1, "4")
    net.mg = topology.create_nxgraph(net)
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
    s = pd.Series(dtype=object)
    s["test"] = [(1, 2), (3, 4)]
    json_string = json.dumps(s, cls=PPJSONEncoder)
    s1 = json.loads(json_string, cls=PPJSONDecoder)
    assert (type(s["test"][0]) == type(s1["test"][0]))


def test_new_pp_object_io():
    net = networks.mv_oberrhein()
    ds = DFData(pd.DataFrame(data=np.array([[0, 1, 2], [7, 8, 9]])))
    control.ConstControl(net, 'sgen', 'p_mw', 42, profile_name=0, data_source=ds)
    control.ContinuousTapControl(net, 142, 1)

    obj = net.controller.object.at[0]
    obj.run = pp.runpp

    s = json.dumps(net, cls=PPJSONEncoder)

    net1 = json.loads(s, cls=PPJSONDecoder)

    obj1 = net1.controller.object.at[0]
    obj2 = net1.controller.object.at[1]

    assert isinstance(obj1, control.ConstControl)
    assert isinstance(obj2, control.ContinuousTapControl)
    assert obj1.run is pp.runpp
    assert isinstance(obj1.data_source, DFData)
    assert isinstance(obj1.data_source.df, pd.DataFrame)


def test_convert_format_for_pp_objects(net_in):
    pp.create_transformer(net_in, net_in.bus.index.values[0], net_in.bus.index.values[1],
                          '0.25 MVA 20/0.4 kV', tap_pos=0)
    c1 = control.ContinuousTapControl(net_in, 0, 1.02)
    c2 = control.DiscreteTapControl(net_in, 0, 1, 1)
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


def test_json_io_same_net(net_in, tmp_path):
    control.ConstControl(net_in, 'load', 'p_mw', 0)

    s = pp.to_json(net_in)
    net1 = pp.from_json_string(s)
    assert isinstance(net1.controller.object.at[0], control.ConstControl)

    filename = os.path.abspath(str(tmp_path)) + "testfile.json"
    pp.to_json(net_in, filename)
    net2 = pp.from_json(filename)
    assert isinstance(net2.controller.object.at[0], control.ConstControl)


def test_json_different_nets():
    net = networks.mv_oberrhein()
    net2 = networks.simple_four_bus_system()
    control.ContinuousTapControl(net, 114, 1.02)
    net.tuple = (1, "4")
    net.mg = topology.create_nxgraph(net)
    json_string = json.dumps([net, net2], cls=PPJSONEncoder)
    [net_out, net2_out] = json.loads(json_string, cls=PPJSONDecoder)
    assert_net_equal(net_out, net)
    assert_net_equal(net2_out, net2)
    pp.runpp(net_out, run_control=True)
    pp.runpp(net, run_control=True)
    assert_net_equal(net, net_out)


def test_deepcopy_controller():
    net = pp.networks.mv_oberrhein()
    control.ContinuousTapControl(net, 114, 1.01)
    net2 = copy.deepcopy(net)
    ct1 = net.controller.object.iloc[0]
    ct2 = net2.controller.object.iloc[0]
    assert ct1 != ct2
    assert ct1.equals(ct2)
    ct2.vm_set_pu = 1.02
    assert not ct1.equals(ct2)


@pytest.mark.skipif('geopandas' not in sys.modules, reason="requires the GeoPandas library")
def test_empty_geo_dataframe():
    net = pp.create_empty_network()
    net.bus_geodata['geometry'] = None
    net.bus_geodata = gpd.GeoDataFrame(net.bus_geodata)
    s = pp.to_json(net)
    net1 = pp.from_json_string(s)
    assert assert_net_equal(net, net1)

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
