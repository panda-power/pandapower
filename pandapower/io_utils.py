# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import copy
import importlib
import json
import numbers
import os
import pickle
import sys
import types
import weakref
from collections import defaultdict
from functools import partial
from inspect import isclass, _findclass
from json.decoder import WHITESPACE, WHITESPACE_STR, JSONArray
from warnings import warn

import networkx
import numpy
import pandas as pd
from networkx.readwrite import json_graph
from numpy import ndarray, generic, equal, isnan, allclose, any as anynp
from packaging import version
from pandapower.auxiliary import pandapowerNet
from pandapower.create import create_empty_network
from pandas.testing import assert_series_equal, assert_frame_equal

try:
    from functools import singledispatch
except ImportError:
    # Python 2.7
    from singledispatch import singledispatch

try:
    import fiona
    import fiona.crs
    import geopandas

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

try:
    import shapely.geometry

    SHAPELY_INSTALLED = True
except (ImportError, OSError):
    SHAPELY_INSTALLED = False

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


OBJ_ADDR = "_object_with_address"
WEAKREF_ADDR = "_weakref_to_address"


def coords_to_df(value, geotype="line"):
    columns = ["x", "y", "coords"] if geotype == "bus" else ["coords"]
    geo = pd.DataFrame(columns=columns, index=value.index)
    if any(~value.coords.isnull()):
        k = max(len(v) for v in value.coords.values)
        v = numpy.empty((len(value), k * 2))
        v.fill(numpy.nan)
        for i, idx in enumerate(value.index):
            # get coords and convert them to x1, y1, x2, y2...
            coords = value.at[idx, 'coords']
            if coords is None:
                continue
            v[i, :len(coords) * 2] = numpy.array(coords).flatten()
        geo = pd.DataFrame(v, index=value.index)
        geo.columns = ["%s%i" % (w, i) for i in range(k) for w in "xy"]
    if geotype == "bus":
        geo["x"] = value["x"].values
        geo["y"] = value["y"].values
    return geo


def to_dict_of_dfs(net, include_results=False, fallback_to_pickle=True, include_empty_tables=True):
    dodfs = dict()
    dtypes = []
    dodfs["parameters"] = dict()  # pd.DataFrame(columns=["parameter"])
    for item, value in net.items():
        # dont save internal variables and results (if not explicitely specified)
        if item.startswith("_") or (item.startswith("res") and not include_results):
            continue
        elif item == "std_types":
            for t in net.std_types.keys():  # which are ["line", "trafo", "trafo3w"]
                if net.std_types[t]:  # avoid empty excel sheets for std_types if empty
                    dodfs["%s_std_types" % t] = pd.DataFrame(net.std_types[t]).T
            continue
        elif item == "profiles":
            for t in net.profiles.keys():  # which could be e.g. "sgen", "gen", "load", ...
                if net.profiles[t].shape[0]:  # avoid empty excel sheets for std_types if empty
                    dodfs["%s_profiles" % t] = pd.DataFrame(net.profiles[t])
            continue
        elif item == "user_pf_options":
            if len(value) > 0:
                dodfs["user_pf_options"] = pd.DataFrame(value, index=[0])
            continue
        elif isinstance(value, (int, float, bool, str)):
            # attributes of primitive types are just stored in a DataFrame "parameters"
            dodfs["parameters"][item] = net[item]
            continue
        elif not isinstance(value, pd.DataFrame):
            logger.warning("Could not serialize net.%s" % item)
            continue

        # value is pandas DataFrame
        if not include_empty_tables and value.empty:
            continue

        if item == "bus_geodata":
            geo = coords_to_df(value, geotype="bus")
            if GEOPANDAS_INSTALLED and isinstance(value, geopandas.GeoDataFrame):
                geo["geometry"] = [s.to_wkt() for s in net.bus_geodata.geometry.values]
            dodfs[item] = geo
        elif item == "line_geodata":
            geo = coords_to_df(value, geotype="line")
            if GEOPANDAS_INSTALLED and isinstance(value, geopandas.GeoDataFrame):
                geo["geometry"] = [s.to_wkt() for s in net.line_geodata.geometry.values]
            dodfs[item] = geo
        else:
            dodfs[item] = value
        # save dtypes
        for column, dtype in value.dtypes.iteritems():
            dtypes.append((item, column, str(dtype)))
    dodfs["dtypes"] = pd.DataFrame(dtypes, columns=["element", "column", "dtype"])
    dodfs["parameters"] = pd.DataFrame(dodfs["parameters"], index=[0])
    return dodfs


def dicts_to_pandas(json_dict):
    warn("This function is deprecated and will be removed in a future release.\r\n"
         "Please resave your grid using the current pandapower version.", DeprecationWarning)
    pd_dict = dict()
    for k in sorted(json_dict.keys()):
        if isinstance(json_dict[k], dict):
            pd_dict[k] = pd.DataFrame.from_dict(json_dict[k], orient="columns")
            if pd_dict[k].shape[0] == 0:  # skip empty dataframes
                continue
            if pd_dict[k].index[0].isdigit():
                pd_dict[k].set_index(pd_dict[k].index.astype(numpy.int64), inplace=True)
        else:
            raise UserWarning("The network is an old version or corrupt. "
                              "Try to use the old load function")
    return pd_dict


def df_to_coords(net, item, table):
    # converts dataframe to coords in net
    num_points = len(table.columns) // 2
    net[item] = pd.DataFrame(index=table.index, columns=net[item].columns)
    if item == "bus_geodata":
        num_points -= 1
        net[item].loc[:, ['x', 'y']] = table.loc[:, ['x', 'y']]

    for i in table.index:
        coords = table.loc[i]
        # for i, coords in table.iterrows():
        coord = [(coords["x%u" % nr], coords["y%u" % nr]) for nr in range(num_points)
                 if pd.notnull(coords["x%u" % nr])]
        if len(coord):
            net[item].loc[i, "coords"] = coord


def from_dict_of_dfs(dodfs):
    net = create_empty_network()
    for c in dodfs["parameters"].columns:
        net[c] = dodfs["parameters"].at[0, c]
    for item, table in dodfs.items():
        if item in ("parameters", "dtypes"):
            continue
        elif item in ["line_geodata", "bus_geodata"]:
            df_to_coords(net, item, table)
        elif item.endswith("_std_types"):
            net["std_types"][item[:-10]] = table.T.to_dict()
            continue  # don't go into try..except
        elif item.endswith("_profiles"):
            if "profiles" not in net.keys():
                net["profiles"] = dict()
            net["profiles"][item[:-9]] = table
            continue  # don't go into try..except
        elif item == "user_pf_options":
            net['user_pf_options'] = {c: v for c, v in zip(table.columns, table.values[0])}
            continue  # don't go into try..except
        else:
            net[item] = table
        # set the index to be Int64Index
        try:
            net[item].set_index(net[item].index.astype(numpy.int64), inplace=True)
        except TypeError:
            # TypeError: if not int64 index (e.g. str)
            pass
    restore_all_dtypes(net, dodfs["dtypes"])
    return net


def restore_all_dtypes(net, dtypes):
    for _, v in dtypes.iterrows():
        try:
            if v["dtype"] == "object":
                c = net[v.element][v.column]
                net[v.element][v.column] = numpy.where(c.isnull(), None, c)
                # net[v.element][v.column] = net[v.element][v.column].fillna(value=None)
            net[v.element][v.column] = net[v.element][v.column].astype(v["dtype"])
        except KeyError:
            pass


def to_dict_with_coord_transform(net, point_geo_columns, line_geo_columns):
    save_net = dict()
    for key, item in net.items():
        if hasattr(item, "columns") and "geometry" in item.columns:
            # we convert shapely-objects to primitive data-types on a deepcopy
            item = copy.deepcopy(item)
            if key in point_geo_columns and not isinstance(item.geometry.values[0], tuple):
                item["geometry"] = item.geometry.apply(lambda x: (x.x, x.y))
            elif key in line_geo_columns and not isinstance(item.geometry.values[0], list):
                item["geometry"] = item.geometry.apply(lambda x: list(x.coords))

        save_net[key] = {"DF": item.to_dict("split"),
                         "dtypes": {col: dt for col, dt in zip(item.columns, item.dtypes)}} \
            if isinstance(item, pd.DataFrame) else item
    return save_net


def get_raw_data_from_pickle(filename):
    def read(f):
        if sys.version_info >= (3, 0):
            return pickle.load(f, encoding='latin1')
        else:
            return pickle.load(f)

    if hasattr(filename, 'read'):
        net = read(filename)
    elif not os.path.isfile(filename):
        raise UserWarning("File %s does not exist!!" % filename)
    else:
        with open(filename, "rb") as f:
            net = read(f)
    return net


def transform_net_with_df_and_geo(net, point_geo_columns, line_geo_columns):
    try:
        epsg = net.gis_epsg_code
    except AttributeError:
        epsg = None

    for key, item in net.items():
        if isinstance(item, dict) and "DF" in item:
            df_dict = item["DF"]
            if "columns" in df_dict:
                # make sure the index is Int64Index
                try:
                    df_index = pd.Int64Index(df_dict['index'])
                except TypeError:
                    df_index = df_dict['index']
                if GEOPANDAS_INSTALLED and "geometry" in df_dict["columns"] \
                        and epsg is not None:
                    # convert primitive data-types to shapely-objects
                    if key in point_geo_columns:
                        data = {"x": [row[0] for row in df_dict["data"]],
                                "y": [row[1] for row in df_dict["data"]]}
                        geo = [shapely.geometry.Point(row[2][0], row[2][1]) for row in df_dict["data"]]
                    elif key in line_geo_columns:
                        data = {"coords": [row[0] for row in df_dict["data"]]}
                        geo = [shapely.geometry.LineString(row[1]) for row in df_dict["data"]]

                    net[key] = geopandas.GeoDataFrame(data, crs=fiona.crs.from_epsg(epsg),
                                                      geometry=geo, index=df_index)
                else:
                    net[key] = pd.DataFrame(columns=df_dict["columns"], index=df_index,
                                            data=df_dict["data"])
            else:
                net[key] = pd.DataFrame.from_dict(df_dict)
                if "columns" in item:
                    if version.parse(pd.__version__) < version.parse("0.21"):
                        net[key] = net[key].reindex_axis(item["columns"], axis=1)
                    else:
                        net[key] = net[key].reindex(item["columns"], axis=1)

            if "dtypes" in item:
                if "columns" in df_dict and "geometry" in df_dict["columns"]:
                    pass
                else:
                    try:
                        # only works with pandas 0.19 or newer
                        net[key] = net[key].astype(item["dtypes"])
                    except:
                        # works with pandas <0.19
                        for column in net[key].columns:
                            net[key][column] = net[key][column].astype(item["dtypes"][column])


def isinstance_partial(obj, cls):
    if isinstance(obj, (pandapowerNet, tuple)):
        return False
    return isinstance(obj, cls)


class PPJSONEncoder(json.JSONEncoder):
    def __init__(self, isinstance_func=isinstance_partial, obj_memo=None, **kwargs):
        """
        Special JSON Encoder for pandapower. Differences from normal encoding:
          - the default function is overwritten so that a to_serializable function can be defined\
            for different classes
          - the isinstance function includes pandapower networks and tuples, because they shall be\
            saved in a special format (not normal list or dict) if possible
          - an onj_memo dict collects objects that have already been serialized to avoid duplicated\
            serialization and deserialization of two distinct objects of which one is just a\
            reference

        :param isinstance_func: function to be handed to the encoder (to identify lists/ dicts)
        :type isinstance_func: function
        :param obj_memo: object memory dictionary (can be filled if an upper level encoder has\
                         already collected objects and wants to hand them over to a lower level\
                         encoder)
        :type obj_memo: dict
        :param kwargs: keyword arguments handed to json.JSONEncoder
        :type kwargs: dict
        """
        super(PPJSONEncoder, self).__init__(**kwargs)
        self.isinstance_func = isinstance_func
        self.obj_memo = list() if obj_memo is None else obj_memo

    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string
        representation as available.

        For example::

            for chunk in JSONEncoder().iterencode(bigobject):
                mysocket.write(chunk)

        """
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = json.encoder.encode_basestring_ascii
        else:
            _encoder = json.encoder.encode_basestring

        def floatstr(o, allow_nan=self.allow_nan, _repr=float.__repr__, _inf=json.encoder.INFINITY,
                     _neginf=-json.encoder.INFINITY):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if o != o:
                text = 'NaN'
            elif o == _inf:
                text = 'Infinity'
            elif o == _neginf:
                text = '-Infinity'
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(
                    "Out of range float values are not JSON compliant: " + repr(o))

            return text

        _iterencode = json.encoder._make_iterencode(
            markers, self.default, _encoder, self.indent, floatstr,
            self.key_separator, self.item_separator, self.sort_keys,
            self.skipkeys, _one_shot, isinstance=self.isinstance_func)
        return _iterencode(o, 0)

    def default(self, o):
        try:
            s = to_serializable(o, self.obj_memo)
        except TypeError:
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, o)
        else:
            return s


def check_object_at_address(o):
    return isinstance(o, dict) and OBJ_ADDR in o.keys()


def check_weakref_at_address(o):
    return isinstance(o, dict) and WEAKREF_ADDR in o.keys()


def check_dict(o):
    return isinstance(o, dict)


check_obj_vect = numpy.vectorize(check_object_at_address)
check_weakref_vect = numpy.vectorize(check_weakref_at_address)
check_dict_vect = numpy.vectorize(check_dict)


def get_obj_idx_addr_series(s):
    return [(idx, oa[OBJ_ADDR]) for idx, oa in s[check_obj_vect(s.values)].items()]


def get_weakref_idx_addr_series(s):
    return [(idx, oa[WEAKREF_ADDR]) for idx, oa in s[check_obj_vect(s.values)].items()]


def get_obj_idx_addr_iterator(iterator):
    return [(k, v[OBJ_ADDR]) for k, v in iterator if check_object_at_address(v)]


def get_weakref_idx_addr_iterator(iterator):
    return [(k, v[WEAKREF_ADDR]) for k, v in iterator if check_weakref_at_address(v)]


def get_idx_addr_array(a, ref_type=OBJ_ADDR):
    idx_addr = list()

    if a.shape == (0,):
        return idx_addr

    check_func = check_weakref_vect if "weakref" in ref_type else check_obj_vect

    if len(a.shape) == 1:
        has_obj = check_func(a)
        return list(zip(numpy.where(has_obj)[0], [v[ref_type] for v in a[has_obj]]))

    shapelist = [list(range(dim)) for dim in a.shape[1:]]
    all_idx_combinations = [[]]
    for x in shapelist:
        all_idx_combinations = [i + [y] for y in x for i in all_idx_combinations]
    for idx_comb in all_idx_combinations:
        current_slice = tuple([slice(None, None, None)] + idx_comb)
        has_obj = check_func(a[current_slice])
        current_idx = [tuple([ho] + idx_comb) for ho in numpy.where(has_obj)[0]]
        idx_addr.extend([(ho, a[ho][ref_type]) for ho in current_idx])


def change_objects_series(s, hook):
    if s.dtype == "O" and len(s) > 0:
        if numpy.any(check_dict_vect(s)):
            underlying_objects = get_obj_idx_addr_series(s)
            weakrefs = get_weakref_idx_addr_series(s)
            s = s.apply(lambda obj: hook(obj) if isinstance(obj, dict) else obj)
            s = s.astype("O")
            return s, underlying_objects, weakrefs
    return s, [], []


class FromSerializable:
    def __init__(self):
        self.class_name = 'class_name'
        self.module_name = 'module_name'
        self.registry = {}

    def __get__(self, instance, owner):
        if instance is None:
            return self
        class_module = getattr(instance, self.class_name), getattr(instance, self.module_name)
        if not class_module in self.registry:
            class_module = ('', '')
        method = self.registry[class_module]
        return method.__get__(instance, owner)

    def register(self, class_name='', module_name=''):
        def decorator(method):
            self.registry[(class_name, module_name)] = method
            return method

        return decorator


class FromSerializableRegistry:
    from_serializable = FromSerializable()
    class_name = ''
    module_name = ''

    def __init__(self, obj, d, pp_hook_funct, memo_pp, addresses_to_fill, weakrefs_to_fill):
        """
        Class containing all registered function for deserialization.

        :param obj: object to be deserialized (usually string)
        :type obj: str
        :param d: dictionary with additional information
        :type d: dict
        :param pp_hook_funct: the object hook to be handed to underlying decoders
        :type pp_hook_funct: function
        :param memo_pp: object memory dict to be filled with the object if necessary
        :type memo_pp: dict
        :param addresses_to_fill: dict mapping addresses to setter functions where to insert the\
                                  objects restored for these addresses (saved in memo_pp)
        :type addresses_to_fill: defaultdict
        :param weakrefs_to_fill: dict mapping addresses to setter functions where to insert the\
                                 weakrefs of objects restored for these addresses (saved in memo_pp)
        :type weakrefs_to_fill: defaultdict
        """
        self.obj = obj
        self.d = d
        self.pp_hook = pp_hook_funct
        self.memo_pp = memo_pp
        self.addresses_to_fill = addresses_to_fill
        self.weakrefs_to_fill = weakrefs_to_fill
        self.underlying_objects = list()
        self.weakrefs = list()

    @from_serializable.register(class_name='Series', module_name='pandas.core.series')
    def series(self):
        s = pd.read_json(self.obj, precise_float=True, **self.d)
        s, self.underlying_objects, self.weakrefs = change_objects_series(s, self.pp_hook)
        return s

    @from_serializable.register(class_name='DataFrame', module_name='pandas.core.frame')
    def dataframe(self):
        df = pd.read_json(self.obj, precise_float=True, convert_axes=False, **self.d)
        try:
            df.set_index(df.index.astype(numpy.int64), inplace=True)
        except (ValueError, TypeError, AttributeError):
            logger.debug("failed setting int64 index")
        for col in df.columns:
            df[col], uo, wr = change_objects_series(df[col], self.pp_hook)
            self.underlying_objects.extend(uo)
            self.weakrefs.extend(wr)
        return df

    @from_serializable.register(class_name='pandapowerNet', module_name='pandapower.auxiliary')
    def pandapower_net(self):
        if isinstance(self.obj, str):  # backwards compatibility
            from pandapower import from_json_string
            net = from_json_string(self.obj)
        else:
            net = create_empty_network()
            net.update(self.obj)
        self.underlying_objects = get_obj_idx_addr_iterator(net.items())
        self.weakrefs = get_weakref_idx_addr_iterator(net.items())
        return net

    @from_serializable.register(class_name="MultiGraph", module_name="networkx")
    def networkx(self):
        return json_graph.adjacency_graph(self.obj, attrs={'id': 'json_id', 'key': 'json_key'})

    @from_serializable.register(class_name="method")
    def method(self):
        logger.warning('deserializing of method not implemented')
        # class_ = getattr(module, obj) # doesn't work
        return self.obj

    @from_serializable.register(class_name='function', module_name='pandapower.run')
    def function(self):
        module = importlib.import_module(self.module_name)
        class_ = getattr(module, self.obj)  # works
        return class_

    @from_serializable.register()
    def rest(self):
        module = importlib.import_module(self.module_name)
        class_ = getattr(module, self.class_name)
        if isclass(class_) and issubclass(class_, JSONSerializableClass):
            if isinstance(self.obj, str):
                self.obj = json.loads(
                    self.obj, cls=PPJSONDecoder, memo_pp=self.memo_pp,
                    addresses_to_fill=self.addresses_to_fill,
                    weakrefs_to_fill=self.weakrefs_to_fill, set_addresses=False)
                # backwards compatibility
            js_obj = class_.from_dict(self.obj)
            self.underlying_objects = get_obj_idx_addr_iterator(js_obj.__dict__.items())
            self.weakrefs = get_weakref_idx_addr_iterator(js_obj.__dict__.items())
        else:
            # for non-pp objects, e.g. tuple
            js_obj = class_(self.obj, **self.d)
            typ = type(js_obj)
            if typ in [list, tuple]:
                self.underlying_objects = get_obj_idx_addr_iterator(enumerate(js_obj))
                self.weakrefs = get_weakref_idx_addr_iterator(enumerate(js_obj))
            elif typ == dict:
                self.underlying_objects = get_obj_idx_addr_iterator(js_obj.items())
                self.weakrefs = get_weakref_idx_addr_iterator(js_obj.items())
            elif typ == numpy.ndarray:
                self.underlying_objects = get_idx_addr_array(js_obj)
                self.weakrefs = get_idx_addr_array(js_obj, ref_type=WEAKREF_ADDR)
            else:
                logger.warning("The object of type %s cannot be checked for underlying objects!"
                               % type(js_obj))
        return js_obj

    if GEOPANDAS_INSTALLED:
        @from_serializable.register(class_name='GeoDataFrame')
        def geodataframe(self):
            df = geopandas.GeoDataFrame.from_features(fiona.Collection(self.obj),
                                                      crs=self.d['crs']).astype(self.d['dtype'])
            if "id" in df:
                df.set_index(df['id'].values.astype(numpy.int64), inplace=True)
            # coords column is not handled properly when using from_features
            if 'coords' in df:
                # df['coords'] = df.coords.apply(json.loads)
                valid_coords = ~pd.isnull(df.coords)
                df.loc[valid_coords, 'coords'] = df.loc[valid_coords, "coords"].apply(json.loads)
            df = df.reindex(columns=self.d['columns'])
            for col in df.columns:
                df[col], uo, wr = change_objects_series(df[col], self.pp_hook)
                self.underlying_objects.extend(uo)
                self.weakrefs.extend(wr)
            return df

    if SHAPELY_INSTALLED:
        @from_serializable.register(module_name='shapely')
        def shapely(self):
            return shapely.geometry.shape(self.obj)


class PPJSONDecoder(json.JSONDecoder):
    def __init__(self, **kwargs):
        self.memo_pp = kwargs.pop("memo_pp", dict())
        self.addresses_to_fill = kwargs.pop("addresses_to_fill", defaultdict(list))
        self.weakrefs_to_fill = kwargs.pop("weakrefs_to_fill", defaultdict(list))
        self.set_addresses = kwargs.pop("set_addresses", True)
        self.currently_to_fill = list()
        super_kwargs = {"object_hook": self.pp_hook}
        super_kwargs.update(kwargs)
        super().__init__(**super_kwargs)
        self.parse_array = self.from_json_list

    def decode(self, s, _w=WHITESPACE.match):
        obj = super().decode(s, _w)
        typ = type(obj)
        if typ == list:
            add_obj_weakref_key_addr_list(obj, self.addresses_to_fill, self.weakrefs_to_fill)
        elif typ == dict:
            add_obj_weakref_key_addr_dict(obj, self.addresses_to_fill, self.weakrefs_to_fill)
        if self.set_addresses:
            for addr, fill_list in self.addresses_to_fill.items():
                for obj_to_fill, setfunc, key in fill_list:
                    setfunc(obj_to_fill, key, self.memo_pp[addr])
            for addr, fill_list in self.weakrefs_to_fill.items():
                for obj_to_fill, setfunc, key in fill_list:
                    setfunc(obj_to_fill, key, weakref.ref(self.memo_pp[addr]))
        return obj

    def pp_hook(self, d, registry_class=FromSerializableRegistry):
        try:
            if '_module' in d and '_class' in d:
                if "_object" in d:
                    obj = d.pop('_object')
                elif "_state" in d:
                    obj = d['_state']
                    if d['has_net']:
                        obj['net'] = 'net'
                    if '_init' in obj:
                        del obj['_init']
                    return obj  # backwards compatibility
                else:
                    # obj = {"_init": d, "_state": dict()}  # backwards compatibility
                    obj = {key: val for key, val in d.items() if key not in ['_module', '_class']}
                fs = registry_class(obj, d, self.pp_hook, self.memo_pp, self.addresses_to_fill,
                                    self.weakrefs_to_fill)
                fs.class_name = d.pop('_class', '')
                fs.module_name = d.pop('_module', '')
                obj = fs.from_serializable()
                if len(fs.underlying_objects) > 0:
                    print(fs.underlying_objects)
                if "_address" in d:
                    self.memo_pp[str(d["_address"])] = obj
                for key, addr in fs.underlying_objects:
                    self.addresses_to_fill[str(addr)].append((obj, get_setter_method(obj), key))
                for key, addr in fs.weakrefs:
                    self.weakrefs_to_fill[str(addr)].append((obj, get_setter_method(obj), key))
                return obj
            else:
                add_obj_weakref_key_addr_dict(d, self.addresses_to_fill, self.weakrefs_to_fill)
                return d
        except TypeError:
            logger.debug('Loading your grid raised a TypeError. %s raised this exception' % d)
            return d

    def from_json_list(self, s_and_end, scan_once, _w=WHITESPACE.match, _ws=WHITESPACE_STR):
        value, end = JSONArray(s_and_end=s_and_end, scan_once=scan_once, _w=_w, _ws=_ws)
        if isinstance(value, list):
            add_obj_weakref_key_addr_list(value, self.addresses_to_fill, self.weakrefs_to_fill)
        return value, end


def get_setter_method(obj):
    if isinstance(obj, pd.Series):
        return pd.Series._set_value
    if isinstance(obj, pd.DataFrame):
        return pd.DataFrame._set_value
    if hasattr(obj, "__setitem__"):
        return type(obj).__setitem__
    return setattr


def add_obj_weakref_key_addr_dict(d, addresses_to_fill, weakrefs_to_fill):
    for k, v in d.items():
        if check_object_at_address(v):
            addresses_to_fill[v[OBJ_ADDR]].append((d, dict.__setitem__, k))
        elif check_weakref_at_address(v):
            weakrefs_to_fill[v[WEAKREF_ADDR]].append((d, dict.__setitem__, k))


def add_obj_weakref_key_addr_list(l, addresses_to_fill, weakrefs_to_fill):
    if isinstance(l, list):
        for i, val in enumerate(l):
            if check_object_at_address(val):
                addresses_to_fill[val[OBJ_ADDR]].append((l, list.__setitem__, i))
            elif check_weakref_at_address(val):
                weakrefs_to_fill[val[WEAKREF_ADDR]].append((l, list.__setitem__, i))


def encrypt_string(s, key, compress=True):
    from cryptography.fernet import Fernet
    import hashlib
    import base64
    key_base = hashlib.sha256(key.encode())
    key = base64.urlsafe_b64encode(key_base.digest())
    cipher_suite = Fernet(key)

    s = s.encode()
    if compress:
        import zlib
        s = zlib.compress(s)
    s = cipher_suite.encrypt(s)
    s = s.decode()
    return s


def decrypt_string(s, key):
    from cryptography.fernet import Fernet
    import hashlib
    import base64
    key_base = hashlib.sha256(key.encode())
    key = base64.urlsafe_b64encode(key_base.digest())
    cipher_suite = Fernet(key)

    s = s.encode()
    s = cipher_suite.decrypt(s)
    try:
        import zlib
        s = zlib.decompress(s)
    except:
        pass
    s = s.decode()
    return s


class JSONSerializableClass(object):
    # json_excludes = ["net", "_net", "self", "__class__"]
    json_excludes = ["self", "__class__", "_net"]
    json_to_addr = ["net"]

    def __init__(self, **kwargs):
        pass

    @property
    def net(self):
        return self._net()

    @net.setter
    def net(self, net):
        self._net = weakref.ref(net)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'net':
                setattr(result, k, memo[id(self.net)])
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

    def to_json(self, obj_memo=None):
        """
        Each controller should have this method implemented. The resulting json string should be
        readable by the controller's from_json function and by the function add_ctrl_from_json in
        control_handler.
        """
        return json.dumps(self.to_dict(), cls=PPJSONEncoder, obj_memo=obj_memo)

    def to_dict(self):
        def consider_callable(value):
            if callable(value) and value.__class__ in (types.MethodType, types.FunctionType):
                if value.__class__ == types.MethodType and _findclass(value) is not None:
                    return with_signature(value, value.__name__, obj_module=_findclass(value))
                return with_signature(value, value.__name__)
            return value

        d = {key: consider_callable(val) for key, val in self.__dict__.items()
             if key not in self.json_excludes and key not in self.json_to_addr}
        if hasattr(self, "net"):
            d.update({'net': address_string(self.net)})
        return d

    def add_to_net(self, element, index, column="object", overwrite=False):
        if element not in self.net:
            self.net[element] = pd.DataFrame(columns=[column])
        if index in self.net[element].index.values:
            obj = self.net[element].object.at[index]
            if overwrite or not isinstance(obj, JSONSerializableClass):
                logger.info("Updating %s with index %s" % (element, index))
            else:
                raise UserWarning("%s with index %s already exists" % (element, index))
        self.net[element].at[index, column] = self

    def __eq__(self, other):

        class UnequalityFound(Exception):
            pass

        def check_equality(obj1, obj2):
            if isinstance(obj1, (ndarray, generic)) or isinstance(obj2, (ndarray, generic)):
                unequal = True
                if equal(obj1, obj2):
                    unequal = False
                elif anynp(isnan(obj1)):
                    if allclose(obj1, obj2, atol=0, rtol=0, equal_nan=True):
                        unequal = False
                if unequal:
                    raise UnequalityFound
            elif not isinstance(obj2, type(obj1)):
                raise UnequalityFound
            elif isinstance(obj1, pandapowerNet):
                pass
            elif isinstance(obj1, pd.DataFrame):
                if len(obj1) > 0:
                    try:
                        assert_frame_equal(obj1, obj2)
                    except:
                        raise UnequalityFound
            elif isinstance(obj2, pd.Series):
                if len(obj1) > 0:
                    try:
                        assert_series_equal(obj1, obj2)
                    except:
                        raise UnequalityFound
            elif isinstance(obj1, dict):
                check_dictionary_equality(obj1, obj2)
            elif obj1 != obj1 and obj2 != obj2:
                pass
            elif callable(obj1):
                check_callable_equality(obj1, obj2)
            elif obj1 != obj2:
                try:
                    if not (isnan(obj1) and isnan(obj2)):
                        raise UnequalityFound
                except:
                    raise UnequalityFound

        def check_dictionary_equality(obj1, obj2):
            if set(obj1.keys()) != set(obj2.keys()):
                raise UnequalityFound
            for key in obj1.keys():
                if key != "_init":
                    check_equality(obj1[key], obj2[key])

        def check_callable_equality(obj1, obj2):
            if isinstance(obj1, weakref.ref) and isinstance(obj2, weakref.ref):
                return
            if str(obj1) != str(obj2):
                raise UnequalityFound

        if isinstance(other, self.__class__):
            try:
                check_equality(self.__dict__, other.__dict__)
                return True
            except UnequalityFound as e:
                return False
        else:
            return False

    @classmethod
    def from_dict(cls, d):
        obj = JSONSerializableClass.__new__(cls)
        obj.__dict__.update(d)
        return obj

    @classmethod
    def from_json(cls, json_string):
        d = json.loads(json_string, cls=PPJSONDecoder)
        return cls.from_dict(d)


def with_signature(obj, val, obj_module=None, obj_class=None, with_address=False):
    if obj_module is None:
        obj_module = obj.__module__.__str__()
    if obj_class is None:
        obj_class = obj.__class__.__name__
    d = {'_module': obj_module, '_class': obj_class, '_object': val}
    if with_address:
        d.update({'_address': id(obj)})
    if hasattr(obj, 'dtype'):
        d.update({'dtype': str(obj.dtype)})
    return d


def address_string(obj):
    return {OBJ_ADDR: str(id(obj))}


def weakref_string(obj):
    return {WEAKREF_ADDR: str(id(obj))}


@singledispatch
def to_serializable(obj, memo=None):
    logger.debug('standard case')
    return str(obj)


@to_serializable.register(pandapowerNet)
def json_pandapowernet(obj, memo=None):
    if memo is not None:
        if obj in memo:
            return address_string(obj)
        memo.append(obj)
    net_dict = {k: item for k, item in obj.items() if not k.startswith("_")}
    d = with_signature(obj, net_dict, with_address=True)
    return d


@to_serializable.register(pd.DataFrame)
def json_dataframe(obj, memo=None):
    logger.debug('DataFrame')
    orient = "split"
    json_string = obj.to_json(
        orient=orient, default_handler=partial(to_serializable, memo=memo), double_precision=15)
    d = with_signature(obj, json_string)
    d['orient'] = orient
    if len(obj.columns) > 0 and isinstance(obj.columns[0], str):
        d['dtype'] = obj.dtypes.astype('str').to_dict()
    return d


if GEOPANDAS_INSTALLED:
    @to_serializable.register(geopandas.GeoDataFrame)
    def json_geodataframe(obj, memo=None):
        logger.debug('GeoDataFrame')
        d = with_signature(obj, obj.to_json())
        d.update({'dtype': obj.dtypes.astype('str').to_dict(),
                  'crs': obj.crs, 'columns': obj.columns})
        return d


@to_serializable.register(pd.Series)
def json_series(obj, memo=None):
    logger.debug('Series')
    d = with_signature(obj, obj.to_json(
        orient='split', default_handler=partial(to_serializable, memo=memo), double_precision=15))
    d.update({'dtype': str(obj.dtypes), 'orient': 'split', 'typ': 'series'})
    return d


@to_serializable.register(numpy.ndarray)
def json_array(obj, memo=None):
    logger.debug("ndarray")
    d = with_signature(obj, list(obj), obj_module='numpy', obj_class='array')
    return d


@to_serializable.register(numpy.integer)
def json_npint(obj, memo=None):
    logger.debug("integer")
    return int(obj)


@to_serializable.register(numpy.floating)
def json_npfloat(obj, memo=None):
    logger.debug("floating")
    return float(obj)


@to_serializable.register(numbers.Number)
def json_num(obj, memo=None):
    logger.debug("numbers.Number")
    return str(obj)


@to_serializable.register(complex)
def json_complex(obj, memo=None):
    logger.debug("complex")
    d = with_signature(obj, str(obj), obj_module='builtins', obj_class='complex')
    d.pop('dtype')
    return d


@to_serializable.register(pd.Index)
def json_pdindex(obj, memo=None):
    logger.debug("pd.Index")
    return with_signature(obj, list(obj), obj_module='pandas')


@to_serializable.register(bool)
def json_bool(obj, memo=None):
    logger.debug("bool")
    return "true" if obj else "false"


@to_serializable.register(tuple)
def json_tuple(obj, memo=None):
    logger.debug("tuple")
    d = with_signature(obj, list(obj), obj_module='builtins', obj_class='tuple')
    return d


@to_serializable.register(set)
def json_set(obj, memo=None):
    logger.debug("set")
    d = with_signature(obj, list(obj), obj_module='builtins', obj_class='set')
    return d

@to_serializable.register(frozenset)
def json_frozenset(obj, memo=None):
    logger.debug("frozenset")
    d = with_signature(obj, list(obj), obj_module='builtins', obj_class='frozenset')
    return d


@to_serializable.register(networkx.Graph)
def json_networkx(obj, memo=None):
    logger.debug("nx graph")
    json_string = json_graph.adjacency_data(obj, attrs={'id': 'json_id', 'key': 'json_key'})
    d = with_signature(obj, json_string, obj_module="networkx")
    return d


@to_serializable.register(JSONSerializableClass)
def controller_to_serializable(obj, memo=None):
    logger.debug('JSONSerializableClass')
    if memo is not None:
        if obj in memo:
            return address_string(obj)
        memo.append(obj)
    d = with_signature(obj, obj.to_json(memo), with_address=True)
    return d


@to_serializable.register(weakref.ReferenceType)
def json_weakref(obj, memo=None):
    logger.debug('weakref')
    # d = with_signature(obj, json.dumps(weakref_string(obj())), obj_module="weakref",
    #                    obj_class="ReferenceType")
    return weakref_string(obj)


def mkdirs_if_not_existent(dir_to_create):
    already_exist = os.path.isdir(dir_to_create)
    os.makedirs(dir_to_create, exist_ok=True)
    return ~already_exist


if SHAPELY_INSTALLED:
    @to_serializable.register(shapely.geometry.LineString)
    def json_linestring(obj, memo=None):
        logger.debug("shapely linestring")
        json_string = shapely.geometry.mapping(obj)
        d = with_signature(obj, json_string, obj_module="shapely")
        return d


    @to_serializable.register(shapely.geometry.Point)
    def json_point(obj, memo=None):
        logger.debug("shapely Point")
        json_string = shapely.geometry.mapping(obj)
        d = with_signature(obj, json_string, obj_module="shapely")
        return d


    @to_serializable.register(shapely.geometry.Polygon)
    def json_polygon(obj, memo=None):
        logger.debug("shapely Polygon")
        json_string = shapely.geometry.mapping(obj)
        d = with_signature(obj, json_string, obj_module="shapely")
        return d
