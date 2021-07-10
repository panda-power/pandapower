# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


# Additional copyright for modified code by Brendan Curran-Johnson (ADict class):
# Copyright (c) 2013 Brendan Curran-Johnson
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# (https://github.com/bcj/AttrDict/blob/master/LICENSE.txt)

import copy
from collections.abc import MutableMapping

import numpy as np
import numpy.core.numeric as ncn
import pandas as pd
import scipy as sp
import six
from packaging import version

from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_STATUS
from pandapower.pypower.idx_bus import BUS_I, BUS_TYPE, NONE, PD, QD, VM, VA, REF, VMIN, VMAX, PV
from pandapower.pypower.idx_gen import PMIN, PMAX, QMIN, QMAX

try:
    from numba import jit
    from numba._version import version_version as numba_version
except ImportError:
    from .pf.no_numba import jit

try:
    from lightsim2grid.newtonpf import newtonpf as newtonpf_ls
except ImportError:
    newtonpf_ls = None
try:
    import pplog as logging
except ImportError:
    import logging

lightsim2grid_available = True if newtonpf_ls is not None else False
logger = logging.getLogger(__name__)


class ADict(dict, MutableMapping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # to prevent overwrite of internal attributes by new keys
        # see _valid_name()
        self._setattr('_allow_invalid_attributes', False)

    def _build(self, obj, **kwargs):
        """
        We only want dict like elements to be treated as recursive AttrDicts.
        """
        return obj

    # --- taken from AttrDict

    def __getstate__(self):
        return self.copy(), self._allow_invalid_attributes

    def __dir__(self):
        return list(six.iterkeys(self))

    def __setstate__(self, state):
        mapping, allow_invalid_attributes = state
        self.update(mapping)
        self._setattr('_allow_invalid_attributes', allow_invalid_attributes)

    @classmethod
    def _constructor(cls, mapping):
        return cls(mapping)

    # --- taken from MutableAttr

    def _setattr(self, key, value):
        """
        Add an attribute to the object, without attempting to add it as
        a key to the mapping (i.e. internals)
        """
        super(MutableMapping, self).__setattr__(key, value)

    def __setattr__(self, key, value):
        """
        Add an attribute.

        key: The name of the attribute
        value: The attributes contents
        """
        if self._valid_name(key):
            self[key] = value
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__setattr__(key, value)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute creation.".format(
                    cls=self.__class__.__name__
                )
            )

    def _delattr(self, key):
        """
        Delete an attribute from the object, without attempting to
        remove it from the mapping (i.e. internals)
        """
        super(MutableMapping, self).__delattr__(key)

    def __delattr__(self, key, force=False):
        """
        Delete an attribute.

        key: The name of the attribute
        """
        if self._valid_name(key):
            del self[key]
        elif getattr(self, '_allow_invalid_attributes', True):
            super(MutableMapping, self).__delattr__(key)
        else:
            raise TypeError(
                "'{cls}' does not allow attribute deletion.".format(
                    cls=self.__class__.__name__
                )
            )

    def __call__(self, key):
        """
        Dynamically access a key-value pair.

        key: A key associated with a value in the mapping.

        This differs from __getitem__, because it returns a new instance
        of an Attr (if the value is a Mapping object).
        """
        if key not in self:
            raise AttributeError(
                "'{cls} instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    def __getattr__(self, key):
        """
        Access an item as an attribute.
        """
        if key not in self or not self._valid_name(key):
            raise AttributeError(
                "'{cls}' instance has no attribute '{name}'".format(
                    cls=self.__class__.__name__, name=key
                )
            )

        return self._build(self[key])

    def __deepcopy__(self, memo):
        """
        overloads the deepcopy function of pandapower if at least one DataFrame with column "object" is in net

        in addition, line geodata can contain mutable objects like lists, and it is also treated specially

        reason: some of these objects contain a reference to net which breaks the default deepcopy function.
        Also, the DataFrame doesn't deepcopy its elements if geodata changes in the lists, it affects both net instances
        This fix was introduced in pandapower 2.2.1

        """
        deep_columns = {'object', 'coords', 'geometry'}
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            if isinstance(v, pd.DataFrame) and not set(v.columns).isdisjoint(deep_columns):
                if k not in result:
                    result[k] = v.__class__(index=v.index, columns=v.columns)
                for col in v.columns:
                    if col in deep_columns:
                        result[k][col] = v[col].apply(lambda x: copy.deepcopy(x, memo))
                    else:
                        result[k][col] = copy.deepcopy(v[col], memo)
                _preserve_dtypes(result[k], v.dtypes)
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        result._setattr('_allow_invalid_attributes', self._allow_invalid_attributes)
        return result

    @classmethod
    def _valid_name(cls, key):
        """
        Check whether a key is a valid attribute name.

        A key may be used as an attribute if:
         * It is a string
         * The key doesn't overlap with any class attributes (for Attr,
            those would be 'get', 'items', 'keys', 'values', 'mro', and
            'register').
        """
        return (
                isinstance(key, six.string_types) and
                not hasattr(cls, key)
        )


class pandapowerNet(ADict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(args[0], self.__class__):
            net = args[0]
            self.clear()
            self.update(**net.deepcopy())

    def deepcopy(self):
        return copy.deepcopy(self)

    def __repr__(self):  # pragma: no cover
        r = "This pandapower network includes the following parameter tables:"
        par = []
        res = []
        for tb in list(self.keys()):
            if not tb.startswith("_") and isinstance(self[tb], pd.DataFrame) and len(self[tb]) > 0:
                if 'res_' in tb:
                    res.append(tb)
                else:
                    par.append(tb)
        for tb in par:
            length = len(self[tb])
            r += "\n   - %s (%s %s)" % (tb, length, "elements" if length > 1 else "element")
        if res:
            r += "\n and the following results tables:"
            for tb in res:
                length = len(self[tb])
                r += "\n   - %s (%s %s)" % (tb, length, "elements" if length > 1 else "element")
        return r


def _preserve_dtypes(df, dtypes):
    for item, dtype in list(dtypes.iteritems()):
        if df.dtypes.at[item] != dtype:
            try:
                df[item] = df[item].astype(dtype)
            except ValueError:
                df[item] = df[item].astype(float)


def get_free_id(df):
    """
    Returns next free ID in a dataframe
    """
    return np.int64(0) if len(df) == 0 else df.index.values.max() + 1


class ppException(Exception):
    """
    General pandapower custom parent exception.
    """
    pass


def _sum_by_group(bus, first_val, second_val):
    order = np.argsort(bus)
    bus = bus[order]
    index = np.ones(len(bus), 'bool')
    index[:-1] = bus[1:] != bus[:-1]
    bus = bus[index]
    first_val = first_val[order]
    first_val.cumsum(out=first_val)
    first_val = first_val[index]
    first_val[1:] = first_val[1:] - first_val[:-1]
    second_val = second_val[order]
    second_val.cumsum(out=second_val)
    second_val = second_val[index]
    second_val[1:] = second_val[1:] - second_val[:-1]
    return bus, first_val, second_val


def _sum_by_group_nvals(bus, *vals):
    order = np.argsort(bus)
    bus = bus[order]
    index = np.ones(len(bus), 'bool')
    index[:-1] = bus[1:] != bus[:-1]
    bus = bus[index]
    newvals = tuple(np.zeros((len(vals), len(bus))))
    for val, newval in zip(vals, newvals):
        val = val[order]
        val.cumsum(out=val)
        val = val[index]
        val[1:] = val[1:] - val[:-1]
        newval[:] = val
        # Returning vals keeps the original array dimensions, which causes an error if more than one element is
        # connected to the same bus. Instead, we create a second tuple of arrays on which we map the results.
        # Todo: Check if this workaround causes no problems
    return (bus,) + newvals


def get_indices(selection, lookup, fused_indices=True):
    """
    Helper function during pd2mpc conversion. It resolves the mapping from a
    given selection of indices to the actual indices, using a dict lookup being
    passed as well.

    :param selection: Indices we want to select
    :param lookup: The mapping itself
    :param fused_indices: Flag which way the conversion is working.
    :return:
    """
    if fused_indices:
        return np.array([lookup[k] for k in selection], dtype="int")
    else:
        return np.array([lookup["before_fuse"][k] for k in selection], dtype="int")


def _get_values(source, selection, lookup):
    """
    Returns values for a selection of values after a lookup.

    :param source: The array of values to select from.
    :param selection: An array of keys, for the selection.
    :param lookup: The mapping to resolve actual indices of the
    value array from the selection.
    :return:
    """
    v = np.zeros(len(selection))
    for i, k in enumerate(selection):
        v[i] = source[lookup[np.int(k)]]
    return v


def _set_isolated_nodes_out_of_service(ppc, bus_not_reachable):
    isolated_nodes = np.where(bus_not_reachable)[0]
    if len(isolated_nodes) > 0:
        logger.debug("There are isolated buses in the network!")
        # set buses in ppc out of service
        ppc['bus'][isolated_nodes, BUS_TYPE] = NONE

        pus = abs(ppc['bus'][isolated_nodes, PD] * 1e3).sum()
        qus = abs(ppc['bus'][isolated_nodes, QD] * 1e3).sum()
        if pus > 0 or qus > 0:
            logger.debug("%.0f kW active and %.0f kVar reactive power are unsupplied" % (pus, qus))
    else:
        pus = qus = 0

    return isolated_nodes, pus, qus, ppc


def _check_connectivity_opf(ppc):
    """
    Checks if the ppc contains isolated buses and changes slacks to PV nodes if multiple slacks are
    in net.
    :param ppc: pypower case file
    :return:
    """
    br_status = ppc['branch'][:, BR_STATUS] == True
    nobranch = ppc['branch'][br_status, :].shape[0]
    nobus = ppc['bus'].shape[0]
    bus_from = ppc['branch'][br_status, F_BUS].real.astype(int)
    bus_to = ppc['branch'][br_status, T_BUS].real.astype(int)
    slacks = ppc['bus'][ppc['bus'][:, BUS_TYPE] == 3, BUS_I].astype(int)

    adj_matrix = sp.sparse.coo_matrix((np.ones(nobranch),
                                       (bus_from, bus_to)),
                                      shape=(nobus, nobus))

    bus_not_reachable = np.ones(ppc["bus"].shape[0], dtype=bool)
    slack_set = set(slacks)
    for slack in slacks:
        if ppc['bus'][slack, BUS_TYPE] == PV:
            continue
        reachable = sp.sparse.csgraph.breadth_first_order(adj_matrix, slack, False, False)
        bus_not_reachable[reachable] = False
        reach_set = set(reachable)
        intersection = slack_set & reach_set
        if len(intersection) > 1:
            # if slack is in reachable other slacks are connected to this one. Set it to Gen bus
            demoted_slacks = list(intersection - {slack})
            ppc['bus'][demoted_slacks, BUS_TYPE] = PV
            logger.warning("Multiple connected slacks in one area found. This would probably lead "
                           "to non-convergence of the OPF. I'll change all but one slack (ext_grid)"
                           " to gens. To avoid undesired behaviour, rather convert the slacks to "
                           "gens yourself and set slack=True for one of them.")

    isolated_nodes, pus, qus, ppc = _set_isolated_nodes_out_of_service(ppc, bus_not_reachable)
    return isolated_nodes, pus, qus


def _check_connectivity(ppc):
    """
    Checks if the ppc contains isolated buses. If yes this isolated buses are set out of service
    :param ppc: pypower case file
    :return:
    """
    br_status = ppc['branch'][:, BR_STATUS] == True
    nobranch = ppc['branch'][br_status, :].shape[0]
    nobus = ppc['bus'].shape[0]
    bus_from = ppc['branch'][br_status, F_BUS].real.astype(int)
    bus_to = ppc['branch'][br_status, T_BUS].real.astype(int)
    slacks = ppc['bus'][ppc['bus'][:, BUS_TYPE] == 3, BUS_I]

    # we create a "virtual" bus thats connected to all slack nodes and start the connectivity
    # search at this bus
    bus_from = np.hstack([bus_from, slacks])
    bus_to = np.hstack([bus_to, np.ones(len(slacks)) * nobus])

    adj_matrix = sp.sparse.coo_matrix((np.ones(nobranch + len(slacks)),
                                       (bus_from, bus_to)),
                                      shape=(nobus + 1, nobus + 1))

    reachable = sp.sparse.csgraph.breadth_first_order(adj_matrix, nobus, False, False)
    # TODO: the former impl. excluded ppc buses that are already oos, but is this necessary ?
    # if so: bus_not_reachable = np.hstack([ppc['bus'][:, BUS_TYPE] != 4, np.array([False])])
    bus_not_reachable = np.ones(ppc["bus"].shape[0] + 1, dtype=bool)
    bus_not_reachable[reachable] = False
    isolated_nodes, pus, qus, ppc = _set_isolated_nodes_out_of_service(ppc, bus_not_reachable)
    return isolated_nodes, pus, qus


def _python_set_elements_oos(ti, tis, bis, lis):  # pragma: no cover
    for i in range(len(ti)):
        if tis[i] and bis[ti[i]]:
            lis[i] = True


def _python_set_isolated_buses_oos(bus_in_service, ppc_bus_isolated, bus_lookup):  # pragma: no cover
    for k in range(len(bus_in_service)):
        if ppc_bus_isolated[bus_lookup[k]]:
            bus_in_service[k] = False


try:
    get_values = jit(nopython=True, cache=True)(_get_values)
    set_elements_oos = jit(nopython=True, cache=True)(_python_set_elements_oos)
    set_isolated_buses_oos = jit(nopython=True, cache=True)(_python_set_isolated_buses_oos)
except RuntimeError:
    get_values = jit(nopython=True, cache=False)(_get_values)
    set_elements_oos = jit(nopython=True, cache=False)(_python_set_elements_oos)
    set_isolated_buses_oos = jit(nopython=True, cache=False)(_python_set_isolated_buses_oos)


def _select_is_elements_numba(net, isolated_nodes=None, sequence=None):
    # is missing sgen_controllable and load_controllable
    max_bus_idx = np.max(net["bus"].index.values)
    bus_in_service = np.zeros(max_bus_idx + 1, dtype=bool)
    bus_in_service[net["bus"].index.values] = net["bus"]["in_service"].values.astype(bool)
    if isolated_nodes is not None and len(isolated_nodes) > 0:
        ppc = net["_ppc"] if sequence is None else net["_ppc%s" % sequence]
        ppc_bus_isolated = np.zeros(ppc["bus"].shape[0], dtype=bool)
        ppc_bus_isolated[isolated_nodes] = True
        set_isolated_buses_oos(bus_in_service, ppc_bus_isolated, net["_pd2ppc_lookups"]["bus"])
    #    mode = net["_options"]["mode"]
    elements = ["load", "motor", "sgen", "asymmetric_load", "asymmetric_sgen", "gen" \
        , "ward", "xward", "shunt", "ext_grid", "storage"]  # ,"impedance_load"
    is_elements = dict()
    for element in elements:
        len_ = len(net[element].index)
        element_in_service = np.zeros(len_, dtype=bool)
        if len_ > 0:
            element_df = net[element]
            set_elements_oos(element_df["bus"].values, element_df["in_service"].values,
                             bus_in_service, element_in_service)
        if net["_options"]["mode"] == "opf" and element in ["load", "sgen", "storage"]:
            if "controllable" in net[element]:
                controllable = net[element].controllable.fillna(False).values.astype(bool)
                controllable_is = controllable & element_in_service
                if controllable_is.any():
                    is_elements["%s_controllable" % element] = controllable_is
                    element_in_service = element_in_service & ~controllable_is
        is_elements[element] = element_in_service

    is_elements["bus_is_idx"] = net["bus"].index.values[bus_in_service[net["bus"].index.values]]
    is_elements["line_is_idx"] = net["line"].index[net["line"].in_service.values]
    return is_elements


def _add_ppc_options(net, calculate_voltage_angles, trafo_model, check_connectivity, mode,
                     switch_rx_ratio, enforce_q_lims, recycle, delta=1e-10,
                     voltage_depend_loads=False, trafo3w_losses="hv", init_vm_pu=1.0,
                     init_va_degree=0, p_lim_default=1e9, q_lim_default=1e9,
                     neglect_open_switch_branches=False, consider_line_temperature=False):
    """
    creates dictionary for pf, opf and short circuit calculations from input parameters.
    """
    # if recycle is None:
    #     recycle = dict(trafo=False, bus_pq=False, bfsw=False)

    init_results = (isinstance(init_vm_pu, str) and (init_vm_pu == "results")) or \
                   (isinstance(init_va_degree, str) and (init_va_degree == "results"))

    options = {
        "calculate_voltage_angles": calculate_voltage_angles,
        "trafo_model": trafo_model,
        "check_connectivity": check_connectivity,
        "mode": mode,
        "switch_rx_ratio": switch_rx_ratio,
        "enforce_q_lims": enforce_q_lims,
        "recycle": recycle,
        "voltage_depend_loads": voltage_depend_loads,
        "consider_line_temperature": consider_line_temperature,
        "delta": delta,
        "trafo3w_losses": trafo3w_losses,
        "init_vm_pu": init_vm_pu,
        "init_va_degree": init_va_degree,
        "init_results": init_results,
        "p_lim_default": p_lim_default,
        "q_lim_default": q_lim_default,
        "neglect_open_switch_branches": neglect_open_switch_branches,
    }
    _add_options(net, options)


def _check_bus_index_and_print_warning_if_high(net, n_max=1e7):
    max_bus = max(net.bus.index.values)
    if max_bus >= n_max > len(net["bus"]):
        logger.warning("Maximum bus index is high (%i). You should avoid high bus indices because "
                       "of perfomance reasons. Try resetting the bus indices with the toolbox "
                       "function create_continuous_bus_index()" % max_bus)


def _check_gen_index_and_print_warning_if_high(net, n_max=1e7):
    if net.gen.empty:
        return
    max_gen = max(net.gen.index.values)
    if max_gen >= n_max > len(net["gen"]):
        logger.warning("Maximum generator index is high (%i). You should avoid high generator "
                       "indices because of perfomance reasons. Try resetting the bus indices with "
                       "the toolbox function create_continuous_elements_index()" % max_gen)


def _add_pf_options(net, tolerance_mva, trafo_loading, numba, ac,
                    algorithm, max_iteration, **kwargs):
    """
    creates dictionary for pf, opf and short circuit calculations from input parameters.
    """

    options = {
        "tolerance_mva": tolerance_mva,
        "trafo_loading": trafo_loading,
        "numba": numba,
        "ac": ac,
        "algorithm": algorithm,
        "max_iteration": max_iteration
    }

    options.update(kwargs)  # update options with some algorithm-specific parameters
    _add_options(net, options)


def _add_opf_options(net, trafo_loading, ac, v_debug=False, **kwargs):
    """
    creates dictionary for pf, opf and short circuit calculations from input parameters.
    """
    options = {
        "trafo_loading": trafo_loading,
        "ac": ac,
        "v_debug": v_debug
    }

    options.update(kwargs)  # update options with some algorithm-specific parameters
    _add_options(net, options)


def _add_sc_options(net, fault, case, lv_tol_percent, tk_s, topology, r_fault_ohm,
                    x_fault_ohm, kappa, ip, ith, branch_results,
                    kappa_method, return_all_currents,
                    inverse_y):
    """
    creates dictionary for pf, opf and short circuit calculations from input parameters.
    """
    options = {
        "fault": fault,
        "case": case,
        "lv_tol_percent": lv_tol_percent,
        "tk_s": tk_s,
        "topology": topology,
        "r_fault_ohm": r_fault_ohm,
        "x_fault_ohm": x_fault_ohm,
        "kappa": kappa,
        "ip": ip,
        "ith": ith,
        "branch_results": branch_results,
        "kappa_method": kappa_method,
        "return_all_currents": return_all_currents,
        "inverse_y": inverse_y
    }
    _add_options(net, options)


def _add_options(net, options):
    # double_parameters = set(net.__internal_options.keys()) & set(options.keys())
    double_parameters = set(net._options.keys()) & set(options.keys())
    if len(double_parameters) > 0:
        raise UserWarning(
            "Parameters always have to be unique! The following parameters where specified " +
            "twice: %s" % double_parameters)
    # net.__internal_options.update(options)
    net._options.update(options)


def _clean_up(net, res=True):
    # mode = net.__internal_options["mode"]

    # set internal selected _is_elements to None. This way it is not stored (saves disk space)
    # net._is_elements = None

    #    mode = net._options["mode"]
    #    if res:
    #        res_bus = net["res_bus_sc"] if mode == "sc" else \
    #            net["res_bus_3ph"] if mode == "pf_3ph" else \
    #                net["res_bus"]
    #    if len(net["trafo3w"]) > 0:
    #        buses_3w = net.trafo3w["ad_bus"].values
    #        net["bus"].drop(buses_3w, inplace=True)
    #        net["trafo3w"].drop(["ad_bus"], axis=1, inplace=True)
    #        if res:
    #            res_bus.drop(buses_3w, inplace=True)
    #
    #    if len(net["xward"]) > 0:
    #        xward_buses = net["xward"]["ad_bus"].values
    #        net["bus"].drop(xward_buses, inplace=True)
    #        net["xward"].drop(["ad_bus"], axis=1, inplace=True)
    #        if res:
    #            res_bus.drop(xward_buses, inplace=True)
    if len(net["dcline"]) > 0:
        dc_gens = net.gen.index[(len(net.gen) - len(net.dcline) * 2):]
        net.gen.drop(dc_gens, inplace=True)
        if res:
            net.res_gen.drop(dc_gens, inplace=True)


def _set_isolated_buses_out_of_service(net, ppc):
    # set disconnected buses out of service
    # first check if buses are connected to branches
    disco = np.setxor1d(ppc["bus"][:, 0].astype(int),
                        ppc["branch"][ppc["branch"][:, 10] == 1, :2].real.astype(int).flatten())

    # but also check if they may be the only connection to an ext_grid
    net._isolated_buses = np.setdiff1d(disco, ppc['bus'][ppc['bus'][:, 1] == REF, :1].real.astype(int))
    ppc["bus"][net._isolated_buses, 1] = NONE


def _write_lookup_to_net(net, element, element_lookup):
    """
    Updates selected lookups in net
    """
    net["_pd2ppc_lookups"][element] = element_lookup


def _check_if_numba_is_installed(numba):
    numba_warning_str = 'numba cannot be imported and numba functions are disabled.\n' \
                        'Probably the execution is slow.\n' \
                        'Please install numba to gain a massive speedup.\n' \
                        '(or if you prefer slow execution, set the flag numba=False to avoid ' + \
                        'this warning!)\n'

    try:
        # get numba Version (in order to use it it must be > 0.25)
        if version.parse(numba_version) < version.parse("0.2.5"):
            logger.warning('Warning: numba version too old -> Upgrade to a version > 0.25.\n' +
                           numba_warning_str)
            numba = False
    except:
        logger.warning(numba_warning_str)
        numba = False

    return numba


def _deactive(msg):
    logger.error(msg)
    return False


def _check_lightsim2grid_compatibility(net, lightsim2grid, voltage_dependend_loads, algorithm, enforce_q_lims):
    if lightsim2grid:
        if not lightsim2grid_available:
            return _deactive("option 'lightsim2grid' is True activates but module cannot be imported. "
                             "I'll deactive lightsim2grid.")
        if algorithm != 'nr':
            raise ValueError("option 'lightsim2grid' is True activates but algorithm is not 'nr'.")
        if voltage_dependend_loads:
            return _deactive("option 'lightsim2grid' is True but voltage dependend loads are in your grid."
                             "I'll deactive lightsim2grid.")
        if enforce_q_lims:
            return _deactive("option 'lightsim2grid' is True and enforce_q_lims is True. This is not supported."
                             "I'll deactive lightsim2grid.")
        if len(net.ext_grid) > 1:
            return _deactive("option 'lightsim2grid' is True and multiple ext_grids are in the grid."
                             "I'll deactive lightsim2grid.")
        if np.any(net.gen.bus.isin(net.ext_grid.bus)):
            return _deactive("option 'lightsim2grid' is True and gens are at slack buses."
                             "I'll deactive lightsim2grid.")

    return lightsim2grid


# =============================================================================
# Functions for 3 Phase Unbalanced Load Flow
# =============================================================================

# =============================================================================
# Convert to three decoupled sequence networks
# =============================================================================


def X012_to_X0(X012):
    return np.transpose(X012[0, :])


def X012_to_X1(X012):
    return np.transpose(X012[1, :])


def X012_to_X2(X012):
    return np.transpose(X012[2, :])


# =============================================================================
# Three decoupled sequence network to 012 matrix conversion
# =============================================================================

def combine_X012(X0, X1, X2):
    comb = np.vstack((X0, X1, X2))
    return comb


# =============================================================================
# Symmetrical transformation matrix
# Tabc : 012 > abc
# T012 : abc >012
# =============================================================================

def phase_shift_unit_operator(angle_deg):
    return 1 * np.exp(1j * np.deg2rad(angle_deg))


a = phase_shift_unit_operator(120)
asq = phase_shift_unit_operator(-120)
Tabc = np.array(
    [
        [1, 1, 1],
        [1, asq, a],
        [1, a, asq]
    ])

T012 = np.divide(np.array(
    [
        [1, 1, 1],
        [1, a, asq],
        [1, asq, a]
    ]), 3)


def sequence_to_phase(X012):
    return np.asarray(np.matmul(Tabc, X012))


def phase_to_sequence(Xabc):
    return np.asarray(np.matmul(T012, Xabc))


# def Y_phase_to_sequence(Xabc):
#   return np.asarray(np.matmul(T012,Xabc,Tabc))
# =============================================================================
# Calculating Sequence Current from sequence Voltages
# =============================================================================

def I0_from_V012(V012, Y):
    V0 = X012_to_X0(V012)
    if type(Y) in [sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        return np.asarray(np.matmul(Y.todense(), V0))
    else:
        return np.asarray(np.matmul(Y, V0))


def I1_from_V012(V012, Y):
    V1 = X012_to_X1(V012)[:, np.newaxis]
    if type(Y) in [sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        i1 = np.asarray(np.matmul(Y.todense(), V1))
        return np.transpose(i1)
    else:
        i1 = np.asarray(np.matmul(Y, V1))
        return np.transpose(i1)


def I2_from_V012(V012, Y):
    V2 = X012_to_X2(V012)
    if type(Y) in [sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        return np.asarray(np.matmul(Y.todense(), V2))
    else:
        return np.asarray(np.matmul(Y, V2))


def V1_from_ppc(ppc):
    return np.transpose(
        np.array(
            ppc["bus"][:, VM] * np.exp(1j * np.deg2rad(ppc["bus"][:, VA]))
        )
    )


def V_from_I(Y, I):
    return np.transpose(np.array(sp.sparse.linalg.spsolve(Y, I)))


def I_from_V(Y, V):
    if type(Y) in [sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        return np.asarray(np.matmul(Y.todense(), V))
    else:
        return np.asarray(np.matmul(Y, V))


# =============================================================================
# Calculating Power
# =============================================================================

def S_from_VI_elementwise(V, I):
    return np.multiply(V, I.conjugate())


def I_from_SV_elementwise(S, V):
    return np.conjugate(np.divide(S, V, out=np.zeros_like(S), where=V != 0))  # Return zero if div by zero


def SVabc_from_SV012(S012, V012, n_res=None, idx=None):
    if n_res is None:
        n_res = S012.shape[1]
    if idx is None:
        idx = np.ones(n_res, dtype="bool")
    I012 = np.array(np.zeros((3, n_res)), dtype=np.complex128)
    I012[:, idx] = I_from_SV_elementwise(S012[:, idx], V012[:, idx])
    Vabc = sequence_to_phase(V012[:, idx])
    Iabc = sequence_to_phase(I012[:, idx])
    Sabc = S_from_VI_elementwise(Vabc, Iabc)
    return Sabc, Vabc


def _add_auxiliary_elements(net):
    if len(net.dcline) > 0:
        _add_dcline_gens(net)


def _add_dcline_gens(net):
    from pandapower.create import create_gen
    for dctab in net.dcline.itertuples():
        pfrom = dctab.p_mw
        pto = (pfrom * (1 - dctab.loss_percent / 100) - dctab.loss_mw)
        pmax = dctab.max_p_mw
        create_gen(net, bus=dctab.to_bus, p_mw=pto, vm_pu=dctab.vm_to_pu,
                   min_p_mw=0, max_p_mw=pmax,
                   max_q_mvar=dctab.max_q_to_mvar, min_q_mvar=dctab.min_q_to_mvar,
                   in_service=dctab.in_service)
        create_gen(net, bus=dctab.from_bus, p_mw=-pfrom, vm_pu=dctab.vm_from_pu,
                   min_p_mw=-pmax, max_p_mw=0,
                   max_q_mvar=dctab.max_q_from_mvar, min_q_mvar=dctab.min_q_from_mvar,
                   in_service=dctab.in_service)


def _replace_nans_with_default_limits(net, ppc):
    qlim = net._options["q_lim_default"]
    plim = net._options["p_lim_default"]

    for matrix, column, default in [("gen", QMAX, qlim), ("gen", QMIN, -qlim), ("gen", PMIN, -plim),
                                    ("gen", PMAX, plim), ("bus", VMAX, 2.0), ("bus", VMIN, 0.0)]:
        limits = ppc[matrix][:, [column]]
        ncn.copyto(limits, default, where=np.isnan(limits))
        ppc[matrix][:, [column]] = limits


def _init_runpp_options(net, algorithm, calculate_voltage_angles, init,
                        max_iteration, tolerance_mva, trafo_model,
                        trafo_loading, enforce_q_lims, check_connectivity,
                        voltage_depend_loads, passed_parameters=None,
                        consider_line_temperature=False, **kwargs):
    """
    Inits _options in net for runpp.
    """
    overrule_options = {}
    if passed_parameters is not None:
        overrule_options = {key: val for key, val in net.user_pf_options.items()
                            if key not in passed_parameters.keys()}

    kwargs.update(overrule_options)

    trafo3w_losses = kwargs.get("trafo3w_losses", "hv")
    v_debug = kwargs.get("v_debug", False)
    delta_q = kwargs.get("delta_q", 0)
    switch_rx_ratio = kwargs.get("switch_rx_ratio", 2)
    numba = kwargs.get("numba", True)
    init_vm_pu = kwargs.get("init_vm_pu", None)
    init_va_degree = kwargs.get("init_va_degree", None)
    neglect_open_switch_branches = kwargs.get("neglect_open_switch_branches", False)
    # recycle options
    recycle = kwargs.get("recycle", None)
    only_v_results = kwargs.get("only_v_results", False)
    # scipy spsolve options in NR power flow
    use_umfpack = kwargs.get("use_umfpack", True)
    permc_spec = kwargs.get("permc_spec", None)
    lightsim2grid = kwargs.get("lightsim2grid", False)

    if "init" in overrule_options:
        init = overrule_options["init"]

    # check if numba is available and the corresponding flag
    if numba:
        numba = _check_if_numba_is_installed(numba)

    if voltage_depend_loads:
        if not (np.any(net["load"]["const_z_percent"].values)
                or np.any(net["load"]["const_i_percent"].values)):
            voltage_depend_loads = False

    if algorithm not in ['nr', 'bfsw', 'iwamoto_nr'] and voltage_depend_loads == True:
        logger.warning("voltage-dependent loads not supported for {0} power flow algorithm -> "
                       "loads will be considered as constant power".format(algorithm))

    lightsim2grid = _check_lightsim2grid_compatibility(net, lightsim2grid, voltage_depend_loads,
                                                       algorithm, enforce_q_lims)

    ac = True
    mode = "pf"
    if calculate_voltage_angles == "auto":
        calculate_voltage_angles = False
        is_hv_bus = np.where(net.bus.vn_kv.values > 70)[0]
        if any(is_hv_bus) > 0:
            line_buses = set(net.line.from_bus.values) & set(net.line.to_bus.values)
            hv_buses = net.bus.index[is_hv_bus]
            if any(a in line_buses for a in hv_buses):
                calculate_voltage_angles = True

    default_max_iteration = {"nr": 10, "iwamoto_nr": 10, "bfsw": 100, "gs": 10000, "fdxb": 30,
                             "fdbx": 30}
    if max_iteration == "auto":
        max_iteration = default_max_iteration[algorithm]

    if init != "auto" and (init_va_degree is not None or init_vm_pu is not None):
        raise ValueError("Either define initialization through 'init' or through 'init_vm_pu' and "
                         "'init_va_degree'.")

    init_from_results = init == "results" or \
                        (isinstance(init_vm_pu, str) and init_vm_pu == "results") or \
                        (isinstance(init_va_degree, str) and init_va_degree == "results")
    if init_from_results and len(net.res_bus) == 0:
        init = "auto"
        init_vm_pu = None
        init_va_degree = None

    if init == "auto":
        if init_va_degree is None or (isinstance(init_va_degree, str) and init_va_degree == "auto"):
            init_va_degree = "dc" if calculate_voltage_angles else "flat"
        if init_vm_pu is None or (isinstance(init_vm_pu, str) and init_vm_pu == "auto"):
            init_vm_pu = (net.ext_grid.vm_pu.values.sum() + net.gen.vm_pu.values.sum()) / \
                         (len(net.ext_grid.vm_pu.values) + len(net.gen.vm_pu.values))
    elif init == "dc":
        init_vm_pu = "flat"
        init_va_degree = "dc"
    else:
        init_vm_pu = init
        init_va_degree = init

    # init options
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, switch_rx_ratio=switch_rx_ratio, init_vm_pu=init_vm_pu,
                     init_va_degree=init_va_degree, enforce_q_lims=enforce_q_lims, recycle=recycle,
                     voltage_depend_loads=voltage_depend_loads, delta=delta_q,
                     trafo3w_losses=trafo3w_losses,
                     neglect_open_switch_branches=neglect_open_switch_branches,
                     consider_line_temperature=consider_line_temperature)
    _add_pf_options(net, tolerance_mva=tolerance_mva, trafo_loading=trafo_loading,
                    numba=numba, ac=ac, algorithm=algorithm, max_iteration=max_iteration,
                    v_debug=v_debug, only_v_results=only_v_results, use_umfpack=use_umfpack,
                    permc_spec=permc_spec, lightsim2grid=lightsim2grid)
    net._options.update(overrule_options)


def _init_nx_options(net):
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=False,
                     trafo_model="t", check_connectivity=False,
                     mode="nx", switch_rx_ratio=2, init_vm_pu='flat', init_va_degree="flat",
                     enforce_q_lims=False, recycle=False,
                     voltage_depend_loads=False, delta=0, trafo3w_losses="hv")


def _init_rundcpp_options(net, trafo_model, trafo_loading, recycle, check_connectivity,
                          switch_rx_ratio, trafo3w_losses, **kwargs):
    ac = False
    numba = True
    mode = "pf"
    init = 'flat'

    numba = _check_if_numba_is_installed(numba)

    # the following parameters have no effect if ac = False
    calculate_voltage_angles = True
    enforce_q_lims = False
    algorithm = None
    max_iteration = None
    tolerance_mva = None
    only_v_results = kwargs.get("only_v_results", False)
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, switch_rx_ratio=switch_rx_ratio, init_vm_pu=init,
                     init_va_degree=init, enforce_q_lims=enforce_q_lims, recycle=recycle,
                     voltage_depend_loads=False, delta=0, trafo3w_losses=trafo3w_losses)
    _add_pf_options(net, tolerance_mva=tolerance_mva, trafo_loading=trafo_loading,
                    numba=numba, ac=ac, algorithm=algorithm, max_iteration=max_iteration,
                    only_v_results=only_v_results)


def _init_runopp_options(net, calculate_voltage_angles, check_connectivity, switch_rx_ratio, delta,
                         init, numba, trafo3w_losses, consider_line_temperature=False, **kwargs):
    if numba:
        numba = _check_if_numba_is_installed(numba)
    mode = "opf"
    ac = True
    trafo_model = "t"
    trafo_loading = 'current'
    enforce_q_lims = True
    recycle = None
    only_v_results = False
    # scipy spsolve options in NR power flow
    use_umfpack = kwargs.get("use_umfpack", True)
    permc_spec = kwargs.get("permc_spec", None)
    lightsim2grid = kwargs.get("lightsim2grid", False)

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, switch_rx_ratio=switch_rx_ratio, init_vm_pu=init,
                     init_va_degree=init, enforce_q_lims=enforce_q_lims, recycle=recycle,
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses,
                     consider_line_temperature=consider_line_temperature)
    _add_opf_options(net, trafo_loading=trafo_loading, ac=ac, init=init, numba=numba, lightsim2grid=lightsim2grid,
                     only_v_results=only_v_results, use_umfpack=use_umfpack, permc_spec=permc_spec)


def _init_rundcopp_options(net, check_connectivity, switch_rx_ratio, delta, trafo3w_losses, **kwargs):
    mode = "opf"
    ac = False
    init = "flat"
    trafo_model = "t"
    trafo_loading = 'current'
    calculate_voltage_angles = True
    enforce_q_lims = True
    recycle = None
    only_v_results = False
    # scipy spsolve options in NR power flow
    use_umfpack = kwargs.get("use_umfpack", True)
    permc_spec = kwargs.get("permc_spec", None)
    # net.__internal_options = {}
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode=mode, switch_rx_ratio=switch_rx_ratio, init_vm_pu=init,
                     init_va_degree=init, enforce_q_lims=enforce_q_lims, recycle=recycle,
                     voltage_depend_loads=False, delta=delta, trafo3w_losses=trafo3w_losses)
    _add_opf_options(net, trafo_loading=trafo_loading, init=init, ac=ac, only_v_results=only_v_results,
                     use_umfpack=use_umfpack, permc_spec=permc_spec)


def _init_runse_options(net, v_start, delta_start, calculate_voltage_angles,
                        **kwargs):

    check_connectivity = kwargs.get("check_connectivity", True)
    trafo_model = kwargs.get("trafo_model", "t")
    trafo3w_losses = kwargs.get("trafo3w_losses", "hv")
    switch_rx_ratio = kwargs.get("switch_rx_ratio", 2)

    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=calculate_voltage_angles,
                     trafo_model=trafo_model, check_connectivity=check_connectivity,
                     mode="pf", switch_rx_ratio=switch_rx_ratio, init_vm_pu=v_start,
                     init_va_degree=delta_start, enforce_q_lims=False, recycle=None,
                     voltage_depend_loads=False, trafo3w_losses=trafo3w_losses)
    _add_pf_options(net, tolerance_mva="1e-8", trafo_loading="power",
                    numba=False, ac=True, algorithm="nr", max_iteration="auto",
                    only_v_results=False)


def _internal_stored(net):
    """

    The function newtonpf() needs these variables as inputs:
    Ybus, Sbus, V0, pv, pq, ppci, options

    Parameters
    ----------
    net - the pandapower net

    Returns
    -------
    True if all variables are stored False otherwise

    """
    # checks if all internal variables are stored in net, which are needed for a power flow

    if net["_ppc"] is None:
        return False

    mandatory_pf_variables = ["J", "bus", "gen", "branch", "baseMVA", "V", "pv", "pq", "ref",
                              "Ybus", "Yf", "Yt", "Sbus", "ref_gens"]
    for var in mandatory_pf_variables:
        if "internal" not in net["_ppc"] or var not in net["_ppc"]["internal"]:
            logger.warning("recycle is set to True, but internal variables are missing")
            return False
    return True
