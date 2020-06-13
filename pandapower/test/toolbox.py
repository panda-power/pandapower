# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
from math import isnan

import numpy as np
import pandas.testing as pdt
import pytest
import pandapower as pp


def assert_mpc_equal(mpc1, mpc2):
    for name in ['bus', 'gen', 'branch', 'baseMVA']:
        try:
            assert np.allclose(mpc1[name], mpc2[name])
        except AssertionError:
            print(
                "Conversion from pandapower net to Matpower case failed creating %s table" % name)
            raise
    try:
        assert np.array_equal(mpc1['version'], mpc2['version'])
    except AssertionError:
        print("Pypower version changed from {} to {}".format(
            mpc2['version'], mpc1['version']))


def assert_net_equal(a_net, b_net, **kwargs):
    """Returns True if the given pandapower networks are equal.
    Raises AssertionError if grids are not equal.
    """
    status = True
    namelist = ['bus', 'bus_geodata', 'load', 'sgen', 'ext_grid', 'line', 'shunt', 'line_geodata',
                'trafo', 'switch', 'trafo3w', 'gen', 'ext_grid', 'res_line', 'res_bus', 'res_sgen',
                'res_gen', 'res_shunt', 'res_load', 'res_ext_grid', 'res_trafo']
    for name in namelist:
        if name in a_net or name in b_net:
            if not (a_net[name] is None and b_net[name] is None):
                try:
                    df1 = a_net[name].sort_index().sort_index(axis=1)  # workaround for bug in
                    df2 = b_net[name].sort_index().sort_index(axis=1)  # pandas, dont use
                    pdt.assert_frame_equal(df1, df2, check_dtype=True, **kwargs)  # check_like here
                except AssertionError:
                    pytest.fail("Tables are not equal: %s" % name)
                    status = False

    return status


def assert_res_equal(a, b, **kwargs):
    """Returns True if the result tables of the given pandapower networks are equal.
    Raises AssertionError if results are not equal.
    """
    namelist = ['res_line', 'res_bus', 'res_gen', 'res_sgen',
                'res_load', 'res_ext_grid', 'res_trafo', 'res_trafo3w']

    for name in namelist:
        if name in a or name in b:
            if not (a[name] is None and b[name] is None):
                try:
                    pdt.assert_frame_equal(a[name], b[name], **kwargs)
                except AssertionError:
                    pytest.fail("Result tables are not equal: %s" % name)
                    raise


def assert_res_out_of_service(net, idx, name):
    """Returns True if the result tables of the given pandapower network contain NaNs resp. 0
    Raises AssertionError if datatype is not according to specifications.

    Specifications are:

        res_bus["vm_pu"]  		     nan
        res_bus.va_degree  	         nan
        res_bus["p_mw"]       	     0
        res_bus["q_mvar"]        	  0

        res_line.p_from_mw  		  0
        res_line.q_from_mvar		  0
        res_line.p_to_mw		     0
        res_line.q_to_mvar		     0
        res_line.i_ka			     0
        res_line["loading_percent"] 0

        res_trafo			        all 0

        res_load			        all 0

        res_ext_grid		        all nan

        res_gen["p_mw"]		         0
        res_gen-q_mvar 		         0
        res_gen_va_degree	         nan

        res_sgen 		           all  0

      Future: Elements out of service will not appear in result table!

"""

    status = True
    try:
        if name == 'bus':
            assert isnan(net["res_bus"]["vm_pu"].at[idx])
            assert isnan(net["res_bus"].va_degree.at[idx])
            assert net["res_bus"]["p_mw"].at[idx] == 0
            assert net["res_bus"]["q_mvar"].at[idx] == 0
        elif name == 'gen':
            if net.gen.in_service.any():
                assert net["res_gen"]["p_mw"].at[idx] == 0
                assert net["res_gen"]["q_mvar"].at[idx] == 0
                assert isnan(net["res_gen"].va_degree.at[idx])
            else:
                assert net.res_gen is None
        elif name in ('load', 'trafo', 'sgen', 'line'):
            assert (net['res_' + name].loc[idx] == 0).all()
        elif name == 'ext_grid':
            assert idx not in net.res_ext_grid.index
        else:
            print("Element res_{} does not exist!".format(name, name))
            status = False

    except AssertionError:
        pytest.fail(
            "res_{} table is not according to specifications if {} is out of service".format(name, name))
        status = False
        raise

    return status


def create_test_network():
    """Creates a simple pandapower test network
    """
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, name="bus1", vn_kv=10.)
    pp.create_ext_grid(net, b1)
    b2 = pp.create_bus(net, name="bus2", geodata=(1, 2), vn_kv=.4)
    b3 = pp.create_bus(net, name="bus3", geodata=(1, 3), vn_kv=.4, index=7)
    b4 = pp.create_bus(net, name="bus4", vn_kv=10.)
    pp.create_transformer_from_parameters(net, b4, b2, vk_percent=3.75,
                                          tap_max=2, vn_lv_kv=0.4,
                                          shift_degree=150, tap_neutral=0,
                                          vn_hv_kv=10.0, vkr_percent=2.8125,
                                          tap_pos=0, tap_side="hv", tap_min=-2,
                                          tap_step_percent=2.5, i0_percent=0.68751,
                                          sn_mva=0.016, pfe_kw=0.11, name=None,
                                          in_service=True, index=None)
    # 0.016 MVA 10/0.4 kV ET 16/23  SGB

    pp.create_line_from_parameters(net, b2, b3, 1, name="line1", r_ohm_per_km=0.2067,
                                   ices=0.389985, c_nf_per_km=720.0, max_i_ka=0.328,
                                   x_ohm_per_km=0.1897522, geodata=np.array([[1, 2], [3, 4]]))
    # NAYY 1x150RM 0.6/1kV ir
    pp.create_line_from_parameters(net, b1, b4, 1, name="line2", r_ohm_per_km=0.876,
                                   c_nf_per_km=260.0, max_i_ka=0.123, x_ohm_per_km=0.1159876)

    # NAYSEY 3x35rm/16 6/10kV

    pp.create_load(net, b2, p_mw=0.010, q_mvar=0, name="load1")
    pp.create_load(net, b3, p_mw=0.040, q_mvar=0.002, name="load2")
    pp.create_gen(net, b4, p_mw=0.200, vm_pu=1.0)
    pp.create_sgen(net, b3, p_mw=0.050, sn_mva=0.1)

    return net


def create_test_network2():
    """Creates a simple pandapower test network
    """
    net = pp.from_pickle(os.path.join(pp.pp_dir, "test", "loadflow", "testgrid.p"))
    #    net = pp.file_io.from_pickle("testgrid.p")

    return net


def add_grid_connection(net, vn_kv=20., zone=None):
    """Creates a new grid connection for create_result_test_network()
    """
    b1 = pp.create_bus(net, vn_kv=vn_kv, zone=zone)
    pp.create_ext_grid(net, b1, vm_pu=1.01)
    b2 = pp.get_free_id(net.bus) + 2  # shake up the indices so that non-consecutive indices are tested
    b2 = pp.create_bus(net, vn_kv=vn_kv, zone=zone, index=b2)
    l1 = create_test_line(net, b1, b2)
    return b1, b2, l1


def create_test_line(net, b1, b2, in_service=True):
    return pp.create_line_from_parameters(net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                          c_nf_per_km=300, max_i_ka=.2, df=.8,
                                          in_service=in_service, index=pp.get_free_id(net.line) + 1)
