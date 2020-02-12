# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import pandapower as pp
import pandapower.control as ct
from pandapower.control import ConstControl
from pandapower.test.timeseries.test_timeseries import create_data_source, simple_test_net
from pandapower.timeseries import DFData, OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.networks import simple_four_bus_system
logger = logging.getLogger(__name__)
ow_logger = logging.getLogger("hp.control.output_writer")
logger.setLevel(logging.ERROR)
ow_logger.setLevel(logging.CRITICAL)


def test_output_writer_log(simple_test_net):
    net = simple_test_net

    # timeseries data
    df = pd.DataFrame([[15, 30, 2], [12, 27, 1.5], [7, 29, 2.1]])
    ds = DFData(df)

    # Create gen controller with datasource
    ct.ConstControl(net, element="load", variable="p_mw", element_index=[0, 2], data_source=ds,
                    profile_name=[0, 2])

    # Create, add output and set outputwriter
    ow = OutputWriter(net, output_path=tempfile.gettempdir())
    ow.remove_log_variable("res_bus")
    orig_index = [0, 1]
    ow.log_variable("res_bus", "vm_pu", orig_index)
    ow.log_variable("res_sgen", "p_mw")
    ow.log_variable("res_sgen", "q_mvar")

    # Run timeseries
    run_timeseries(net, time_steps=range(2))

    # --- double logged variables handling
    ow2 = copy.deepcopy(ow)
    new_idx = 2
    ow2.log_variable("res_bus", "vm_pu", new_idx, eval_name="test")
    run_timeseries(net, time_steps=range(2), output_writer=ow2)
    assert all(ow2.output["res_bus.vm_pu"].columns == orig_index + [new_idx])

    # Todo: This test makes no sense if res_bus is logged by default
    # ow3 = copy.deepcopy(ow)
    # new_idx = [2, 3]
    # ow3.log_variable("res_bus", "vm_pu", new_idx)
    # run_timeseries(net, time_steps=range(2), output_writer=ow3)
    # assert all(ow3.output["res_bus.vm_pu"].columns == orig_index + new_idx)

    ow4 = copy.deepcopy(ow)
    new_idx = [2, 4]
    ow4.log_variable("res_bus", "vm_pu", new_idx, eval_name=["test1", "test2"])
    run_timeseries(net, time_steps=range(2), output_writer=ow4)
    assert all(ow4.output["res_bus.vm_pu"].columns == orig_index + new_idx)


def test_output_writer_with_timesteps_set(simple_test_net):
    net = simple_test_net

    n_timesteps = 10
    profiles, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'i_ka')
    run_timeseries(net, time_steps)
    assert len(ow.output["res_bus.vm_pu"]) == n_timesteps
    assert len(ow.output["res_line.i_ka"]) == n_timesteps


def test_output_writer_without_timesteps_set(simple_test_net):
    net = simple_test_net
    n_timesteps = 5
    profiles, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'i_ka')
    run_timeseries(net, time_steps)
    assert len(ow.output["res_bus.vm_pu"]) == n_timesteps
    assert len(ow.output["res_line.i_ka"]) == n_timesteps


def test_output_writer_without_timesteps_set_repeat(simple_test_net):
    net = simple_test_net
    # the same outputwriter should be able to run repeated time series

    time_steps_to_check = [8, 5, 10]
    profiles, ds = create_data_source(max(time_steps_to_check))
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'i_ka')

    for n_timesteps in time_steps_to_check:
        time_steps = range(0, n_timesteps)
        run_timeseries(net, time_steps)
        assert len(ow.output["res_bus.vm_pu"].index) == n_timesteps


def test_output_writer_short_data_source(simple_test_net):
    net = simple_test_net
    # outputwriter should fail if data source is shorter than time steps

    n_timesteps = 10
    profiles, ds = create_data_source(5)
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'i_ka')

    with pytest.raises(KeyError):
        run_timeseries(net, time_steps)


def test_default_output_writer(simple_test_net):
    net = simple_test_net

    n_timesteps = 5
    profiles, ds = create_data_source(n_timesteps)
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    time_steps = range(0, n_timesteps)
    run_timeseries(net, time_steps)
    ow = net.output_writer.iloc[0, 0]
    loading_percent = ow.output["res_line.loading_percent"]
    vm_pu = ow.output["res_bus.vm_pu"]
    assert loading_percent.shape[0] == n_timesteps and loading_percent.shape[1] == len(net.line)
    assert vm_pu.shape[0] == n_timesteps and vm_pu.shape[1] == len(net.bus)


def test_output_writer_eval_simple(simple_test_net):
    net = simple_test_net

    n_timesteps = 1
    profiles, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])
    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu', eval_function=max, eval_name="max")
    run_timeseries(net, time_steps)
    assert len(ow.output["res_bus.vm_pu"]["max"]) == n_timesteps


def test_output_writer_multiple_index_definition(simple_test_net):
    net = simple_test_net

    n_timesteps = 1
    profiles, ds = create_data_source(n_timesteps)
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])
    time_steps = range(0, n_timesteps)

    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu', net.load.bus[[0, 1]])
    ow.log_variable('res_bus', 'vm_pu', index=[1, 2])
    ow.log_variable('res_bus', 'vm_pu', index=[3, 2, 1])
    ow.log_variable('res_bus', 'vm_pu', net.load.bus)
    ow.log_variable('res_bus', 'vm_pu', index=[3, 4])
    ow.log_variable('res_bus', 'vm_pu', net.bus.index)
    ow.log_variable('res_bus', 'vm_pu', 0)
    run_timeseries(net, time_steps)
    backup_result = copy.deepcopy(ow.output["res_bus.vm_pu"].loc[:, net.bus.index])
    del ow

    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu', net.bus.index)

    # run_timeseries(net, time_steps)
    run_timeseries(net, time_steps)
    # assert all are considered
    assert len(ow.output["res_bus.vm_pu"].columns) == len(net.bus.index)
    # assert correct order of values
    assert np.allclose(backup_result, ow.output["res_bus.vm_pu"].loc[:, net.bus.index])


def test_remove_variable(simple_test_net):
    net = simple_test_net

    ow = OutputWriter(net)
    # test printing
    logger.info(ow)
    assert len(ow.log_variables) == 2
    assert ow.log_variables[0][0] == "res_bus" and ow.log_variables[0][1] == "vm_pu"
    assert ow.log_variables[1][0] == "res_line" and ow.log_variables[1][1] == "loading_percent"
    ow.remove_log_variable("res_bus")
    assert ow.log_variables[0][0] == "res_line" and ow.log_variables[0][1] == "loading_percent"
    ow.remove_log_variable("res_line", "loading_percent")
    assert len(ow.log_variables) == 0


def test_store_and_load(simple_test_net):
    net = simple_test_net

    n_timesteps = 2
    profiles, ds = create_data_source(n_timesteps)
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])
    dirname = tempfile.gettempdir()
    ow = OutputWriter(net, output_path=dirname, output_file_type=".json")
    ow.remove_log_variable("res_bus")
    tmp_file = os.path.join(dirname, "net.json")
    pp.to_json(net, tmp_file)
    del net
    del ow
    res_line_file = os.path.join(dirname, "res_line", "loading_percent.json")
    # del result file is one is present
    if os.path.isfile(res_line_file):
        os.remove(res_line_file)
    net = pp.from_json(tmp_file)
    ow = net.output_writer.iat[0, 0]
    assert len(ow.log_variables) == 1
    assert ow.output_path == dirname
    time_steps = range(0, n_timesteps)
    run_timeseries(net, time_steps=time_steps)
    # check if results were written
    assert os.path.isfile(res_line_file)


def test_ppc_log(simple_test_net):
    net = simple_test_net
    n_timesteps = 5
    profiles, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"],
                 recycle=True)

    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json", log_variables=list())
    ow.log_variable('ppc_bus', 'vm')
    ow.log_variable('ppc_bus', 'va')
    pp.runpp(net, only_v_results=True, recycle={"bus_pq": True, "gen": False, "trafo": False})
    run_timeseries(net, time_steps, recycle={"bus_pq": True, "gen": False, "trafo": False}, only_v_results=True)
    assert len(ow.output["ppc_bus.vm"]) == n_timesteps
    assert len(ow.output["ppc_bus.va"]) == n_timesteps


def test_ow_index():
    net = simple_four_bus_system()
    steps = [3,5,7]
    p_data = pd.DataFrame(index=steps, columns=["0", "1"], data=[[0.01, 0.02], 
                                                                 [0.03, 0.04],
                                                                 [0.05, 0.06],
                                                                 ])
    v_data = pd.DataFrame(index=steps, columns=["0"], data=[1.01, 1.03, 1.02])
        
    ds_p = DFData(p_data)
    ds_v = DFData(v_data)
    
    ct.ConstControl(net, element='load', variable='p_mw',
                 element_index=net.load.index.tolist(), data_source=ds_p,
                 profile_name=p_data.columns)
    ct.ConstControl(net, element='ext_grid', variable='vm_pu', 
                 element_index=0, data_source=ds_v, 
                 profile_name='0')
    
    ow = OutputWriter(net)
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    
    run_timeseries(net, time_steps=p_data.index)
    
    assert np.all(ow.output["res_line.loading_percent"].index == p_data.index)
    
if __name__ == '__main__':
    pytest.main(['-s', __file__])