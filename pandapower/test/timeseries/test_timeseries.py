# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import tempfile

import numpy as np
import pandas as pd
import pytest

import pandapower.control.util.diagnostic
import pandapower as pp
import logging
from pandapower.control import ContinuousTapControl, ConstControl

from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries, control_diagnostic

logger = logging.getLogger(__name__)

@pytest.fixture
def simple_test_net():
    net = pp.create_empty_network()
    pp.set_user_pf_options(net, init='dc', calculate_voltage_angles=True)
    b0 = pp.create_bus(net, 110)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 20)
    b3 = pp.create_bus(net, 20)
    b4 = pp.create_bus(net, 6)

    pp.create_ext_grid(net, b0)
    pp.create_line(net, b0, b1, 10, "149-AL1/24-ST1A 110.0")

    pp.create_transformer(net, b1, b2, "25 MVA 110/20 kV", name='tr1')

    pp.create_transformer3w_from_parameters(net, b1, b3, b4, 110, 20, 6, 1e2, 1e2, 1e1, 3, 2, 2, 1,
                                            1, 1, 100, 1, 60, 30, 'hv', tap_step_percent=1.5,
                                            tap_step_degree=0, tap_pos=0, tap_neutral=0, tap_max=10,
                                            tap_min=-10, name='tr2')

    pp.create_load(net, b2, 1.5e1, 1, name='trafo1')
    pp.create_load(net, b3, 3e1, 1.5, name='trafo2_mv')
    pp.create_load(net, b4, 2, -0.15, name='trafo2_lv')

    return net


def create_rand_data_source(net, n_timesteps=10):
    profiles = dict()
    elements = ["load", "sgen"]
    for el in elements:
        element = net[el]
        profiles[el] = pd.DataFrame()
        for idx in element.index:
            p_mw = element.loc[idx, "p_mw"]
            profiles[el][el + str(idx)] = np.random.random(n_timesteps) * p_mw

    el = "trafo3w"
    element = net[el]
    profiles[el] = pd.DataFrame()
    for idx in element.index:
        profiles[el][el + str(idx)] = np.random.randint(-3, 3, n_timesteps)

    return profiles


def create_data_source(n_timesteps=10):
    profiles = pd.DataFrame()
    profiles['load1'] = np.random.random(n_timesteps) * 2e1
    profiles['load2_mv_p'] = np.random.random(n_timesteps) * 4e1
    profiles['load2_mv_q'] = np.random.random(n_timesteps) * 1e1

    profiles['load3_hv_p'] = profiles.load2_mv_p + abs(np.random.random())
    profiles['load3_hv_q'] = profiles.load2_mv_q + abs(np.random.random())

    profiles['slack_v'] = np.clip(np.random.random(n_timesteps) + 0.5, 0.8, 1.2)
    profiles['trafo_v'] = np.clip(np.random.random(n_timesteps) + 0.5, 0.9, 1.1)

    profiles["trafo_tap"] = np.random.randint(-3, 3, n_timesteps)

    ds = DFData(profiles)

    return profiles, ds


def setup_output_writer(net, time_steps):
    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir())
    ow.log_variable('load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_trafo3w', 'p_hv_mw')
    ow.log_variable('res_trafo3w', 'q_hv_mvar')
    return ow


def test_const_control(simple_test_net):
    net = simple_test_net
    profiles, ds = create_data_source()
    time_steps = range(0, 10)
    ow = setup_output_writer(net, time_steps)

    ConstControl(net, 'load', 'p_mw', element_index=0, data_source=ds, profile_name='load1',
                 scale_factor=0.85)

    ConstControl(net, 'ext_grid', 'vm_pu', element_index=0, data_source=ds, profile_name='slack_v')

    run_timeseries(net, time_steps, output_writer=ow, verbose=False)

    assert np.alltrue(profiles['load1'].values * 0.85 == ow.output['load.p_mw'][0].values)
    assert np.alltrue(profiles['slack_v'].values == ow.output['res_bus.vm_pu'][0].values)


def test_const_control_write_to_object_attribute(simple_test_net):
    net = simple_test_net
    profiles, ds = create_data_source()
    time_steps = range(0, 10)
    ow = setup_output_writer(net, time_steps)

    ContinuousTapControl(net, 0, 1., level=1, check_tap_bounds=False)

    ConstControl(net, 'load', 'p_mw', element_index=0, data_source=ds, profile_name='load1',
                 scale_factor=0.85)

    ConstControl(net, 'ext_grid', 'vm_pu', element_index=0, data_source=ds, profile_name='slack_v')

    ConstControl(net, 'controller', 'object.vm_set_pu', element_index=0, data_source=ds, profile_name='trafo_v')

    run_timeseries(net, time_steps, output_writer=ow, verbose=False)

    assert np.alltrue(profiles['load1'].values * 0.85 == ow.output['load.p_mw'][0].values)
    assert np.alltrue(profiles['slack_v'].values == ow.output['res_bus.vm_pu'][0].values)
    assert np.allclose(profiles['trafo_v'].values, ow.output['res_bus.vm_pu'][net.trafo.at[0, 'lv_bus']].values, atol=1e-3, rtol=0)


def test_false_alarm_trafos(simple_test_net):
    net = simple_test_net

    import io
    s = io.StringIO()
    h = logging.StreamHandler(stream=s)
    pandapower.control.util.diagnostic.logger.addHandler(h)

    ContinuousTapControl(net, 0, 1)
    ContinuousTapControl(net, 0, 1, trafotype='3W')

    if 'convergence problems' in s.getvalue():
        raise UserWarning('Control diagnostic raises false alarm! Controllers are fine, '
                          'but warning is raised: %s' % s.getvalue())

    control_diagnostic(net)
    if 'convergence problems' in s.getvalue():
        raise UserWarning('Control diagnostic raises false alarm! Controllers are fine, '
                          'but warning is raised: %s' % s.getvalue())

    pandapower.control.util.diagnostic.logger.removeHandler(h)
    del h
    del s


def test_timeseries_results(simple_test_net):
    # This test compares output writer results with input
    # test net
    net = simple_test_net
    net.user_pf_options = dict()

    n_timesteps = 5
    profiles, ds = create_data_source(n_timesteps)

    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"],
                 scale_factor=0.5)

    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')

    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    run_timeseries(net, time_steps, output_writer=ow, verbose=False)
    assert np.allclose(ow.output['res_load.p_mw'].sum().values * 2,
                       profiles[["load1", "load2_mv_p", "load3_hv_p"]].sum().values)

    # 3load - @Rieke What is this test for compared to the first one?
    # @Flo in / out of service testen ...
    ow.log_variable('res_load', 'p_mw')
    net.controller.in_service = False  # set the first controller out of service
    ConstControl(net, 'load', 'p_mw', element_index=0, data_source=ds, profile_name='load1')

    run_timeseries(net, time_steps, output_writer=ow, verbose=False)
    assert np.allclose(ow.output['res_load.p_mw'][0].sum(), profiles["load1"].sum())


def test_timeseries_var_func(simple_test_net):
    # This test checks if the output writer works with a user defined function

    # test net
    net = simple_test_net

    n_timesteps = 5
    profiles, ds = create_data_source(n_timesteps)

    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"],
                 scale_factor=0.5)

    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_load', 'p_mw', eval_function=np.max)
    ow.log_variable('res_bus', 'vm_pu', eval_function=np.min)
    ow.log_variable('res_bus', 'q_mvar', eval_function=np.sum)

    run_timeseries(net, time_steps, output_writer=ow, verbose=False)
    # asserts if last value of output_writers output is the minimum value
    assert net["res_load"]["p_mw"].max() == ow.output["res_load.p_mw"].iloc[-1].values
    assert net["res_bus"]["vm_pu"].min() == ow.output["res_bus.vm_pu"].iloc[-1, -1]
    assert net["res_bus"]["q_mvar"].sum() == ow.output["res_bus.q_mvar"].iloc[-1].values

    # get minimum voltage of all hv busses
    mask = (net.bus.vn_kv > 70.0) & (net.bus.vn_kv < 380.0)
    hv_busses_index = net.bus.loc[mask].index
    mask = (net.bus.vn_kv > 1.0) & (net.bus.vn_kv < 70.0)
    mv_busses_index = net.bus.loc[mask].index
    ow.log_variable('res_bus', 'vm_pu', index=hv_busses_index, eval_function=np.min,
                    eval_name="hv_bus_min")
    ow.log_variable('res_bus', 'vm_pu', index=mv_busses_index, eval_function=np.min,
                    eval_name="mv_bus_min")
    run_timeseries(net, time_steps, output_writer=ow, verbose=False)
    assert net["res_bus"].loc[hv_busses_index, "vm_pu"].min() == ow.output["res_bus.vm_pu"].loc[
        time_steps[-1], "hv_bus_min"]
    assert net["res_bus"].loc[mv_busses_index, "vm_pu"].min() == ow.output["res_bus.vm_pu"].loc[
        time_steps[-1], "mv_bus_min"]


def test_time_steps(simple_test_net):
    net = simple_test_net
    n_timesteps = 11
    profiles, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    # correct
    run_timeseries(net, time_steps=range(0, n_timesteps), verbose=False)
    # also correct
    run_timeseries(net, time_steps=[0, 2, 4, 8, 9], verbose=False)
    # ok. missing time_step list -> should check the datasource
    run_timeseries(net, verbose=False)
    # depricated
    run_timeseries(net, time_steps=(0, 10), verbose=False)


def test_output_dump_after_time(simple_test_net):
    net = simple_test_net

    n_timesteps = 100
    profiles, ds = create_data_source(n_timesteps)

    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    time_steps = range(0, n_timesteps)
    # write output after 0.1 minutes to disk
    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json",
                      write_time=0.05)
    ow.log_variable('res_load', 'p_mw')
    ow.log_variable('res_bus', 'vm_pu')

    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    run_timeseries(net, time_steps, output_writer=ow, verbose=False)
    # ToDo: read partially dumped results and compare with all stored results


if __name__ == '__main__':
    pytest.main(['-s', __file__])