# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower as pp
import pytest
import pandapower.networks as nw
import logging as log
import numpy as np

logger = log.getLogger(__name__)
from pandapower.control import ContinuousTapControl


def test_continuous_tap_control_lv():
    # --- load system and run power flow
    net = nw.simple_four_bus_system()
    pp.set_user_pf_options(net, init='dc', calculate_voltage_angles=True)
    # --- initial tap data
    net.trafo.tap_side = 'lv'
    net.trafo.tap_neutral = 0
    net.trafo.tap_min = -2
    net.trafo.tap_max = 2
    net.trafo.tap_step_percent = 1.25
    net.trafo.tap_pos = 0
    # --- run loadflow
    pp.runpp(net)

    # todo: rewrite to not compare to hardcoded values
    tid = 0
    ContinuousTapControl(net, tid=tid, vm_set_pu=0.99, side='lv')
    # DiscreteTapControl(net, tid=0, side='lv', vm_lower_pu=0.95, vm_upper_pu=0.99)

    logger.info("case1: low voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    pp.runpp(net)
    logger.info(
        "after ContinuousTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_trafo.vm_lv_pu.values, net.trafo.tap_pos.values))

    assert np.isclose(net.res_trafo.vm_lv_pu.at[tid], 0.99, atol=1e-3)
    assert np.isclose(net.trafo.tap_pos.values, -.528643)
    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 1.03
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    pp.runpp(net)
    logger.info(
        "after ContinuousTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))

    assert np.isclose(net.trafo.tap_pos.values, -2)
    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 0.98
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    pp.runpp(net)
    logger.info(
        "after ContinuousTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))
    assert np.isclose(net.res_trafo.vm_lv_pu.at[tid], 0.99, atol=1e-3)
    assert np.isclose(net.trafo.tap_pos.values, 1.077656)


def test_continuous_tap_control_hv():
    # --- load system and run power flow
    net = nw.simple_four_bus_system()
    # --- initial tap data
    net.trafo.tap_side = 'hv'
    net.trafo.tap_neutral = 0
    net.trafo.tap_min = -2
    net.trafo.tap_max = 2
    net.trafo.tap_step_percent = 1.25
    net.trafo.tap_pos = 0
    # --- run loadflow
    pp.runpp(net)
    tid = 0
    ContinuousTapControl(net, tid=tid, vm_set_pu=0.99, side='lv')
    # td = control.DiscreteTapControl(net, tid=0, side='lv', vm_lower_pu=0.95, vm_upper_pu=0.99)

    logger.info("case1: low voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    pp.runpp(net)
    logger.info(
        "after ContinuousTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))

    assert np.isclose(net.res_trafo.vm_lv_pu.at[tid], 0.99, atol=1e-3)
    assert np.isclose(net.trafo.tap_pos.values, 0.528643)

    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 1.03
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    pp.runpp(net)
    logger.info(
        "after ContinuousTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))
    assert np.isclose(net.trafo.tap_pos.values, 2)

    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 0.98
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    pp.runpp(net)
    logger.info(
        "after ContinuousTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_trafo.vm_lv_pu.at[tid], net.trafo.tap_pos.values))
    assert np.isclose(net.res_trafo.vm_lv_pu.at[tid], 0.99, atol=1e-3)
    assert np.isclose(net.trafo.tap_pos.values, -1.07765621)


if __name__ == '__main__':
    pytest.main(['-xs', __file__])
