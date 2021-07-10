# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower as pp
from pandapower.control import DiscreteTapControl
import pytest
import pandapower.networks as nw
import logging as log

logger = log.getLogger(__name__)


def test_discrete_tap_control_lv():
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

    DiscreteTapControl(net, tid=0, side='lv', vm_lower_pu=0.95, vm_upper_pu=0.99)

    logger.info("case1: low voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] ==  -1

    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 1.03
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == -2
    # reduce voltage from 1.03 pu to 0.949 pu
    net.ext_grid.vm_pu = 0.949
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == 1


def test_discrete_tap_control_hv():
    # --- load system and run power flow
    net = nw.simple_four_bus_system()
    pp.set_user_pf_options(net, init='dc', calculate_voltage_angles=True)
    # --- initial tap data
    net.trafo.tap_side = 'hv'
    net.trafo.tap_neutral = 0
    net.trafo.tap_min = -2
    net.trafo.tap_max = 2
    net.trafo.tap_step_percent = 1.25
    net.trafo.tap_pos = 0
    # --- run loadflow
    pp.runpp(net)

    DiscreteTapControl(net, tid=0, side='lv', vm_lower_pu=0.95, vm_upper_pu=0.99)

    logger.info("case1: low voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == 1
    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 1.03
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == 2
    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 0.949
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == -1


def test_discrete_tap_control_lv_from_tap_step_percent():
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

    DiscreteTapControl.from_tap_step_percent(net, tid=0, side='lv', vm_set_pu=0.98)

    logger.info("case1: low voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] ==  -1

    # check if it changes the lower and upper limits
    net.controller.object.at[0].vm_set_pu = 1
    pp.runpp(net, run_control=True)
    assert abs(net.controller.object.at[0].vm_upper_pu - 1.00725) < 1e-6
    assert abs(net.controller.object.at[0].vm_lower_pu - 0.99275) < 1e-6
    net.controller.object.at[0].vm_set_pu = 0.98
    pp.runpp(net, run_control=True)
    assert abs(net.controller.object.at[0].vm_upper_pu - 0.98725) < 1e-6
    assert abs(net.controller.object.at[0].vm_lower_pu - 0.97275) < 1e-6

    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 1.03
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == -2
    # reduce voltage from 1.03 pu to 0.969 pu
    net.ext_grid.vm_pu = 0.969
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == 1


def test_discrete_tap_control_hv_from_tap_step_percent():
    # --- load system and run power flow
    net = nw.simple_four_bus_system()
    pp.set_user_pf_options(net, init='dc', calculate_voltage_angles=True)
    # --- initial tap data
    net.trafo.tap_side = 'hv'
    net.trafo.tap_neutral = 0
    net.trafo.tap_min = -2
    net.trafo.tap_max = 2
    net.trafo.tap_step_percent = 1.25
    net.trafo.tap_pos = 0
    # --- run loadflow
    pp.runpp(net)

    DiscreteTapControl.from_tap_step_percent(net, tid=0, side='lv', vm_set_pu=0.98)

    logger.info("case1: low voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] ==  1

    # check if it changes the lower and upper limits
    net.controller.object.at[0].vm_set_pu = 1
    pp.runpp(net, run_control=True)
    assert abs(net.controller.object.at[0].vm_upper_pu - 1.00725) < 1e-6
    assert abs(net.controller.object.at[0].vm_lower_pu - 0.99275) < 1e-6
    net.controller.object.at[0].vm_set_pu = 0.98
    pp.runpp(net, run_control=True)
    assert abs(net.controller.object.at[0].vm_upper_pu - 0.98725) < 1e-6
    assert abs(net.controller.object.at[0].vm_lower_pu - 0.97275) < 1e-6

    # increase voltage from 1.0 pu to 1.03 pu
    net.ext_grid.vm_pu = 1.03
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == 2
    # reduce voltage from 1.03 pu to 0.969 pu
    net.ext_grid.vm_pu = 0.969
    # switch back tap position
    net.trafo.tap_pos.at[0] = 0
    pp.runpp(net)

    logger.info("case2: high voltage")
    logger.info("before control: trafo voltage at low voltage bus is %f, tap position is %u"
                % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))

    # run control
    pp.runpp(net, run_control=True)
    logger.info(
        "after DiscreteTapControl: trafo voltage at low voltage bus is %f, tap position is %f"
        % (net.res_bus.vm_pu[net.trafo.lv_bus].values, net.trafo.tap_pos.values))
    assert net.trafo.tap_pos.at[0] == -1


if __name__ == '__main__':
    pytest.main(['-xs', __file__])
