# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.control.controller.trafo_control import TrafoController

class DiscreteTapControl(TrafoController):
    """
    Trafo Controller with local tap changer voltage control.

    INPUT:
        **net** (attrdict) - Pandapower struct

        **tid** (int) - ID of the trafo that is controlled

        **vm_lower_pu** (float) - Lower voltage limit in pu

        **vm_upper_pu** (float) - Upper voltage limit in pu

    OPTIONAL:

        **side** (string, "lv") - Side of the transformer where the voltage is controlled (hv or lv)

        **trafotype** (float, "2W") - Trafo type ("2W" or "3W")

        **tol** (float, 0.001) - Voltage tolerance band at bus in Percent (default: 1% = 0.01pu)

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """

    def __init__(self, net, tid, vm_lower_pu, vm_upper_pu, side="lv", trafotype="2W",
                 tol=1e-3, in_service=True, order=0, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"tid": tid, 'trafotype': trafotype}
        super(DiscreteTapControl, self).__init__(
            net, tid, side, tol=tol, in_service=in_service, order=order, trafotype=trafotype,
            drop_same_existing_ctrl=drop_same_existing_ctrl, matching_params=matching_params,
            **kwargs)

        self.vm_lower_pu = vm_lower_pu
        self.vm_upper_pu = vm_upper_pu

        self.tap_pos = net[self.trafotable].at[tid, "tap_pos"]
        self.vm_delta_pu = net[self.trafotable].at[tid, "tap_step_percent"] / 100. * .5 + self.tol
        self.vm_set_pu = kwargs.get("vm_set_pu")

    @classmethod
    def from_tap_step_percent(cls, net, tid, vm_set_pu, side="lv", trafotype="2W", tol=1e-3, in_service=True, order=0,
                              drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        """
        Alternative mode of the controller, which uses a set point for voltage and the value of net.trafo.tap_step_percent to calculate
        vm_upper_pu and vm_lower_pu. To this end, the parameter vm_set_pu should be provided, instead of vm_lower_pu and vm_upper_pu.
        To use this mode of the controller, the controller can be initialized as following:

        >>> c = DiscreteTapControl.from_tap_step_percent(net, tid, vm_set_pu)

        INPUT:
            **net** (attrdict) - Pandapower struct

            **tid** (int) - ID of the trafo that is controlled

            **vm_set_pu** (float) - Voltage setpoint in pu
        """
        self = cls(net, tid=tid, vm_lower_pu=None, vm_upper_pu=None, side=side, trafotype=trafotype, tol=tol,
                   in_service=in_service, order=order, drop_same_existing_ctrl=drop_same_existing_ctrl,
                   matching_params=matching_params, vm_set_pu=vm_set_pu, **kwargs)
        return self

    @property
    def vm_set_pu(self):
        return self._vm_set_pu

    @vm_set_pu.setter
    def vm_set_pu(self, value):
        self._vm_set_pu = value
        if value is None:
            return
        self.vm_lower_pu = value - self.vm_delta_pu
        self.vm_upper_pu = value + self.vm_delta_pu

    def initialize_control(self, net):
        if hasattr(self, 'vm_set_pu') and self.vm_set_pu is not None:
            self.vm_delta_pu = net[self.trafotable].at[self.tid, "tap_step_percent"] / 100. * .5 + self.tol

    def control_step(self, net):
        """
        Implements one step of the Discrete controller, always stepping only one tap position up or down
        """
        vm_pu = net.res_bus.at[self.controlled_bus, "vm_pu"]
        self.tap_pos = net[self.trafotable].at[self.tid, "tap_pos"]

        if self.tap_side_coeff * self.tap_sign == 1:
            if vm_pu < self.vm_lower_pu and self.tap_pos > self.tap_min:
                self.tap_pos -= 1
            elif vm_pu > self.vm_upper_pu and self.tap_pos < self.tap_max:
                self.tap_pos += 1
        elif self.tap_side_coeff * self.tap_sign == -1:
            if vm_pu < self.vm_lower_pu and self.tap_pos < self.tap_max:
                self.tap_pos += 1
            elif vm_pu > self.vm_upper_pu and self.tap_pos > self.tap_min:
                self.tap_pos -= 1

        # WRITE TO NET
        net[self.trafotable].at[self.tid, "tap_pos"] = self.tap_pos

    def is_converged(self, net):
        """
        Checks if the voltage is within the desired voltage band, then returns True
        """
        if not self.tid in net[self.trafotable].index or \
           not net[self.trafotable].at[self.tid, 'in_service']:
            return True
        vm_pu = net.res_bus.at[self.controlled_bus, "vm_pu"]
        self.tap_pos = net[self.trafotable].at[self.tid, "tap_pos"]

        # render this controller converged if he cant reach the desired point
        if self.tap_side_coeff * self.tap_sign == 1:
            if vm_pu < self.vm_lower_pu and self.tap_pos == self.tap_min:
                return True
            elif vm_pu > self.vm_upper_pu and self.tap_pos == self.tap_max:
                return True
        elif self.tap_side_coeff * self.tap_sign == -1:
            if vm_pu < self.vm_lower_pu and self.tap_pos == self.tap_max:
                return True
            elif vm_pu > self.vm_upper_pu and self.tap_pos == self.tap_min:
                return True
        return self.vm_lower_pu < vm_pu < self.vm_upper_pu

