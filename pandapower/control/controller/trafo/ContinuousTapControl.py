# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from pandapower.control.controller.trafo_control import TrafoController


class ContinuousTapControl(TrafoController):
    """
    Trafo Controller with local tap changer voltage control.

    INPUT:
        **net** (attrdict) - Pandapower struct

        **tid** (int) - ID of the trafo that is controlled

        **vm_set_pu** (float) - Maximum OLTC target voltage at bus in pu

    OPTIONAL:

        **tol** (float, 0.001) - Voltage tolerance band at bus in percent (default: 1% = 0.01pu)

        **side** (string, "lv") - Side of the transformer where the voltage is controlled

        **trafo_type** (float, "2W") - Trafo type ("2W" or "3W")

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **check_tap_bounds** (bool, True) - In case of true the tap_bounds will be considered

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """

    def __init__(self, net, tid, vm_set_pu, tol=1e-3, side="lv", trafotype="2W", in_service=True,
                 check_tap_bounds=True, level=0, order=0, drop_same_existing_ctrl=False,
                 matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"tid": tid, 'trafotype': trafotype}
        super().__init__(net, tid=tid, side=side, tol=tol, in_service=in_service,
                         trafotype=trafotype, level=level, order=order,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)

        t = net[self.trafotable]
        b = net.bus
        if trafotype == "2W":
            self.t_nom = t.at[tid, "vn_lv_kv"] / t.at[tid, "vn_hv_kv"] * \
                         b.at[net[self.trafotable].at[tid, "hv_bus"], "vn_kv"] / \
                         b.at[net[self.trafotable].at[tid, "lv_bus"], "vn_kv"]
        elif side == "lv":
            self.t_nom = t.at[tid, "vn_lv_kv"] / t.at[tid, "vn_hv_kv"] * \
                         b.at[net[self.trafotable].at[tid, "hv_bus"], "vn_kv"] / \
                         b.at[net[self.trafotable].at[tid, "lv_bus"], "vn_kv"]
        elif side == "mv":
            self.t_nom = t.at[tid, "vn_mv_kv"] / t.at[tid, "vn_hv_kv"] * \
                         b.at[net[self.trafotable].at[tid, "hv_bus"], "vn_kv"] / \
                         b.at[net[self.trafotable].at[tid, "mv_bus"], "vn_kv"]

        self.check_tap_bounds = check_tap_bounds
        self.vm_set_pu = vm_set_pu
        self.trafotype = trafotype
        if trafotype == "2W":
            net.trafo["tap_pos"] = net.trafo.tap_pos.astype(float)
        elif trafotype == "3W":
            net.trafo3w["tap_pos"] = net.trafo3w.tap_pos.astype(float)
        self.tol = tol

    def control_step(self, net):
        """
        Implements one step of the ContinuousTapControl
        """
        delta_vm_pu = net.res_bus.at[self.controlled_bus, "vm_pu"] - self.vm_set_pu
        tc = delta_vm_pu / self.tap_step_percent * 100 / self.t_nom
        self.tap_pos += tc * self.tap_side_coeff * self.tap_sign
        if self.check_tap_bounds:
            self.tap_pos = np.clip(self.tap_pos, self.tap_min, self.tap_max)

        # WRITE TO NET
        net[self.trafotable].at[self.tid, "tap_pos"] = self.tap_pos

    def is_converged(self, net):
        """
        The ContinuousTapControl is converged, when the difference of the voltage between control steps is smaller
        than the Tolerance (tol).
        """

        if not net[self.trafotable].at[self.tid, 'in_service']:
            return True
        vm_pu = net.res_bus.at[self.controlled_bus, "vm_pu"]
        self.tap_pos = net[self.trafotable].at[self.tid, 'tap_pos']
        difference = 1 - self.vm_set_pu / vm_pu

        if self.check_tap_bounds:
            if self.tap_side_coeff * self.tap_sign == 1:
                if vm_pu < self.vm_set_pu and self.tap_pos == self.tap_min:
                    return True
                elif vm_pu > self.vm_set_pu and self.tap_pos == self.tap_max:
                    return True
            elif self.tap_side_coeff * self.tap_sign == -1:
                if vm_pu > self.vm_set_pu and self.tap_pos == self.tap_min:
                    return True
                elif vm_pu < self.vm_set_pu and self.tap_pos == self.tap_max:
                    return True
        return abs(difference) < self.tol
