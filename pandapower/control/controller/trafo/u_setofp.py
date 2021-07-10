

from pandapower.control import ContinuousTapControl
from pandapower.control import DiscreteTapControl
from pandapower.control.basic_controller import Controller

__author__ = 'jdollichon'

from pandapower.control.controller.trafo_control import TrafoController
from control.util.characteristic import Characteristic

try:
    import pplog
except:
    import logging as pplog

logger = pplog.getLogger(__name__)


class UsetOfP(Controller):
    """
    Trafo Controller which changes its voltage setpoint in respect to the powerflow at the
    transformator. It can be used as a con

    INPUT:
       **net** (attrdict) - Pandapower struct

       **tid** (int) - ID of the trafo that is controlled

    OPTIONAL:

       **continuous** (boolean, True) - Switch for using continuous or discrete controlling

        **discrete_band** (float, None) - Only used when continuous is False. Voltage limits being
        used around the set-point. E.g. 0.05 would result in an upper limit of set_point + 0.05
        and a lower limit of set_point - 0.05.

       **side** (string) - Side of the transformer where the voltage is controlled ("hv" or "lv")

       **tol** (float, 1e-3) - Voltage tolerance band at bus in Percent

        **characteristic** (Characteristic, [10000, 20000]) - Expects a characteristic curve as an
            instance of control.util.characteristic (also accepts an epsilon for tolerance)

        **in_service** (bool, True) - Indicates if the controller is currently in_service

    """

    def __init__(self, net, tid, continuous=True, discrete_band=None, side="lv", tol=1e-3,
                 characteristic=Characteristic([10, 20], [0.95, 1.05]), in_service=True, **kwargs):
        super(UsetOfP, self).__init__(net, in_service=in_service, **kwargs)

        self.controlled_bus = net.trafo.at[tid, side+"_bus"]

        self.hv_bus = net.trafo.at[tid, "hv_bus"]
        self.lv_bus = net.trafo.at[tid, "lv_bus"]

        # characteristic curve
        self.cc = characteristic
        self.u_target = None
        self.diff = None

        self.continuous = continuous

        if continuous:
            if len(net.ext_grid.query("bus==%u" % self.hv_bus).index.values) != 1:
                raise NotImplementedError("Continuous control is only available for \
                    transformers connected to an external grid.")

            self.t_nom = net.trafo.at[tid, "vn_lv_kv"]/net.trafo.at[tid, "vn_hv_kv"] * \
                net.bus.at[self.hv_bus, "vn_kv"] / net.bus.at[self.lv_bus, "vn_kv"]

            self.ctrl = ContinuousTapControl(net, tid, 1.0, tol=tol, side=side,
                                             trafotype="2W", in_service=in_service,
                                             check_tap_bounds=True)
        else:
            self.discrete_band = discrete_band
            self.ctrl = DiscreteTapControl(net, tid, -discrete_band + 1, discrete_band + 1, side=side,
                                           trafotype="2W", tol=tol, in_service=in_service)

    def control_step(self, net):
        # empty control step: control is executd by underlying trafo controller
        pass

    def is_converged(self, net):
        self.setpoint = self.cc(net.res_bus.at[self.controlled_bus, "p_mw"])

        if self.continuous:
            self.ctrl.vm_set_pu = self.setpoint
        else:
            self.ctrl.vm_lower_pu = -self.discrete_band + self.setpoint
            self.ctrl.vm_upper_pu = self.discrete_band + self.setpoint

        return self.ctrl.is_converged(net)
