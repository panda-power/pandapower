__author__ = 'lthurner'

import numpy as np
from pandapower.control.basic_controller import Controller

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)


class TrafoController(Controller):
    """
    Trafo Controller with local tap changer voltage control.

    INPUT:
       **net** (attrdict) - Pandapower struct

       **tid** (int) - ID of the trafo that is controlled

       **side** (string) - Side of the transformer where the voltage is controlled ("hv", "mv"
       or "lv")

       **tol** (float) - Voltage tolerance band at bus in Percent

       **in_service** (bool) - Indicates if the element is currently active

       **trafotype** (string) - Type of the controlled trafo ("2W" or "3W")

    OPTIONAL:
        **recycle** (bool, True) - Re-use of internal-data in a time series loop.
    """

    def __init__(self, net, tid, side, tol, in_service, trafotype, level=0, order=0, recycle=True,
                 **kwargs):
        super().__init__(net, in_service=in_service, level=level, order=order, recycle=recycle,
                         **kwargs)
        self.tid = tid
        self.trafotype = trafotype

        if trafotype == "2W":
            if side not in ["hv", "lv"]:
                raise UserWarning("side has to be 'hv' or 'lv' for high/low voltage, "
                                  "received %s" % side)
            self.trafotable = "trafo"
        elif trafotype == "3W":
            if side not in ["hv", "mv", "lv"]:
                raise UserWarning("side has to be 'hv', 'mv' or 'lv' for high/middle/low voltage, "
                                  "received %s" % side)
            self.trafotable = "trafo3w"
        else:
            raise UserWarning("unknown trafo type, received %s" % trafotype)

        self.element_in_service = self.net[self.trafotable].at[self.tid, "in_service"]

        self.side = side
        self.controlled_bus = self.net[self.trafotable].at[tid, side + "_bus"]
        if self.controlled_bus in self.net.ext_grid.loc[self.net.ext_grid.in_service, 'bus'].values:
            logger.warning("Controlled Bus is Slack Bus - deactivating controller")
            self.set_active(False)
        elif self.controlled_bus in self.net.ext_grid.loc[
            ~self.net.ext_grid.in_service, 'bus'].values:
            logger.warning("Controlled Bus is Slack Bus with slack out of service - "
                           "not deactivating controller")

        self.tap_min = self.net[self.trafotable].at[tid, "tap_min"]
        self.tap_max = self.net[self.trafotable].at[tid, "tap_max"]
        self.tap_step_percent = self.net[self.trafotable].at[tid, "tap_step_percent"]
        self.tap_step_degree = self.net[self.trafotable].at[tid, "tap_step_degree"]
        if not np.isnan(self.tap_step_degree):
            self.tap_sign = np.sign(np.cos(np.deg2rad(self.tap_step_degree)))
        else:
            self.tap_sign = 1
        if self.tap_sign == 0 or np.isnan(self.tap_sign):
            # 0 is very unprobable case because of numpy, but still checking to be 100 % sure
            self.tap_sign = 1
        self.tap_pos = self.net[self.trafotable].at[self.tid, "tap_pos"]
        if np.isnan(self.tap_pos):
            self.tap_pos = self.net[self.trafotable].at[tid, "tap_neutral"]

        if np.isnan(self.tap_min) or \
                np.isnan(self.tap_max) or \
                np.isnan(self.tap_step_percent):
            logger.error("Trafo-Controller has been initialized with NaN values, check "
                         "net.trafo.tap_pos etc. if they are set correctly!")

        self.tol = tol
        tap_side = self.net[self.trafotable].tap_side.at[tid]
        if trafotype == "2W":
            if tap_side == "hv":
                self.tap_side_coeff = 1
            elif tap_side == "lv":
                self.tap_side_coeff = -1
            else:
                raise ValueError("Trafo tap side (in net.%s) has to be either hv or lv, "
                                 "but received: %s for trafo %s" % (self.trafotable, tap_side,
                                                                    self.tid))
        elif trafotype == "3W":
            if tap_side == "hv":
                self.tap_side_coeff = 1
            elif tap_side in ["mv", "lv"]:
                self.tap_side_coeff = -1
            else:
                raise ValueError("Trafo tap side (in net.%s) has to be either hv, mv or lv, "
                                 "but received %s for trafo %s" % (self.trafotable, tap_side,
                                                                   self.tid))
        if self.net[self.trafotable].at[self.tid, "tap_step_percent"] < 0:
            self.tap_side_coeff *= -1
        self.set_recycle()

    def set_recycle(self):
        allowed_elements = ["2W", "3W"]
        if self.recycle is False or self.trafotype not in allowed_elements:
            # if recycle is set to False by the user when creating the controller it is deactivated or when
            # const control controls an element which is not able to be recycled
            self.recycle = False
            return
        # these variables determine what is re-calculated during a time series run
        recycle = dict(trafo=True, gen=False, bus_pq=False)
        self.recycle = recycle
            
    def timestep(self):
        self.tap_pos = self.net[self.trafotable].at[self.tid, "tap_pos"]

    def __repr__(self):
        s = '%s of %s %d' % (self.__class__.__name__, self.trafotable, self.tid)
        return s

    def __str__(self):
        s = '%s of %s %d' % (self.__class__.__name__, self.trafotable, self.tid)
        return s
