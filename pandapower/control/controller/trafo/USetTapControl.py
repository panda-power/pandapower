# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.control.controller.characteristic_control import CharacteristicControl
from pandapower.control.util.characteristic import Characteristic

class USetTapControl(CharacteristicControl):
    """
    Controller that adjusts the setpoint of a local tap changer voltage control based on a load flow result (e.g. p_lv_mw, i_lv_ka etc.)
    according to a defined characteristic.

    INPUT:
        **net** (attrdict) - Pandapower net

        **cid** (int) - ID of the tap changer controller, an attribute of which is controlled

        **variable** (float) - Variable from the result table that is used for the characteristic

    OPTIONAL:

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """

    def __init__(self, net, cid, variable='p_hv_mw', characteristic=Characteristic([10, 20], [0.95, 1.05]), tol=1e-3, in_service=True,
                 order=0, level=0, drop_same_existing_ctrl=False, matching_params=None, **kwargs):
        if matching_params is None:
            matching_params = {"cid": cid, 'variable': variable}
        c = net.controller.at[cid, 'object']
        super().__init__(net, output_element="controller", output_variable="object.vm_set_pu", output_element_index=cid,
                         input_element="res_" + c.trafotable, input_variable=variable, input_element_index=c.tid,
                         characteristic=characteristic, tol=tol, in_service=in_service, order=order,
                         level=level, drop_same_existing_ctrl=drop_same_existing_ctrl, matching_params=matching_params, **kwargs)