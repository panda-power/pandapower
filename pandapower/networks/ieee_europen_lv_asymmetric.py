# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:15:13 2019

@author: uk067483
"""

import os
import pandapower as pp
from pandapower.pf.runpp_3ph import runpp_3ph
from pandapower import pp_dir


def ieee_european_lv_asymmetric(scenario="on_peak_566"):
    """
    Loads the IEEE European LV network, a generic 0.416 kV network serviced by one 0.8 MVA MV/LV transformer
    station. The network supplies 906 LV buses and 55 1-PH loads
    The network layout is mostly radial

    The network can be loaded with three different scenarios for On-Peak and Off-Peak load
    which are defined by scaling factors for loads / generators . 
    These scenarios are a good starting point for working with this
    network, but you are of course free to parametrize the network for your use case.

    The network also includes geographical information of lines and buses for plotting.

    OPTIONAL:
        **scenario** - (str, "on_mid"): defines the scaling for load and generation

                - "on_mid": high load scenario
                - "off_start": low load scenario at 12:01 AM
                - "off_end": low load scenario at mignight

    OUTPUT:
         **net** - pandapower network

    EXAMPLE:

    import pandapower.networks
    net = pandapower.networks.ieee_european_lv_asymmetric("off_start")
    """
    if scenario == "off_peak_1":
        net = pp.from_json(os.path.join(pp_dir, "networks", "IEEE_European_LV_Off_Peak_1.json"))
    elif scenario == "on_peak_566":
        net = pp.from_json(os.path.join(pp_dir, "networks", "IEEE_European_LV_On_Peak_566.json"))
    elif scenario == "off_peak_1440":
        net = pp.from_json(os.path.join(pp_dir, "networks", "IEEE_European_LV_Off_Peak_1440.json"))
    else:
        raise ValueError("Unknown scenario %s - chose 'on_peak_566' or 'off_peak_1' or 'off_peak_1440'" % scenario)
    return net