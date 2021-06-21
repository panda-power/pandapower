# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from pandapower import pp_dir
import pandapower as pp
import numpy as np
import os
import pytest
from packaging import version as vs

folder = os.path.join(pp_dir, "test", "test_files", "old_versions")
found_versions = [file.split("_")[1].split(".json")[0] for _, _, files
                  in os.walk(folder) for file in files]


@pytest.mark.slow
@pytest.mark.parametrize("version", found_versions)
def test_convert_format(version):
    filename = os.path.join(folder, "example_%s.json" % version)
    if not os.path.isfile(filename):
        raise ValueError("File for version %s does not exist" % version)
    try:
        net = pp.from_json(filename, convert=False)
        if ('version' in net) and (vs.parse(str(net.version)) > vs.parse('2.0.1')):
            _ = pp.from_json(filename, table_selection = ['bus', 'load'])
    except:
        raise UserWarning("Can not load network saved in pandapower version %s" % version)
    vm_pu_old = net.res_bus.vm_pu.copy()
    pp.convert_format(net)
    try:
        pp.runpp(net, run_control="controller" in net and len(net.controller) > 0)
    except:
        raise UserWarning("Can not run power flow in network "
                          "saved with pandapower version %s" % version)
    if not np.allclose(vm_pu_old.values, net.res_bus.vm_pu.values):
        raise UserWarning("Power flow results mismatch "
                          "with pandapower version %s" % version)

def test_convert_format_pq_bus_meas():
    net = pp.from_json(os.path.join(folder, "example_2.3.1.json"), convert=False)
    net = pp.convert_format(net)
    pp.runpp(net)

    bus_p_meas = net.measurement.query("element_type=='bus' and measurement_type=='p'").set_index("element", drop=True)
    assert np.allclose(net.res_bus.p_mw, bus_p_meas["value"])

def test_convert_format_controller():
    net = pp.from_json(os.path.join(folder, "example_2.3.0.json"), convert=True)
    controller = net.controller.object.iloc[0]
    assert not hasattr(controller, "net")

if __name__ == '__main__':
    pytest.main([__file__])
