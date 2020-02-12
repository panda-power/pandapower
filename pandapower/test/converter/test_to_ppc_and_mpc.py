# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest
import pandapower.converter as cv
from pandapower.test.converter.test_from_ppc import get_testgrids
from pandapower.pypower.idx_bus import VM, BUS_I, VA
from pandapower.powerflow import LoadflowNotConverged, reset_results
from pandapower.pf.runpf_pypower import _runpf_pypower


def test_to_ppc_and_mpc():
    # pypower cases to validate
    functions = ['case4gs', 'case6ww', 'case30', 'case39']
    for fn in functions:
        # get pypower grids with results
        ppc_net = get_testgrids(fn, 'pypower_cases.p')

        # get pandapower grids
        pandapower_module = __import__('pandapower', fromlist=['networks'])
        pandapower_function = getattr(pandapower_module.networks, fn)
        net = pandapower_function()
        reset_results(net)

        # convert pandapower grids to ppc
        ppc = cv.to_ppc(net)
        # convert pandapower grids to mpc (no result validation)
        mpc = cv.to_mpc(net)

        # validate voltage results of pandapower-to-ppc-converted grids vs. original pypower results
        net["_options"]['ac'] = True
        net["_options"]['numba'] = True
        net["_options"]['tolerance_mva'] = 1e-8
        net["_options"]['algorithm'] = "fdbx"
        net["_options"]['max_iteration'] = 30
        net["_options"]['enforce_q_lims'] = False
        net["_options"]['calculate_voltage_angles'] = True
        res_converted_pp, status_converted_pp = _runpf_pypower(ppc, net["_options"])

        if status_converted_pp:
            # get lookup pp2ppc
            bus_lookup = net["_pd2ppc_lookups"]["bus"]
            # check for equality in bus voltages
            pp_buses = bus_lookup[res_converted_pp['bus'][:, BUS_I].astype(int)]
            res1 = res_converted_pp['bus'][pp_buses, VM:VA + 1]
            res2 = ppc_net['bus'][:, VM:VA + 1]
            assert np.allclose(res1, res2)
        else:
            raise LoadflowNotConverged("Loadflow did not converge!")


if __name__ == "__main__":
    pytest.main(["test_to_ppc_and_mpc.py", "-s"])
