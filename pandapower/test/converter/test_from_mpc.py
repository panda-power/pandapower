# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import pytest

import pandapower as pp
import pandapower.networks as pn
from pandapower.converter import from_mpc

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def test_from_mpc():
    case24 = pn.case24_ieee_rts()
    pp.set_user_pf_options(case24)
    this_folder = os.path.join(pp.pp_dir, "test", "converter")
    mat_case_path = os.path.join(this_folder, 'case24_ieee_rts.mat')
    case24_from_mpc = from_mpc(mat_case_path, f_hz=60, casename_mpc_file='mpc')

    pp.runpp(case24)
    pp.runpp(case24_from_mpc)

    assert case24_from_mpc.converged
    assert pp.nets_equal(case24, case24_from_mpc, check_only_results=True)


if __name__ == '__main__':
    pytest.main(["test_from_mpc.py"])
