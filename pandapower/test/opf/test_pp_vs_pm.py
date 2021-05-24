# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.pypower.idx_bus import BUS_I, VMAX, VMIN, BUS_TYPE, REF
import numpy as np
import pytest
import copy
from numpy import array
from pandapower.converter.pypower import from_ppc
import pandapower as pp
from pandapower.pd2ppc import _pd2ppc

try:
    from julia.core import UnsupportedPythonError
except ImportError:
    UnsupportedPythonError = Exception
try:
    from julia import Main

    julia_installed = True
except (ImportError, RuntimeError, UnsupportedPythonError) as e:
    julia_installed = False
    print(e)

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def case14_pm_file():
    
    mpc = {"branch": array([[ 0.0000e+00,  1.0000e+00,  1.9380e-04,  5.9170e-04,  5.2800e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 0.0000e+00,  4.0000e+00,  5.4030e-04,  2.2304e-03,  4.9200e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 1.0000e+00,  2.0000e+00,  4.6990e-04,  1.9797e-03,  4.3800e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 1.0000e+00,  3.0000e+00,  5.8110e-04,  1.7632e-03,  3.4000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 1.0000e+00,  4.0000e+00,  5.6950e-04,  1.7388e-03,  3.4600e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 2.0000e+00,  3.0000e+00,  6.7010e-04,  1.7103e-03,  1.2800e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 3.0000e+00,  4.0000e+00,  1.3350e-04,  4.2110e-04,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 5.0000e+00,  1.0000e+01,  9.4980e-04,  1.9890e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 5.0000e+00,  1.1000e+01,  1.2291e-03,  2.5581e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 5.0000e+00,  1.2000e+01,  6.6150e-04,  1.3027e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 8.0000e+00,  9.0000e+00,  3.1810e-04,  8.4500e-04,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 8.0000e+00,  1.3000e+01,  1.2711e-03,  2.7038e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 9.0000e+00,  1.0000e+01,  8.2050e-04,  1.9207e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 1.1000e+01,  1.2000e+01,  2.2092e-03,  1.9988e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 1.2000e+01,  1.3000e+01,  1.7093e-03,  3.4802e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 3.0000e+00,  6.0000e+00,  0.0000e+00,  2.0912e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  9.7800e-01,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 3.0000e+00,  8.0000e+00,  0.0000e+00,  5.5618e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  9.6900e-01,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 4.0000e+00,  5.0000e+00,  0.0000e+00,  2.5202e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  9.3200e-01,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 6.0000e+00,  7.0000e+00,  0.0000e+00,  1.7615e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00],
       [ 6.0000e+00,  8.0000e+00,  0.0000e+00,  1.1001e-03,  0.0000e+00,
         9.9000e+03,  2.5000e+02,  2.5000e+02,  1.0000e+00,  0.0000e+00,
         1.0000e+00, -3.6000e+02,  3.6000e+02,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
         0.0000e+00,  0.0000e+00,  0.0000e+00]]), 
       "bus": array([[ 0. , 3. , 0. , 0. , 0. , 0. , 1. , 1.06 ,
                      0. , 135. , 1. , 1.06 , 1.06 , 0. , 0. ],
                     [ 1. , 2. , 21.7 , 12.7 , 0. , 0. , 1. , 1.045 ,
                      -3.95523766, 135. , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 2. , 2. , 94.2 , 19. , 0. , 0. , 1. , 1.01 ,
                      -10.40760708, 135. , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 3. , 1. , 47.8 , -3.9 , 0. , 0. , 1. , 1.01744145,
                      -8.21095265, 135. , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 4. , 1. , 7.6 , 1.6 , 0. , 0. , 1. , 1.01883931,
                      -7.01636257, 135. , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 5. , 2. , 11.2 , 7.5 , 0. , 0. , 1. , 1.07 ,
                      -11.61937248, 0.208 , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 6. , 1. , -21. , 0. , 0. , 0. , 1. , 1.04887992,
                      -9.4740081 , 14. , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 7. , 2. , 0. , 0. , 0. , 0. , 1. , 1.09 ,
                          -9.47400522, 12. , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 8. , 1. , 29.5 , 16.6 , 0. , 19. , 1. , 1.04712183,
                      -11.33959824, 0.208 , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 9. , 1. , 1. , 5.8 , 0. , 0. , 1. , 1.04418246,
                      -11.38690514, 0.208 , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 10. , 1. , 3.5 , 1.8 , 0. , 0. , 1. , 1.0487667 ,
                      -11.62833811, 0.208 , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 11. , 1. , 6.1 , 1.6 , 0. , 0. , 1. , 1.04497585,
                      -12.41956695, 0.208 , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 12. , 1. , 13.5 , 5.8 , 0. , 0. , 1. , 1.04055949,
                      -12.42974608, 0.208 , 1. , 1.06 , 0.94 , 0. , 0. ],
                     [ 13. , 1. , 14.9 , 5. , 0. , 0. , 1. , 1.02613375,
                      -12.83129875, 0.208 , 1. , 1.06 , 0.94 , 0. , 0. ]]), 
       "gen": array([[ 0.000e+00,  0.000e+00,  0.000e+00,  1.000e+01, -1.000e-10,
         1.060e+00,  1.000e+00,  1.000e+00,  3.324e+02, -1.000e-10,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00],
       [ 1.000e+00,  4.000e+01,  0.000e+00,  5.000e+01, -4.000e+01,
         1.045e+00,     np.nan,  1.000e+00,  1.400e+02, -1.000e-10,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00],
       [ 2.000e+00,  0.000e+00,  0.000e+00,  4.000e+01, -1.000e-10,
         1.010e+00,     np.nan,  1.000e+00,  1.000e+02, -1.000e-10,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00],
       [ 5.000e+00,  0.000e+00,  0.000e+00,  2.400e+01, -6.000e+00,
         1.070e+00,     np.nan,  1.000e+00,  1.000e+02, -1.000e-10,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00],
       [ 7.000e+00,  0.000e+00,  0.000e+00,  2.400e+01, -6.000e+00,
         1.090e+00,     np.nan,  1.000e+00,  1.000e+02, -1.000e-10,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,
         0.000e+00]]), 
       "gencost": array([[2.00000e+00, 0.00000e+00, 0.00000e+00, 3.00000e+00, 4.30293e-02,
        2.00000e+01, 0.00000e+00],
       [2.00000e+00, 0.00000e+00, 0.00000e+00, 3.00000e+00, 2.50000e-01,
        2.00000e+01, 0.00000e+00],
       [2.00000e+00, 0.00000e+00, 0.00000e+00, 3.00000e+00, 1.00000e-02,
        4.00000e+01, 0.00000e+00],
       [2.00000e+00, 0.00000e+00, 0.00000e+00, 3.00000e+00, 1.00000e-02,
        4.00000e+01, 0.00000e+00],
       [2.00000e+00, 0.00000e+00, 0.00000e+00, 3.00000e+00, 1.00000e-02,
        4.00000e+01, 0.00000e+00]]), 
       "version": 2, "baseMVA": 1.0}

    net = from_ppc(mpc, f_hz=50)
    
    return net
    

@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_case5_pm_pd2ppc():

    # load net
    net = case14_pm_file()
    # run pd2ppc with ext_grid controllable = False
    pp.runpp(net)
    assert "controllable" not in net.ext_grid
    net["_options"]["mode"] = "opf"
    ppc = _pd2ppc(net)
    # check which one is the ref bus in ppc
    ref_idx = int(ppc[0]["bus"][:, BUS_I][ppc[0]["bus"][:, BUS_TYPE] == REF])
    vmax = ppc[0]["bus"][ref_idx, VMAX]
    vmin = ppc[0]["bus"][ref_idx, VMIN]

    assert net.ext_grid.vm_pu[0] == vmin
    assert net.ext_grid.vm_pu[0] == vmax

    # run pd2ppc with ext_grd controllable = True
    net.ext_grid["controllable"] = True
    ppc = _pd2ppc(net)
    ref_idx = int(ppc[0]["bus"][:, BUS_I][ppc[0]["bus"][:, BUS_TYPE] == REF])
    vmax = ppc[0]["bus"][ref_idx, VMAX]
    vmin = ppc[0]["bus"][ref_idx, VMIN]

    assert net.bus.min_vm_pu[net.ext_grid.bus].values[0] == vmin
    assert net.bus.max_vm_pu[net.ext_grid.bus].values[0] == vmax

    assert net.ext_grid["in_service"].values.dtype == bool
    assert net.ext_grid["bus"].values.dtype == "uint32"
    pp.create_ext_grid(net, bus=4, vm_pu=net.res_bus.vm_pu.loc[4])

    assert net.ext_grid["bus"].values.dtype == "uint32"
    assert net.ext_grid["in_service"].values.dtype == bool

    ppc = _pd2ppc(net)
    ref_idx = int(ppc[0]["bus"][:, BUS_I][ppc[0]["bus"][:, BUS_TYPE] == REF])

    bus2 = net._pd2ppc_lookups["bus"][net.ext_grid.bus[1]]
    vmax0 = ppc[0]["bus"][ref_idx, VMAX]
    vmin0 = ppc[0]["bus"][ref_idx, VMIN]

    vmax1 = ppc[0]["bus"][bus2, VMAX]
    vmin1 = ppc[0]["bus"][bus2, VMIN]

    assert net.bus.min_vm_pu[net.ext_grid.bus].values[0] == vmin0
    assert net.bus.max_vm_pu[net.ext_grid.bus].values[0] == vmax0

    assert net.ext_grid.vm_pu.values[1] == vmin1
    assert net.ext_grid.vm_pu.values[1] == vmax1


def test_opf_ext_grid_controllable():
    # load net
    net = case14_pm_file()
    net_old = copy.deepcopy(net)
    net_new = copy.deepcopy(net)
    # run pd2ppc with ext_grid controllable = False
    pp.runopp(net_old)
    net_new.ext_grid["controllable"] = True
    pp.runopp(net_new)
    assert np.isclose(net_new.res_bus.vm_pu[net.ext_grid.bus[0]], 1.0599999600072285)
    assert np.isclose(net_old.res_bus.vm_pu[net.ext_grid.bus[0]], 1.060000000015124)

    assert np.isclose(net_old.res_cost, 6925.04868)
    assert np.isclose(net_new.res_cost, 6925.0486)


def test_opf_ext_grid_controllable():
    # load net
    net = case14_pm_file()
    # run pd2ppc with ext_grid controllable = False
    pp.create_ext_grid(net, bus=1, controllable=True)
    pp.runopp(net)
    assert np.isclose(net.res_bus.vm_pu[net.ext_grid.bus[0]], 1.0599999999755336)


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_opf_ext_grid_controllable_pm():
    # load net
    net = case14_pm_file()
    net_old = copy.deepcopy(net)
    net_new = copy.deepcopy(net)
    # run pd2ppc with ext_grid controllable = False
    pp.runpm_ac_opf(net_old, calculate_voltage_angles=True, correct_pm_network_data=False, opf_flow_lim="I")
    net_new.ext_grid["controllable"] = True
    pp.runpm_ac_opf(net_new, calculate_voltage_angles=True, correct_pm_network_data=False,
                    delete_buffer_file=False, pm_file_path="buffer_file.json", opf_flow_lim="I")

    assert np.isclose(net_new.res_bus.vm_pu[net.ext_grid.bus[0]], 1.0599999600072285)
    assert np.isclose(net_old.res_bus.vm_pu[net.ext_grid.bus[0]], 1.060000000015124)

    assert np.isclose(net_old.res_cost, 6925.04868)
    assert np.isclose(net_new.res_cost, 6925.04868)


if __name__ == "__main__":
    test_opf_ext_grid_controllable()
    # pytest.main([__file__, "-xs"])
