# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import pytest

import pandapower as pp
import pandapower.shortcircuit as sc


@pytest.fixture
def meshed_grid():
    net = pp.from_json(os.path.join(pp.pp_dir, "test", "shortcircuit", "sc_test_meshed_grid.json"))
    bid = pp.create_bus(net, vn_kv=10.)
    pp.create_switch(net, net.ext_grid.bus.iloc[0], bid, et="b")
    net.ext_grid.bus.iloc[0] = bid
    pp.create_bus(net, vn_kv=0.4, in_service=False)
    return net

def test_max_10_meshed_grid(meshed_grid):
    net = meshed_grid
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent= 10., kappa_method="B")
    assert (abs(net.res_bus_sc.ikss_ka.at[0] - 5.773503) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[1] - 14.82619) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[2] - 4.606440) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[3] - 4.068637) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[4] - 13.61509) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[5] - 2.812111) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[6] - 1.212288) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[7] - 1.525655) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[8] - 1.781087) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[9] - 1.568337) <1e-5)

    assert (abs(net.res_bus_sc.ip_ka.at[0] - 14.256050) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[1] - 33.751300) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[2] - 6.759302) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[3] - 6.359403) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[4] - 26.49241) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[5] - 4.726619) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[6] - 2.015958) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[7] - 2.538654) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[8] - 2.576375) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[9] - 2.608065) <1e-5)

    assert (abs(net.res_bus_sc.ith_ka.at[0] - 5.871191) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[1] - 14.97527) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[2] - 4.613454) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[3] - 4.077662) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[4] - 13.68449) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[5] - 2.820525) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[6] - 1.215770) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[7] - 1.530048) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[8] - 1.783442) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[9] - 1.572843) <1e-5)


def test_max_6_meshed_grid(meshed_grid):
    net = meshed_grid
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent = 6., kappa_method="B")
    assert (abs(net.res_bus_sc.ikss_ka.at[0] - 5.773503) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[1] - 14.75419) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[2] - 4.437882) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[3] - 4.068637) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[4] - 13.53425) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[5] - 2.701411) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[6] - 1.159945) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[7] - 1.460757) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[8] - 1.705172) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[9] - 1.501673) <1e-5)

    assert (abs(net.res_bus_sc.ip_ka.at[0] - 14.25605) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[1] - 33.59996) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[2] - 6.50406) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[3] - 6.359403) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[4] - 26.28476) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[5] - 4.537759) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[6] - 1.928734) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[7] - 2.430331) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[8] - 2.466185) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[9] - 2.496901) <1e-5)

    assert (abs(net.res_bus_sc.ith_ka.at[0] - 5.871191) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[1] - 14.90284) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[2] - 4.44457) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[3] - 4.077662) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[4] - 13.60275) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[5] - 2.709475) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[6] - 1.163276) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[7] - 1.464961) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[8] - 1.707423) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[9] - 1.505985) <1e-5)


def test_min_10_meshed_grid(meshed_grid):
    net = meshed_grid
    sc.calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent= 10., kappa_method="B")
    assert (abs(net.res_bus_sc.ikss_ka.at[0] - 2.309401) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[1] - 11.3267) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[2] - 2.879343) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[3] - 1.884323) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[4] - 10.40083) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[5] - 1.693922) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[6] - 0.7107017) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[7] - 0.9000445) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[8] - 1.055881) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[9] - 0.928488) <1e-5)

    assert (abs(net.res_bus_sc.ip_ka.at[0] - 5.702418) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[1] - 26.01655) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[2] - 4.166047) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[3] - 3.124163) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[4] - 20.04053) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[5] - 2.813883) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[6] - 1.179085) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[7] - 1.493293) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[8] - 1.523338) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[9] - 1.540432) <1e-5)

    assert (abs(net.res_bus_sc.ith_ka.at[0] - 2.348476) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[1] - 11.44622) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[2] - 2.883161) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[3] - 1.889675) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[4] - 10.45195) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[5] - 1.698768) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[6] - 0.712725) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[7] - 0.9026074) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[8] - 1.057233) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[9] - 0.9311316) <1e-5)

def test_min_6_meshed_grid(meshed_grid):
    net = meshed_grid
    sc.calc_sc(net, case='min', ip=True, ith=True, lv_tol_percent = 6., kappa_method="B")
    assert (abs(net.res_bus_sc.ikss_ka.at[0] - 2.309401) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[1] - 11.75072) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[2] - 2.895465) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[3] - 1.884323) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[4] - 10.77961) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[5] - 1.700202) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[6] - 0.7116519) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[7] - 0.9016006) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[8] - 1.0576) <1e-5)
    assert (abs(net.res_bus_sc.ikss_ka.at[9] - 0.9301236) <1e-5)

    assert (abs(net.res_bus_sc.ip_ka.at[0] - 5.702418) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[1] - 27.00861) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[2] - 4.18812) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[3] - 3.124163) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[4] - 20.72881) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[5] - 2.824028) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[6] - 1.180654) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[7] - 1.495858) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[8] - 1.525799) <1e-5)
    assert (abs(net.res_bus_sc.ip_ka.at[9] - 1.543131) <1e-5)

    assert (abs(net.res_bus_sc.ith_ka.at[0] - 2.348476) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[1] - 11.87518) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[2] - 2.899291) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[3] - 1.889675) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[4] - 10.8322) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[5] - 1.705064) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[6] - 0.7136779) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[7] - 0.9041679) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[8] - 1.058954) <1e-5)
    assert (abs(net.res_bus_sc.ith_ka.at[9] - 0.9327717) <1e-5)

if __name__ == '__main__':
    pytest.main(['-xs'])
