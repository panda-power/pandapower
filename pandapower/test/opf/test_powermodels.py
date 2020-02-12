# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import json
import os
from functools import partial

import numpy as np
import pandas as pd
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.converter.powermodels.from_pm import read_pm_results_to_net
from pandapower.pd2ppc import _pd2ppc
from pandapower.test.consistency_checks import consistency_checks
from pandapower.test.toolbox import add_grid_connection, create_test_line
from pandapower.converter import convert_pp_to_pm
from pandapower.test.opf.test_basic import simple_opf_test_net

try:
    from julia import Main

    julia_installed = True
except (ImportError, RuntimeError) as e:
    julia_installed = False
    print(e)


@pytest.fixture
def net_3w_trafo_opf():
    net = pp.create_empty_network()

    # create buses
    bus1 = pp.create_bus(net, vn_kv=220.)
    bus2 = pp.create_bus(net, vn_kv=110.)
    bus3 = pp.create_bus(net, vn_kv=110.)
    bus4 = pp.create_bus(net, vn_kv=110.)
    bus5 = pp.create_bus(net, vn_kv=110.)

    pp.create_bus(net, vn_kv=110., in_service=False)

    # create 220/110 kV transformer
    pp.create_transformer3w_from_parameters(net, bus1, bus2, bus5, vn_hv_kv=220, vn_mv_kv=110,
                                            vn_lv_kv=110, vk_hv_percent=10., vk_mv_percent=10.,
                                            vk_lv_percent=10., vkr_hv_percent=0.5,
                                            vkr_mv_percent=0.5, vkr_lv_percent=0.5, pfe_kw=100,
                                            i0_percent=0.1, shift_mv_degree=0, shift_lv_degree=0,
                                            sn_hv_mva=100, sn_mv_mva=50, sn_lv_mva=50)

    # create 110 kV lines
    pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus3, bus4, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus2, length_km=40., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus5, length_km=30., std_type='149-AL1/24-ST1A 110.0')

    # create loads
    pp.create_load(net, bus2, p_mw=60, controllable=False)
    pp.create_load(net, bus3, p_mw=70, controllable=False)
    pp.create_sgen(net, bus3, p_mw=10, controllable=False)

    # create generators
    pp.create_ext_grid(net, bus1, min_p_mw=0, max_p_mw=1000, max_q_mvar=0.01, min_q_mvar=0)
    pp.create_gen(net, bus3, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01)
    pp.create_gen(net, bus4, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01)
    net.gen["controllable"] = False
    return net


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_compare_pwl_and_poly(net_3w_trafo_opf):
    net = net_3w_trafo_opf
    net.ext_grid.loc[:, "min_p_mw"] = -999.
    net.ext_grid.loc[:, "max_p_mw"] = 999.
    net.ext_grid.loc[:, "max_q_mvar"] = 999.
    net.ext_grid.loc[:, "min_q_mvar"] = -999.
    pp.create_pwl_cost(net, 0, 'ext_grid', [[0, 1, 1]])
    pp.create_pwl_cost(net, 0, 'gen', [[0, 30, 3], [30, 80, 3]])
    pp.create_pwl_cost(net, 1, 'gen', [[0, 80, 2]])
    net.bus.loc[:, "max_vm_pu"] = 1.1
    net.bus.loc[:, "min_vm_pu"] = .9
    pp.runpm_ac_opf(net)
    consistency_checks(net)

    p_gen = net.res_gen.p_mw.values
    q_gen = net.res_gen.q_mvar.values
    vm_bus = net.res_bus.vm_pu.values
    va_bus = net.res_bus.va_degree.values

    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)

    pp.create_poly_cost(net, 0, 'ext_grid', cp1_eur_per_mw=1)
    pp.create_poly_cost(net, 0, 'gen', cp1_eur_per_mw=3)
    pp.create_poly_cost(net, 1, 'gen', cp1_eur_per_mw=2)

    # pp.runopp(net)
    pp.runpm_ac_opf(net, correct_pm_network_data=False)
    consistency_checks(net)

    np.allclose(p_gen, net.res_gen.p_mw.values)
    np.allclose(q_gen, net.res_gen.q_mvar.values)
    np.allclose(vm_bus, net.res_bus.vm_pu.values)
    np.allclose(va_bus, net.res_bus.va_degree.values)

    # pp.rundcopp(net)
    pp.runpm_dc_opf(net, correct_pm_network_data=False)
    consistency_checks(net, test_q=False)

    np.allclose(p_gen, net.res_gen.p_mw.values)
    np.allclose(va_bus, net.res_bus.va_degree.values)


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_pwl():
    net = pp.create_empty_network()

    # create buses
    bus1 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)
    bus2 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)
    bus3 = pp.create_bus(net, vn_kv=110., min_vm_pu=0.9, max_vm_pu=1.1)

    # create 110 kV lines
    pp.create_line(net, bus1, bus2, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus2, bus3, length_km=50., std_type='149-AL1/24-ST1A 110.0')

    # create loads
    pp.create_load(net, bus2, p_mw=80, controllable=False)

    # create generators
    g1 = pp.create_gen(net, bus1, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01, slack=True)
    g2 = pp.create_gen(net, bus3, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01)
    #    net.gen["controllable"] = False

    pp.create_pwl_cost(net, g1, 'gen', [[0, 2, 2], [2, 80, 5]])
    pp.create_pwl_cost(net, g2, 'gen', [[0, 2, 2], [2, 80, 5]])

    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.iloc[0], net.res_gen.p_mw.iloc[1])
    assert np.isclose(net.res_gen.q_mvar.iloc[0], net.res_gen.q_mvar.iloc[1])

    net.pwl_cost.drop(net.pwl_cost.index, inplace=True)
    g3 = pp.create_gen(net, bus1, p_mw=80, min_p_mw=0, max_p_mw=80, vm_pu=1.01)

    pp.create_pwl_cost(net, g1, 'gen', [[0, 2, 1.], [2, 80, 8.]])
    pp.create_pwl_cost(net, g2, 'gen', [[0, 3, 2.], [3, 80, 14]])
    pp.create_pwl_cost(net, g3, 'gen', [[0, 1, 3.], [1, 80, 10.]])

    net.load.p_mw = 1
    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g2], 0)
    assert np.isclose(net.res_gen.p_mw.at[g3], 0)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1])

    net.load.p_mw = 3
    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g3], 0)
    assert np.isclose(net.res_gen.p_mw.at[g1], 2)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] + net.res_gen.p_mw.at[g2] * 2)

    net.load.p_mw = 5
    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g1], 2)
    assert np.isclose(net.res_gen.p_mw.at[g2], 3)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] + net.res_gen.p_mw.at[g2] * 2 +
                      net.res_gen.p_mw.at[g3] * 3)


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_without_ext_grid():
    net = pp.create_empty_network()

    min_vm_pu = 0.95
    max_vm_pu = 1.05

    # create buses
    bus1 = pp.create_bus(net, vn_kv=220., geodata=(5, 9))
    bus2 = pp.create_bus(net, vn_kv=110., geodata=(6, 10), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus3 = pp.create_bus(net, vn_kv=110., geodata=(10, 9), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus4 = pp.create_bus(net, vn_kv=110., geodata=(8, 8), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus5 = pp.create_bus(net, vn_kv=110., geodata=(6, 8), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)

    # create 220/110/110 kV 3W-transformer
    pp.create_transformer3w_from_parameters(net, bus1, bus2, bus5, vn_hv_kv=220, vn_mv_kv=110,
                                            vn_lv_kv=110, vk_hv_percent=10., vk_mv_percent=10.,
                                            vk_lv_percent=10., vkr_hv_percent=0.5,
                                            vkr_mv_percent=0.5, vkr_lv_percent=0.5, pfe_kw=100,
                                            i0_percent=0.1, shift_mv_degree=0, shift_lv_degree=0,
                                            sn_hv_mva=100, sn_mv_mva=50, sn_lv_mva=50)

    # create 110 kV lines
    pp.create_line(net, bus2, bus3, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus3, bus4, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus2, length_km=40., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus4, bus5, length_km=30., std_type='149-AL1/24-ST1A 110.0')

    # create loads
    pp.create_load(net, bus2, p_mw=60, controllable=False)
    pp.create_load(net, bus3, p_mw=70, controllable=False)
    pp.create_load(net, bus4, p_mw=10, controllable=False)

    # create generators
    g1 = pp.create_gen(net, bus1, p_mw=40, min_p_mw=0, min_q_mvar=-20, max_q_mvar=20, slack=True, min_vm_pu=min_vm_pu,
                       max_vm_pu=max_vm_pu)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_mw=1000)

    g2 = pp.create_gen(net, bus3, p_mw=40, min_p_mw=0, min_q_mvar=-20, max_q_mvar=20, vm_pu=1.01,
                       min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu, max_p_mw=40.)
    pp.create_poly_cost(net, g2, 'gen', cp1_eur_per_mw=2000)

    g3 = pp.create_gen(net, bus4, p_mw=0.050, min_p_mw=0, min_q_mvar=-20, max_q_mvar=20, vm_pu=1.01,
                       min_vm_pu=min_vm_pu,
                       max_vm_pu=max_vm_pu, max_p_mw=0.05)
    pp.create_poly_cost(net, g3, 'gen', cp1_eur_per_mw=3000)

    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert np.isclose(net.res_gen.p_mw.at[g2], 0, atol=1e-5, rtol=1e-5)
    assert np.isclose(net.res_gen.p_mw.at[g3], 0, atol=1e-5, rtol=1e-5)
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] * 1e3)
    net.trafo3w["max_loading_percent"] = 150.

    pp.runpm_ac_opf(net)
    consistency_checks(net, rtol=1e-3)
    assert 149. < net.res_trafo3w.loading_percent.values[0] < 150.01
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] * 1e3 + net.res_gen.p_mw.at[g2] * 2e3)

    pp.runpm_dc_opf(net)
    consistency_checks(net, rtol=1e-3, test_q=False)
    assert 149. < net.res_trafo3w.loading_percent.values[0] < 150.01
    assert np.isclose(net.res_cost, net.res_gen.p_mw.at[g1] * 1e3 + net.res_gen.p_mw.at[g2] * 2e3)


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_multiple_ext_grids():
    net = pp.create_empty_network()
    # generate three ext grids
    b11, b12, l11 = add_grid_connection(net, vn_kv=110.)
    b21, b22, l21 = add_grid_connection(net, vn_kv=110.)
    b31, b32, l31 = add_grid_connection(net, vn_kv=110.)
    # connect them
    l12_22 = create_test_line(net, b12, b22)
    l22_32 = create_test_line(net, b22, b32)

    # create load and sgen to optimize
    pp.create_load(net, b12, p_mw=60)

    g3 = pp.create_sgen(net, b12, p_mw=50, min_p_mw=20, max_p_mw=200, controllable=True)
    pp.create_poly_cost(net, g3, 'sgen', cp1_eur_per_mw=10.)
    # set positive costs for ext_grid -> minimize ext_grid usage
    ext_grids = net.ext_grid.index
    net["ext_grid"].loc[0, "vm_pu"] = .99
    net["ext_grid"].loc[1, "vm_pu"] = 1.0
    net["ext_grid"].loc[2, "vm_pu"] = 1.01
    for idx in ext_grids:
        # eg = net["ext_grid"].loc[idx]
        pp.create_poly_cost(net, idx, 'ext_grid', cp1_eur_per_mw=10.)

    pp.runpm_ac_opf(net)
    assert np.allclose(net.res_sgen.loc[0, "p_mw"], 60.)


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_voltage_angles():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=20.)
    b4 = pp.create_bus(net, vn_kv=10.)
    b5 = pp.create_bus(net, vn_kv=10., in_service=False)
    tidx = pp.create_transformer3w(
        net, b2, b3, b4, std_type='63/25/38 MVA 110/20/10 kV', max_loading_percent=120)
    pp.create_load(net, b3, p_mw=5, controllable=False)
    load_id = pp.create_load(net, b4, p_mw=5, controllable=True, max_p_mw=25, min_p_mw=0, min_q_mvar=-1e-6,
                             max_q_mvar=1e-6)
    pp.create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=1)
    pp.create_poly_cost(net, load_id, "load", cp1_eur_per_mw=1000)
    net.trafo3w.shift_lv_degree.at[tidx] = 10
    net.trafo3w.shift_mv_degree.at[tidx] = 30
    net.bus.loc[:, "max_vm_pu"] = 1.1
    net.bus.loc[:, "min_vm_pu"] = .9

    custom_file = os.path.join(os.path.abspath(os.path.dirname(pp.test.__file__)),
                               "test_files", "run_powermodels_custom.jl")

    # load is zero since costs are high. PF results should be the same as OPF
    net.load.loc[1, "p_mw"] = 0.
    pp.runpp(net, calculate_voltage_angles=True)
    va_degree = net.res_bus.loc[:, "va_degree"].values
    vm_pu = net.res_bus.loc[:, "vm_pu"].values
    loading3w = net.res_trafo3w.loc[:, "loading_percent"].values

    for run in [pp.runpm_ac_opf, partial(pp.runpm, julia_file=custom_file)]:
        run(net, calculate_voltage_angles=True)
        consistency_checks(net)

        assert 30. < (net.res_bus.va_degree.at[b1] - net.res_bus.va_degree.at[b3]) % 360 < 32.
        assert 10. < (net.res_bus.va_degree.at[b1] - net.res_bus.va_degree.at[b4]) % 360 < 11.
        assert np.isnan(net.res_bus.va_degree.at[b5])
        assert np.allclose(net.res_bus.va_degree.values, va_degree, atol=1e-6, rtol=1e-6, equal_nan=True)
        assert np.allclose(net.res_bus.vm_pu.values, vm_pu, atol=1e-6, rtol=1e-6, equal_nan=True)
        assert np.allclose(net.res_trafo3w.loading_percent, loading3w, atol=1e-2, rtol=1e-2, equal_nan=True)


def init_ne_line(net, new_line_index, construction_costs=None):
    """
    init function for new line dataframe, which specifies the possible new lines being built by power models opt

    Parameters
    ----------
    net - pp net
    new_line_index (list) - indices of new lines. These are copied to the new dataframe net["ne_line"] from net["line"]
    construction_costs (list, 0.) - costs of newly constructed lines

    Returns
    -------

    """
    # init dataframe
    net["ne_line"] = net["line"].loc[new_line_index, :]
    # add costs, if None -> init with zeros
    construction_costs = np.zeros(
        len(new_line_index)) if construction_costs is None else construction_costs
    net["ne_line"].loc[new_line_index, "construction_cost"] = construction_costs
    # set in service, but only in ne line dataframe
    net["ne_line"].loc[new_line_index, "in_service"] = True
    # init res_ne_line to save built status afterwards
    net["res_ne_line"] = pd.DataFrame(data=0, index=new_line_index, columns=["built"], dtype=int)


def tnep_grid():
    net = pp.create_empty_network()

    min_vm_pu = 0.95
    max_vm_pu = 1.05

    # create buses
    bus1 = pp.create_bus(net, vn_kv=110., geodata=(5, 9), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus2 = pp.create_bus(net, vn_kv=110., geodata=(6, 10), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus3 = pp.create_bus(net, vn_kv=110., geodata=(10, 9), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)
    bus4 = pp.create_bus(net, vn_kv=110., geodata=(8, 8), min_vm_pu=min_vm_pu, max_vm_pu=max_vm_pu)

    # create 110 kV lines
    pp.create_line(net, bus1, bus2, length_km=70., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus1, bus3, length_km=50., std_type='149-AL1/24-ST1A 110.0')
    pp.create_line(net, bus1, bus4, length_km=100., std_type='149-AL1/24-ST1A 110.0')

    # create loads
    pp.create_load(net, bus2, p_mw=60)
    pp.create_load(net, bus3, p_mw=70)
    pp.create_load(net, bus4, p_mw=50)

    # create generators
    g1 = pp.create_gen(net, bus1, p_mw=9.513270, min_p_mw=0, max_p_mw=200, vm_pu=1.01, slack=True)
    pp.create_poly_cost(net, g1, 'gen', cp1_eur_per_mw=1)

    g2 = pp.create_gen(net, bus2, p_mw=78.403291, min_p_mw=0, max_p_mw=200, vm_pu=1.01)
    pp.create_poly_cost(net, g2, 'gen', cp1_eur_per_mw=3)

    g3 = pp.create_gen(net, bus3, p_mw=92.375601, min_p_mw=0, max_p_mw=200, vm_pu=1.01)
    pp.create_poly_cost(net, g3, 'gen', cp1_eur_per_mw=3)

    net.line["max_loading_percent"] = 20

    # possible new lines (set out of service in line DataFrame)
    l1 = pp.create_line(net, bus1, bus4, 10., std_type="305-AL1/39-ST1A 110.0", name="new_line1",
                        max_loading_percent=20., in_service=False)
    l2 = pp.create_line(net, bus2, bus4, 20., std_type="149-AL1/24-ST1A 110.0", name="new_line2",
                        max_loading_percent=20., in_service=False)
    l3 = pp.create_line(net, bus3, bus4, 30., std_type='149-AL1/24-ST1A 110.0', name="new_line3",
                        max_loading_percent=20., in_service=False)
    l4 = pp.create_line(net, bus3, bus4, 40., std_type='149-AL1/24-ST1A 110.0', name="new_line4",
                        max_loading_percent=20., in_service=False)

    new_line_index = [l1, l2, l3, l4]
    construction_costs = [10., 20., 30., 45.]
    # create new line dataframe
    init_ne_line(net, new_line_index, construction_costs)

    return net


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_pm_tnep():
    net = tnep_grid()
    # check if max line loading percent is violated (should be)
    pp.runpp(net)
    assert np.any(net["res_line"].loc[:, "loading_percent"] >
                  net["line"].loc[:, "max_loading_percent"])

    # run power models tnep optimization
    pp.runpm_tnep(net, pm_model="ACPPowerModel")
    # set lines to be built in service
    lines_to_built = net["res_ne_line"].loc[net["res_ne_line"].loc[:, "built"], "built"].index
    net["line"].loc[lines_to_built, "in_service"] = True
    # run a power flow calculation again and check if max_loading percent is still violated
    pp.runpp(net)
    # check max line loading results
    assert not np.any(net["res_line"].loc[:, "loading_percent"] >
                      net["line"].loc[:, "max_loading_percent"])


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
@pytest.mark.xfail(reason="experimental. SOC is wrong since update")
def test_storage_opt():
    net = nw.case5()
    pp.create_storage(net, 2, p_mw=1., max_e_mwh=.2, soc_percent=100., q_mvar=1.)
    pp.create_storage(net, 3, p_mw=1., max_e_mwh=.3, soc_percent=100., q_mvar=1.)

    # optimize for 24 time steps. At the end the SOC is 0%
    storage_results = pp.runpm_storage_opf(net, n_timesteps=24)
    assert np.allclose(storage_results[0].loc[22, "soc_mwh"], 0.004960, rtol=1e-4, atol=1e-4)
    assert np.allclose(storage_results[0].loc[23, "soc_mwh"], 0.)
    assert np.allclose(storage_results[1].loc[22, "soc_percent"], 29.998074, rtol=1e-4, atol=1e-4)
    assert np.allclose(storage_results[1].loc[23, "soc_mwh"], 0.)


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_ots_opt():
    net = nw.case5()
    branch_status = net["line"].loc[:, "in_service"].values
    assert np.array_equal(np.array([1, 1, 1, 1, 1, 1]).astype(bool), branch_status.astype(bool))
    pp.runpm_ots(net)
    branch_status = net["res_line"].loc[:, "in_service"].values
    pp.runpp(net)
    net.line.loc[:, "in_service"] = branch_status.astype(bool)
    pp.runpp(net)
    try:
        assert np.array_equal(np.array([1, 1, 1, 0, 1, 0]).astype(bool), branch_status.astype(bool))
    except AssertionError:
        assert np.array_equal(np.array([0, 1, 1, 1, 1, 0]).astype(bool), branch_status.astype(bool))


def assert_pf(net, dc=False):
    custom_file = os.path.join(os.path.abspath(os.path.dirname(pp.__file__)),
                               "opf", "run_powermodels_powerflow.jl")
    if dc:
        # see https://github.com/lanl-ansi/PowerModels.jl/issues/612 for details
        pp.runpm(net, julia_file=custom_file, pm_model="DCMPPowerModel")
    else:
        pp.runpm(net, julia_file=custom_file, pm_model="ACPPowerModel")

    va_pm = copy.copy(net.res_bus.va_degree)
    vm_pm = copy.copy(net.res_bus.vm_pu)

    if dc:
        pp.rundcpp(net, calculate_voltage_angles=True)
    else:
        pp.runpp(net, calculate_voltage_angles=True)
    va_pp = copy.copy(net.res_bus.va_degree)
    vm_pp = copy.copy(net.res_bus.vm_pu)

    assert np.allclose(va_pm, va_pp)
    if not dc:
        assert np.allclose(vm_pm, vm_pp)


@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_pm_ac_powerflow_simple():
    net = nw.simple_four_bus_system()
    net.trafo.loc[0, "shift_degree"] = 0.
    assert_pf(net)


@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
@pytest.mark.xfail(reason="DCMPPowerModel not released yet")
def test_pm_dc_powerflow_simple():
    net = nw.simple_four_bus_system()
    net.trafo.loc[0, "shift_degree"] = 0.
    assert_pf(net, dc=True)


@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_pm_ac_powerflow_shunt():
    net = nw.simple_four_bus_system()
    pp.create_shunt(net, 2, q_mvar=-0.5)
    net.trafo.loc[0, "shift_degree"] = 0.
    assert_pf(net)


@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
@pytest.mark.xfail(reason="DCMPPowerModel not released yet")
def test_pm_dc_powerflow_shunt():
    net = nw.simple_four_bus_system()
    pp.create_shunt(net, 2, q_mvar=-0.5)
    net.trafo.loc[0, "shift_degree"] = 0.
    assert_pf(net, dc=True)


@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_pm_ac_powerflow_tap():
    net = nw.simple_four_bus_system()
    net.trafo.loc[0, "shift_degree"] = 30.
    net.trafo.loc[0, "tap_pos"] = -2.
    assert_pf(net)


@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
@pytest.mark.xfail(reason="DCMPPowerModel not released yet")
def test_pm_dc_powerflow_tap():
    net = nw.simple_four_bus_system()
    net.trafo.loc[0, "shift_degree"] = 0.
    assert_pf(net, dc=True)
    net.trafo.loc[0, "shift_degree"] = 30.
    net.trafo.loc[0, "tap_pos"] = -2.
    assert_pf(net, dc=True)


def test_pp_to_pm_conversion(net_3w_trafo_opf):
    # tests if the conversion to power models works
    net = net_3w_trafo_opf
    pm = convert_pp_to_pm(net)


def test_pm_to_pp_conversion(simple_opf_test_net):
    # this tests checks if the runopp results are the same as the ones from powermodels.
    # Results are read from a result file containing the simple_opf_test_net

    net = simple_opf_test_net
    pp.create_poly_cost(net, 0, "gen", cp1_eur_per_mw=100)

    # get pandapower opf results
    pp.runopp(net, delta=1e-13)
    va_degree = copy.deepcopy(net.res_bus.va_degree)
    vm_pu = copy.deepcopy(net.res_bus.vm_pu)

    # get previously calculated power models results
    pm_res_file = os.path.join(os.path.abspath(os.path.dirname(pp.test.__file__)),
                               "test_files", "pm_example_res.json")

    with open(pm_res_file, "r") as fp:
        result_pm = json.load(fp)
    net._options["correct_pm_network_data"] = True
    ppc, ppci = _pd2ppc(net)
    read_pm_results_to_net(net, ppc, ppci, result_pm)
    assert np.allclose(net.res_bus.vm_pu, vm_pu, atol=1e-4)
    assert np.allclose(net.res_bus.va_degree, va_degree, atol=1e-2, rtol=1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
