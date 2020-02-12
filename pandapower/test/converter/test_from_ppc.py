# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
import pickle

import pytest

import pandapower as pp
import pandapower.networks as pn
from pandapower.converter import from_ppc, validate_from_ppc, to_ppc

try:
    import pypower.case24_ieee_rts as c24

    pypower_installed = True
except ImportError:
    pypower_installed = False

try:
    import pplog as logging
except:
    import logging

logger = logging.getLogger(__name__)
max_diff_values1 = {"bus_vm_pu": 1e-6, "bus_va_degree": 1e-5, "branch_p_mw": 1e-3,
                    "branch_q_mvar": 1e-3, "gen_p_mw": 1e-3, "gen_q_mvar": 1e-3}


def get_testgrids(name, filename):
    """
    This function return the ppc (or pp net) which is saved in ppc_testgrids.p to validate the
    from_ppc function via validate_from_ppc.
    """
    folder = os.path.join(pp.pp_dir, 'test', 'converter')
    ppcs = pickle.load(open(os.path.join(folder, filename), "rb"))
    return ppcs[name]


def test_from_ppc():
    ppc = get_testgrids('case2_2', 'ppc_testgrids.p')
    net_by_ppc = from_ppc(ppc)
    net_by_code = pp.from_json(os.path.join(pp.pp_dir, 'test', 'converter', 'case2_2_by_code.json'))
    pp.set_user_pf_options(net_by_code)  # for assertion of nets_equal
    pp.runpp(net_by_ppc, trafo_model="pi")
    pp.runpp(net_by_code, trafo_model="pi")

    assert type(net_by_ppc) == type(net_by_code)
    assert net_by_ppc.converged
    assert len(net_by_ppc.bus) == len(net_by_code.bus)
    assert len(net_by_ppc.trafo) == len(net_by_code.trafo)
    assert len(net_by_ppc.ext_grid) == len(net_by_code.ext_grid)
    assert pp.nets_equal(net_by_ppc, net_by_code, check_only_results=True, tol=1.e-9)


def test_validate_from_ppc():
    ppc = get_testgrids('case2_2', 'ppc_testgrids.p')
    net = pp.from_json(os.path.join(pp.pp_dir, 'test', 'converter', 'case2_2_by_code.json'))
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)


def test_ppc_testgrids():
    # check ppc_testgrids
    name = ['case2_1', 'case2_2', 'case2_3', 'case2_4', 'case3_1', 'case3_2', 'case6', 'case14', 'case57']
    for i in name:
        ppc = get_testgrids(i, 'ppc_testgrids.p')
        net = from_ppc(ppc, f_hz=60)
        assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)
        logger.debug('%s has been checked successfully.' % i)


@pytest.mark.slow
def test_pypower_cases():
    # check pypower cases
    name = ['case4gs', 'case6ww', 'case24_ieee_rts', 'case30', 'case39',
            'case118', 'case300']
    for i in name:
        ppc = get_testgrids(i, 'pypower_cases.p')
        net = from_ppc(ppc, f_hz=60)
        assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values1)
        logger.debug('%s has been checked successfully.' % i)
    # --- Because there is a pypower power flow failure in generator results in case9 (which is not
    # in matpower) another max_diff_values must be used to receive an successful validation
    max_diff_values2 = {"vm_pu": 1e-6, "va_degree": 1e-5, "p_branch_mw": 1e-3,
                        "q_branch_mvar": 1e-3, "p_gen_mw": 1e3, "q_gen_mvar": 1e3}
    ppc = get_testgrids('case9', 'pypower_cases.p')
    net = from_ppc(ppc, f_hz=60)
    assert validate_from_ppc(ppc, net, max_diff_values=max_diff_values2)
    logger.debug('case9 has been checked successfully.')


def test_case9_conversion():
    net = pn.case9()
    # set max_loading_percent to enable line limit conversion
    net.line["max_loading_percent"] = 100
    ppc = to_ppc(net, mode="opf")
    # correction because voltage limits are set to 1.0 at slack buses
    ppc["bus"][0, 12] = 0.9
    ppc["bus"][0, 11] = 1.1

    net2 = from_ppc(ppc, f_hz=net.f_hz)
    # again add max_loading_percent to enable valid comparison
    net2.line["max_loading_percent"] = 100

    # compare loadflow results
    pp.runpp(net)
    pp.runpp(net2)
    assert pp.nets_equal(net, net2, check_only_results=True, tol=1e-10)

    # compare optimal powerflow results
    pp.runopp(net, delta=1e-16)
    pp.runopp(net2, delta=1e-16)
    assert pp.nets_equal(net, net2, check_only_results=True, tol=1e-10)


@pytest.mark.skipif(pypower_installed == False, reason="needs pypower installation")
def test_case24():
    net = from_ppc(c24.case24_ieee_rts())
    pp.runopp(net)
    assert net.OPF_converged


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])
