import pytest
import pandapower.networks as nw
from pandapower.converter.powermodels.to_pm import init_ne_line
import pandas as pd
import numpy as np
import pandapower as pp
from pandapower.converter import convert_pp_to_pm

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


def cigre_grid():
    net = nw.create_cigre_network_mv()

    net["bus"].loc[:, "min_vm_pu"] = 0.95
    net["bus"].loc[:, "max_vm_pu"] = 1.05

    net["line"].loc[:, "max_loading_percent"] = 60.
    return net

def define_possible_new_lines(net):
    # Here the possible new lines are a copy of all the lines which are already in the grid
    max_idx = max(net["line"].index)
    net["line"] = pd.concat([net["line"]] * 2, ignore_index=True) # duplicate
    # they must be set out of service in the line DataFrame (otherwise they are already "built")
    net["line"].loc[max_idx + 1:, "in_service"] = False
    # get the index of the new lines
    new_lines = net["line"].loc[max_idx + 1:].index

    # creates the new line DataFrame net["ne_line"] which defines the measures to choose from. The costs are defined
    # exemplary as 1. for every line.
    init_ne_line(net, new_lines, construction_costs=np.ones(len(new_lines)))

    return net

@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_pm_tnep_cigre_dc():
    # get the grid
    net = cigre_grid()
    # add the possible new lines
    define_possible_new_lines(net)
    # check if max line loading percent is violated (should be)
    pp.runpp(net)
    print("Max line loading prior to optimization:")
    print(net.res_line.loading_percent.max())
    assert np.any(net["res_line"].loc[:, "loading_percent"] > net["line"].loc[:, "max_loading_percent"])

    # run power models tnep optimization

    pp.runpm_tnep(net, pm_solver= "juniper", pm_model="DCMPPowerModel") # gurobi is a better option, but not for travis
  
    # print the information about the newly built lines
    print("These lines are to be built:")
    print(net["res_ne_line"])

    # set lines to be built in service
    lines_to_built = net["res_ne_line"].loc[net["res_ne_line"].loc[:, "built"], "built"].index
    net["line"].loc[lines_to_built, "in_service"] = True

    # run a power flow calculation again and check if max_loading percent is still violated
    pp.runpp(net)

    # check max line loading results
    assert not np.any(net["res_line"].loc[:, "loading_percent"] > net["line"].loc[:, "max_loading_percent"])

    print("Max line loading after the optimization:")
    print(net.res_line.loading_percent.max())

def define_ext_grid_limits(net):
    # define limits
    net["ext_grid"].loc[:, "min_p_mw"] = -9999.
    net["ext_grid"].loc[:, "max_p_mw"] = 9999.
    net["ext_grid"].loc[:, "min_q_mvar"] = -9999.
    net["ext_grid"].loc[:, "max_q_mvar"] = 9999.
    # define costs
    for i in net.ext_grid.index:
        pp.create_poly_cost(net, i, 'ext_grid', cp1_eur_per_mw=1)

def test_pm_tnep_cigre_only_conversion():
    # get the grid
    net = cigre_grid()
    # add the possible new lines
    define_possible_new_lines(net)
    # check if max line loading percent is violated (should be)
    pp.runpp(net)
    print("Max line loading prior to optimization:")
    print(net.res_line.loading_percent.max())
    assert np.any(net["res_line"].loc[:, "loading_percent"] > net["line"].loc[:, "max_loading_percent"])

    # run power models tnep optimization
    convert_pp_to_pm(net)


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
def test_pm_tnep_cigre_ac_S():
    # get the grid
    net = cigre_grid()
    # add the possible new lines
    define_possible_new_lines(net)
    # check if max line loading percent is violated (should be)
    pp.runpp(net)
    print("Max line loading prior to optimization:")
    print(net.res_line.loading_percent.max())
    assert np.any(net["res_line"].loc[:, "loading_percent"] > net["line"].loc[:, "max_loading_percent"])

    # run power models tnep optimization
    pp.runpm_tnep(net, pm_solver= "juniper", pm_model="ACPPowerModel", opf_flow_lim="S") # gurobi is a better option, but not for travis
    # print the information about the newly built lines
    print("These lines are to be built:")
    print(net["res_ne_line"])

    # set lines to be built in service
    lines_to_built = net["res_ne_line"].loc[net["res_ne_line"].loc[:, "built"], "built"].index
    net["line"].loc[lines_to_built, "in_service"] = True

    # run a power flow calculation again and check if max_loading percent is still violated
    pp.runpp(net)

    # check max line loading results
    assert not np.any(net["res_line"].loc[:, "loading_percent"] > net["line"].loc[:, "max_loading_percent"])

    print("Max line loading after the optimization:")
    print(net.res_line.loading_percent.max())


@pytest.mark.slow
@pytest.mark.skipif(julia_installed == False, reason="requires julia installation")
@pytest.mark.xfail(reason="Not yet implemented in the pm pp interface for TNEP")
def test_pm_tnep_cigre_ac_I():
    # get the grid
    net = cigre_grid()
    # add the possible new lines
    define_possible_new_lines(net)
    # check if max line loading percent is violated (should be)
    pp.runpp(net)
    print("Max line loading prior to optimization:")
    print(net.res_line.loading_percent.max())
    assert np.any(net["res_line"].loc[:, "loading_percent"] > net["line"].loc[:, "max_loading_percent"])

    # run power models tnep optimization
    pp.runpm_tnep(net, pm_solver= "juniper", pm_model="ACPPowerModel", opf_flow_lim="I") # gurobi is a better option, but not for travis
    # print the information about the newly built lines
    print("These lines are to be built:")
    print(net["res_ne_line"])

    # set lines to be built in service
    lines_to_built = net["res_ne_line"].loc[net["res_ne_line"].loc[:, "built"], "built"].index
    net["line"].loc[lines_to_built, "in_service"] = True

    # run a power flow calculation again and check if max_loading percent is still violated
    pp.runpp(net)

    # check max line loading results
    assert not np.any(net["res_line"].loc[:, "loading_percent"] > net["line"].loc[:, "max_loading_percent"])

    print("Max line loading after the optimization:")
    print(net.res_line.loading_percent.max())



if __name__ == '__main__':

    # pytest.main([__file__])
    # test_pm_tnep_cigre_ac_I()
    # test_pm_tnep_cigre()
    # test_pm_tnep_cigre_only_conversion()
    test_pm_tnep_cigre_dc()