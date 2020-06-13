# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:06:25 2018
Tests 3 phase power flow algorithm
@author: gbanerjee
"""
import pandapower as pp
import numpy as np
import pytest

@pytest.fixture
def net():
    v_base = 20                     # 20kV Base Voltage
    mva_base = 100                      # 100 MVA
    Net = pp.create_empty_network(sn_mva=mva_base)
    
    bus0 = pp.create_bus(Net, vn_kv=v_base, name="Bus 0")
    
    pp.create_ext_grid(Net, bus=bus0, vm_pu=1.0, name="Grid Connection", s_sc_max_mva=5000, 
                           rx_max=0.1, r0x0_max=0.1, x0x_max=1.0)
    
    bus1 = pp.create_bus(Net, name="Bus1", vn_kv=20, type="b")
    
    #pp.add_zero_impedance_parameters(Net)
        
    pp.create_asymmetric_load(Net, bus1, p_a_mw=0.3, q_a_mvar=0.003, p_b_mw=0.2, q_b_mvar=0.002,
                                  p_c_mw=0.1, q_c_mvar=0.001, scaling=1.0, in_service=True, type='wye')
    
        
    pp.create_line_from_parameters(Net, from_bus=bus0, to_bus=bus1, length_km=2.0, r0_ohm_per_km=.789,
                                       x0_ohm_per_km=0.306, c0_nf_per_km=272.9, max_i_ka=0.496,
                                       r_ohm_per_km=0.184, x_ohm_per_km=0.1900664, c_nf_per_km=273)
    return Net


def test_check_it(net):
    pp.runpp_3ph(net)


    line_pp = np.abs(net.res_line_3ph[~np.isnan(net.res_line_3ph.i_a_from_ka)]
                     [['i_a_from_ka', 'i_a_to_ka', 'i_b_from_ka', 'i_b_to_ka',
                       'i_c_from_ka', 'i_c_to_ka', 'i_n_from_ka', 'i_n_to_ka',
                       'p_a_from_mw', 'p_a_to_mw', 'q_a_from_mvar', 'q_a_to_mvar',
                       'p_b_from_mw', 'p_b_to_mw', 'q_b_from_mvar', 'q_b_to_mvar',
                       'p_c_from_mw', 'p_c_to_mw', 'q_c_from_mvar', 'q_c_to_mvar',
                       'loading_percent'
                       ]].values)
    line_pf = np.abs(np.array(
                        [[0.0260710, 0.0260123, 0.0174208, 0.0173288,
                          0.0088636, 0.0086592, 0.0150228,  0.0150217,
                          0.3003988, (-0.3), (-0.0196415), (-0.003),
                          0.2000966, (-0.2), (-0.0206405), (-0.002),
                          0.0999835, (-0.1), (-0.0218763), (-0.001),
                          5.25625]]))
    assert np.max(np.abs(line_pp - line_pf)) < 1.1e-5
   

 

    

if __name__ == "__main__":
    pytest.main(["test_runpp_3ph_n_line.py"])
    
