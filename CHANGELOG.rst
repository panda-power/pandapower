Change Log
=============

[2.2.1]- 2020-01-29
----------------------
- [FIXED] Missing csv files #625
- [FIXED] deepcopy speed and missing DataFrames in net #620, #631
- [FIXED] simple plotly error with generic coords #619
- [FIXED] create line with passed geodata #610
- [FIXED] ConstControl write to and all_index attribute #609
- [FIXED] collection plotting issue #608


[2.2.0]- 2020-01-17
----------------------
- [ADDED] control and timeseries module
- [ADDED] Support phasor measurement in state estimation
- [ADDED] Support recycle in state estimation
- [ADDED] PowerModels.jl converter callable without running the PowerModels optimization
- [ADDED] Other PowerModels features via interface callable (e.g. network data check and different solver)
- [ADDED] toolbox function select_subnet now also copies cost data and net parameters
- [ADDED] consideration of result tables in toolbox functions drop
- [ADDED] new jupyter notebook examples for time series, controller and PowerModels.jl interface
- [ADDED] reindex_buses() toolbox function

- [FIXED] Bugfixes in PowerModels conversion, OPF in general and tests
- [FIXED] renew opf_task() toolbox function which got outdated
- [FIXED] dtype at element parameter in cost tables
- [FIXED] convert_format.py: added the renaming of controller column and of the controller attributes, added tests for version 2.1.0

- [CHANGED] Unified the mesurement unit conversion of state estimation in ppc conversion
- [CHANGED] OPF bounds and settings for gens. limits or fixed values can now be enforced. See #511
- [CHANGED] OPF documentation and _check_necessary_opf_parameters()
- [CHANGED] JSON I/O: pandapower objects that are derived from JSONSerializableClass are now instantiated using __new__ instead of __init__ (as before), and the serialization has been adjusted; self.update_initialized(locals()) is not necessary anymore and has been removed; restore_json_objects is not needed anymore and has been removed
- [CHANGED] column name in net.controller: "controller" -> "object"
- [CHANGED] variable names in ContinuousTapControl ("u_set" -> "vm_set_pu") and in DiscreteTapControl ("u_lower" -> "vm_lower_pu", "u_upper" -> "vm_upper_pu")
- [CHANGED] __version__ is now changed to 2.2.0

[2.1.0]- 2019-07-08
----------------------
- [ADDED] calc_single_sc function to analyse a single fault instead of vectorized fault
- [ADDED] convenience function for logarithmic colormaps in plotting
- [CHANGED] corrected spelling 'continous' to 'continuous' in several functions
- [ADDED] additional standard types for overhead lines
- [CHANGED] make pp.to_json format closer to the JSON standard #406
- [ADDED] PowerModels.jl storage interface for time series based storage optimization.
- [ADDED] PowerModels.jl OTS interface for optimize transmission switching optimization.
- [ADDED] PowerModels.jl TNEP interface for transmission expansion optimization. See Jupyter Notebook
- [ADDED] pytest slow marker for tests and functions to run all, slow or fast tests
- [ADDED] Graph-Tool interface
- [ADDED] Multiple new algorithms and robust estimators in state estimation
- [ADDED] Support measurements for trafo3w in state estimation
- [ADDED] Auto zero-injection bus handling in state estimation

[2.0.1]- 2019-03-28
----------------------
- [FIXED] bug in short-circuit impedance of gens
- [ADDED] use estimation of rdss_pu defined in IEC 60909 of gens if not defined

[2.0.0]- 2019-03-21
----------------------
- [CHANGED] units from kW/kVAr/kVA to MW/MVAr/MVA in all elements #73
- [CHANGED] signing system from load to generation in gen, sgen and ext_grid #208
- [CHANGED] all trafo tap parameters from 'tp' to 'tap', tp_mid to tap_neutral #246
- [CHANGED] all trafo short-circuit voltage parameter names from "vsc" to "vk" #246
- [CHANGED] definition of cost functions #211
- [CHANGED] definition of measurements in measurement table #343
- [ADDED] interface to PowerModels.jl for OPF #207
- [CHANGED] removed Python 2 support #224
- [ADDED] load flow and OPF for user-defined temperature of lines, with the optional columns in line table "alpha" and "temperature_degree_celsius" #283
- [ADDED] z_ohm parameter in net.switch to assign resistance to switches #259
- [FIXED] initializing from results also considers auxiliary buses #236
- [ADDED] trafo3w switches are supported in create_nxgraph #271
- [CHANGED] create_nxgraph adds edges in multigraph with key=(element, idx) instead of key=0,1.. #85
- [CHANGED] patch size in create_bus_collection is not duplicated for rectangles anymore #181

[1.6.1] - 2019-02-18
----------------------
- [CHANGED] Patch size in create_bus_collection is not duplicated for rectangles anymore #181
- [CHANGED] Mask colormap z array to ensure nan handling
- [FIXED] active power distribution in DC OPF for multiple generators at one bus
- [ADDED] support for networkx graphs in json IO
- [ADDED] support for shapely objects in json IO
- [ADDED] switches for three winding transformers #30
- [ADDED] net.bus_geodata.coords to store line representation of busbars and create_busbar_collection to plot them
- [CHANGED] draw_collections also supports tuples of collections
- [ADDED] OPF logging output for verbose=True
- [ADDED] compatibility for pandas 0.24
- [FIXED] bug for single bus networks in DC PF #288

[1.6.0] - 2018-09-18
----------------------
- [CHANGED] Cost definition changed for optimal powerflow, see OPF documentation (http://pandapower.readthedocs.io/en/v1.6.0/powerflow/opf.html) and opf_changes-may18.ipynb
- [ADDED] OPF data (controllable, max_loading, costs, min_p_kw, ...) in Power System Test Cases
- [ADDED] case_ieee30, case5, case_illinois200
- [FIXED] 1 additional Trafo in case39, vn_kv change in case118, sgen indices in polynomial_cost in case 1888rte, case2848rte
- [ADDED] toolbox functions replace_impedance_by_line(), replace_line_by_impedance() and get_element_indices() including tests
- [CHANGED] new implementation of to_json, from_json for loading and saving grids using functools.singledispatch
- [FIXED] checking similar to "if x: ..." or "x = x or ..." when it is meant "if x is None: ...", because it is potentially problematic with some types
- [FIXED] convert_format: some older pandapower grids had "0" as "tp_side" in net.trafo, this is checked now as well
- [FIXED] create_buses: accepts a single tuple (set the same geodata for all buses) or an array of the corresponding shape (for individual geodata)
- [CHANGED] create_ext_grid_collection (plotting): ext_grid and ext_grid buses can be specified if a collection should only include some of ext grids
- [ADDED] ability to define phase shifting transformers with tp_st_percent #117
- [ADDED] support for multiple voltage controlling elements (ext_grid, gen, dcline) at one bus #134
- [CHANGED] reduced number of arguments in runpp by moving some less important arguments to kwargs #122
- [ADDED] parameters init_vm_pu and init_va_degree to allow independent initialization of bus magnitude and angle #113
- [ADDED] number of power flow iterations are now saved
- [ADDED] calculation of r, x and z for networkx branches
- [ADDED] support for plotly 3.2
- [FIXED] plotly bugfixes for trafo traces and result representation
- [ADDED] Iwamoto algorithm for solving ill-conditioned power flow problems

[1.5.1] - 2018-05-04
----------------------
- [FIXED] delta-wye transformation for 3W-transformers #54
- [ADDED] bus-bus switches collection #76
- [FIXED] some broken documentation links

[1.5.0] - 2018-04-25
----------------------
- [FIXED] plotly hover function for edges (only if use_line_geodata == False)
- [FIXED] from_ppc trafo parameter calculation now also considers baseMVA != 100
- [CHANGED] update create_collection docstrings
- [CHANGED] update HV/MV transformer standard type data
- [ADDED] pp_elements() toolbox function
- [ADDED] new parameter g_us_per_km to model dielectric losses in lines
- [ADDED] single phase short-circuit calculation with negative sequence models
- [ADDED] generic storage model (sgen/load like element with negative / positive power allowed)
- [ADDED] modelling of the complex (voltage magnitude and angle) tap changer for cross control
- [ADDED] modelling of the tap changer of a 3-winding transformer at star point or terminals
- [ADDED] losses of 3W transformers can be modeled at star point, HV, MV or LV side

[1.4.3] - 2018-02-06
----------------------
- [CHANGED] change of collection function names
- [ADDED] sgen collections and ration functionality for sgen and load collections
- [ADDED] cosphi_from_pq toolbox function
- [ADDED] create_nxgraph: respect_switches includes transformer switches

[1.4.2] - 2017-12-05
----------------------
- [ADDED] compatbility with networkx 2.0 (see #82)
- [ADDED] compatibility with pandas 0.21 (see #83)
- [CHANGED] implementation of ZIP loads changed to constant current magnitude paradigm (see #62)
- [ADDED] max_step parameter for shunt
- [ADDED] added warning for large bus index values
- [FIXED] bug in short-circuit results of trafo3w
- [FIXED] bugfix in find_bridges and refactoring
- [CHANGED] faster implementation of result cleanup
- [CHANGED] faster implementation of line index handling in power flow
- [FIXED] bug in plotly label display (#75)
- [ADDED] several fixes, extensions, tests for toolbox
- [ADDED] additional MV line standard types
- [FIXED] kerber extrem vorstadtnetz mv bus voltage
- [FIXED] removed incorrect estimation result tables for load, sgen, gen

[1.4.1] - 2017-09-19
----------------------
- [FIXED] ZIP load issue that led to incorrect calculation of I part with voltage angle shifts
- [FIXED] Bug that set voltage constraints to 0.9/1.2 if no voltage constraints was given in OPF
- [ADDED] possibility to access J matrix after power flow
- [ADDED] opf cost conversion
- [ADDED] opf costs in power system test cases

[1.4.0] - 2017-07-27
----------------------

- [ADDED] possibility to save networks to an sql database
- [CHANGED] major change in fileIO: all networks are converted to a uniform dataframe only version before they are saved as excel, json or sql. Old files can still be loaded, but all files saved with v1.4 can only be loaded with v1.4!
- [FIXED] all tests now pass if numba is not installed (although pandapower might be slow without numba)
- [FIXED] state estimation bug with phase shift transformers
- [CHANGED] OPF now raises specific warning if parameters are missing instead of generic exception
- [ADDED] geographical data for cigre and IEEE case networks
- [ADDED] Dickert LV Networks

[1.3.1] - 2017-06-16
----------------------
- [CHANGED] to_pickle saves only python datatypes and no pickle objects
- [ADDED] html representation of pandapower nets
- [ADDED] collections for trafos, loads, ext_grids
- [CHANGED] renamed create_shunt_as_condensator to create_shunt_as_capacitor
- [FIXED] mock problem in create docstrings
- [ADDED] Synthetic Voltage Control LV Networks

[1.3.0] - 2017-05-10
----------------------
- [ADDED] ZIP loads integrated in power flow
- [ADDED] numba implementation of dissolving switch buses
- [ADDED] Current source representation of full converter elements in short circuit calculations
- [ADDED] Method C for calculation of factor kappa in short circuit calculation
- [CHANGED] Speedup for calculation of branch short circuit currents
- [CHANGED] Branch results for minimum short circuit calculations are calculated as minimal currents
- [ADDED] Interactive plots with plotly
- [CHANGED] included pypower files for power flow and index files
- [FIXED] compatibility with numpy 1.12
- [CHANGED] -1 is a valid value for net.bus_geodata.x
- [CHANGED] allow transformers with negative xk to provide large scale IEEE cases (RTE, PEGASE, Polish)
- [ADDED] large scale IEEE cases (RTE, PEGASE, Polish)
- [ADDED] rated voltage and step variable for shunts
- [ADDED] lagrange multiplier included in bus results after OPF

[1.2.2] - 2017-03-22
--------------------
- [CHANGED] Minor refactoring in pd2ppc
- [ADDED] Technical Report

[1.2.1] - 2017-03-21
--------------------
- [FIXED] Readme for PyPi

[1.2.0] - 2017-03-21
--------------------
- [CHANGED] net.line.imax_ka to net.line.max_i_ka for consistency reasons
- [ADDED] net.line.tp_st_degree for phase shift in trafo tap changers
- [ADDED] sn_kva parameter in create_empty network for per unit system reference power
- [ADDED] parameter parallel for trafo element
- [ADDED] connectivity check for power flow to deal with disconnected network areas
- [ADDED] backward/forward sweep power flow algorithm specially suited for radial and weakly-meshed networks
- [ADDED] linear piece wise and polynomial OPF cost functions
- [ADDED] possibility to make loads controllable in OPF
- [ADDED] to_json and from_json functions to save/load networks with a JSON format
- [ADDED] generator lookup to allow multiple generators at one bus
- [CHANGED] Initialization of calculate_voltage_angles and init for high voltage networks
- [ADDED] bad data detection for state estimation
- [CHANGED] from_ppc: no detect_trafo anymore, several gen at each node possible
- [CHANGED] validate_from_ppc: improved validation behaviour by means of duplicated gen and branch rearangement
- [ADDED] networks: case33bw, case118, case300, case1354pegase, case2869pegase, case9241pegase, GBreducednetwork, GBnetwork, iceland, cigre_network_mv with_der='all' der
- [ADDED] possibility to add fault impedance for short-circuit current calculation
- [ADDED] branch results for short circuits
- [ADDED] static generator model for short circuits
- [ADDED] three winding transformer model for short circuits
- [FIXED] correctly neglecting shunts and tap changer position for short-circuits
- [ADDED] two phase short-circuit current calculation
- [ADDED] tests for short circuit currents with validation against DIgSILENT PowerFactory


[1.1.1] - 2017-01-12
----------------------
- [ADDED] installation description and pypi files from github
- [ADDED] automatic inversion of active power limits in convert format to account for convention change in version 1.1.0
- [CHANGED] install_requires in setup.py


[1.1.0] - 2017-01-11
----------------------
- [ADDED] impedance element can now be used with unsymetric impedances zij != zji
- [ADDED] dcline element that allows modelling DC lines in PF and OPF
- [ADDED] simple plotting function: call pp.simple_plot(net) to directly plot the network
- [ADDED] measurement table for networks. Enables the definition of measurements for real-time simulations.
- [ADDED] estimation module, which provides state estimation functionality with weighted least squares algorithm
- [ADDED] shortcircuit module in beta version for short-circuit calculation according to IEC-60909
- [ADDED] documentation of model validation and tests
- [ADDED] case14, case24_ieee_rts, case39, case57 networks
- [ADDED] mpc and ppc converter
- [CHANGED] convention for active power limits of generators. Generator with max. feed in of 50kW before: p_min_kw=0, p_max_kw=-50. Now p_max_kw=0, p_min_kw=50
- [ADDED] DC power flow function pp.rundcopp
- [FIXED] bug in create_transformer function for tp_pos parameter
- [FIXED] bug in voltage ratio for low voltage side tap changers
- [FIXED] bug in rated voltage calculation for opf line constraints

[1.0.2] - 2016-11-30
----------------------

- [CHANGED] changed in_service dtype from f8 to bool for shunt, ward, xward
- [CHANGED] included i_from_ka and i_to_ka in net.res_line
- [ADDED] recycle parameter added. ppc, Ybus, _is_elements and bus_lookup can be reused between multiple powerflows if recycle["ppc"] == True, ppc values (P,Q,V) only get updated.
- [FIXED] OPF bugfixes: cost scaling, correct calculation of res_bus.p_kw for sgens
- [ADDED] loadcase added as pypower_extension since unnecessary deepcopies were removed
- [CHANGED] supress warnings parameter removed from loadflow, casting warnings are automatically supressed

[1.0.1] - 2016-11-09
----------------------

- [CHANGED] update short introduction example to include transformer
- [CHANGED] included pypower in setup.py requirements (only pypower, not numpy, scipy etc.)
- [CHANGED] mpc / ppc renamed to ppci / ppc
- [FIXED] MANIFEST.ini includes all relevant doc files and exclude report
- [FIXED] handling of tp_pos parameter in create_trafo and create_trafo3w
- [FIXED] init="result" for open bus-line switches
