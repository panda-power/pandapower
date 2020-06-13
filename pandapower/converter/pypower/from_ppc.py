# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from math import pi
from numpy import sign, nan, append, zeros, array, sqrt, where
from numpy import max as max_
from pandas import Series, DataFrame, concat
from pandapower.pypower.idx_gen import GEN_BUS, PMIN, PMAX, QMIN, QMAX, GEN_STATUS
from pandapower.pypower.idx_cost import COST, NCOST
from pandapower.pypower.idx_bus import BUS_I, BASE_KV
import pandapower as pp

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

try:
    from pypower import ppoption, runpf, runopf, rundcpf, rundcopf
    ppopt = ppoption.ppoption(VERBOSE=0, OUT_ALL=0)
    pypower_import = True
except ImportError:
    pypower_import = False

ppc_elms = ["bus", "branch", "gen"]


def _create_costs(net, ppc, gen_lookup, type, idx):
    if ppc['gencost'][idx, 0] == 1:
        if not len(ppc['gencost'][idx, COST:]) == 2*ppc['gencost'][idx, NCOST]:
            logger.error("In gencost line %s, the number n does not fit to the number of values" %
                         idx)
        raise NotImplementedError
        pp.create_pwl_cost(net, gen_lookup.element.at[idx],
                           gen_lookup.element_type.at[idx],
                           ppc['gencost'][idx, 4:], type)
    elif ppc['gencost'][idx, 0] == 2:
        ncost = ppc['gencost'][idx, NCOST]
        if ncost == 1:
            cp2 = 0
            cp1 = 0
            cp0 = ppc['gencost'][idx, COST]
        elif ncost == 2:
            cp2 = 0
            cp1 = ppc['gencost'][idx, COST]
            cp0 = ppc['gencost'][idx, COST + 1]
        elif ncost == 3:
            cp2 = ppc['gencost'][idx, COST]
            cp1 = ppc['gencost'][idx, COST + 1]
            cp0 = ppc['gencost'][idx, COST + 2]
        elif ncost > 3:
            logger.warning("The pandapower poly_cost table only supports up to 2nd order " +
                           "polynomials. The ppc higher order polynomials cannot be converted.")
            cp2 = ppc['gencost'][idx, COST + ncost - 3]
            cp1 = ppc['gencost'][idx, COST + ncost - 2]
            cp0 = ppc['gencost'][idx, COST + ncost - 1]
        else:
            raise ValueError("'ncost' must be an positve integer but is " + str(ncost))
        pp.create_poly_cost(net, gen_lookup.element.at[idx], gen_lookup.element_type.at[idx],
                                  cp1_eur_per_mw=cp1, cp2_eur_per_mw2=cp2, cp0_eur=cp0)
    else:
        logger.info("Cost mode of gencost line %s is unknown." % idx)


def _gen_bus_info(ppc, idx_gen):
    bus_name = int(ppc["gen"][idx_gen, GEN_BUS])
    # assumption: there is only one bus with this bus_name:
    idx_bus = int(where(ppc["bus"][:, BUS_I] == bus_name)[0][0])
    current_bus_type = int(ppc["bus"][idx_bus, 1])

    same_bus_gen_idx = where(ppc["gen"][:, GEN_BUS] == ppc["gen"][idx_gen, GEN_BUS])[0].astype(int)
    same_bus_in_service_gen_idx = same_bus_gen_idx[where(ppc["gen"][same_bus_gen_idx, GEN_STATUS] > 0)]
    first_same_bus_in_service_gen_idx = same_bus_in_service_gen_idx[0] if len(
        same_bus_in_service_gen_idx) else None
    last_same_bus_in_service_gen_idx = same_bus_in_service_gen_idx[-1] if len(
        same_bus_in_service_gen_idx) else None

    return current_bus_type, idx_bus, same_bus_gen_idx, first_same_bus_in_service_gen_idx, \
        last_same_bus_in_service_gen_idx


def from_ppc(ppc, f_hz=50, validate_conversion=False, **kwargs):
    """
    This function converts pypower case files to pandapower net structure.

    INPUT:

        **ppc** : The pypower case file.

    OPTIONAL:

        **f_hz** (float, 50) - The frequency of the network.

        **validate_conversion** (bool, False) - If True, validate_from_ppc is run after conversion.
            For running the validation, the ppc must already contain the pypower
            powerflow results or pypower must be importable.

        ****kwargs** keyword arguments for validate_from_ppc if validate_conversion is True

    OUTPUT:

        **net** : pandapower net.

    EXAMPLE:

        import pandapower.converter as pc

        from pypower import case4gs

        ppc_net = case4gs.case4gs()

        net = pc.from_ppc(ppc_net, f_hz=60)

    """
    # --- catch common failures
    if Series(ppc['bus'][:, BASE_KV] <= 0).any():
        logger.info('There are false baseKV given in the pypower case file.')

    # --- general_parameters
    baseMVA = ppc['baseMVA']  # MVA
    omega = pi * f_hz  # 1/s
    MAX_VAL = 99999.

    net = pp.create_empty_network(f_hz=f_hz, sn_mva=baseMVA)

    # --- bus data -> create buses, sgen, load, shunt
    for i in range(len(ppc['bus'])):
        # create buses
        pp.create_bus(net, name=int(ppc['bus'][i, 0]), vn_kv=ppc['bus'][i, 9], type="b",
                      zone=ppc['bus'][i, 10], in_service=bool(ppc['bus'][i, 1] != 4),
                      max_vm_pu=ppc['bus'][i, 11], min_vm_pu=ppc['bus'][i, 12])
        # create sgen, load
        if ppc['bus'][i, 2] > 0:
            pp.create_load(net, i, p_mw=ppc['bus'][i, 2], q_mvar=ppc['bus'][i, 3],
                           controllable=False)
        elif ppc['bus'][i, 2] < 0:
            pp.create_sgen(net, i, p_mw=-ppc['bus'][i, 2], q_mvar=-ppc['bus'][i, 3],
                           type="", controllable=False)
        elif ppc['bus'][i, 3] != 0:
            pp.create_load(net, i, p_mw=ppc['bus'][i, 2], q_mvar=ppc['bus'][i, 3],
                           controllable=False)
        # create shunt
        if ppc['bus'][i, 4] != 0 or ppc['bus'][i, 5] != 0:
            pp.create_shunt(net, i, p_mw=ppc['bus'][i, 4],
                            q_mvar=-ppc['bus'][i, 5])
    # unused data of ppc: Vm, Va (partwise: in ext_grid), zone

    # --- gen data -> create ext_grid, gen, sgen
    gen_lookup = DataFrame(nan, columns=['element', 'element_type'],
                           index=range(len(ppc['gen'][:, 0])))
    # if in ppc is only one gen -> numpy initially uses one dim array -> change to two dim array
    if len(ppc["gen"].shape) == 1:
        ppc["gen"] = array(ppc["gen"], ndmin=2)
    for i in range(len(ppc['gen'][:, 0])):
        current_bus_type, current_bus_idx, same_bus_gen_idx, first_same_bus_in_service_gen_idx, \
            last_same_bus_in_service_gen_idx = _gen_bus_info(ppc, i)
        # create ext_grid
        if current_bus_type == 3:
            if i == first_same_bus_in_service_gen_idx:
                gen_lookup.element.loc[i] = pp.create_ext_grid(
                    net, bus=current_bus_idx, vm_pu=ppc['gen'][last_same_bus_in_service_gen_idx, 5],
                    va_degree=ppc['bus'][current_bus_idx, 8], in_service=bool(ppc['gen'][i, 7] > 0),
                    max_p_mw=ppc['gen'][i, PMAX], min_p_mw=ppc['gen'][i, PMIN],
                    max_q_mvar=ppc['gen'][i, QMAX], min_q_mvar=ppc['gen'][i, QMIN])
                gen_lookup.element_type.loc[i] = 'ext_grid'
                if ppc['gen'][i, 4] > ppc['gen'][i, 3]:
                    logger.info('min_q_mvar of gen %d must be less than max_q_mvar but is not.' % i)
                if -ppc['gen'][i, 9] < -ppc['gen'][i, 8]:
                    logger.info('max_p_mw of gen %d must be less than min_p_mw but is not.' % i)
            else:
                current_bus_type = 1
        # create gen
        elif current_bus_type == 2:
            if i == first_same_bus_in_service_gen_idx:
                gen_lookup.element.loc[i] = pp.create_gen(
                    net, bus=current_bus_idx, vm_pu=ppc['gen'][last_same_bus_in_service_gen_idx, 5],
                    p_mw=ppc['gen'][i, 1],
                    in_service=bool(ppc['gen'][i, 7] > 0), controllable=True,
                    max_p_mw=ppc['gen'][i, PMAX], min_p_mw=ppc['gen'][i, PMIN],
                    max_q_mvar=ppc['gen'][i, QMAX], min_q_mvar=ppc['gen'][i, QMIN])
                gen_lookup.element_type.loc[i] = 'gen'
                if ppc['gen'][i, 1] < 0:
                    logger.info('p_mw of gen %d must be less than zero but is not.' % i)
                if ppc['gen'][i, 4] > ppc['gen'][i, 3]:
                    logger.info('min_q_mvar of gen %d must be less than max_q_mvar but is not.' % i)
                if -ppc['gen'][i, 9] < -ppc['gen'][i, 8]:
                    logger.info('max_p_mw of gen %d must be less than min_p_mw but is not.' % i)
            else:
                current_bus_type = 1
        # create sgen
        if current_bus_type == 1:
            gen_lookup.element.loc[i] = pp.create_sgen(
                net, bus=current_bus_idx, p_mw=ppc['gen'][i, 1],
                q_mvar=ppc['gen'][i, 2], type="", in_service=bool(ppc['gen'][i, 7] > 0),
                max_p_mw=ppc['gen'][i, PMAX], min_p_mw=ppc['gen'][i, PMIN],
                max_q_mvar=ppc['gen'][i, QMAX], min_q_mvar=ppc['gen'][i, QMIN],
                controllable=True)
            gen_lookup.element_type.loc[i] = 'sgen'
            if ppc['gen'][i, 1] < 0:
                logger.info('p_mw of sgen %d must be less than zero but is not.' % i)
            if ppc['gen'][i, 4] > ppc['gen'][i, 3]:
                logger.info('min_q_mvar of gen %d must be less than max_q_mvar but is not.' % i)
            if -ppc['gen'][i, 9] < -ppc['gen'][i, 8]:
                logger.info('max_p_mw of gen %d must be less than min_p_mw but is not.' % i)
    # unused data of ppc: Vg (partwise: in ext_grid and gen), mBase, Pc1, Pc2, Qc1min, Qc1max,
    # Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30,ramp_q, apf

    # --- branch data -> create line, trafo
    for i in range(len(ppc['branch'])):
        from_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 0]))
        to_bus = pp.get_element_index(net, 'bus', name=int(ppc['branch'][i, 1]))

        from_vn_kv = ppc['bus'][from_bus, 9]
        to_vn_kv = ppc['bus'][to_bus, 9]
        if (from_vn_kv == to_vn_kv) & ((ppc['branch'][i, 8] == 0) | (ppc['branch'][i, 8] == 1)) & \
           (ppc['branch'][i, 9] == 0):  # create line
            Zni = ppc['bus'][to_bus, 9]**2/baseMVA  # ohm
            max_i_ka = ppc['branch'][i, 5]/ppc['bus'][to_bus, 9]/sqrt(3)
            if max_i_ka == 0.0:
                max_i_ka = MAX_VAL
                logger.debug("ppc branch rateA is zero -> Using MAX_VAL instead to calculate " +
                             "maximum branch flow")
            pp.create_line_from_parameters(
                net, from_bus=from_bus, to_bus=to_bus, length_km=1,
                r_ohm_per_km=ppc['branch'][i, 2]*Zni, x_ohm_per_km=ppc['branch'][i, 3]*Zni,
                c_nf_per_km=ppc['branch'][i, 4]/Zni/omega*1e9/2,
                max_i_ka=max_i_ka, type='ol', max_loading_percent=100,
                in_service=bool(ppc['branch'][i, 10]))

        else:  # create transformer
            if from_vn_kv >= to_vn_kv:
                hv_bus = from_bus
                vn_hv_kv = from_vn_kv
                lv_bus = to_bus
                vn_lv_kv = to_vn_kv
                tap_side = 'hv'
            else:
                hv_bus = to_bus
                vn_hv_kv = to_vn_kv
                lv_bus = from_bus
                vn_lv_kv = from_vn_kv
                tap_side = 'lv'
                if from_vn_kv == to_vn_kv:
                    logger.warning('The pypower branch %d (from_bus, to_bus)=(%d, %d) is considered'
                                   ' as a transformer because of a ratio != 0 | 1 but it connects '
                                   'the same voltage level', i, ppc['branch'][i, 0],
                                   ppc['branch'][i, 1])
            rk = ppc['branch'][i, 2]
            xk = ppc['branch'][i, 3]
            zk = (rk ** 2 + xk ** 2) ** 0.5
            sn = ppc['branch'][i, 5]
            if sn == 0.0:
                sn = MAX_VAL
                logger.debug("ppc branch rateA is zero -> Using MAX_VAL instead to calculate " +
                             "apparent power")
            ratio_1 = 0 if ppc['branch'][i, 8] == 0 else (ppc['branch'][i, 8] - 1) * 100
            i0_percent = -ppc['branch'][i, 4] * 100 * baseMVA / sn
            if i0_percent < 0:
                logger.info('A transformer always behaves inductive consumpting but the '
                            'susceptance of pypower branch %d (from_bus, to_bus)=(%d, %d) is '
                            'positive.', i, ppc['branch'][i, 0], ppc['branch'][i, 1])

            pp.create_transformer_from_parameters(
                net, hv_bus=hv_bus, lv_bus=lv_bus, sn_mva=sn, vn_hv_kv=vn_hv_kv,
                vn_lv_kv=vn_lv_kv, vk_percent=sign(xk) * zk * sn * 100 / baseMVA,
                vkr_percent=rk * sn * 100 / baseMVA, max_loading_percent=100,
                pfe_kw=0, i0_percent=i0_percent, shift_degree=ppc['branch'][i, 9],
                tap_step_percent=abs(ratio_1) if ratio_1 else nan,
                tap_pos=sign(ratio_1) if ratio_1 else nan,
                tap_side=tap_side if ratio_1 else None, tap_neutral=0 if ratio_1 else nan)
    # unused data of ppc: rateB, rateC

    # --- gencost -> create polynomial_cost, piecewise_cost
    if 'gencost' in ppc:
        if len(ppc['gencost'].shape) == 1:
            # reshape gencost if only one gencost is given -> no indexError
            ppc['gencost'] = ppc['gencost'].reshape((1, -1))
        if ppc['gencost'].shape[0] <= gen_lookup.shape[0]:
            idx_p = range(ppc['gencost'].shape[0])
            idx_q = []
        elif ppc['gencost'].shape[0] > gen_lookup.shape[0]:
            idx_p = range(gen_lookup.shape[0])
            idx_q = range(gen_lookup.shape[0], ppc['gencost'].shape[0])
        if ppc['gencost'].shape[0] >= 2*gen_lookup.shape[0]:
            idx_p = range(gen_lookup.shape[0])
            idx_q = range(gen_lookup.shape[0], 2*gen_lookup.shape[0])
        for idx in idx_p:
            _create_costs(net, ppc, gen_lookup, 'p', idx)
        for idx in idx_q:
            _create_costs(net, ppc, gen_lookup, 'q', idx)

    # areas are unconverted

    if validate_conversion:
        logger.setLevel(logging.DEBUG)
        if not validate_from_ppc(ppc, net, **kwargs):
            logger.error("Validation failed.")

    return net


def _validate_diff_res(diff_res, max_diff_values):
    to_iterate = set(max_diff_values.keys()) & {'gen_q_mvar', 'branch_p_mw', 'branch_q_mvar',
                                                'gen_p_mw', 'bus_va_degree', 'bus_vm_pu'}
    if not len(to_iterate):
        logger.warning("There are no keys to validate.")
    val = True
    for i in to_iterate:
        elm = i.split("_")[0]
        sought = ["p", "q"] if elm != "bus" else ["vm", "va"]
        col = int(array([0, 1])[[j in i for j in sought]][0]) if elm != "branch" else \
            list(array([[0, 2], [1, 3]])[[j in i for j in sought]][0])
        val &= bool(max_(abs(diff_res[elm][:, col])) < max_diff_values[i])
    return val


def validate_from_ppc(ppc_net, net, pf_type="runpp", max_diff_values={
    "bus_vm_pu": 1e-6, "bus_va_degree": 1e-5, "branch_p_mw": 1e-6, "branch_q_mvar": 1e-6,
        "gen_p_mw": 1e-6, "gen_q_mvar": 1e-6}, run=True):
    """
    This function validates the pypower case files to pandapower net structure conversion via a \
    comparison of loadflow calculation results. (Hence the opf cost conversion is not validated.)

    INPUT:

        **ppc_net** - The pypower case file, which must already contain the pypower powerflow
            results or pypower must be importable.

        **net** - The pandapower network.

    OPTIONAL:

        **pf_type** ("runpp", string) - Type of validated power flow. Possible are ("runpp",
            "rundcpp", "runopp", "rundcopp")

        **max_diff_values** - Dict of maximal allowed difference values. The keys must be
        'vm_pu', 'va_degree', 'p_branch_mw', 'q_branch_mvar', 'p_gen_mw' and 'q_gen_mvar' and
        the values floats.

        **run** (True, bool or list of two bools) - changing the value to False avoids trying to run
            (optimal) loadflows. Giving a list of two bools addresses first pypower and second
            pandapower.

    OUTPUT:

        **conversion_success** - conversion_success is returned as False if pypower or pandapower
        cannot calculate a powerflow or if the maximum difference values (max_diff_values )
        cannot be hold.

    EXAMPLE:

        import pandapower.converter as pc

        net = cv.from_ppc(ppc_net, f_hz=50)

        conversion_success = cv.validate_from_ppc(ppc_net, net)

    NOTE:

        The user has to take care that the loadflow results already are included in the provided \
        ppc_net or pypower is importable.
    """
    # check in case of optimal powerflow comparison whether cost information exist
    if "opp" in pf_type:
        if not (len(net.polynomial_cost) | len(net.piecewise_linear_cost)):
            if "gencost" in ppc_net:
                if not len(ppc_net["gencost"]):
                    logger.debug('ppc and pandapower net do not include cost information.')
                    return True
                else:
                    logger.error('The pandapower net does not include cost information.')
                    return False
            else:
                logger.debug('ppc and pandapower net do not include cost information.')
                return True

    # guarantee run parameter as list, for pypower and pandapower (optimal) powerflow run
    run = [run, run] if isinstance(run, bool) else run

    # --- check pypower powerflow success, if possible
    if pypower_import and run[0]:
        try:
            if pf_type == "runpp":
                ppc_net = runpf.runpf(ppc_net, ppopt)[0]
            elif pf_type == "rundcpp":
                ppc_net = rundcpf.rundcpf(ppc_net, ppopt)[0]
            elif pf_type == "runopp":
                ppc_net = runopf.runopf(ppc_net, ppopt)
            elif pf_type == "rundcopp":
                ppc_net = rundcopf.rundcopf(ppc_net, ppopt)
            else:
                raise ValueError("The pf_type %s is unknown" % pf_type)
        except:
            logger.debug("The pypower run did not work.")
    ppc_success = True
    if 'success' in ppc_net.keys():
        if ppc_net['success'] != 1:
            ppc_success = False
            logger.error("The given ppc data indicates an unsuccessful pypower powerflow: " +
                         "'ppc_net['success'] != 1'")
    if (ppc_net['branch'].shape[1] < 17):
        ppc_success = False
        logger.error("The shape of given ppc data indicates missing pypower powerflow results.")

    # --- try to run a pandapower powerflow
    if run[1]:
        if pf_type == "runpp":
            try:
                pp.runpp(net, init="dc", calculate_voltage_angles=True, trafo_model="pi")
            except pp.LoadflowNotConverged:
                try:
                    pp.runpp(net, calculate_voltage_angles=True, init="flat", trafo_model="pi")
                except pp.LoadflowNotConverged:
                    try:
                        pp.runpp(net, trafo_model="pi", calculate_voltage_angles=False)
                        if "bus_va_degree" in max_diff_values.keys():
                            max_diff_values["bus_va_degree"] = 1e2 if max_diff_values[
                                "bus_va_degree"] < 1e2 else max_diff_values["bus_va_degree"]
                        logger.info("voltage_angles could be calculated.")
                    except pp.LoadflowNotConverged:
                        logger.error('The pandapower powerflow does not converge.')
        elif pf_type == "rundcpp":
            try:
                pp.rundcpp(net, trafo_model="pi")
            except pp.LoadflowNotConverged:
                logger.error('The pandapower dc powerflow does not converge.')
        elif pf_type == "runopp":
                try:
                    pp.runopp(net, init="flat", calculate_voltage_angles=True)
                except pp.OPFNotConverged:
                    try:
                        pp.runopp(net, init="pf", calculate_voltage_angles=True)
                    except (pp.OPFNotConverged, pp.LoadflowNotConverged, KeyError):
                        try:
                            pp.runopp(net, init="flat", calculate_voltage_angles=False)
                            logger.info("voltage_angles could be calculated.")
                            if "bus_va_degree" in max_diff_values.keys():
                                max_diff_values["bus_va_degree"] = 1e2 if max_diff_values[
                                    "bus_va_degree"] < 1e2 else max_diff_values["bus_va_degree"]
                        except pp.OPFNotConverged:
                            try:
                                pp.runopp(net, init="pf", calculate_voltage_angles=False)
                                if "bus_va_degree" in max_diff_values.keys():
                                    max_diff_values["bus_va_degree"] = 1e2 if max_diff_values[
                                        "bus_va_degree"] < 1e2 else max_diff_values["bus_va_degree"]
                                logger.info("voltage_angles could be calculated.")
                            except (pp.OPFNotConverged, pp.LoadflowNotConverged, KeyError):
                                logger.error('The pandapower optimal powerflow does not converge.')
        elif pf_type == "rundcopp":
            try:
                pp.rundcopp(net)
            except pp.LoadflowNotConverged:
                logger.error('The pandapower dc optimal powerflow does not converge.')
        else:
            raise ValueError("The pf_type %s is unknown" % pf_type)

    # --- prepare powerflow result comparison by reordering pp results as they are in ppc results
    if not ppc_success:
        return False
    if "opp" in pf_type:
        if not net.OPF_converged:
            return
    elif not net.converged:
        return False

    # --- store pypower powerflow results
    ppc_res = dict.fromkeys(ppc_elms)
    ppc_res["branch"] = ppc_net['branch'][:, 13:17]
    ppc_res["bus"] = ppc_net['bus'][:, 7:9]
    ppc_res["gen"] = ppc_net['gen'][:, 1:3]

    # --- pandapower bus result table
    pp_res = dict.fromkeys(ppc_elms)
    pp_res["bus"] = array(net.res_bus.sort_index()[['vm_pu', 'va_degree']])

    # --- pandapower gen result table
    pp_res["gen"] = zeros([1, 2])
    # consideration of parallel generators via storing how much generators have been considered
    # each node
    # if in ppc is only one gen -> numpy initially uses one dim array -> change to two dim array
    if len(ppc_net["gen"].shape) == 1:
        ppc_net["gen"] = array(ppc_net["gen"], ndmin=2)
    GENS = DataFrame(ppc_net['gen'][:, [0]].astype(int))
    GEN_uniq = GENS.drop_duplicates()
    already_used_gen = Series(zeros(GEN_uniq.shape[0]).astype(int),
                              index=[int(v) for v in GEN_uniq.values])
    change_q_compare = []
    for i, j in GENS.iterrows():
        current_bus_type, current_bus_idx, same_bus_gen_idx, first_same_bus_in_service_gen_idx, \
            last_same_bus_in_service_gen_idx = _gen_bus_info(ppc_net, i)
        if current_bus_type == 3 and i == first_same_bus_in_service_gen_idx:
            pp_res["gen"] = append(pp_res["gen"], array(net.res_ext_grid[
                    net.ext_grid.bus == current_bus_idx][['p_mw', 'q_mvar']]).reshape((1, 2)), 0)
        elif current_bus_type == 2 and i == first_same_bus_in_service_gen_idx:
            pp_res["gen"] = append(pp_res["gen"], array(net.res_gen[
                    net.gen.bus == current_bus_idx][['p_mw', 'q_mvar']]).reshape((1, 2)), 0)
        else:
            pp_res["gen"] = append(pp_res["gen"], array(net.res_sgen[
                net.sgen.bus == current_bus_idx][['p_mw', 'q_mvar']])[
                already_used_gen.at[int(j)]].reshape((1, 2)), 0)
            already_used_gen.at[int(j)] += 1
            change_q_compare += [int(j)]
    pp_res["gen"] = pp_res["gen"][1:, :]  # delete initial zero row

    # --- pandapower branch result table
    pp_res["branch"] = zeros([1, 4])
    # consideration of parallel branches via storing how often branches were considered
    # each node-to-node-connection
    try:
        init1 = concat([net.line.from_bus, net.line.to_bus], axis=1,
                       sort=True).drop_duplicates()
        init2 = concat([net.trafo.hv_bus, net.trafo.lv_bus], axis=1,
                       sort=True).drop_duplicates()
    except TypeError:
        # legacy pandas < 0.21
        init1 = concat([net.line.from_bus, net.line.to_bus], axis=1).drop_duplicates()
        init2 = concat([net.trafo.hv_bus, net.trafo.lv_bus], axis=1).drop_duplicates()
    init1['hv_bus'] = nan
    init1['lv_bus'] = nan
    init2['from_bus'] = nan
    init2['to_bus'] = nan
    try:
        already_used_branches = concat([init1, init2], axis=0, sort=True)
    except TypeError:
        # pandas < 0.21 legacy
        already_used_branches = concat([init1, init2], axis=0)
    already_used_branches['number'] = zeros([already_used_branches.shape[0], 1]).astype(int)
    BRANCHES = DataFrame(ppc_net['branch'][:, [0, 1, 8, 9]])
    for i in BRANCHES.index:
        from_bus = pp.get_element_index(net, 'bus', name=int(ppc_net['branch'][i, 0]))
        to_bus = pp.get_element_index(net, 'bus', name=int(ppc_net['branch'][i, 1]))
        from_vn_kv = ppc_net['bus'][from_bus, 9]
        to_vn_kv = ppc_net['bus'][to_bus, 9]
        ratio = BRANCHES[2].at[i]
        angle = BRANCHES[3].at[i]
        # from line results
        if (from_vn_kv == to_vn_kv) & ((ratio == 0) | (ratio == 1)) & (angle == 0):
            pp_res["branch"] = append(pp_res["branch"], array(net.res_line[
                (net.line.from_bus == from_bus) &
                (net.line.to_bus == to_bus)]
                [['p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar']])[
                int(already_used_branches.number.loc[
                   (already_used_branches.from_bus == from_bus) &
                   (already_used_branches.to_bus == to_bus)].values)].reshape(1, 4), 0)
            already_used_branches.number.loc[(already_used_branches.from_bus == from_bus) &
                                             (already_used_branches.to_bus == to_bus)] += 1
        # from trafo results
        else:
            if from_vn_kv >= to_vn_kv:
                pp_res["branch"] = append(pp_res["branch"], array(net.res_trafo[
                    (net.trafo.hv_bus == from_bus) &
                    (net.trafo.lv_bus == to_bus)]
                    [['p_hv_mw', 'q_hv_mvar', 'p_lv_mw', 'q_lv_mvar']])[
                    int(already_used_branches.number.loc[
                        (already_used_branches.hv_bus == from_bus) &
                        (already_used_branches.lv_bus == to_bus)].values)].reshape(1, 4), 0)
                already_used_branches.number.loc[(already_used_branches.hv_bus == from_bus) &
                                                 (already_used_branches.lv_bus == to_bus)] += 1
            else:  # switch hv-lv-connection of pypower connection buses
                pp_res["branch"] = append(pp_res["branch"], array(net.res_trafo[
                    (net.trafo.hv_bus == to_bus) &
                    (net.trafo.lv_bus == from_bus)]
                    [['p_lv_mw', 'q_lv_mvar', 'p_hv_mw', 'q_hv_mvar']])[
                    int(already_used_branches.number.loc[
                        (already_used_branches.hv_bus == to_bus) &
                        (already_used_branches.lv_bus == from_bus)].values)].reshape(1, 4), 0)
                already_used_branches.number.loc[
                    (already_used_branches.hv_bus == to_bus) &
                    (already_used_branches.lv_bus == from_bus)] += 1
    pp_res["branch"] = pp_res["branch"][1:, :]  # delete initial zero row

    # --- do the powerflow result comparison
    diff_res = dict.fromkeys(ppc_elms)
    diff_res["bus"] = ppc_res["bus"] - pp_res["bus"]
    diff_res["bus"][:, 1] -= diff_res["bus"][0, 1]  # remove va_degree offset
    diff_res["branch"] = ppc_res["branch"] - pp_res["branch"]
    diff_res["gen"] = ppc_res["gen"] - pp_res["gen"]
    # comparison of buses with several generator units only as q sum
    for i in GEN_uniq.loc[GEN_uniq[0].isin(change_q_compare)].index:
        next_is = GEN_uniq.index[GEN_uniq.index > i]
        if len(next_is) > 0:
            next_i = next_is[0]
        else:
            next_i = GENS.index[-1] + 1
        if (next_i - i) > 1:
            diff_res["gen"][i:next_i, 1] = sum(diff_res["gen"][i:next_i, 1])
    # logger info
    logger.debug("Maximum voltage magnitude difference between pypower and pandapower: "
                 "%.2e pu" % max_(abs(diff_res["bus"][:, 0])))
    logger.debug("Maximum voltage angle difference between pypower and pandapower: "
                 "%.2e degree" % max_(abs(diff_res["bus"][:, 1])))
    logger.debug("Maximum branch flow active power difference between pypower and pandapower: "
                 "%.2e MW" % max_(abs(diff_res["branch"][:, [0, 2]])))
    logger.debug("Maximum branch flow reactive power difference between pypower and "
                 "pandapower: %.2e MVAr" % max_(abs(diff_res["branch"][:, [1, 3]])))
    logger.debug("Maximum active power generation difference between pypower and pandapower: "
                 "%.2e MW" % max_(abs(diff_res["gen"][:, 0])))
    logger.debug("Maximum reactive power generation difference between pypower and pandapower: "
                 "%.2e MVAr" % max_(abs(diff_res["gen"][:, 1])))
    if _validate_diff_res(diff_res, {"bus_vm_pu": 1e-3, "bus_va_degree": 1e-3, "branch_p_mw": 1e-6,
                                     "branch_q_mvar": 1e-6}) and \
            (max_(abs(diff_res["gen"])) > 1e-1).any():
        logger.debug("The active/reactive power generation difference possibly results "
                     "because of a pypower error. Please validate "
                     "the results via pypower loadflow.")  # this occurs e.g. at ppc case9
    # give a return
    if isinstance(max_diff_values, dict):
        return _validate_diff_res(diff_res, max_diff_values)
    else:
        logger.debug("'max_diff_values' must be a dict.")
