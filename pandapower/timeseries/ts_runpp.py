import inspect
import collections
import functools

import numpy as np
from numpy import complex128, zeros

import pandapower as pp
import pandapower.pf.run_newton_raphson_pf as nr_pf

from pandapower.control.controller.const_control import ConstControl
from pandapower.control.controller.trafo_control import TrafoController
from pandapower.auxiliary import _clean_up
from pandapower.build_branch import _calc_trafo_parameter, _calc_trafo3w_parameter
from pandapower.build_bus import _calc_pq_elements_and_add_on_ppc, \
    _calc_shunts_and_add_on_ppc
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, TAP, SHIFT, BR_STATUS, RATE_A
from pandapower.pypower.idx_bus import PD, QD
from pandapower.pd2ppc import _pd2ppc
from pandapower.pypower.makeSbus import _get_Sbus, _get_Cg, makeSbus
from pandapower.pf.pfsoln_numba import pfsoln as pfsoln_full, pf_solution_single_slack
from pandapower.powerflow import LoadflowNotConverged, _add_auxiliary_elements
from pandapower.results import _copy_results_ppci_to_ppc, _extract_results, _get_aranged_lookup
from pandapower.results_branch import _get_branch_flows, _get_line_results, _get_trafo3w_results, _get_trafo_results
from pandapower.results_bus import write_pq_results_to_element, _get_bus_v_results, _get_bus_results
from pandapower.results_gen import _get_gen_results
from pandapower.timeseries.output_writer import OutputWriter

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class TimeSeriesRunpp:
    """
    Class for time series runpp.
    The goal of this class is a runpp function, which reuses many NR variables in each time step of a time series
    calculation.
    Also not all powerflow results are calculated in each runpp. Instead only the results the output writer wants
    are calculated. Therefore this class must be initiliazed with an output writer
    """

    def __init__(self, net):
        self.net = net
        self.output_writer = net.output_writer.iat[0, 0]

        # update functions
        self.update_pq = False
        self.update_trafo = False
        self.init_timeseries_newton()

    def ts_newtonpf(self, net):
        options = net["_options"]

        bus = self.ppci["bus"]
        branch = self.ppci["branch"]
        gen = self.ppci["gen"]
        # compute complex bus power injections [generation - load]
        # self.Cg = _get_Cg(gen_on, bus)
        # Sbus = _get_Sbus(self.baseMVA, bus, gen, self.Cg)
        Sbus = makeSbus(self.baseMVA, bus, gen)

        # run the newton power  flow
        V, success, _, _, _, _ = nr_pf.newtonpf(self.Ybus, Sbus, self.V, self.pv, self.pq, self.ppci, options)

        if not success:
            logger.warning("Loadflow not converged")
            logger.info("Lines of of service:")
            logger.info(net.line[~net.line.in_service])
            raise LoadflowNotConverged("Power Flow did not converge after")

        if self.ppci["gen"].shape[0] == 1 and not options["voltage_depend_loads"]:
            pfsoln = pf_solution_single_slack
        else:
            pfsoln = pfsoln_full

        bus, gen, branch = pfsoln(self.baseMVA, bus, gen, branch, self.Ybus, self.Yf, self.Yt, V, self.ref,
                                  self.ref_gens, Ibus=self.Ibus)

        self.ppci["bus"] = bus
        self.ppci["branch"] = branch
        self.ppci["gen"] = gen
        self.ppci["success"] = success
        self.ppci["et"] = None

        # ppci doesn't contain out of service elements, but ppc does -> copy results accordingly
        self.ppc = _copy_results_ppci_to_ppc(self.ppci, self.ppc, options["mode"])

        # raise if PF was not successful. If DC -> success is always 1
        if self.ppc["success"] != 1:
            _clean_up(net, res=False)
        else:
            net["_ppc"] = self.ppc
            net["converged"] = True

        self.V = V

        _extract_results(net, self.ppc)

        return net

    def _get_bus_p_q_results_from_ppc(self, net, ppc, net_bus_idx, ppc_bus_idx):
        """
        reads p, q results from ppc to net. Note: This function returns wrong vales if shunts, xwards and wards are in net
        @param net:
        @param ppc:
        @return:
        """
        # read bus_pq array which contains p and q values for each bus in net
        bus_pq = np.zeros(shape=(len(net["bus"].index), 2), dtype=np.float)
        bus_pq[net_bus_idx, 0] = ppc["bus"][ppc_bus_idx, PD] * 1e3
        bus_pq[net_bus_idx, 1] = ppc["bus"][ppc_bus_idx, QD] * 1e3

        bus_lookup_aranged = _get_aranged_lookup(net)
        _get_gen_results(net, ppc, bus_lookup_aranged, bus_pq)
        _get_bus_results(net, ppc, bus_pq)

        net["res_bus"].index = net["bus"].index

    def get_branch_results(self, net, ppc, branch_update):
        i_ft, s_ft = _get_branch_flows(ppc)
        if "line" in branch_update:
            _get_line_results(net, ppc, i_ft)
        if "trafo" in branch_update:
            _get_trafo_results(net, ppc, s_ft, i_ft)
        if "trafo3w" in branch_update:
            _get_trafo3w_results(net, ppc, s_ft, i_ft)

    def init_timeseries_newton(self):
        """
        This function is called in the first iteration and variables needed in every loop are stored (Ybus, ppci...)
        @param net:
        @return:
        """
        self.init_newton_variables()
        net = self.net
        # get ppc and ppci
        # pp.runpp(net, init_vm_pu="flat", init_va_degree="dc")
        # pp.runpp(net, init_vm_pu="results", init_va_degree="results")
        pp.runpp(net, init="dc")
        pp.runpp(net, init="results")
        net._options["init_results"] = True
        net._options["init_vm_pu"] = "results"
        net._options["init_va_degree"] = "results"
        options = net._options
        _add_auxiliary_elements(net)
        self.ppc, self.ppci = _pd2ppc(net)
        net["_ppc"] = self.ppc

        self.baseMVA, bus, gen, branch, self.ref, self.pv, self.pq, _, _, self.V, self.ref_gens = \
            nr_pf._get_pf_variables_from_ppci(self.ppci)
        self.ppci, self.Ybus, self.Yf, self.Yt = \
            nr_pf._get_Y_bus(self.ppci, options, nr_pf.makeYbus_numba, self.baseMVA, bus, branch)
        self.Ibus = zeros(len(self.V), dtype=complex128)

        # self.Cg = _get_Cg(gen, bus)  # assumes that all gens are on!

        if "controller" in net:
            self.get_update_ctrl()

        return net

    def _update_nr_variables(self):
        """
        This function updates the stored NR variables for the powerflow, depending on the defined controllers for the
        time series calculation

        example:
        If only PQ updates are needed -> only pq values (in ppci.bus[:,PD]...) are updated
        If taps of trafos are updated -> Ybus must be recalculated
        Todo implement stuff for the controllers you use...

        @return:
        """
        ### PQ Updates
        if self.update_pq:
            # update P, Q values
            _calc_pq_elements_and_add_on_ppc(self.net, self.ppci)
            # adds P and Q for shunts, wards and xwards (to PQ nodes)
            # Todo: update shunts
            # _calc_shunts_and_add_on_ppc(self.net, self.ppci)

        ### Ybus recalculation
        if self.update_trafo:
            # update branch SHIFT entries for transformers (if tap changed)
            self.update_trafos()
            if "ybus_handler" in self.net:
                logger.error("Ybus update by trafo tap controller and ybus_handler is not possible simultaneously")

        if "ybus_handler" in self.net:
            self.Ybus, self.Yf, self.Yt = self.net["ybus_handler"].get_y()

    def cleanup(self):
        _clean_up(self.net)
        self.init_newton_variables()

    def ts_runpp(self, net, **kwargs):
        # update pq values in ppci
        self._update_nr_variables()

        # ----- run the powerflow -----
        net = self.ts_newtonpf(net)

        return net

    def init_newton_variables(self):
        self.baseMVA = None
        self.ref = None
        self.pv = None
        self.pq = None
        self.V = None
        self.ppc = None
        self.ppci = None
        self.Ybus = None
        self.Yf = None
        self.Yt = None
        self.Ibus = None
        self.Cg = None

    def update_trafos(self):
        net = self.net
        ppci = self.ppci

        # update branch SHIFT entries for transfomers (if tap changed)
        lookup = net._pd2ppc_lookups["branch"]
        if "trafo" in lookup:
            _calc_trafo_parameter(net, ppci)
        if "trafo3w" in lookup:
            _calc_trafo3w_parameter(net, ppci)

        # update Ybus based on this
        options = net._options
        baseMVA, bus, gen, branch, ref, pv, pq, _, _, V, _ = nr_pf._get_pf_variables_from_ppci(ppci)
        self.ppci, self.Ybus, self.Yf, self.Yt = nr_pf._get_Y_bus(ppci, options, nr_pf.makeYbus_numba, baseMVA, bus,
                                                                  branch)

    def get_update_ctrl(self):
        controllers = self.net.controller["object"]
        update_pq = False
        update_trafo = False
        for controller in controllers:
            base_clases = inspect.getmro(controller.__class__)
            if ConstControl in base_clases:
                update_pq = True
            elif TrafoController in controller.__class__.__bases__:
                update_trafo = True
            else:
                raise TypeError("controller class not supported for recycle %s" % controller.__class__)

        self.update_pq = update_pq
        self.update_trafo = update_trafo
