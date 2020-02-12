# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix, vstack, hstack
from scipy.sparse.linalg import spsolve

from pandapower.pypower.idx_bus import BUS_TYPE, VA, VM, bus_cols
from pandapower.estimation.idx_bus import ZERO_INJ_FLAG, P, P_STD, Q, Q_STD
from pandapower.estimation.ppc_conversion import _build_measurement_vectors, ExtendedPPCI

from pandapower.estimation.algorithm.matrix_base import BaseAlgebra,\
    BaseAlgebraZeroInjConstraints
from pandapower.estimation.algorithm.estimator import BaseEstimatorIRWLS, get_estimator
 
try:
    import pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)

__all__ = ["WLSAlgorithm", "WLSZeroInjectionConstraintsAlgorithm", "IRWLSAlgorithm"]


class BaseAlgorithm:
    def __init__(self, tolerance, maximum_iterations, logger=std_logger):
        self.tolerance = tolerance
        self.max_iterations = maximum_iterations
        self.logger = logger
        self.successful = False
        self.iterations = None

        # Parameters for estimate
        self.eppci = None
        self.pp_meas_indices = None

    def check_observability(self, eppci: ExtendedPPCI, z):
        # Check if observability criterion is fulfilled and the state estimation is possible
        if len(z) < 2 * eppci["bus"].shape[0] - 1:
            self.logger.error("System is not observable (cancelling)")
            self.logger.error("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * eppci["bus"].shape[0] - 1))
            raise UserWarning("Measurements available: %d. Measurements required: %d" %
                              (len(z), 2 * eppci["bus"].shape[0] - 1))

    def check_result(self, current_error, cur_it):
        # print output for results
        if current_error <= self.tolerance:
            self.successful = True
            self.logger.debug("State Estimation successful ({:d} iterations)".format(cur_it))
        else:
            self.successful = False
            self.logger.debug("State Estimation not successful ({:d}/{:d} iterations)".format(cur_it,
                                                                                              self.max_iterations))

    def initialize(self, eppci: ExtendedPPCI):
        # Check observability
        self.eppci = eppci
        self.pp_meas_indices = eppci.pp_meas_indices
        self.check_observability(eppci, eppci.z)

    def estimate(self, ppci: ExtendedPPCI, **kwargs):
        # Must be implemented individually!!
        pass 


class WLSAlgorithm(BaseAlgorithm):
    def __init__(self, tolerance, maximum_iterations, logger=std_logger):
        super(WLSAlgorithm, self).__init__(tolerance, maximum_iterations, logger)

        # Parameters for Bad data detection
        self.R_inv = None
        self.Gm = None
        self.r = None
        self.H = None
        self.hx = None

    def estimate(self, eppci, **kwargs):
        self.initialize(eppci)
        # matrix calculation object
        sem = BaseAlgebra(eppci)

        current_error, cur_it = 100., 0
        # invert covariance matrix
        r_inv = csr_matrix(np.diagflat(1/eppci.r_cov ** 2))
        E = eppci.E
        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = csr_matrix(sem.create_rx(E)).T

                # jacobian matrix H
                H = csr_matrix(sem.create_hx_jacobian(E))

                # gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)

                # state vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = spsolve(G_m, H.T * (r_inv * r))

                # Update E with d_E
                E += d_E.ravel()
                eppci.update_E(E)

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E))
                self.logger.debug("Current error: {:.7f}".format(current_error))
            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        if self.successful:
            # store variables required for chi^2 and r_N_max test:
            self.R_inv = r_inv.toarray()
            self.Gm = G_m.toarray()
            self.r = r.toarray()
            self.H = H.toarray()
            # create h(x) for the current iteration
            self.hx = sem.create_hx(eppci.E)
        return eppci


class WLSZeroInjectionConstraintsAlgorithm(BaseAlgorithm):
    def estimate(self, eppci, **kwargs):
        # state vector built from delta, |V| and zero injections
        # Find pq bus with zero p,q and shunt admittance
        zero_injection_bus = np.argwhere(eppci["bus"][:, bus_cols+ZERO_INJ_FLAG] == True).ravel()
        eppci["bus"][zero_injection_bus, [bus_cols+P, bus_cols+P_STD, bus_cols+Q, bus_cols+Q_STD]] = np.NaN
        # Withn pq buses with zero injection identify those who have also no p or q measurement
        p_zero_injections = zero_injection_bus
        q_zero_injections = zero_injection_bus
        new_states = np.zeros(len(p_zero_injections) + len(q_zero_injections))

        num_bus = eppci["bus"].shape[0]

        # matrix calculation object
        sem = BaseAlgebraZeroInjConstraints(eppci)

        current_error, cur_it = 100., 0
        r_inv = csr_matrix((np.diagflat(1/eppci.r_cov) ** 2))
        E = eppci.E
        # update the E matrix
        E_ext = np.r_[eppci.E, new_states]

        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                c_x = sem.create_cx(E, p_zero_injections, q_zero_injections)

                # residual r
                r = csr_matrix(sem.create_rx(E)).T
                c_rxh = csr_matrix(c_x).T

                # jacobian matrix H
                H_temp = sem.create_hx_jacobian(E)
                C_temp = sem.create_cx_jacobian(E, p_zero_injections, q_zero_injections)
                H, C = csr_matrix(H_temp), csr_matrix(C_temp)

                # gain matrix G_m
                # G_m = H^t * R^-1 * H
                G_m = H.T * (r_inv * H)

                # building a new gain matrix for new constraints.
                A_1 = vstack([G_m, C])
                c_ax = hstack([C, np.zeros((C.shape[0], C.shape[0]))])
                c_xT = c_ax.T
                M_tx = csr_matrix(hstack((A_1, c_xT)))  # again adding to the new gain matrix
                rhs = H.T * (r_inv * r)  # original right hand side
                C_rhs = vstack((rhs, -c_rxh))  # creating the righ hand side with new constraints

                # state vector difference d_E and update E
                d_E_ext = spsolve(M_tx, C_rhs)
                E_ext += d_E_ext.ravel()
                E = E_ext[:E.shape[0]]
                eppci.update_E(E)

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E_ext[:len(eppci.non_slack_buses) + num_bus]))
                self.logger.debug("Current error: {:.7f}".format(current_error))
            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        return eppci
    

class IRWLSAlgorithm(BaseAlgorithm):
    def estimate(self, eppci, estimator="wls", **kwargs):
        self.initialize(eppci)

        # matrix calculation object
        sem = get_estimator(BaseEstimatorIRWLS, estimator)(eppci, **kwargs)

        current_error, cur_it = 100., 0
        E = eppci.E
        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = csr_matrix(sem.create_rx(E)).T

                # jacobian matrix H
                H = csr_matrix(sem.create_hx_jacobian(E))

                # gain matrix G_m
                # G_m = H^t * Phi * H
                phi = csr_matrix(sem.create_phi(E))
                G_m = H.T * (phi * H)

                # state vector difference d_E and update E
                d_E = spsolve(G_m, H.T * (phi * r))
                E += d_E.ravel()
                eppci.update_E(E)

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E))
                self.logger.debug("Current error: {:.7f}".format(current_error))
            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        # update V/delta
        return eppci

