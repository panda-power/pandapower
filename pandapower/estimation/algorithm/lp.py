# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.optimize import linprog

from pandapower.estimation.algorithm.matrix_base import BaseAlgebra
from pandapower.estimation.algorithm.base import BaseAlgorithm


class LPAlgorithm(BaseAlgorithm):
    def estimate(self, eppci, **kwargs):
        if "estimator" in kwargs and kwargs["estimator"].lower() != "lav": # pragma: no cover
            self.logger.warning("LP Algorithm supports only LAV Estimator!! Set to LAV!!")

        # matrix calculation object
        sem = BaseAlgebra(eppci)

        current_error, cur_it = 100., 0
        E = eppci.E
        while current_error > self.tolerance and cur_it < self.max_iterations:
            self.logger.debug("Starting iteration {:d}".format(1 + cur_it))
            try:
                # residual r
                r = sem.create_rx(E)

                # jacobian matrix H
                H = sem.create_hx_jacobian(E)

                # state vector difference d_E
                # d_E = G_m^-1 * (H' * R^-1 * r)
                d_E = self.solve_lp(H, E, r)
                E += d_E
                eppci.update_E(E)

                # prepare next iteration
                cur_it += 1
                current_error = np.max(np.abs(d_E))
                self.logger.debug("Current error: {:.7f}".format(current_error))
            except np.linalg.linalg.LinAlgError: # pragma: no cover
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

        # check if the estimation is successfull
        self.check_result(current_error, cur_it)
        return eppci

    def solve_lp(self, H, x, r):
        n, m = H.shape[1], H.shape[0]
        zero_n = np.zeros((n, 1))
        one_m = np.ones((m, 1))
        Im = np.eye(m)

        c_T = np.r_[zero_n, zero_n, one_m, one_m] 
        A = np.c_[H, -H, Im, -Im]

        res = linprog(c_T.ravel(), A_eq=A, b_eq=r,
                      method="simplex", options={'tol': 1e-5, 'disp': True, 'maxiter':20000})
        if res.success:
            d_x = np.array(res['x'][:n]).ravel() - np.array(res['x'][n:2*n]).ravel()
            return d_x
        else: # pragma: no cover
            raise np.linalg.linalg.LinAlgError 

