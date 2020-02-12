# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from scipy.optimize import minimize

from pandapower.estimation.algorithm.base import BaseAlgorithm
from pandapower.estimation.algorithm.estimator import BaseEstimatorOpt, get_estimator

#DEFAULT_OPT_METHOD = "Newton-CG"
DEFAULT_OPT_METHOD = "TNC"
#DEFAULT_OPT_METHOD = "SLSQP"
#DEFAULT_OPT_METHOD = 'L-BFGS-B'


class OptAlgorithm(BaseAlgorithm):
    def estimate(self, eppci, estimator="wls", **opt_vars):
        opt_method = DEFAULT_OPT_METHOD if 'opt_method' not in opt_vars else opt_vars['opt_method']

        # matrix calculation object
        estm = get_estimator(BaseEstimatorOpt, estimator)(eppci, **opt_vars)

        jac = estm.create_cost_jacobian
        res = minimize(estm.cost_function, x0=eppci.E, 
                       method=opt_method, jac=jac, tol=self.tolerance,
                       options={"disp":True})

        self.successful = res.success
        if self.successful:
            E = res.x
            eppci.update_E(E)
            return eppci
        else:
            raise Exception("Optimiaztion failed! State Estimation not successful!")
