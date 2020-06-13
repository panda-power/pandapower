# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Quadratic Program Solver for PYPOWER.
"""

import sys

from pandapower.pypower.qps_pips import qps_pips
#from pandapower.pypower.qps_ipopt import qps_ipopt
#from pandapower.pypower.qps_cplex import qps_cplex
#from pandapower.pypower.qps_mosek import qps_mosek
#from pandapower.pypower.qps_gurobi import qps_gurobi

from pandapower.pypower.util import have_fcn


def qps_pypower(H, c=None, A=None, l=None, u=None, xmin=None, xmax=None,
                x0=None, opt=None):
    """Quadratic Program Solver for PYPOWER.

    A common wrapper function for various QP solvers.
    Solves the following QP (quadratic programming) problem::

        min 1/2 x'*H*x + c'*x
         x

    subject to::

        l <= A*x <= u       (linear constraints)
        xmin <= x <= xmax   (variable bounds)

    Inputs (all optional except C{H}, C{c}, C{A} and C{l}):
        - C{H} : matrix (possibly sparse) of quadratic cost coefficients
        - C{c} : vector of linear cost coefficients
        - C{A, l, u} : define the optional linear constraints. Default
        values for the elements of C{l} and C{u} are -Inf and Inf,
        respectively.
        - C{xmin}, C{xmax} : optional lower and upper bounds on the
        C{x} variables, defaults are -Inf and Inf, respectively.
        - C{x0} : optional starting value of optimization vector C{x}
        - C{opt} : optional options structure with the following fields,
        all of which are also optional (default values shown in parentheses)
            - C{alg} (0) - determines which solver to use
                -   0 = automatic, first available of BPMPD_MEX, CPLEX,
                        Gurobi, PIPS
                - 100 = BPMPD_MEX
                - 200 = PIPS, Python Interior Point Solver
                pure Python implementation of a primal-dual
                interior point method
                - 250 = PIPS-sc, a step controlled variant of PIPS
                - 300 = Optimization Toolbox, QUADPROG or LINPROG
                - 400 = IPOPT
                - 500 = CPLEX
                - 600 = MOSEK
                - 700 = Gurobi
            - C{verbose} (0) - controls level of progress output displayed
                - 0 = no progress output
                - 1 = some progress output
                - 2 = verbose progress output
            - C{max_it} (0) - maximum number of iterations allowed
                - 0 = use algorithm default
            - C{bp_opt} - options vector for BP
            - C{cplex_opt} - options dict for CPLEX
            - C{grb_opt}   - options dict for gurobipy
            - C{ipopt_opt} - options dict for IPOPT
            - C{pips_opt}  - options dict for L{qps_pips}
            - C{mosek_opt} - options dict for MOSEK
            - C{ot_opt}    - options dict for QUADPROG/LINPROG
        - C{problem} : The inputs can alternatively be supplied in a single
        C{problem} dict with fields corresponding to the input arguments
        described above: C{H, c, A, l, u, xmin, xmax, x0, opt}

    Outputs:
        - C{x} : solution vector
        - C{f} : final objective function value
        - C{exitflag} : exit flag
            - 1 = converged
            - 0 or negative values = algorithm specific failure codes
        - C{output} : output struct with the following fields:
            - C{alg} - algorithm code of solver used
            - (others) - algorithm specific fields
        - C{lmbda} : dict containing the Langrange and Kuhn-Tucker
        multipliers on the constraints, with fields:
            - C{mu_l} - lower (left-hand) limit on linear constraints
            - C{mu_u} - upper (right-hand) limit on linear constraints
            - C{lower} - lower bound on optimization variables
            - C{upper} - upper bound on optimization variables


    Example from U{http://www.uc.edu/sashtml/iml/chap8/sect12.htm}:

        >>> from numpy import array, zeros, Inf
        >>> from scipy.sparse import csr_matrix
        >>> H = csr_matrix(array([[1003.1,  4.3,     6.3,     5.9],
        ...                       [4.3,     2.2,     2.1,     3.9],
        ...                       [6.3,     2.1,     3.5,     4.8],
        ...                       [5.9,     3.9,     4.8,     10 ]]))
        >>> c = zeros(4)
        >>> A = csr_matrix(array([[1,       1,       1,       1   ],
        ...                       [0.17,    0.11,    0.10,    0.18]]))
        >>> l = array([1, 0.10])
        >>> u = array([1, Inf])
        >>> xmin = zeros(4)
        >>> xmax = None
        >>> x0 = array([1, 0, 0, 1])
        >>> solution = qps_pips(H, c, A, l, u, xmin, xmax, x0)
        >>> round(solution["f"], 11) == 1.09666678128
        True
        >>> solution["converged"]
        True
        >>> solution["output"]["iterations"]
        10

    @author: Ray Zimmerman (PSERC Cornell)
    """
    if opt is None:
        opt = {}
#    if x0 is None:
#        x0 = array([])
#    if xmax is None:
#        xmax = array([])
#    if xmin is None:
#        xmin = array([])

    ## default options
    if 'alg' in opt:
        alg = opt['alg']
    else:
        alg = 0

    if 'verbose' in opt:
        verbose = opt['verbose']
    else:
        verbose = 0

    ##----- call the appropriate solver  -----
    # if alg == 0 or alg == 200 or alg == 250:    ## use MIPS or sc-MIPS
    ## set up options
    if 'pips_opt' in opt:
        pips_opt = opt['pips_opt']
    else:
        pips_opt = {}

    if 'max_it' in opt:
        pips_opt['max_it'] = opt['max_it']

    if alg == 200:
        pips_opt['step_control'] = False
    else:
        pips_opt['step_control'] = True

    pips_opt['verbose'] = verbose

    ## call solver
    x, f, eflag, output, lmbda = \
        qps_pips(H, c, A, l, u, xmin, xmax, x0, pips_opt)
#    elif alg == 400:                    ## use IPOPT
#        x, f, eflag, output, lmbda = \
#            qps_ipopt(H, c, A, l, u, xmin, xmax, x0, opt)
#    elif alg == 500:                    ## use CPLEX
#        x, f, eflag, output, lmbda = \
#            qps_cplex(H, c, A, l, u, xmin, xmax, x0, opt)
#    elif alg == 600:                    ## use MOSEK
#        x, f, eflag, output, lmbda = \
#            qps_mosek(H, c, A, l, u, xmin, xmax, x0, opt)
#    elif 700:                           ## use Gurobi
#        x, f, eflag, output, lmbda = \
#            qps_gurobi(H, c, A, l, u, xmin, xmax, x0, opt)
#     else:
#         print('qps_pypower: {} is not a valid algorithm code\n'.format(alg))

    if 'alg' not in output:
        output['alg'] = alg

    return x, f, eflag, output, lmbda