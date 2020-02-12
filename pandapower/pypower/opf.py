# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

"""Solves an optimal power flow.
"""

from time import time

from numpy import zeros, c_, shape
from pandapower.pypower.idx_brch import MU_ANGMAX
from pandapower.pypower.idx_bus import MU_VMIN
from pandapower.pypower.idx_gen import MU_QMIN

from pandapower.pypower.opf_args import opf_args2
from pandapower.pypower.opf_execute import opf_execute
from pandapower.pypower.opf_setup import opf_setup


def opf(ppc, ppopt):
    """Solves an optimal power flow.

    Returns a C{results} dict.

    The data for the problem can be specified in one of three ways:
      1. a string (ppc) containing the file name of a PYPOWER case
      which defines the data matrices baseMVA, bus, gen, branch, and
      gencost (areas is not used at all, it is only included for
      backward compatibility of the API).
      2. a dict (ppc) containing the data matrices as fields.
      3. the individual data matrices themselves.

    The optional user parameters for user constraints (C{A, l, u}), user costs
    (C{N, fparm, H, Cw}), user variable initializer (C{z0}), and user variable
    limits (C{zl, zu}) can also be specified as fields in a case dict,
    either passed in directly or defined in a case file referenced by name.

    When specified, C{A, l, u} represent additional linear constraints on the
    optimization variables, C{l <= A*[x z] <= u}. If the user specifies an C{A}
    matrix that has more columns than the number of "C{x}" (OPF) variables,
    then there are extra linearly constrained "C{z}" variables. For an
    explanation of the formulation used and instructions for forming the
    C{A} matrix, see the MATPOWER manual.

    A generalized cost on all variables can be applied if input arguments
    C{N}, C{fparm}, C{H} and C{Cw} are specified. First, a linear transformation
    of the optimization variables is defined by means of C{r = N * [x z]}.
    Then, to each element of C{r} a function is applied as encoded in the
    C{fparm} matrix (see MATPOWER manual). If the resulting vector is named
    C{w}, then C{H} and C{Cw} define a quadratic cost on w:
    C{(1/2)*w'*H*w + Cw * w}. C{H} and C{N} should be sparse matrices and C{H}
    should also be symmetric.

    The optional C{ppopt} vector specifies PYPOWER options. If the OPF
    algorithm is not explicitly set in the options PYPOWER will use the default
    solver, based on a primal-dual interior point method. For the AC OPF this
    is C{OPF_ALG = 560}. For the DC OPF, the default is C{OPF_ALG_DC = 200}.
    See L{ppoption} for more details on the available OPF solvers and other OPF
    options and their default values.

    The solved case is returned in a single results dict (described
    below). Also returned are the final objective function value (C{f}) and a
    flag which is C{True} if the algorithm was successful in finding a solution
    (success). Additional optional return values are an algorithm specific
    return status (C{info}), elapsed time in seconds (C{et}), the constraint
    vector (C{g}), the Jacobian matrix (C{jac}), and the vector of variables
    (C{xr}) as well as the constraint multipliers (C{pimul}).

    The single results dict is a PYPOWER case struct (ppc) with the
    usual baseMVA, bus, branch, gen, gencost fields, along with the
    following additional fields:

        - C{order}      see 'help ext2int' for details of this field
        - C{et}         elapsed time in seconds for solving OPF
        - C{success}    1 if solver converged successfully, 0 otherwise
        - C{om}         OPF model object, see 'help opf_model'
        - C{x}          final value of optimization variables (internal order)
        - C{f}          final objective function value
        - C{mu}         shadow prices on ...
            - C{var}
                - C{l}  lower bounds on variables
                - C{u}  upper bounds on variables
            - C{nln}
                - C{l}  lower bounds on nonlinear constraints
                - C{u}  upper bounds on nonlinear constraints
            - C{lin}
                - C{l}  lower bounds on linear constraints
                - C{u}  upper bounds on linear constraints
        - C{g}          (optional) constraint values
        - C{dg}         (optional) constraint 1st derivatives
        - C{df}         (optional) obj fun 1st derivatives (not yet implemented)
        - C{d2f}        (optional) obj fun 2nd derivatives (not yet implemented)
        - C{raw}        raw solver output in form returned by MINOS, and more
            - C{xr}     final value of optimization variables
            - C{pimul}  constraint multipliers
            - C{info}   solver specific termination code
            - C{output} solver specific output information
               - C{alg} algorithm code of solver used
        - C{var}
            - C{val}    optimization variable values, by named block
                - C{Va}     voltage angles
                - C{Vm}     voltage magnitudes (AC only)
                - C{Pg}     real power injections
                - C{Qg}     reactive power injections (AC only)
                - C{y}      constrained cost variable (only if have pwl costs)
                - (other) any user defined variable blocks
            - C{mu}     variable bound shadow prices, by named block
                - C{l}  lower bound shadow prices
                    - C{Va}, C{Vm}, C{Pg}, C{Qg}, C{y}, (other)
                - C{u}  upper bound shadow prices
                    - C{Va}, C{Vm}, C{Pg}, C{Qg}, C{y}, (other)
        - C{nln}    (AC only)
            - C{mu}     shadow prices on nonlinear constraints, by named block
                - C{l}  lower bounds
                    - C{Pmis}   real power mismatch equations
                    - C{Qmis}   reactive power mismatch equations
                    - C{Sf}     flow limits at "from" end of branches
                    - C{St}     flow limits at "to" end of branches
                - C{u}  upper bounds
                    - C{Pmis}, C{Qmis}, C{Sf}, C{St}
        - C{lin}
            - C{mu}     shadow prices on linear constraints, by named block
                - C{l}  lower bounds
                    - C{Pmis}   real power mistmatch equations (DC only)
                    - C{Pf}     flow limits at "from" end of branches (DC only)
                    - C{Pt}     flow limits at "to" end of branches (DC only)
                    - C{PQh}    upper portion of gen PQ-capability curve(AC only)
                    - C{PQl}    lower portion of gen PQ-capability curve(AC only)
                    - C{vl}     constant power factor constraint for loads
                    - C{ycon}   basin constraints for CCV for pwl costs
                    - (other) any user defined constraint blocks
                - C{u}  upper bounds
                    - C{Pmis}, C{Pf}, C{Pf}, C{PQh}, C{PQl}, C{vl}, C{ycon},
                    - (other)
        - C{cost}       user defined cost values, by named block

    @see: L{runopf}, L{dcopf}, L{uopf}, L{caseformat}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Richard Lincoln
    """
    ##----- initialization -----
    t0 = time()         ## start timer

    ## process input arguments
    ppc, ppopt = opf_args2(ppc, ppopt)

    ## add zero columns to bus, gen, branch for multipliers, etc if needed
    nb   = shape(ppc['bus'])[0]    ## number of buses
    nl   = shape(ppc['branch'])[0] ## number of branches
    ng   = shape(ppc['gen'])[0]    ## number of dispatchable injections
    if shape(ppc['bus'])[1] < MU_VMIN + 1:
        ppc['bus'] = c_[ppc['bus'], zeros((nb, MU_VMIN + 1 - shape(ppc['bus'])[1]))]

    if shape(ppc['gen'])[1] < MU_QMIN + 1:
        ppc['gen'] = c_[ppc['gen'], zeros((ng, MU_QMIN + 1 - shape(ppc['gen'])[1]))]

    if shape(ppc['branch'])[1] < MU_ANGMAX + 1:
        ppc['branch'] = c_[ppc['branch'], zeros((nl, MU_ANGMAX + 1 - shape(ppc['branch'])[1]))]

    ##-----  convert to internal numbering, remove out-of-service stuff  -----
    # ppc = ext2int(ppc)

    ##-----  construct OPF model object  -----
    om = opf_setup(ppc, ppopt)

    ##-----  execute the OPF  -----
    results, success, raw = opf_execute(om, ppopt)

    ##-----  revert to original ordering, including out-of-service stuff  -----
    # results = int2ext(results)

    ## zero out result fields of out-of-service gens & branches
    # if len(results['order']['gen']['status']['off']) > 0:
    #     results['gen'][ ix_(results['order']['gen']['status']['off'], [PG, QG, MU_PMAX, MU_PMIN]) ] = 0
    #
    # if len(results['order']['branch']['status']['off']) > 0:
    #     results['branch'][ ix_(results['order']['branch']['status']['off'], [PF, QF, PT, QT, MU_SF, MU_ST, MU_ANGMIN, MU_ANGMAX]) ] = 0

    ##-----  finish preparing output  -----
    et = time() - t0      ## compute elapsed time

    results['et'] = et
    results['success'] = success
    results['raw'] = raw

    return results
