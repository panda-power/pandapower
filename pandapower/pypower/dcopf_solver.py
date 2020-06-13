# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Solves a DC optimal power flow.
"""

from sys import stderr

from copy import deepcopy

from numpy import \
    array, zeros, ones, any, diag, r_, pi, Inf, isnan, arange, c_, dot

from numpy import flatnonzero as find

from scipy.sparse import vstack, hstack, csr_matrix as sparse

from pandapower.pypower.idx_bus import BUS_TYPE, REF, VA, LAM_P, LAM_Q, MU_VMAX, MU_VMIN
from pandapower.pypower.idx_gen import PG, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN
from pandapower.pypower.idx_brch import PF, PT, QF, QT, RATE_A, MU_SF, MU_ST
from pandapower.pypower.idx_cost import MODEL, POLYNOMIAL, PW_LINEAR, NCOST, COST

from pandapower.pypower.util import sub2ind, have_fcn
#from pandapower.pypower.ipopt_options import ipopt_options
#from pandapower.pypower.cplex_options import cplex_options
#from pandapower.pypower.mosek_options import mosek_options
#from pandapower.pypower.gurobi_options import gurobi_options
from pandapower.pypower.qps_pypower import qps_pypower


def dcopf_solver(om, ppopt, out_opt=None):
    """Solves a DC optimal power flow.

    Inputs are an OPF model object, a PYPOWER options dict and
    a dict containing fields (can be empty) for each of the desired
    optional output fields.

    Outputs are a C{results} dict, C{success} flag and C{raw} output dict.

    C{results} is a PYPOWER case dict (ppc) with the usual baseMVA, bus
    branch, gen, gencost fields, along with the following additional
    fields:
        - C{order}      see 'help ext2int' for details of this field
        - C{x}          final value of optimization variables (internal order)
        - C{f}          final objective function value
        - C{mu}         shadow prices on ...
            - C{var}
                - C{l}  lower bounds on variables
                - C{u}  upper bounds on variables
            - C{lin}
                - C{l}  lower bounds on linear constraints
                - C{u}  upper bounds on linear constraints
        - C{g}          (optional) constraint values
        - C{dg}         (optional) constraint 1st derivatives
        - C{df}         (optional) obj fun 1st derivatives (not yet implemented)
        - C{d2f}        (optional) obj fun 2nd derivatives (not yet implemented)

    C{success} is C{True} if solver converged successfully, C{False} otherwise.

    C{raw} is a raw output dict in form returned by MINOS
        - C{xr}     final value of optimization variables
        - C{pimul}  constraint multipliers
        - C{info}   solver specific termination code
        - C{output} solver specific output information

    @see: L{opf}, L{qps_pypower}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Richard Lincoln
    """
    if out_opt is None:
        out_opt = {}

    ## options
    verbose = ppopt['VERBOSE']
    alg     = ppopt['OPF_ALG_DC']

    if alg == 0:
        if have_fcn('cplex'):        ## use CPLEX by default, if available
            alg = 500
        # MOSEK is currently not supported
#        elif have_fcn('mosek'):      ## if not, then MOSEK, if available
#            alg = 600
        elif have_fcn('gurobi'):     ## if not, then Gurobi, if available
            # Error in Gurobi pypower solver -> Issue with pypower 5.1.4. Gurobi won't work. Using alg 200 instead
            # Reason for failure: In qps_gurobi of pypower len(H) raises Error:
            # TypeError: sparse matrix length is ambiguous; use getnnz() or shape[0]
            # Todo: Fix Gurobi and activate 700 again. ATM: Fallback on 200
            # alg = 700
            alg = 200
            UserWarning("Gurobi not working with pypower 5.1.4")
        else:                        ## otherwise PIPS
            alg = 200

    ## unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"], ppc["gencost"]
    cp = om.get_cost_params()
    N, H, Cw = cp["N"], cp["H"], cp["Cw"]
    fparm = array(c_[cp["dd"], cp["rh"], cp["kk"], cp["mm"]])
    Bf = om.userdata('Bf')
    Pfinj = om.userdata('Pfinj')
    vv, ll, _, _ = om.get_idx()

    ## problem dimensions
    ipol = find(gencost[:, MODEL] == POLYNOMIAL) ## polynomial costs
    ipwl = find(gencost[:, MODEL] == PW_LINEAR)  ## piece-wise linear costs
    nb = bus.shape[0]              ## number of buses
    nl = branch.shape[0]           ## number of branches
    nw = N.shape[0]                ## number of general cost vars, w
    ny = om.getN('var', 'y')       ## number of piece-wise linear costs
    nxyz = om.getN('var')          ## total number of control vars of all types

    ## linear constraints & variable bounds
    A, l, u = om.linear_constraints()
    x0, xmin, xmax = om.getv()

    ## set up objective function of the form: f = 1/2 * X'*HH*X + CC'*X
    ## where X = [x;y;z]. First set up as quadratic function of w,
    ## f = 1/2 * w'*HHw*w + CCw'*w, where w = diag(M) * (N*X - Rhat). We
    ## will be building on the (optionally present) user supplied parameters.

    ## piece-wise linear costs
    any_pwl = int(ny > 0)
    if any_pwl:
        # Sum of y vars.
        Npwl = sparse((ones(ny), (zeros(ny), arange(vv["i1"]["y"], vv["iN"]["y"]))), (1, nxyz))
        Hpwl = sparse((1, 1))
        Cpwl = array([1])
        fparm_pwl = array([[1, 0, 0, 1]])
    else:
        Npwl = None#zeros((0, nxyz))
        Hpwl = None#array([])
        Cpwl = array([])
        fparm_pwl = zeros((0, 4))

    ## quadratic costs
    npol = len(ipol)
    if any(len(gencost[ipol, NCOST] > 3)) and sum(gencost[find(gencost[ipol, NCOST] > 3)][:][NCOST+1:]):
        stderr.write('DC opf cannot handle polynomial costs with higher '
                     'than quadratic order.\n')
    iqdr = find(gencost[ipol, NCOST] == 3)
    ilin = find(gencost[ipol, NCOST] == 2)
    polycf = zeros((npol, 3))         ## quadratic coeffs for Pg
    if len(iqdr) > 0:
        polycf[iqdr, :] = gencost[ipol[iqdr], COST:COST + 3]
    if npol:
        polycf[ilin, 1:3] = gencost[ipol[ilin], COST:COST + 2]
    polycf = dot(polycf, diag([ baseMVA**2, baseMVA, 1]))     ## convert to p.u.
    if npol:
        Npol = sparse((ones(npol), (arange(npol), vv["i1"]["Pg"] + ipol)),
                      (npol, nxyz))  # Pg vars
        Hpol = sparse((2 * polycf[:, 0], (arange(npol), arange(npol))),
                      (npol, npol))
    else:
        Npol = None
        Hpol = None
    Cpol = polycf[:, 1]
    fparm_pol = ones((npol, 1)) * array([[1, 0, 0, 1]])

    ## combine with user costs
    NN = vstack([n for n in [Npwl, Npol, N] if n is not None and n.shape[0] > 0], "csr")
    # FIXME: Zero dimension sparse matrices.
    if (Hpwl is not None) and any_pwl and (npol + nw):
        Hpwl = hstack([Hpwl, sparse((any_pwl, npol + nw))])
    if Hpol is not None:
        if any_pwl and npol:
            Hpol = hstack([sparse((npol, any_pwl)), Hpol])
        if npol and nw:
            Hpol = hstack([Hpol, sparse((npol, nw))])
    if (H is not None) and nw and (any_pwl + npol):
        H = hstack([sparse((nw, any_pwl + npol)), H])
    HHw = vstack([h for h in [Hpwl, Hpol, H] if h is not None and h.shape[0] > 0], "csr")
    CCw = r_[Cpwl, Cpol, Cw]
    ffparm = r_[fparm_pwl, fparm_pol, fparm]

    ## transform quadratic coefficients for w into coefficients for X
    nnw = any_pwl + npol + nw
    M = sparse((ffparm[:, 3], (range(nnw), range(nnw))))
    MR = M * ffparm[:, 1]
    HMR = HHw * MR
    MN = M * NN
    HH = MN.T * HHw * MN
    CC = MN.T * (CCw - HMR)
    C0 = 0.5 * dot(MR, HMR) + sum(polycf[:, 2])  # Constant term of cost.

    ## set up input for QP solver
    opt = {'alg': alg, 'verbose': verbose}
    if (alg == 200) or (alg == 250):
        ## try to select an interior initial point
        Varefs = bus[bus[:, BUS_TYPE] == REF, VA] * (pi / 180.0)

        lb, ub = xmin.copy(), xmax.copy()
        lb[xmin == -Inf] = -1e10   ## replace Inf with numerical proxies
        ub[xmax ==  Inf] =  1e10
        x0 = (lb + ub) / 2;
        # angles set to first reference angle
        x0[vv["i1"]["Va"]:vv["iN"]["Va"]] = Varefs[0]
        if ny > 0:
            ipwl = find(gencost[:, MODEL] == PW_LINEAR)
            # largest y-value in CCV data
            c = gencost.flatten('F')[sub2ind(gencost.shape, ipwl,
                                NCOST + 2 * gencost[ipwl, NCOST])]
            x0[vv["i1"]["y"]:vv["iN"]["y"]] = max(c) + 0.1 * abs(max(c))

        ## set up options
        feastol = ppopt['PDIPM_FEASTOL']
        gradtol = ppopt['PDIPM_GRADTOL']
        comptol = ppopt['PDIPM_COMPTOL']
        costtol = ppopt['PDIPM_COSTTOL']
        max_it  = ppopt['PDIPM_MAX_IT']
        max_red = ppopt['SCPDIPM_RED_IT']
        if feastol == 0:
            feastol = ppopt['OPF_VIOLATION']    ## = OPF_VIOLATION by default
        opt["pips_opt"] = {  'feastol': feastol,
                             'gradtol': gradtol,
                             'comptol': comptol,
                             'costtol': costtol,
                             'max_it':  max_it,
                             'max_red': max_red,
                             'cost_mult': 1  }
#    elif alg == 400:
#        opt['ipopt_opt'] = ipopt_options([], ppopt)
#    elif alg == 500:
#        opt['cplex_opt'] = cplex_options([], ppopt)
#    elif alg == 600:
#        opt['mosek_opt'] = mosek_options([], ppopt)
#    elif alg == 700:
#        ppopt['GRB_OPT'] = 0
#        ppopt['GRB_METHOD'] = "automatic"
#        ppopt['GRB_TIMELIMIT'] = Inf
#        ppopt['GRB_THREADS'] = 0
#        opt['GRB_OPT'] = gurobi_options(None, ppopt)
#    else:
#        raise ValueError("Unrecognised solver [%d]." % alg)

    ##-----  run opf  -----
    x, f, info, output, lmbda = \
            qps_pypower(HH, CC, A, l, u, xmin, xmax, x0, opt)
    success = (info == 1)

    ##-----  calculate return values  -----
    if not any(isnan(x)):
        ## update solution data
        Va = x[vv["i1"]["Va"]:vv["iN"]["Va"]]
        Pg = x[vv["i1"]["Pg"]:vv["iN"]["Pg"]]
        f = f + C0

        ## update voltages & generator outputs
        bus[:, VA] = Va * 180 / pi
        gen[:, PG] = Pg * baseMVA

        ## compute branch flows
        branch[:, [QF, QT]] = zeros((nl, 2))
        branch[:, PF] = (Bf * Va + Pfinj) * baseMVA
        branch[:, PT] = -branch[:, PF]

    ## package up results
    mu_l = lmbda["mu_l"]
    mu_u = lmbda["mu_u"]
    muLB = lmbda["lower"]
    muUB = lmbda["upper"]

    ## update Lagrange multipliers
    il = find((branch[:, RATE_A] != 0) & (branch[:, RATE_A] < 1e10))
    bus[:, [LAM_P, LAM_Q, MU_VMIN, MU_VMAX]] = zeros((nb, 4))
    gen[:, [MU_PMIN, MU_PMAX, MU_QMIN, MU_QMAX]] = zeros((gen.shape[0], 4))
    branch[:, [MU_SF, MU_ST]] = zeros((nl, 2))
    bus[:, LAM_P]       = (mu_u[ll["i1"]["Pmis"]:ll["iN"]["Pmis"]] -
                           mu_l[ll["i1"]["Pmis"]:ll["iN"]["Pmis"]]) / baseMVA
    branch[il, MU_SF]   = mu_u[ll["i1"]["Pf"]:ll["iN"]["Pf"]] / baseMVA
    branch[il, MU_ST]   = mu_u[ll["i1"]["Pt"]:ll["iN"]["Pt"]] / baseMVA
    gen[:, MU_PMIN]     = muLB[vv["i1"]["Pg"]:vv["iN"]["Pg"]] / baseMVA
    gen[:, MU_PMAX]     = muUB[vv["i1"]["Pg"]:vv["iN"]["Pg"]] / baseMVA

    pimul = r_[
      mu_l - mu_u,
     -ones((ny)), ## dummy entry corresponding to linear cost row in A
      muLB - muUB
    ]

    mu = { 'var': {'l': muLB, 'u': muUB},
           'lin': {'l': mu_l, 'u': mu_u} }

    results = deepcopy(ppc)
    results["bus"], results["branch"], results["gen"], \
        results["om"], results["x"], results["mu"], results["f"] = \
            bus, branch, gen, om, x, mu, f

    raw = {'xr': x, 'pimul': pimul, 'info': info, 'output': output}

    return results, success, raw
