# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Python Interior Point Solver (PIPS).
"""

from numpy import array, Inf, any, isnan, ones, r_, finfo, \
    zeros, dot, absolute, log, flatnonzero as find
from numpy.linalg import norm
from pandapower.pypower.pipsver import pipsver
from scipy.sparse import vstack, hstack, eye, csr_matrix as sparse
from scipy.sparse.linalg import spsolve


EPS = finfo(float).eps


def pips(f_fcn, x0=None, A=None, l=None, u=None, xmin=None, xmax=None,
         gh_fcn=None, hess_fcn=None, opt=None):
    """Primal-dual interior point method for NLP (nonlinear programming).
    Minimize a function F(X) beginning from a starting point M{x0}, subject to
    optional linear and nonlinear constraints and variable bounds::

            min f(x)
             x

    subject to::

            g(x) = 0            (nonlinear equalities)
            h(x) <= 0           (nonlinear inequalities)
            l <= A*x <= u       (linear constraints)
            xmin <= x <= xmax   (variable bounds)

    Note: The calling syntax is almost identical to that of FMINCON from
    MathWorks' Optimization Toolbox. The main difference is that the linear
    constraints are specified with C{A}, C{L}, C{U} instead of C{A}, C{B},
    C{Aeq}, C{Beq}. The functions for evaluating the objective function,
    constraints and Hessian are identical.

    Example from U{http://en.wikipedia.org/wiki/Nonlinear_programming}:
        >>> from numpy import array, r_, float64, dot
        >>> from scipy.sparse import csr_matrix
        >>> def f2(x):
        ...     f = -x[0] * x[1] - x[1] * x[2]
        ...     df = -r_[x[1], x[0] + x[2], x[1]]
        ...     # actually not used since 'hess_fcn' is provided
        ...     d2f = -array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float64)
        ...     return f, df, d2f
        >>> def gh2(x):
        ...     h = dot(array([[1, -1, 1],
        ...                    [1,  1, 1]]), x**2) + array([-2.0, -10.0])
        ...     dh = 2 * csr_matrix(array([[ x[0], x[0]],
        ...                                [-x[1], x[1]],
        ...                                [ x[2], x[2]]]))
        ...     g = array([])
        ...     dg = None
        ...     return h, g, dh, dg
        >>> def hess2(x, lam, cost_mult=1):
        ...     mu = lam["ineqnonlin"]
        ...     a = r_[dot(2 * array([1, 1]), mu), -1, 0]
        ...     b = r_[-1, dot(2 * array([-1, 1]), mu),-1]
        ...     c = r_[0, -1, dot(2 * array([1, 1]), mu)]
        ...     Lxx = csr_matrix(array([a, b, c]))
        ...     return Lxx
        >>> x0 = array([1, 1, 0], float64)
        >>> solution = pips(f2, x0, gh_fcn=gh2, hess_fcn=hess2)
        >>> round(solution["f"], 11) == -7.07106725919
        True
        >>> solution["output"]["iterations"]
        8

    Ported by Richard Lincoln from the MATLAB Interior Point Solver (MIPS)
    (v1.9) by Ray Zimmerman.  MIPS is distributed as part of the MATPOWER
    project, developed at the Power System Engineering Research Center (PSERC) (PSERC),
    Cornell. See U{http://www.pserc.cornell.edu/matpower/} for more info.
    MIPS was ported by Ray Zimmerman from C code written by H. Wang for his
    PhD dissertation:
      - "On the Computation and Application of Multi-period
        Security-Constrained Optimal Power Flow for Real-time
        Electricity Market Operations", Cornell University, May 2007.

    See also:
      - H. Wang, C. E. Murillo-Sanchez, R. D. Zimmerman, R. J. Thomas,
        "On Computational Issues of Market-Based Optimal Power Flow",
        IEEE Transactions on Power Systems, Vol. 22, No. 3, Aug. 2007,
        pp. 1185-1193.

    All parameters are optional except C{f_fcn} and C{x0}.
    @param f_fcn: Function that evaluates the objective function, its gradients
                  and Hessian for a given value of M{x}. If there are
                  nonlinear constraints, the Hessian information is provided
                  by the 'hess_fcn' argument and is not required here.
    @type f_fcn: callable
    @param x0: Starting value of optimization vector M{x}.
    @type x0: array
    @param A: Optional linear constraints.
    @type A: csr_matrix
    @param l: Optional linear constraints. Default values are M{-Inf}.
    @type l: array
    @param u: Optional linear constraints. Default values are M{Inf}.
    @type u: array
    @param xmin: Optional lower bounds on the M{x} variables, defaults are
                 M{-Inf}.
    @type xmin: array
    @param xmax: Optional upper bounds on the M{x} variables, defaults are
                 M{Inf}.
    @type xmax: array
    @param gh_fcn: Function that evaluates the optional nonlinear constraints
                   and their gradients for a given value of M{x}.
    @type gh_fcn: callable
    @param hess_fcn: Handle to function that computes the Hessian of the
                     Lagrangian for given values of M{x}, M{lambda} and M{mu},
                     where M{lambda} and M{mu} are the multipliers on the
                     equality and inequality constraints, M{g} and M{h},
                     respectively.
    @type hess_fcn: callable
    @param opt: optional options dictionary with the following keys, all of
                which are also optional (default values shown in parentheses)
                  - C{verbose} (False) - Controls level of progress output
                    displayed
                  - C{feastol} (1e-6) - termination tolerance for feasibility
                    condition
                  - C{gradtol} (1e-6) - termination tolerance for gradient
                    condition
                  - C{comptol} (1e-6) - termination tolerance for
                    complementarity condition
                  - C{costtol} (1e-6) - termination tolerance for cost
                    condition
                  - C{max_it} (150) - maximum number of iterations
                  - C{step_control} (False) - set to True to enable step-size
                    control
                  - C{max_red} (20) - maximum number of step-size reductions if
                    step-control is on
                  - C{cost_mult} (1.0) - cost multiplier used to scale the
                    objective function for improved conditioning. Note: This
                    value is also passed as the 3rd argument to the Hessian
                    evaluation function so that it can appropriately scale the
                    objective function term in the Hessian of the Lagrangian.
    @type opt: dict

    @rtype: dict
    @return: The solution dictionary has the following keys:
               - C{x} - solution vector
               - C{f} - final objective function value
               - C{converged} - exit status
                   - True = first order optimality conditions satisfied
                   - False = maximum number of iterations reached
                   - None = numerically failed
               - C{output} - output dictionary with keys:
                   - C{iterations} - number of iterations performed
                   - C{hist} - list of arrays with trajectories of the
                     following: feascond, gradcond, compcond, costcond, gamma,
                     stepsize, obj, alphap, alphad
                   - C{message} - exit message
               - C{lmbda} - dictionary containing the Langrange and Kuhn-Tucker
                 multipliers on the constraints, with keys:
                   - C{eqnonlin} - nonlinear equality constraints
                   - C{ineqnonlin} - nonlinear inequality constraints
                   - C{mu_l} - lower (left-hand) limit on linear constraints
                   - C{mu_u} - upper (right-hand) limit on linear constraints
                   - C{lower} - lower bound on optimization variables
                   - C{upper} - upper bound on optimization variables

    @see: U{http://www.pserc.cornell.edu/matpower/}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln
    """
    if isinstance(f_fcn, dict):  ## problem dict
        p = f_fcn
        f_fcn = p['f_fcn']
        x0 = p['x0']
        if 'opt' in p: opt = p['opt']
        if 'hess_fcn' in p: hess_fcn = p['hess_fcn']
        if 'gh_fcn' in p: gh_fcn = p['gh_fcn']
        if 'xmax' in p: xmax = p['xmax']
        if 'xmin' in p: xmin = p['xmin']
        if 'u' in p: u = p['u']
        if 'l' in p: l = p['l']
        if 'A' in p: A = p['A']

    nx = x0.shape[0]                        # number of variables
    nA = A.shape[0] if A is not None else 0 # number of original linear constr

    # default argument values
    if l is None or len(l) == 0: l = -Inf * ones(nA)
    if u is None or len(u) == 0: u =  Inf * ones(nA)
    if xmin is None or len(xmin) == 0: xmin = -Inf * ones(x0.shape[0])
    if xmax is None or len(xmax) == 0: xmax =  Inf * ones(x0.shape[0])
    if gh_fcn is None:
        nonlinear = False
        gn = array([])
        hn = array([])
    else:
        nonlinear = True

    if opt is None: opt = {}
    # options
    if "feastol" not in opt:
        opt["feastol"] = 1e-06
    if "gradtol" not in opt:
        opt["gradtol"] = 1e-06
    if "comptol" not in opt:
        opt["comptol"] = 1e-06
    if "costtol" not in opt:
        opt["costtol"] = 1e-06
    if "max_it" not in opt:
        opt["max_it"] = 150
    if "max_red" not in opt:
        opt["max_red"] = 20
    if "step_control" not in opt:
        opt["step_control"] = False
    if "cost_mult" not in opt:
        opt["cost_mult"] = 1
    if "verbose" not in opt:
        opt["verbose"] = 0

    # initialize history
    hist = []

    # constants
    xi = 0.99995
    sigma = 0.1
    z0 = 1
    alpha_min = 1e-8
    rho_min = 0.95
    rho_max = 1.05
    mu_threshold = 1e-5

    # initialize
    i = 0                       # iteration counter
    converged = False           # flag
    eflag = False               # exit flag

    # add var limits to linear constraints
    eyex = eye(nx, nx, format="csr")
    AA = eyex if A is None else vstack([eyex, A], "csr")
    ll = r_[xmin, l]
    uu = r_[xmax, u]

    # split up linear constraints
    ieq = find( absolute(uu - ll) <= EPS )
    igt = find( (uu >=  1e10) & (ll > -1e10) )
    ilt = find( (ll <= -1e10) & (uu <  1e10) )
    ibx = find( (absolute(uu - ll) > EPS) & (uu < 1e10) & (ll > -1e10) )
    # zero-sized sparse matrices unsupported
    Ae = AA[ieq, :] if len(ieq) else None
    if len(ilt) or len(igt) or len(ibx):
        idxs = [(1, ilt), (-1, igt), (1, ibx), (-1, ibx)]
        Ai = vstack([sig * AA[idx, :] for sig, idx in idxs if len(idx)], 'csr')
    else:
        Ai = None
    be = uu[ieq]
    bi = r_[uu[ilt], -ll[igt], uu[ibx], -ll[ibx]]

    # evaluate cost f(x0) and constraints g(x0), h(x0)
    x = x0
    f, df = f_fcn(x)                 # cost
    f = f * opt["cost_mult"]
    df = df * opt["cost_mult"]
    if nonlinear:
        hn, gn, dhn, dgn = gh_fcn(x)        # nonlinear constraints
        h = hn if Ai is None else r_[hn.reshape(len(hn),), Ai * x - bi] # inequality constraints
        g = gn if Ae is None else r_[gn, Ae * x - be] # equality constraints

        if (dhn is None) and (Ai is None):
            dh = None
        elif dhn is None:
            dh = Ai.T
        elif Ai is None:
            dh = dhn
        else:
            dh = hstack([dhn, Ai.T])

        if (dgn is None) and (Ae is None):
            dg = None
        elif dgn is None:
            dg = Ae.T
        elif Ae is None:
            dg = dgn
        else:
            dg = hstack([dgn, Ae.T])
    else:
        h = -bi if Ai is None else Ai * x - bi        # inequality constraints
        g = -be if Ae is None else Ae * x - be        # equality constraints
        dh = None if Ai is None else Ai.T     # 1st derivative of inequalities
        dg = None if Ae is None else Ae.T     # 1st derivative of equalities

    # some dimensions
    neq = g.shape[0]           # number of equality constraints
    niq = h.shape[0]           # number of inequality constraints
    neqnln = gn.shape[0]       # number of nonlinear equality constraints
    niqnln = hn.shape[0]       # number of nonlinear inequality constraints
    nlt = len(ilt)             # number of upper bounded linear inequalities
    ngt = len(igt)             # number of lower bounded linear inequalities
    nbx = len(ibx)             # number of doubly bounded linear inequalities

    # initialize gamma, lam, mu, z, e
    gamma = 1                  # barrier coefficient
    lam = zeros(neq)
    z = z0 * ones(niq)
    mu = z0 * ones(niq)
    k = find(h < -z0)
    z[k] = -h[k]
    k = find((gamma / z) > z0)
    mu[k] = gamma / z[k]
    e = ones(niq)

    # check tolerance
    f0 = f
    if opt["step_control"]:
        L = f + dot(lam, g) + dot(mu, h + z) - gamma * sum(log(z))

    Lx = df.copy()
    Lx = Lx + dg * lam if dg is not None else Lx
    Lx = Lx + dh * mu  if dh is not None else Lx

    maxh = zeros(1) if len(h) == 0 else max(h)

    gnorm = norm(g, Inf) if len(g) else 0.0
    lam_norm = norm(lam, Inf) if len(lam) else 0.0
    mu_norm = norm(mu, Inf) if len(mu) else 0.0
    znorm = norm(z, Inf) if len(z) else 0.0
    feascond = \
        max([gnorm, maxh]) / (1 + max([norm(x, Inf), znorm]))
    gradcond = \
        norm(Lx, Inf) / (1 + max([lam_norm, mu_norm]))
    compcond = dot(z, mu) / (1 + norm(x, Inf))
    costcond = absolute(f - f0) / (1 + absolute(f0))

    # save history
    hist.append({'feascond': feascond, 'gradcond': gradcond,
        'compcond': compcond, 'costcond': costcond, 'gamma': gamma,
        'stepsize': 0, 'obj': f / opt["cost_mult"], 'alphap': 0, 'alphad': 0})

    if opt["verbose"]: # pragma: no cover
        s = '-sc' if opt["step_control"] else ''
        v = pipsver('all')
        print('Python Interior Point Solver - PIPS%s, Version %s, %s' %
                    (s, v['Version'], v['Date']))
        if opt['verbose'] > 1:
            print(" it    objective   step size   feascond     gradcond     "
                  "compcond     costcond  ")
            print("----  ------------ --------- ------------ ------------ "
                  "------------ ------------")
            print("%3d  %12.8g %10s %12g %12g %12g %12g" %
                (i, (f / opt["cost_mult"]), "",
                 feascond, gradcond, compcond, costcond))

    if feascond < opt["feastol"] and gradcond < opt["gradtol"] and \
        compcond < opt["comptol"] and costcond < opt["costtol"]:
        converged = True
        if opt["verbose"]:
            print("Converged!")

    # do Newton iterations
    while (not converged) and (i < opt["max_it"]):
        # update iteration counter
        i += 1

        # compute update step
        lmbda = {"eqnonlin": lam[range(neqnln)],
                 "ineqnonlin": mu[range(niqnln)]}
        if nonlinear:
            if hess_fcn is None:
                print("pips: Hessian evaluation via finite differences "
                      "not yet implemented.\nPlease provide "
                      "your own hessian evaluation function.")
            Lxx = hess_fcn(x, lmbda, opt["cost_mult"])
        else:
            _, _, d2f = f_fcn(x, True)      # cost
            Lxx = d2f * opt["cost_mult"]
        rz = range(len(z))
        zinvdiag = sparse((1.0 / z, (rz, rz))) if len(z) else None
        rmu = range(len(mu))
        mudiag = sparse((mu, (rmu, rmu))) if len(mu) else None
        dh_zinv = None if dh is None else dh * zinvdiag
        M = Lxx if dh is None else Lxx + dh_zinv * mudiag * dh.T
        N = Lx if dh is None else Lx + dh_zinv * (mudiag * h + gamma * e)

        Ab = sparse(M) if dg is None else vstack([
            hstack([M, dg]),
            hstack([dg.T, sparse((neq, neq))])
        ])
        bb = r_[-N, -g]

        dxdlam = spsolve(Ab.tocsr(), bb)

        if any(isnan(dxdlam)):
            if opt["verbose"]:
                print('\nNumerically Failed\n')
            eflag = -1
            break

        dx = dxdlam[:nx]
        dlam = dxdlam[nx:nx + neq]
        dz = -h - z if dh is None else -h - z - dh.T * dx
        dmu = -mu if dh is None else -mu + zinvdiag * (gamma * e - mudiag * dz)

        # do the update
        k = find(dz < 0.0)
        alphap = min([xi * min(z[k] / -dz[k]), 1]) if len(k) else 1.0
        k = find(dmu < 0.0)
        alphad = min([xi * min(mu[k] / -dmu[k]), 1]) if len(k) else 1.0
        x = x + alphap * dx
        z = z + alphap * dz
        lam = lam + alphad * dlam
        mu = mu + alphad * dmu
        if niq > 0:
            gamma = sigma * dot(z, mu) / niq

        # evaluate cost, constraints, derivatives
        f, df = f_fcn(x)             # cost
        f = f * opt["cost_mult"]
        df = df * opt["cost_mult"]
        if nonlinear:
            hn, gn, dhn, dgn = gh_fcn(x)                   # nln constraints
#            g = gn if Ai is None else r_[gn, Ai * x - bi] # ieq constraints
#            h = hn if Ae is None else r_[hn, Ae * x - be] # eq constraints
            h = hn if Ai is None else r_[hn.reshape(len(hn),), Ai * x - bi] # ieq constr
            g = gn if Ae is None else r_[gn, Ae * x - be]  # eq constr

            if (dhn is None) and (Ai is None):
                dh = None
            elif dhn is None:
                dh = Ai.T
            elif Ai is None:
                dh = dhn
            else:
                dh = hstack([dhn, Ai.T])

            if (dgn is None) and (Ae is None):
                dg = None
            elif dgn is None:
                dg = Ae.T
            elif Ae is None:
                dg = dgn
            else:
                dg = hstack([dgn, Ae.T])
        else:
            h = -bi if Ai is None else Ai * x - bi    # inequality constraints
            g = -be if Ae is None else Ae * x - be    # equality constraints
            # 1st derivatives are constant, still dh = Ai.T, dg = Ae.T

        Lx = df
        Lx = Lx + dg * lam if dg is not None else Lx
        Lx = Lx + dh * mu  if dh is not None else Lx

        if len(h) == 0:
            maxh = zeros(1)
        else:
            maxh = max(h)

        gnorm = norm(g, Inf) if len(g) else 0.0
        lam_norm = norm(lam, Inf) if len(lam) else 0.0
        mu_norm = norm(mu, Inf) if len(mu) else 0.0
        znorm = norm(z, Inf) if len(z) else 0.0
        feascond = \
            max([gnorm, maxh]) / (1 + max([norm(x, Inf), znorm]))
        gradcond = \
            norm(Lx, Inf) / (1 + max([lam_norm, mu_norm]))
        compcond = dot(z, mu) / (1 + norm(x, Inf))
        costcond = float(absolute(f - f0) / (1 + absolute(f0)))

        hist.append({'feascond': feascond, 'gradcond': gradcond,
            'compcond': compcond, 'costcond': costcond, 'gamma': gamma,
            'stepsize': norm(dx), 'obj': f / opt["cost_mult"],
            'alphap': alphap, 'alphad': alphad})

        if opt["verbose"] > 1:
            print("%3d  %12.8g %10.5g %12g %12g %12g %12g" %
                (i, (f / opt["cost_mult"]), norm(dx), feascond, gradcond,
                 compcond, costcond))

        if feascond < opt["feastol"] and gradcond < opt["gradtol"] and \
            compcond < opt["comptol"] and costcond < opt["costtol"]:
            converged = True
            if opt["verbose"]:
                print("Converged!")
        else:
            if any(isnan(x)) or (alphap < alpha_min) or \
                (alphad < alpha_min) or (gamma < EPS) or (gamma > 1.0 / EPS):
                if opt["verbose"]:
                    print("Numerically failed.")
                eflag = -1
                break
            f0 = f

            if opt["step_control"]:
                L = f + dot(lam, g) + dot(mu, (h + z)) - gamma * sum(log(z))

    if opt["verbose"]:
        if not converged:
            print("Did not converge in %d iterations." % i)

    # package results
    if eflag != -1:
        eflag = converged

    if eflag == 0:
        message = 'Did not converge'
    elif eflag == 1:
        message = 'Converged'
    elif eflag == -1:
        message = 'Numerically failed'
    else:
        raise

    output = {"iterations": i, "hist": hist, "message": message}

    # zero out multipliers on non-binding constraints
    mu[find( (h < -opt["feastol"]) & (mu < mu_threshold) )] = 0.0

    # un-scale cost and prices
    f = f / opt["cost_mult"]
    lam = lam / opt["cost_mult"]
    mu = mu / opt["cost_mult"]

    # re-package multipliers into struct
    lam_lin = lam[neqnln:neq]           # lambda for linear constraints
    mu_lin = mu[niqnln:niq]             # mu for linear constraints
    kl = find(lam_lin < 0.0)     # lower bound binding
    ku = find(lam_lin > 0.0)     # upper bound binding

    mu_l = zeros(nx + nA)
    mu_l[ieq[kl]] = -lam_lin[kl]
    mu_l[igt] = mu_lin[nlt:nlt + ngt]
    mu_l[ibx] = mu_lin[nlt + ngt + nbx:nlt + ngt + nbx + nbx]

    mu_u = zeros(nx + nA)
    mu_u[ieq[ku]] = lam_lin[ku]
    mu_u[ilt] = mu_lin[:nlt]
    mu_u[ibx] = mu_lin[nlt + ngt:nlt + ngt + nbx]

    lmbda = {'mu_l': mu_l[nx:], 'mu_u': mu_u[nx:],
             'lower': mu_l[:nx], 'upper': mu_u[:nx]}

    if niqnln > 0:
        lmbda['ineqnonlin'] = mu[:niqnln]
    if neqnln > 0:
        lmbda['eqnonlin'] = lam[:neqnln]

#    lmbda = {"eqnonlin": lam[:neqnln], 'ineqnonlin': mu[:niqnln],
#             "mu_l": mu_l[nx:], "mu_u": mu_u[nx:],
#             "lower": mu_l[:nx], "upper": mu_u[:nx]}

    solution =  {"x": x, "f": f, "eflag": converged,
                 "output": output, "lmbda": lmbda}

    return solution
