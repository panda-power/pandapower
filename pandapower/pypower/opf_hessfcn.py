# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Evaluates Hessian of Lagrangian for AC OPF.
"""

from numpy import array, zeros, ones, exp, arange, r_, flatnonzero as find
from scipy.sparse import vstack, hstack, issparse, csr_matrix as sparse

from pandapower.pypower.d2AIbr_dV2 import d2AIbr_dV2
from pandapower.pypower.d2ASbr_dV2 import d2ASbr_dV2
from pandapower.pypower.d2Sbus_dV2 import d2Sbus_dV2
from pandapower.pypower.dIbr_dV import dIbr_dV
from pandapower.pypower.dSbr_dV import dSbr_dV
from pandapower.pypower.idx_brch import F_BUS, T_BUS
from pandapower.pypower.idx_cost import MODEL, POLYNOMIAL
from pandapower.pypower.idx_gen import PG, QG
from pandapower.pypower.opf_consfcn import opf_consfcn
from pandapower.pypower.opf_costfcn import opf_costfcn
from pandapower.pypower.polycost import polycost


def opf_hessfcn(x, lmbda, om, Ybus, Yf, Yt, ppopt, il=None, cost_mult=1.0):
    """Evaluates Hessian of Lagrangian for AC OPF.

    Hessian evaluation function for AC optimal power flow, suitable
    for use with L{pips}.

    Examples::
        Lxx = opf_hessfcn(x, lmbda, om, Ybus, Yf, Yt, ppopt)
        Lxx = opf_hessfcn(x, lmbda, om, Ybus, Yf, Yt, ppopt, il)
        Lxx = opf_hessfcn(x, lmbda, om, Ybus, Yf, Yt, ppopt, il, cost_mult)

    @param x: optimization vector
    @param lmbda: C{eqnonlin} - Lagrange multipliers on power balance
    equations. C{ineqnonlin} - Kuhn-Tucker multipliers on constrained
    branch flows.
    @param om: OPF model object
    @param Ybus: bus admittance matrix
    @param Yf: admittance matrix for "from" end of constrained branches
    @param Yt: admittance matrix for "to" end of constrained branches
    @param ppopt: PYPOWER options vector
    @param il: (optional) vector of branch indices corresponding to
    branches with flow limits (all others are assumed to be unconstrained).
    The default is C{range(nl)} (all branches). C{Yf} and C{Yt} contain
    only the rows corresponding to C{il}.
    @param cost_mult: (optional) Scale factor to be applied to the cost
    (default = 1).

    @return: Hessian of the Lagrangian.

    @see: L{opf_costfcn}, L{opf_consfcn}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Richard Lincoln

    Modified by University of Kassel (Friederike Meier): Bugfix in line 173
    """
    ##----- initialize -----
    ## unpack data
    ppc = om.get_ppc()
    baseMVA, bus, gen, branch, gencost = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"], ppc["gencost"]
    cp = om.get_cost_params()
    N, Cw, H, dd, rh, kk, mm = \
        cp["N"], cp["Cw"], cp["H"], cp["dd"], cp["rh"], cp["kk"], cp["mm"]
    vv, _, _, _ = om.get_idx()

    ## unpack needed parameters
    nb = bus.shape[0]          ## number of buses
    nl = branch.shape[0]       ## number of branches
    ng = gen.shape[0]          ## number of dispatchable injections
    nxyz = len(x)              ## total number of control vars of all types

    ## set default constrained lines
    if il is None:
        il = arange(nl)            ## all lines have limits by default
    nl2 = len(il)           ## number of constrained lines

    ## grab Pg & Qg
    Pg = x[vv["i1"]["Pg"]:vv["iN"]["Pg"]]  ## active generation in p.u.
    Qg = x[vv["i1"]["Qg"]:vv["iN"]["Qg"]]  ## reactive generation in p.u.

    ## put Pg & Qg back in gen
    gen[:, PG] = Pg * baseMVA  ## active generation in MW
    gen[:, QG] = Qg * baseMVA  ## reactive generation in MVAr

    ## reconstruct V
    Va = x[vv["i1"]["Va"]:vv["iN"]["Va"]]
    Vm = x[vv["i1"]["Vm"]:vv["iN"]["Vm"]]
    V = Vm * exp(1j * Va)
    nxtra = nxyz - 2 * nb
    pcost = gencost[arange(ng), :]
    if gencost.shape[0] > ng:
        qcost = gencost[arange(ng, 2 * ng), :]
    else:
        qcost = array([])

    ## ----- evaluate d2f -----
    d2f_dPg2 = zeros(ng)#sparse((ng, 1))               ## w.r.t. p.u. Pg
    d2f_dQg2 = zeros(ng)#sparse((ng, 1))               ## w.r.t. p.u. Qg
    ipolp = find(pcost[:, MODEL] == POLYNOMIAL)
    if len(ipolp):
        d2f_dPg2[ipolp] = \
            baseMVA**2 * polycost(pcost[ipolp, :], Pg[ipolp] * baseMVA, 2)
    if qcost.any():          ## Qg is not free
        ipolq = find(qcost[:, MODEL] == POLYNOMIAL)
        d2f_dQg2[ipolq] = \
                baseMVA**2 * polycost(qcost[ipolq, :], Qg[ipolq] * baseMVA, 2)
    i = r_[arange(vv["i1"]["Pg"], vv["iN"]["Pg"]),
           arange(vv["i1"]["Qg"], vv["iN"]["Qg"])]
#    d2f = sparse((vstack([d2f_dPg2, d2f_dQg2]).toarray().flatten(),
#                  (i, i)), shape=(nxyz, nxyz))
    d2f = sparse((r_[d2f_dPg2, d2f_dQg2], (i, i)), (nxyz, nxyz))

    ## generalized cost
    if issparse(N) and N.nnz > 0: # pragma: no cover
        nw = N.shape[0]
        r = N * x - rh                    ## Nx - rhat
        iLT = find(r < -kk)               ## below dead zone
        iEQ = find((r == 0) & (kk == 0))  ## dead zone doesn't exist
        iGT = find(r > kk)                ## above dead zone
        iND = r_[iLT, iEQ, iGT]           ## rows that are Not in the Dead region
        iL = find(dd == 1)                ## rows using linear function
        iQ = find(dd == 2)                ## rows using quadratic function
        LL = sparse((ones(len(iL)), (iL, iL)), (nw, nw))
        QQ = sparse((ones(len(iQ)), (iQ, iQ)), (nw, nw))
        kbar = sparse((r_[ones(len(iLT)), zeros(len(iEQ)), -ones(len(iGT))],
                       (iND, iND)), (nw, nw)) * kk
        rr = r + kbar                  ## apply non-dead zone shift
        M = sparse((mm[iND], (iND, iND)), (nw, nw))  ## dead zone or scale
        diagrr = sparse((rr, (arange(nw), arange(nw))), (nw, nw))

        ## linear rows multiplied by rr(i), quadratic rows by rr(i)^2
        w = M * (LL + QQ * diagrr) * rr
        HwC = H * w + Cw
        AA = N.T * M * (LL + 2 * QQ * diagrr)

        d2f = d2f + AA * H * AA.T + 2 * N.T * M * QQ * \
                sparse((HwC, (arange(nw), arange(nw))), (nw, nw)) * N
    d2f = d2f * cost_mult

    ##----- evaluate Hessian of power balance constraints -----
    nlam = len(lmbda["eqnonlin"]) // 2
    lamP = lmbda["eqnonlin"][:nlam]
    lamQ = lmbda["eqnonlin"][nlam:nlam + nlam]
    Gpaa, Gpav, Gpva, Gpvv = d2Sbus_dV2(Ybus, V, lamP)
    Gqaa, Gqav, Gqva, Gqvv = d2Sbus_dV2(Ybus, V, lamQ)

    d2G = vstack([
            hstack([
                vstack([hstack([Gpaa, Gpav]),
                        hstack([Gpva, Gpvv])]).real +
                vstack([hstack([Gqaa, Gqav]),
                        hstack([Gqva, Gqvv])]).imag,
                sparse((2 * nb, nxtra))]),
            hstack([
                sparse((nxtra, 2 * nb)),
                sparse((nxtra, nxtra))
            ])
        ], "csr")

    ##----- evaluate Hessian of flow constraints -----
    nmu = len(lmbda["ineqnonlin"]) // 2
    muF = lmbda["ineqnonlin"][:nmu]
    muT = lmbda["ineqnonlin"][nmu:nmu + nmu]
    if ppopt['OPF_FLOW_LIM'] == 2:       ## current
        if Yf.size:
            dIf_dVa, dIf_dVm, dIt_dVa, dIt_dVm, If, It = dIbr_dV(branch, Yf, Yt, V)
            Hfaa, Hfav, Hfva, Hfvv = d2AIbr_dV2(dIf_dVa, dIf_dVm, If, Yf, V, muF)
            Htaa, Htav, Htva, Htvv = d2AIbr_dV2(dIt_dVa, dIt_dVm, It, Yt, V, muT)
        else:
            Hfaa= Hfav= Hfva= Hfvv= Htaa= Htav= Htva= Htvv = sparse(zeros((nb,nb)))
    else: # pragma: no cover
        f = branch[il, F_BUS].astype(int)    ## list of "from" buses
        t = branch[il, T_BUS].astype(int)    ## list of "to" buses
        ## connection matrix for line & from buses
        Cf = sparse((ones(nl2), (arange(nl2), f)), (nl2, nb))
        ## connection matrix for line & to buses
        Ct = sparse((ones(nl2), (arange(nl2), t)), (nl2, nb))
        dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St = \
                dSbr_dV(branch[il,:], Yf, Yt, V)
        if ppopt['OPF_FLOW_LIM'] == 1:     ## real power
            Hfaa, Hfav, Hfva, Hfvv = d2ASbr_dV2(dSf_dVa.real, dSf_dVm.real,
                                                Sf.real, Cf, Yf, V, muF)
            Htaa, Htav, Htva, Htvv = d2ASbr_dV2(dSt_dVa.real, dSt_dVm.real,
                                                St.real, Ct, Yt, V, muT)
        else:                  ## apparent power
            Hfaa, Hfav, Hfva, Hfvv = \
                    d2ASbr_dV2(dSf_dVa, dSf_dVm, Sf, Cf, Yf, V, muF)
            Htaa, Htav, Htva, Htvv = \
                    d2ASbr_dV2(dSt_dVa, dSt_dVm, St, Ct, Yt, V, muT)

    d2H = vstack([
            hstack([
                vstack([hstack([Hfaa, Hfav]),
				hstack([Hfva, Hfvv])]) +
				vstack([hstack([Htaa, Htav]),
                        hstack([Htva, Htvv])]),
                sparse((2 * nb, nxtra))
            ]),
            hstack([
                sparse((nxtra, 2 * nb)),
                sparse((nxtra, nxtra))
            ])
        ], "csr")

    ##-----  do numerical check using (central) finite differences  -----
    if 0:
        nx = len(x)
        step = 1e-5
        num_d2f = sparse((nx, nx))
        num_d2G = sparse((nx, nx))
        num_d2H = sparse((nx, nx))
        for i in range(nx):
            xp = x
            xm = x
            xp[i] = x[i] + step / 2
            xm[i] = x[i] - step / 2
            # evaluate cost & gradients
            _, dfp = opf_costfcn(xp, om)
            _, dfm = opf_costfcn(xm, om)
            # evaluate constraints & gradients
            _, _, dHp, dGp = opf_consfcn(xp, om, Ybus, Yf, Yt, ppopt, il)
            _, _, dHm, dGm = opf_consfcn(xm, om, Ybus, Yf, Yt, ppopt, il)
            num_d2f[:, i] = cost_mult * (dfp - dfm) / step
            num_d2G[:, i] = (dGp - dGm) * lmbda["eqnonlin"]   / step
            num_d2H[:, i] = (dHp - dHm) * lmbda["ineqnonlin"] / step
        d2f_err = max(max(abs(d2f - num_d2f)))
        d2G_err = max(max(abs(d2G - num_d2G)))
        d2H_err = max(max(abs(d2H - num_d2H)))
        if d2f_err > 1e-6:
            print('Max difference in d2f: %g' % d2f_err)
        if d2G_err > 1e-5:
            print('Max difference in d2G: %g' % d2G_err)
        if d2H_err > 1e-6:
            print('Max difference in d2H: %g' % d2H_err)

    return d2f + d2G + d2H
