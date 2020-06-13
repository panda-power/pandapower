# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Evaluates objective function, gradient and Hessian for OPF.
"""

from numpy import array, ones, zeros, arange, r_, dot, flatnonzero as find
from scipy.sparse import issparse, csr_matrix as sparse

from pandapower.pypower.idx_cost import MODEL, POLYNOMIAL

from pandapower.pypower.totcost import totcost
from pandapower.pypower.polycost import polycost


def opf_costfcn(x, om, return_hessian=False):
    """Evaluates objective function, gradient and Hessian for OPF.

    Objective function evaluation routine for AC optimal power flow,
    suitable for use with L{pips}. Computes objective function value,
    gradient and Hessian.

    @param x: optimization vector
    @param om: OPF model object

    @return: C{F} - value of objective function. C{df} - (optional) gradient
    of objective function (column vector). C{d2f} - (optional) Hessian of
    objective function (sparse matrix).

    @see: L{opf_consfcn}, L{opf_hessfcn}

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)
    """
    ##----- initialize -----
    ## unpack data
    ppc = om.get_ppc()
    baseMVA, gen, gencost = ppc["baseMVA"], ppc["gen"], ppc["gencost"]
    cp = om.get_cost_params()
    N, Cw, H, dd, rh, kk, mm = \
        cp["N"], cp["Cw"], cp["H"], cp["dd"], cp["rh"], cp["kk"], cp["mm"]
    vv, _, _, _ = om.get_idx()

    ## problem dimensions
    ng = gen.shape[0]          ## number of dispatchable injections
    ny = om.getN('var', 'y')   ## number of piece-wise linear costs
    nxyz = len(x)              ## total number of control vars of all types

    ## grab Pg & Qg
    Pg = x[vv["i1"]["Pg"]:vv["iN"]["Pg"]]  ## active generation in p.u.
    Qg = x[vv["i1"]["Qg"]:vv["iN"]["Qg"]]  ## reactive generation in p.u.

    ##----- evaluate objective function -----
    ## polynomial cost of P and Q
    # use totcost only on polynomial cost in the minimization problem
    # formulation, pwl cost is the sum of the y variables.
    ipol = find(gencost[:, MODEL] == POLYNOMIAL)   ## poly MW and MVAr costs
    xx = r_[ Pg, Qg ] * baseMVA
    if len(ipol)>0:
        f = sum( totcost(gencost[ipol, :], xx[ipol]) )  ## cost of poly P or Q
    else:
        f = 0

    ## piecewise linear cost of P and Q
    if ny > 0:
        ccost = sparse((ones(ny),
                        (zeros(ny), arange(vv["i1"]["y"], vv["iN"]["y"]))),
                       (1, nxyz)).toarray().flatten()
        f = f + dot(ccost, x)
    else:
        ccost = zeros(nxyz)

    ##----- evaluate cost gradient -----
    ## index ranges
    iPg = range(vv["i1"]["Pg"], vv["iN"]["Pg"])
    iQg = range(vv["i1"]["Qg"], vv["iN"]["Qg"])

    ## polynomial cost of P and Q
    df_dPgQg = zeros(2 * ng)        ## w.r.t p.u. Pg and Qg
    if len(ipol):
        df_dPgQg[ipol] = baseMVA * polycost(gencost[ipol, :], xx[ipol], 1)
    df = zeros(nxyz)
    df[iPg] = df_dPgQg[:ng]
    df[iQg] = df_dPgQg[ng:ng + ng]

    ## piecewise linear cost of P and Q
    df = df + ccost  # The linear cost row is additive wrt any nonlinear cost.

    if not return_hessian:
        return f, df

    ## ---- evaluate cost Hessian -----
    pcost = gencost[range(ng), :]
    if gencost.shape[0] > ng:
        qcost = gencost[ng + 1:2 * ng, :]
    else:
        qcost = array([])

    ## polynomial generator costs
    d2f_dPg2 = zeros(ng)               ## w.r.t. p.u. Pg
    d2f_dQg2 = zeros(ng)               ## w.r.t. p.u. Qg
    ipolp = find(pcost[:, MODEL] == POLYNOMIAL)
    d2f_dPg2[ipolp] = \
            baseMVA**2 * polycost(pcost[ipolp, :], Pg[ipolp]*baseMVA, 2)
    if any(qcost):          ## Qg is not free
        ipolq = find(qcost[:, MODEL] == POLYNOMIAL)
        d2f_dQg2[ipolq] = \
                baseMVA**2 * polycost(qcost[ipolq, :], Qg[ipolq] * baseMVA, 2)
    i = r_[iPg, iQg].T
    d2f = sparse((r_[d2f_dPg2, d2f_dQg2], (i, i)), (nxyz, nxyz))

    ## generalized cost
    if N is not None and issparse(N):
        d2f = d2f + AA * H * AA.T + 2 * N.T * M * QQ * \
                sparse((HwC, (range(nw), range(nw))), (nw, nw)) * N

    return f, df, d2f
