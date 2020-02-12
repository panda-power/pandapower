# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Constructs an OPF model object from a PYPOWER case dict.
"""

from sys import stdout, stderr

from numpy import array, any, delete, unique, arange, nonzero, pi, r_, ones, Inf, flatnonzero as find
from scipy.sparse import hstack, csr_matrix as sparse
from pandapower.pypower.idx_brch import RATE_A
from pandapower.pypower.idx_bus import BUS_TYPE, REF, VA, VM, PD, GS, VMAX, VMIN
from pandapower.pypower.idx_cost import MODEL, NCOST, PW_LINEAR, COST, POLYNOMIAL
from pandapower.pypower.idx_gen import GEN_BUS, VG, PG, QG, PMAX, PMIN, QMAX, QMIN
from pandapower.pypower.makeAang import makeAang
from pandapower.pypower.makeApq import makeApq
from pandapower.pypower.makeAvl import makeAvl
from pandapower.pypower.makeAy import makeAy
from pandapower.pypower.makeBdc import makeBdc
from pandapower.pypower.opf_args import opf_args
from pandapower.pypower.pqcost import pqcost
from pandapower.pypower.run_userfcn import run_userfcn
from pandapower.pypower.opf_model import opf_model


def opf_setup(ppc, ppopt):
    """Constructs an OPF model object from a PYPOWER case dict.

    Assumes that ppc is a PYPOWER case dict with internal indexing,
    all equipment in-service, etc.

    @see: L{opf}, L{ext2int}, L{opf_execute}

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Richard Lincoln

    Modified by University of Kassel (Friederike Meier): Bugfix in line 110
    """
    ## options
    dc  = ppopt['PF_DC']        ## 1 = DC OPF, 0 = AC OPF
    alg = ppopt['OPF_ALG']
    verbose = ppopt['VERBOSE']

    ## data dimensions
    nb = ppc['bus'].shape[0]    ## number of buses
    nl = ppc['branch'].shape[0] ## number of branches
    ng = ppc['gen'].shape[0]    ## number of dispatchable injections
    if 'A' in ppc:
        nusr = ppc['A'].shape[0]    ## number of linear user constraints
    else:
        nusr = 0

    if 'N' in ppc:
        nw = ppc['N'].shape[0]      ## number of general cost vars, w
    else:
        nw = 0

    if dc:
        ## ignore reactive costs for DC
        ppc['gencost'], _ = pqcost(ppc['gencost'], ng)

        ## reduce A and/or N from AC dimensions to DC dimensions, if needed
        if nusr or nw: # pragma: no cover
            acc = r_[nb + arange(nb), 2 * nb + ng + arange(ng)]   ## Vm and Qg columns

            if nusr and (ppc['A'].shape[1] >= 2*nb + 2*ng):
                ## make sure there aren't any constraints on Vm or Qg
                if ppc['A'][:, acc].nnz > 0:
                    stderr.write('opf_setup: attempting to solve DC OPF with user constraints on Vm or Qg\n')

                # FIXME: delete sparse matrix columns
                bcc = delete(arange(ppc['A'].shape[1]), acc)
                ppc['A'] = ppc['A'].tolil()[:, bcc].tocsr()           ## delete Vm and Qg columns

            if nw and (ppc['N'].shape[1] >= 2*nb + 2*ng):
                ## make sure there aren't any costs on Vm or Qg
                if ppc['N'][:, acc].nnz > 0:
                    ii, _ = nonzero(ppc['N'][:, acc])
                    _, ii = unique(ii, return_index=True)    ## indices of w with potential non-zero cost terms from Vm or Qg
                    if any(ppc['Cw'][ii]) | ( ('H' in ppc) & (len(ppc['H']) > 0) &
                            any(any(ppc['H'][:, ii])) ):
                        stderr.write('opf_setup: attempting to solve DC OPF with user costs on Vm or Qg\n')

                # FIXME: delete sparse matrix columns
                bcc = delete(arange(ppc['N'].shape[1]), acc)
                ppc['N'] = ppc['N'].tolil()[:, bcc].tocsr()               ## delete Vm and Qg columns

#    ## convert single-block piecewise-linear costs into linear polynomial cost
    pwl1 = find((ppc['gencost'][:, MODEL] == PW_LINEAR) & (ppc['gencost'][:, NCOST] == 2))
    # p1 = array([])
    if len(pwl1) > 0:
        x0 = ppc['gencost'][pwl1, COST]
        y0 = ppc['gencost'][pwl1, COST + 1]
        x1 = ppc['gencost'][pwl1, COST + 2]
        y1 = ppc['gencost'][pwl1, COST + 3]
        m = (y1 - y0) / (x1 - x0)
        b = y0 - m * x0
        ppc['gencost'][pwl1, MODEL] = POLYNOMIAL
        ppc['gencost'][pwl1, NCOST] = 2
        ppc['gencost'][pwl1, COST:COST + 2] = r_['1',m.reshape(len(m),1), b.reshape(len(b),1)] # changed from ppc['gencost'][pwl1, COST:COST + 2] = r_[m, b] because we need to make sure, that m and b have the same shape, resulted in a value error due to shape mismatch before

    ## create (read-only) copies of individual fields for convenience
    baseMVA, bus, gen, branch, gencost, _, lbu, ubu, ppopt, \
            _, fparm, H, Cw, z0, zl, zu, userfcn, _ = opf_args(ppc, ppopt)

    ## warn if there is more than one reference bus
    refs = find(bus[:, BUS_TYPE] == REF)
    if len(refs) > 1 and verbose > 0:
        errstr = '\nopf_setup: Warning: Multiple reference buses.\n' + \
            '           For a system with islands, a reference bus in each island\n' + \
            '           may help convergence, but in a fully connected system such\n' + \
            '           a situation is probably not reasonable.\n\n'
        stdout.write(errstr)

    ## set up initial variables and bounds
    gbus = gen[:, GEN_BUS].astype(int)
    Va   = bus[:, VA] * (pi / 180.0)
    Vm   = bus[:, VM].copy()
    Vm[gbus] = gen[:, VG]   ## buses with gens, init Vm from gen data
    Pg   = gen[:, PG] / baseMVA
    Qg   = gen[:, QG] / baseMVA
    Pmin = gen[:, PMIN] / baseMVA
    Pmax = gen[:, PMAX] / baseMVA
    Qmin = gen[:, QMIN] / baseMVA
    Qmax = gen[:, QMAX] / baseMVA

    if dc:               ## DC model
        ## more problem dimensions
        nv    = 0            ## number of voltage magnitude vars
        nq    = 0            ## number of Qg vars
        q1    = array([])    ## index of 1st Qg column in Ay

        ## power mismatch constraints
        B, Bf, Pbusinj, Pfinj = makeBdc(bus, branch)
        neg_Cg = sparse((-ones(ng), (gen[:, GEN_BUS], arange(ng))), (nb, ng))   ## Pbus w.r.t. Pg
        Amis = hstack([B, neg_Cg], 'csr')
        bmis = -(bus[:, PD] + bus[:, GS]) / baseMVA - Pbusinj

        ## branch flow constraints
        il = find((branch[:, RATE_A] != 0) & (branch[:, RATE_A] < 1e10))
        nl2 = len(il)         ## number of constrained lines
        lpf = -Inf * ones(nl2)
        upf = branch[il, RATE_A] / baseMVA - Pfinj[il]
        upt = branch[il, RATE_A] / baseMVA + Pfinj[il]

        user_vars = ['Va', 'Pg']
        ycon_vars = ['Pg', 'y']
    else:                ## AC model
        ## more problem dimensions
        nv    = nb           ## number of voltage magnitude vars
        nq    = ng           ## number of Qg vars
        q1    = ng           ## index of 1st Qg column in Ay

        ## dispatchable load, constant power factor constraints
        Avl, lvl, uvl, _  = makeAvl(baseMVA, gen)

        ## generator PQ capability curve constraints
        Apqh, ubpqh, Apql, ubpql, Apqdata = makeApq(baseMVA, gen)

        user_vars = ['Va', 'Vm', 'Pg', 'Qg']
        ycon_vars = ['Pg', 'Qg', 'y']

    ## voltage angle reference constraints
    Vau = Inf * ones(nb)
    Val = -Vau
    Vau[refs] = Va[refs]
    Val[refs] = Va[refs]

    ## branch voltage angle difference limits
    Aang, lang, uang, iang  = makeAang(baseMVA, branch, nb, ppopt)

    ## basin constraints for piece-wise linear gen cost variables
    if alg == 545 or alg == 550:     ## SC-PDIPM or TRALM, no CCV cost vars # pragma: no cover
        ny = 0
        Ay = None
        by = array([])
    else:
        ipwl = find(gencost[:, MODEL] == PW_LINEAR)  ## piece-wise linear costs
        ny = ipwl.shape[0]   ## number of piece-wise linear cost vars
        Ay, by = makeAy(baseMVA, ng, gencost, 1, q1, 1+ng+nq)

    if any((gencost[:, MODEL] != POLYNOMIAL) & (gencost[:, MODEL] != PW_LINEAR)):
        stderr.write('opf_setup: some generator cost rows have invalid MODEL value\n')

    ## more problem dimensions
    nx = nb+nv + ng+nq  ## number of standard OPF control variables
    if nusr: # pragma: no cover
        nz = ppc['A'].shape[1] - nx  ## number of user z variables
        if nz < 0:
            stderr.write('opf_setup: user supplied A matrix must have at least %d columns.\n' % nx)
    else:
        nz = 0               ## number of user z variables
        if nw:               ## still need to check number of columns of N
            if ppc['N'].shape[1] != nx:
                stderr.write('opf_setup: user supplied N matrix must have %d columns.\n' % nx)

    ## construct OPF model object
    om = opf_model(ppc)
    if len(pwl1) > 0:
        om.userdata('pwl1', pwl1)

    if dc:
        om.userdata('Bf', Bf)
        om.userdata('Pfinj', Pfinj)
        om.userdata('iang', iang)
        om.add_vars('Va', nb, Va, Val, Vau)
        om.add_vars('Pg', ng, Pg, Pmin, Pmax)
        om.add_constraints('Pmis', Amis, bmis, bmis, ['Va', 'Pg']) ## nb
        om.add_constraints('Pf',  Bf[il, :], lpf, upf, ['Va'])     ## nl
        om.add_constraints('Pt', -Bf[il, :], lpf, upt, ['Va'])     ## nl
        om.add_constraints('ang', Aang, lang, uang, ['Va'])        ## nang
    else:
        om.userdata('Apqdata', Apqdata)
        om.userdata('iang', iang)
        om.add_vars('Va', nb, Va, Val, Vau)
        om.add_vars('Vm', nb, Vm, bus[:, VMIN], bus[:, VMAX])
        om.add_vars('Pg', ng, Pg, Pmin, Pmax)
        om.add_vars('Qg', ng, Qg, Qmin, Qmax)
        om.add_constraints('Pmis', nb, 'nonlinear')
        om.add_constraints('Qmis', nb, 'nonlinear')
        om.add_constraints('Sf', nl, 'nonlinear')
        om.add_constraints('St', nl, 'nonlinear')
        om.add_constraints('PQh', Apqh, array([]), ubpqh, ['Pg', 'Qg'])   ## npqh
        om.add_constraints('PQl', Apql, array([]), ubpql, ['Pg', 'Qg'])   ## npql
        om.add_constraints('vl',  Avl, lvl, uvl,   ['Pg', 'Qg'])   ## nvl
        om.add_constraints('ang', Aang, lang, uang, ['Va'])        ## nang

    ## y vars, constraints for piece-wise linear gen costs
    if ny > 0:
        om.add_vars('y', ny)
        om.add_constraints('ycon', Ay, array([]), by, ycon_vars)          ## ncony

    ## execute userfcn callbacks for 'formulation' stage
    run_userfcn(userfcn, 'formulation', om)

    return om
