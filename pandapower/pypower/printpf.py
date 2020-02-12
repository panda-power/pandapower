# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Prints power flow results.
"""

from sys import stdout

from numpy import \
    ones, zeros, r_, sort, exp, pi, diff, arange, min, \
    argmin, argmax, logical_or, real, imag, any

from numpy import flatnonzero as find

from pandapower.pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, \
    VM, VA, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN, REF
from pandapower.pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, GEN_STATUS, \
    PMAX, PMIN, MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN
from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, \
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST

from pandapower.pypower.isload import isload
from pandapower.pypower.run_userfcn import run_userfcn
from pandapower.pypower.ppoption import ppoption


def printpf(baseMVA, bus=None, gen=None, branch=None, f=None, success=None,
            et=None, fd=None, ppopt=None): # pragma: no cover
    """Prints power flow results.

    Prints power flow and optimal power flow results to C{fd} (a file
    descriptor which defaults to C{stdout}), with the details of what
    gets printed controlled by the optional C{ppopt} argument, which is a
    PYPOWER options vector (see L{ppoption} for details).

    The data can either be supplied in a single C{results} dict, or
    in the individual arguments: C{baseMVA}, C{bus}, C{gen}, C{branch}, C{f},
    C{success} and C{et}, where C{f} is the OPF objective function value,
    C{success} is C{True} if the solution converged and C{False} otherwise,
    and C{et} is the elapsed time for the computation in seconds. If C{f} is
    given, it is assumed that the output is from an OPF run, otherwise it is
    assumed to be a simple power flow run.

    Examples::
        ppopt = ppoptions(OUT_GEN=1, OUT_BUS=0, OUT_BRANCH=0)
        fd = open(fname, 'w+b')
        results = runopf(ppc)
        printpf(results)
        printpf(results, fd)
        printpf(results, fd, ppopt)
        printpf(baseMVA, bus, gen, branch, f, success, et)
        printpf(baseMVA, bus, gen, branch, f, success, et, fd)
        printpf(baseMVA, bus, gen, branch, f, success, et, fd, ppopt)
        fd.close()

    @author: Ray Zimmerman (PSERC Cornell)
    """
    ##----- initialization -----
    ## default arguments
    if isinstance(baseMVA, dict):
        have_results_struct = 1
        results = baseMVA
        if gen is None:
            ppopt = ppoption()   ## use default options
        else:
            ppopt = gen
        if (ppopt['OUT_ALL'] == 0):
            return     ## nothin' to see here, bail out now
        if bus is None:
            fd = stdout         ## print to stdout by default
        else:
            fd = bus
        baseMVA, bus, gen, branch, success, et = \
            results["baseMVA"], results["bus"], results["gen"], \
            results["branch"], results["success"], results["et"]
        if 'f' in results:
            f = results["f"]
        else:
            f = None
    else:
        have_results_struct = 0
        if ppopt is None:
            ppopt = ppoption()   ## use default options
            if fd is None:
                fd = stdout         ## print to stdout by default
        if ppopt['OUT_ALL'] == 0:
            return     ## nothin' to see here, bail out now

    isOPF = f is not None    ## FALSE -> only simple PF data, TRUE -> OPF data

    ## options
    isDC            = ppopt['PF_DC']        ## use DC formulation?
    OUT_ALL         = ppopt['OUT_ALL']
    OUT_ANY         = OUT_ALL == 1     ## set to true if any pretty output is to be generated
    OUT_SYS_SUM     = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_SYS_SUM'])
    OUT_AREA_SUM    = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_AREA_SUM'])
    OUT_BUS         = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_BUS'])
    OUT_BRANCH      = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_BRANCH'])
    OUT_GEN         = (OUT_ALL == 1) or ((OUT_ALL == -1) and ppopt['OUT_GEN'])
    OUT_ANY         = OUT_ANY | ((OUT_ALL == -1) and
                        (OUT_SYS_SUM or OUT_AREA_SUM or OUT_BUS or
                         OUT_BRANCH or OUT_GEN))

    if OUT_ALL == -1:
        OUT_ALL_LIM = ppopt['OUT_ALL_LIM']
    elif OUT_ALL == 1:
        OUT_ALL_LIM = 2
    else:
        OUT_ALL_LIM = 0

    OUT_ANY         = OUT_ANY or (OUT_ALL_LIM >= 1)
    if OUT_ALL_LIM == -1:
        OUT_V_LIM       = ppopt['OUT_V_LIM']
        OUT_LINE_LIM    = ppopt['OUT_LINE_LIM']
        OUT_PG_LIM      = ppopt['OUT_PG_LIM']
        OUT_QG_LIM      = ppopt['OUT_QG_LIM']
    else:
        OUT_V_LIM       = OUT_ALL_LIM
        OUT_LINE_LIM    = OUT_ALL_LIM
        OUT_PG_LIM      = OUT_ALL_LIM
        OUT_QG_LIM      = OUT_ALL_LIM

    OUT_ANY         = OUT_ANY or ((OUT_ALL_LIM == -1) and (OUT_V_LIM or OUT_LINE_LIM or OUT_PG_LIM or OUT_QG_LIM))
    ptol = 1e-4        ## tolerance for displaying shadow prices

    ## create map of external bus numbers to bus indices
    i2e = bus[:, BUS_I].astype(int)
    e2i = zeros(max(i2e) + 1, int)
    e2i[i2e] = arange(bus.shape[0])

    ## sizes of things
    nb = bus.shape[0]      ## number of buses
    nl = branch.shape[0]   ## number of branches
    ng = gen.shape[0]      ## number of generators

    ## zero out some data to make printout consistent for DC case
    if isDC:
        bus[:, r_[QD, BS]]          = zeros((nb, 2))
        gen[:, r_[QG, QMAX, QMIN]]  = zeros((ng, 3))
        branch[:, r_[BR_R, BR_B]]   = zeros((nl, 2))

    ## parameters
    ties = find(bus[e2i[branch[:, F_BUS].real.astype(int)], BUS_AREA] !=
                   bus[e2i[branch[:, T_BUS].real.astype(int)], BUS_AREA])
                            ## area inter-ties
    tap = ones(nl)                           ## default tap ratio = 1 for lines
    xfmr = find(branch[:, TAP]).real           ## indices of transformers
    tap[xfmr] = branch[xfmr, TAP].real           ## include transformer tap ratios
    tap = tap * exp(-1j * pi / 180 * branch[:, SHIFT]) ## add phase shifters
    nzld = find((bus[:, PD] != 0.0) | (bus[:, QD] != 0.0))
    sorted_areas = sort(bus[:, BUS_AREA])
    ## area numbers
    s_areas = sorted_areas[r_[1, find(diff(sorted_areas)) + 1]]
    nzsh = find((bus[:, GS] != 0.0) | (bus[:, BS] != 0.0))
    allg = find( ~isload(gen) )
    ong  = find( (gen[:, GEN_STATUS] > 0) & ~isload(gen) )
    onld = find( (gen[:, GEN_STATUS] > 0) &  isload(gen) )
    V = bus[:, VM] * exp(-1j * pi / 180 * bus[:, VA])
    out = find(branch[:, BR_STATUS] == 0)        ## out-of-service branches
    nout = len(out)
    if isDC:
        loss = zeros(nl)
    else:
        loss = baseMVA * abs(V[e2i[ branch[:, F_BUS].real.astype(int) ]] / tap -
                             V[e2i[ branch[:, T_BUS].real.astype(int) ]])**2 / \
                    (branch[:, BR_R] - 1j * branch[:, BR_X])

    fchg = abs(V[e2i[ branch[:, F_BUS].real.astype(int) ]] / tap)**2 * branch[:, BR_B].real * baseMVA / 2
    tchg = abs(V[e2i[ branch[:, T_BUS].real.astype(int) ]]      )**2 * branch[:, BR_B].real * baseMVA / 2
    loss[out] = zeros(nout)
    fchg[out] = zeros(nout)
    tchg[out] = zeros(nout)

    ##----- print the stuff -----
    if OUT_ANY:
        ## convergence & elapsed time
        if success:
            fd.write('\nConverged in %.2f seconds' % et)
        else:
            fd.write('\nDid not converge (%.2f seconds)\n' % et)

        ## objective function value
        if isOPF:
            fd.write('\nObjective Function Value = %.2f $/hr' % f)

    if OUT_SYS_SUM:
        fd.write('\n================================================================================')
        fd.write('\n| PyPower (ppci) System Summary - these are not valid for pandapower DataFrames|')
        fd.write('\n================================================================================')
        fd.write('\n\nHow many?                How much?              P (MW)            Q (MVAr)')
        fd.write('\n---------------------    -------------------  -------------  -----------------')
        fd.write('\nBuses         %6d     Total Gen Capacity   %7.1f       %7.1f to %.1f' % (nb, sum(gen[allg, PMAX]), sum(gen[allg, QMIN]), sum(gen[allg, QMAX])))
        fd.write('\nGenerators     %5d     On-line Capacity     %7.1f       %7.1f to %.1f' % (len(allg), sum(gen[ong, PMAX]), sum(gen[ong, QMIN]), sum(gen[ong, QMAX])))
        fd.write('\nCommitted Gens %5d     Generation (actual)  %7.1f           %7.1f' % (len(ong), sum(gen[ong, PG]), sum(gen[ong, QG])))
        fd.write('\nLoads          %5d     Load                 %7.1f           %7.1f' % (len(nzld)+len(onld), sum(bus[nzld, PD])-sum(gen[onld, PG]), sum(bus[nzld, QD])-sum(gen[onld, QG])))
        fd.write('\n  Fixed        %5d       Fixed              %7.1f           %7.1f' % (len(nzld), sum(bus[nzld, PD]), sum(bus[nzld, QD])))
        fd.write('\n  Dispatchable %5d       Dispatchable       %7.1f of %-7.1f%7.1f' % (len(onld), -sum(gen[onld, PG]), -sum(gen[onld, PMIN]), -sum(gen[onld, QG])))
        fd.write('\nShunts         %5d     Shunt (inj)          %7.1f           %7.1f' % (len(nzsh),
            -sum(bus[nzsh, VM]**2 * bus[nzsh, GS]), sum(bus[nzsh, VM]**2 * bus[nzsh, BS]) ))
        fd.write('\nBranches       %5d     Losses (I^2 * Z)     %8.2f          %8.2f' % (nl, sum(loss.real), sum(loss.imag) ))
        fd.write('\nTransformers   %5d     Branch Charging (inj)     -            %7.1f' % (len(xfmr), sum(fchg) + sum(tchg) ))
        fd.write('\nInter-ties     %5d     Total Inter-tie Flow %7.1f           %7.1f' % (len(ties), sum(abs(branch[ties, PF]-branch[ties, PT])) / 2, sum(abs(branch[ties, QF]-branch[ties, QT])) / 2))
        fd.write('\nAreas          %5d' % len(s_areas))
        fd.write('\n')
        fd.write('\n                          Minimum                      Maximum')
        fd.write('\n                 -------------------------  --------------------------------')
        minv = min(bus[:, VM])
        mini = argmin(bus[:, VM])
        maxv = max(bus[:, VM])
        maxi = argmax(bus[:, VM])
        fd.write('\nVoltage Magnitude %7.3f p.u. @ bus %-4d     %7.3f p.u. @ bus %-4d' % (minv, bus[mini, BUS_I], maxv, bus[maxi, BUS_I]))
        minv = min(bus[:, VA])
        mini = argmin(bus[:, VA])
        maxv = max(bus[:, VA])
        maxi = argmax(bus[:, VA])
        fd.write('\nVoltage Angle   %8.2f deg   @ bus %-4d   %8.2f deg   @ bus %-4d' % (minv, bus[mini, BUS_I], maxv, bus[maxi, BUS_I]))
        if not isDC:
            maxv = max(loss.real)
            maxi = argmax(loss.real)
            fd.write('\nP Losses (I^2*R)             -              %8.2f MW    @ line %d-%d' % (maxv, branch[maxi, F_BUS].real, branch[maxi, T_BUS].real))
            maxv = max(loss.imag)
            maxi = argmax(loss.imag)
            fd.write('\nQ Losses (I^2*X)             -              %8.2f MVAr  @ line %d-%d' % (maxv, branch[maxi, F_BUS].real, branch[maxi, T_BUS].real))
        if isOPF:
            minv = min(bus[:, LAM_P])
            mini = argmin(bus[:, LAM_P])
            maxv = max(bus[:, LAM_P])
            maxi = argmax(bus[:, LAM_P])
            fd.write('\nLambda P        %8.2f $/MWh @ bus %-4d   %8.2f $/MWh @ bus %-4d' % (minv, bus[mini, BUS_I], maxv, bus[maxi, BUS_I]))
            minv = min(bus[:, LAM_Q])
            mini = argmin(bus[:, LAM_Q])
            maxv = max(bus[:, LAM_Q])
            maxi = argmax(bus[:, LAM_Q])
            fd.write('\nLambda Q        %8.2f $/MWh @ bus %-4d   %8.2f $/MWh @ bus %-4d' % (minv, bus[mini, BUS_I], maxv, bus[maxi, BUS_I]))
        fd.write('\n')

    if OUT_AREA_SUM:
        fd.write('\n================================================================================')
        fd.write('\n|     Area Summary                                                             |')
        fd.write('\n================================================================================')
        fd.write('\nArea  # of      # of Gens        # of Loads         # of    # of   # of   # of')
        fd.write('\n Num  Buses   Total  Online   Total  Fixed  Disp    Shunt   Brchs  Xfmrs   Ties')
        fd.write('\n----  -----   -----  ------   -----  -----  -----   -----   -----  -----  -----')
        for i in range(len(s_areas)):
            a = s_areas[i]
            ib = find(bus[:, BUS_AREA] == a)
            ig = find((bus[e2i[gen[:, GEN_BUS].astype(int)], BUS_AREA] == a) & ~isload(gen))
            igon = find((bus[e2i[gen[:, GEN_BUS].astype(int)], BUS_AREA] == a) & (gen[:, GEN_STATUS] > 0) & ~isload(gen))
            ildon = find((bus[e2i[gen[:, GEN_BUS].astype(int)], BUS_AREA] == a) & (gen[:, GEN_STATUS] > 0) & isload(gen))
            inzld = find((bus[:, BUS_AREA] == a) & logical_or(bus[:, PD], bus[:, QD]))
            inzsh = find((bus[:, BUS_AREA] == a) & logical_or(bus[:, GS], bus[:, BS]))
            ibrch = find((bus[e2i[branch[:, F_BUS].real.astype(int)], BUS_AREA] == a) & (bus[e2i[branch[:, T_BUS].real.astype(int)], BUS_AREA] == a))
            in_tie = find((bus[e2i[branch[:, F_BUS].real.astype(int)], BUS_AREA] == a) & (bus[e2i[branch[:, T_BUS].real.astype(int)], BUS_AREA] != a))
            out_tie = find((bus[e2i[branch[:, F_BUS].real.astype(int)], BUS_AREA] != a) & (bus[e2i[branch[:, T_BUS].real.astype(int)], BUS_AREA] == a))
            if not any(xfmr + 1):
                nxfmr = 0
            else:
                nxfmr = len(find((bus[e2i[branch[xfmr, F_BUS].real.astype(int)], BUS_AREA] == a) & (bus[e2i[branch[xfmr, T_BUS].real.astype(int)], BUS_AREA] == a)))
            fd.write('\n%3d  %6d   %5d  %5d   %5d  %5d  %5d   %5d   %5d  %5d  %5d' %
                (a, len(ib), len(ig), len(igon), \
                len(inzld)+len(ildon), len(inzld), len(ildon), \
                len(inzsh), len(ibrch), nxfmr, len(in_tie)+len(out_tie)))

        fd.write('\n----  -----   -----  ------   -----  -----  -----   -----   -----  -----  -----')
        fd.write('\nTot: %6d   %5d  %5d   %5d  %5d  %5d   %5d   %5d  %5d  %5d' %
            (nb, len(allg), len(ong), len(nzld)+len(onld),
            len(nzld), len(onld), len(nzsh), nl, len(xfmr), len(ties)))
        fd.write('\n')
        fd.write('\nArea      Total Gen Capacity           On-line Gen Capacity         Generation')
        fd.write('\n Num     MW           MVAr            MW           MVAr             MW    MVAr')
        fd.write('\n----   ------  ------------------   ------  ------------------    ------  ------')
        for i in range(len(s_areas)):
            a = s_areas[i]
            ig = find((bus[e2i[gen[:, GEN_BUS].astype(int)], BUS_AREA] == a) & ~isload(gen))
            igon = find((bus[e2i[gen[:, GEN_BUS].astype(int)], BUS_AREA] == a) & (gen[:, GEN_STATUS] > 0) & ~isload(gen))
            fd.write('\n%3d   %7.1f  %7.1f to %-.1f  %7.1f  %7.1f to %-7.1f   %7.1f %7.1f' %
                (a, sum(gen[ig, PMAX]), sum(gen[ig, QMIN]), sum(gen[ig, QMAX]),
                sum(gen[igon, PMAX]), sum(gen[igon, QMIN]), sum(gen[igon, QMAX]),
                sum(gen[igon, PG]), sum(gen[igon, QG]) ))

        fd.write('\n----   ------  ------------------   ------  ------------------    ------  ------')
#        fd.write('\nTot:  %7.1f  %7.1f to %-7.1f  %7.1f  %7.1f to %-7.1f   %7.1f %7.1f' %
#                (sum(gen[allg, PMAX]), sum(gen[allg, QMIN]), sum(gen[allg, QMAX]),
#                sum(gen[ong, PMAX]), sum(gen[ong, QMIN]), sum(gen[ong, QMAX]),
#                sum(gen[ong, PG]), sum(gen[ong, QG]) ))
        fd.write('\n')
        fd.write('\nArea    Disp Load Cap       Disp Load         Fixed Load        Total Load')
        fd.write('\n Num      MW     MVAr       MW     MVAr       MW     MVAr       MW     MVAr')
        fd.write('\n----    ------  ------    ------  ------    ------  ------    ------  ------')
        Qlim = (gen[:, QMIN] == 0) * gen[:, QMAX] + (gen[:, QMAX] == 0) * gen[:, QMIN]
        for i in range(len(s_areas)):
            a = s_areas[i]
            ildon = find((bus[e2i[gen[:, GEN_BUS].astype(int)], BUS_AREA] == a) & (gen[:, GEN_STATUS] > 0) & isload(gen))
            inzld = find((bus[:, BUS_AREA] == a) & logical_or(bus[:, PD], bus[:, QD]))
            fd.write('\n%3d    %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f' %
                (a, -sum(gen[ildon, PMIN]),
                -sum(Qlim[ildon]),
                -sum(gen[ildon, PG]), -sum(gen[ildon, QG]),
                sum(bus[inzld, PD]), sum(bus[inzld, QD]),
                -sum(gen[ildon, PG]) + sum(bus[inzld, PD]),
                -sum(gen[ildon, QG]) + sum(bus[inzld, QD]) ))

        fd.write('\n----    ------  ------    ------  ------    ------  ------    ------  ------')
        fd.write('\nTot:   %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f   %7.1f %7.1f' %
                (-sum(gen[onld, PMIN]),
                -sum(Qlim[onld]),
                -sum(gen[onld, PG]), -sum(gen[onld, QG]),
                sum(bus[nzld, PD]), sum(bus[nzld, QD]),
                -sum(gen[onld, PG]) + sum(bus[nzld, PD]),
                -sum(gen[onld, QG]) + sum(bus[nzld, QD])) )
        fd.write('\n')
        fd.write('\nArea      Shunt Inj        Branch      Series Losses      Net Export')
        fd.write('\n Num      MW     MVAr     Charging      MW     MVAr       MW     MVAr')
        fd.write('\n----    ------  ------    --------    ------  ------    ------  ------')
        for i in range(len(s_areas)):
            a = s_areas[i]
            inzsh   = find((bus[:, BUS_AREA] == a) & logical_or(bus[:, GS], bus[:, BS]))
            ibrch   = find((bus[e2i[branch[:, F_BUS].real.astype(int)], BUS_AREA] == a) & (bus[e2i[branch[:, T_BUS].real.astype(int)], BUS_AREA] == a) & branch[:, BR_STATUS].astype(bool))
            in_tie  = find((bus[e2i[branch[:, F_BUS].real.astype(int)], BUS_AREA] != a) & (bus[e2i[branch[:, T_BUS].real.astype(int)], BUS_AREA] == a) & branch[:, BR_STATUS].astype(bool))
            out_tie = find((bus[e2i[branch[:, F_BUS].real.astype(int)], BUS_AREA] == a) & (bus[e2i[branch[:, T_BUS].real.astype(int)], BUS_AREA] != a) & branch[:, BR_STATUS].astype(bool))
            fd.write('\n%3d    %7.1f %7.1f    %7.1f    %7.2f %7.2f   %7.1f %7.1f' %
                (a, -sum(bus[inzsh, VM]**2 * bus[inzsh, GS]),
                 sum(bus[inzsh, VM]**2 * bus[inzsh, BS]),
                 sum(fchg[ibrch]) + sum(tchg[ibrch]) + sum(fchg[out_tie]) + sum(tchg[in_tie]),
                 sum(real(loss[ibrch])) + sum(real(loss[r_[in_tie, out_tie]])) / 2,
                 sum(imag(loss[ibrch])) + sum(imag(loss[r_[in_tie, out_tie]])) / 2,
                 sum(branch[in_tie, PT])+sum(branch[out_tie, PF]) - sum(real(loss[r_[in_tie, out_tie]])) / 2,
                 sum(branch[in_tie, QT])+sum(branch[out_tie, QF]) - sum(imag(loss[r_[in_tie, out_tie]])) / 2  ))

        fd.write('\n----    ------  ------    --------    ------  ------    ------  ------')
        fd.write('\nTot:   %7.1f %7.1f    %7.1f    %7.2f %7.2f       -       -' %
            (-sum(bus[nzsh, VM]**2 * bus[nzsh, GS]),
             sum(bus[nzsh, VM]**2 * bus[nzsh, BS]),
             sum(fchg) + sum(tchg), sum(real(loss)), sum(imag(loss)) ))
        fd.write('\n')

    ## generator data
    if OUT_GEN:
        if isOPF:
            genlamP = bus[e2i[gen[:, GEN_BUS].astype(int)], LAM_P]
            genlamQ = bus[e2i[gen[:, GEN_BUS].astype(int)], LAM_Q]

        fd.write('\n================================================================================')
        fd.write('\n|     Generator Data                                                           |')
        fd.write('\n================================================================================')
        fd.write('\n Gen   Bus   Status     Pg        Qg   ')
        if isOPF: fd.write('   Lambda ($/MVA-hr)')
        fd.write('\n  #     #              (MW)     (MVAr) ')
        if isOPF: fd.write('     P         Q    ')
        fd.write('\n----  -----  ------  --------  --------')
        if isOPF: fd.write('  --------  --------')
        for k in range(len(ong)):
            i = ong[k]
            fd.write('\n%3d %6d     %2d ' % (i, gen[i, GEN_BUS], gen[i, GEN_STATUS]))
            if (gen[i, GEN_STATUS] > 0) & logical_or(gen[i, PG], gen[i, QG]):
                fd.write('%10.2f%10.2f' % (gen[i, PG], gen[i, QG]))
            else:
                fd.write('       -         -  ')
            if isOPF: fd.write('%10.2f%10.2f' % (genlamP[i], genlamQ[i]))

        fd.write('\n                     --------  --------')
        fd.write('\n            Total: %9.2f%10.2f' % (sum(gen[ong, PG]), sum(gen[ong, QG])))
        fd.write('\n')
        if any(onld + 1):
            fd.write('\n================================================================================')
            fd.write('\n|     Dispatchable Load Data                                                   |')
            fd.write('\n================================================================================')
            fd.write('\n Gen   Bus   Status     Pd        Qd   ')
            if isOPF: fd.write('   Lambda ($/MVA-hr)')
            fd.write('\n  #     #              (MW)     (MVAr) ')
            if isOPF: fd.write('     P         Q    ')
            fd.write('\n----  -----  ------  --------  --------')
            if isOPF: fd.write('  --------  --------')
            for k in range(len(onld)):
                i = onld[k]
                fd.write('\n%3d %6d     %2d ' % (i, gen[i, GEN_BUS], gen[i, GEN_STATUS]))
                if (gen[i, GEN_STATUS] > 0) & logical_or(gen[i, PG], gen[i, QG]):
                    fd.write('%10.2f%10.2f' % (-gen[i, PG], -gen[i, QG]))
                else:
                    fd.write('       -         -  ')

                if isOPF: fd.write('%10.2f%10.2f' % (genlamP[i], genlamQ[i]))
            fd.write('\n                     --------  --------')
            fd.write('\n            Total: %9.2f%10.2f' % (-sum(gen[onld, PG]), -sum(gen[onld, QG])))
            fd.write('\n')

    ## bus data
    if OUT_BUS:
        fd.write('\n================================================================================')
        fd.write('\n|     Bus Data                                                                 |')
        fd.write('\n================================================================================')
        fd.write('\n Bus      Voltage          Generation             Load        ')
        if isOPF: fd.write('  Lambda($/MVA-hr)')
        fd.write('\n  #   Mag(pu) Ang(deg)   P (MW)   Q (MVAr)   P (MW)   Q (MVAr)')
        if isOPF: fd.write('     P        Q   ')
        fd.write('\n----- ------- --------  --------  --------  --------  --------')
        if isOPF: fd.write('  -------  -------')
        for i in range(nb):
            fd.write('\n%5d%7.3f%9.3f' % tuple(bus[i, [BUS_I, VM, VA]]))
            if bus[i, BUS_TYPE] == REF:
                fd.write('*')
            else:
                fd.write(' ')
            g  = find((gen[:, GEN_STATUS] > 0) & (gen[:, GEN_BUS] == bus[i, BUS_I]) &
                        ~isload(gen))
            ld = find((gen[:, GEN_STATUS] > 0) & (gen[:, GEN_BUS] == bus[i, BUS_I]) &
                        isload(gen))
            if any(g + 1):
                fd.write('%9.2f%10.2f' % (sum(gen[g, PG]), sum(gen[g, QG])))
            else:
                fd.write('      -         -  ')

            if logical_or(bus[i, PD], bus[i, QD]) | any(ld + 1):
                if any(ld + 1):
                    fd.write('%10.2f*%9.2f*' % (bus[i, PD] - sum(gen[ld, PG]),
                                                bus[i, QD] - sum(gen[ld, QG])))
                else:
                    fd.write('%10.2f%10.2f ' % tuple(bus[i, [PD, QD]]))
            else:
                fd.write('       -         -   ')
            if isOPF:
                fd.write('%9.3f' % bus[i, LAM_P])
                if abs(bus[i, LAM_Q]) > ptol:
                    fd.write('%8.3f' % bus[i, LAM_Q])
                else:
                    fd.write('     -')
        fd.write('\n                        --------  --------  --------  --------')
        fd.write('\n               Total: %9.2f %9.2f %9.2f %9.2f' %
            (sum(gen[ong, PG]), sum(gen[ong, QG]),
             sum(bus[nzld, PD]) - sum(gen[onld, PG]),
             sum(bus[nzld, QD]) - sum(gen[onld, QG])))
        fd.write('\n')

    ## branch data
    if OUT_BRANCH:
        fd.write('\n================================================================================')
        fd.write('\n|     Branch Data                                                              |')
        fd.write('\n================================================================================')
        fd.write('\nBrnch   From   To    From Bus Injection   To Bus Injection     Loss (I^2 * Z)  ')
        fd.write('\n  #     Bus    Bus    P (MW)   Q (MVAr)   P (MW)   Q (MVAr)   P (MW)   Q (MVAr)')
        fd.write('\n-----  -----  -----  --------  --------  --------  --------  --------  --------')
        for i in range(nl):
            fd.write('\n%4d%7d%7d%10.2f%10.2f%10.2f%10.2f%10.3f%10.2f' %
                (i, branch[i, F_BUS].real, branch[i, T_BUS].real,
                     branch[i, PF].real, branch[i, QF].real, branch[i, PT].real, branch[i, QT].real,
                     loss[i].real, loss[i].imag))
        fd.write('\n                                                             --------  --------')
        fd.write('\n                                                    Total:%10.3f%10.2f' %
                (sum(real(loss)), sum(imag(loss))))
        fd.write('\n')

    ##-----  constraint data  -----
    if isOPF:
        ctol = ppopt['OPF_VIOLATION']   ## constraint violation tolerance
        ## voltage constraints
        if (not isDC) & (OUT_V_LIM == 2 | (OUT_V_LIM == 1 &
                             (any(bus[:, VM] < bus[:, VMIN] + ctol) |
                              any(bus[:, VM] > bus[:, VMAX] - ctol) |
                              any(bus[:, MU_VMIN] > ptol) |
                              any(bus[:, MU_VMAX] > ptol)))):
            fd.write('\n================================================================================')
            fd.write('\n|     Voltage Constraints                                                      |')
            fd.write('\n================================================================================')
            fd.write('\nBus #  Vmin mu    Vmin    |V|   Vmax    Vmax mu')
            fd.write('\n-----  --------   -----  -----  -----   --------')
            for i in range(nb):
                if (OUT_V_LIM == 2) | (OUT_V_LIM == 1 &
                             ((bus[i, VM] < bus[i, VMIN] + ctol) |
                              (bus[i, VM] > bus[i, VMAX] - ctol) |
                              (bus[i, MU_VMIN] > ptol) |
                              (bus[i, MU_VMAX] > ptol))):
                    fd.write('\n%5d' % bus[i, BUS_I])
                    if ((bus[i, VM] < bus[i, VMIN] + ctol) |
                            (bus[i, MU_VMIN] > ptol)):
                        fd.write('%10.3f' % bus[i, MU_VMIN])
                    else:
                        fd.write('      -   ')

                    fd.write('%8.3f%7.3f%7.3f' % tuple(bus[i, [VMIN, VM, VMAX]]))
                    if (bus[i, VM] > bus[i, VMAX] - ctol) | (bus[i, MU_VMAX] > ptol):
                        fd.write('%10.3f' % bus[i, MU_VMAX])
                    else:
                        fd.write('      -    ')
            fd.write('\n')

        ## generator P constraints
        if (OUT_PG_LIM == 2) | \
                ((OUT_PG_LIM == 1) & (any(gen[ong, PG] < gen[ong, PMIN] + ctol) |
                                      any(gen[ong, PG] > gen[ong, PMAX] - ctol) |
                                      any(gen[ong, MU_PMIN] > ptol) |
                                      any(gen[ong, MU_PMAX] > ptol))) | \
                ((not isDC) & ((OUT_QG_LIM == 2) |
                ((OUT_QG_LIM == 1) & (any(gen[ong, QG] < gen[ong, QMIN] + ctol) |
                                      any(gen[ong, QG] > gen[ong, QMAX] - ctol) |
                                      any(gen[ong, MU_QMIN] > ptol) |
                                      any(gen[ong, MU_QMAX] > ptol))))):
            fd.write('\n================================================================================')
            fd.write('\n|     Generation Constraints                                                   |')
            fd.write('\n================================================================================')

        if (OUT_PG_LIM == 2) | ((OUT_PG_LIM == 1) &
                                 (any(gen[ong, PG] < gen[ong, PMIN] + ctol) |
                                  any(gen[ong, PG] > gen[ong, PMAX] - ctol) |
                                  any(gen[ong, MU_PMIN] > ptol) |
                                  any(gen[ong, MU_PMAX] > ptol))):
            fd.write('\n Gen   Bus                Active Power Limits')
            fd.write('\n  #     #    Pmin mu    Pmin       Pg       Pmax    Pmax mu')
            fd.write('\n----  -----  -------  --------  --------  --------  -------')
            for k in range(len(ong)):
                i = ong[k]
                if (OUT_PG_LIM == 2) | ((OUT_PG_LIM == 1) &
                            ((gen[i, PG] < gen[i, PMIN] + ctol) |
                             (gen[i, PG] > gen[i, PMAX] - ctol) |
                             (gen[i, MU_PMIN] > ptol) | (gen[i, MU_PMAX] > ptol))):
                    fd.write('\n%4d%6d ' % (i, gen[i, GEN_BUS]))
                    if (gen[i, PG] < gen[i, PMIN] + ctol) | (gen[i, MU_PMIN] > ptol):
                        fd.write('%8.3f' % gen[i, MU_PMIN])
                    else:
                        fd.write('     -  ')
                    if gen[i, PG]:
                        fd.write('%10.2f%10.2f%10.2f' % tuple(gen[i, [PMIN, PG, PMAX]]))
                    else:
                        fd.write('%10.2f       -  %10.2f' % tuple(gen[i, [PMIN, PMAX]]))
                    if (gen[i, PG] > gen[i, PMAX] - ctol) | (gen[i, MU_PMAX] > ptol):
                        fd.write('%9.3f' % gen[i, MU_PMAX])
                    else:
                        fd.write('      -  ')
            fd.write('\n')

        ## generator Q constraints
        if (not isDC) & ((OUT_QG_LIM == 2) | ((OUT_QG_LIM == 1) &
                                 (any(gen[ong, QG] < gen[ong, QMIN] + ctol) |
                                  any(gen[ong, QG] > gen[ong, QMAX] - ctol) |
                                  any(gen[ong, MU_QMIN] > ptol) |
                                  any(gen[ong, MU_QMAX] > ptol)))):
            fd.write('\nGen  Bus              Reactive Power Limits')
            fd.write('\n #    #   Qmin mu    Qmin       Qg       Qmax    Qmax mu')
            fd.write('\n---  ---  -------  --------  --------  --------  -------')
            for k in range(len(ong)):
                i = ong[k]
                if (OUT_QG_LIM == 2) | ((OUT_QG_LIM == 1) &
                            ((gen[i, QG] < gen[i, QMIN] + ctol) |
                             (gen[i, QG] > gen[i, QMAX] - ctol) |
                             (gen[i, MU_QMIN] > ptol) |
                             (gen[i, MU_QMAX] > ptol))):
                    fd.write('\n%3d%5d' % (i, gen[i, GEN_BUS]))
                    if (gen[i, QG] < gen[i, QMIN] + ctol) | (gen[i, MU_QMIN] > ptol):
                        fd.write('%8.3f' % gen[i, MU_QMIN])
                    else:
                        fd.write('     -  ')
                    if gen[i, QG]:
                        fd.write('%10.2f%10.2f%10.2f' % tuple(gen[i, [QMIN, QG, QMAX]]))
                    else:
                        fd.write('%10.2f       -  %10.2f' % tuple(gen[i, [QMIN, QMAX]]))

                    if (gen[i, QG] > gen[i, QMAX] - ctol) | (gen[i, MU_QMAX] > ptol):
                        fd.write('%9.3f' % gen[i, MU_QMAX])
                    else:
                        fd.write('      -  ')
            fd.write('\n')

        ## dispatchable load P constraints
        if (OUT_PG_LIM == 2) | (OUT_QG_LIM == 2) | \
                ((OUT_PG_LIM == 1) & (any(gen[onld, PG] < gen[onld, PMIN] + ctol) |
                                      any(gen[onld, PG] > gen[onld, PMAX] - ctol) |
                                      any(gen[onld, MU_PMIN] > ptol) |
                                      any(gen[onld, MU_PMAX] > ptol))) | \
                ((OUT_QG_LIM == 1) & (any(gen[onld, QG] < gen[onld, QMIN] + ctol) |
                                      any(gen[onld, QG] > gen[onld, QMAX] - ctol) |
                                      any(gen[onld, MU_QMIN] > ptol) |
                                      any(gen[onld, MU_QMAX] > ptol))):
            fd.write('\n================================================================================')
            fd.write('\n|     Dispatchable Load Constraints                                            |')
            fd.write('\n================================================================================')
        if (OUT_PG_LIM == 2) | ((OUT_PG_LIM == 1) &
                                 (any(gen[onld, PG] < gen[onld, PMIN] + ctol) |
                                  any(gen[onld, PG] > gen[onld, PMAX] - ctol) |
                                  any(gen[onld, MU_PMIN] > ptol) |
                                  any(gen[onld, MU_PMAX] > ptol))):
            fd.write('\nGen  Bus               Active Power Limits')
            fd.write('\n #    #   Pmin mu    Pmin       Pg       Pmax    Pmax mu')
            fd.write('\n---  ---  -------  --------  --------  --------  -------')
            for k in range(len(onld)):
                i = onld[k]
                if (OUT_PG_LIM == 2) | ((OUT_PG_LIM == 1) &
                            ((gen[i, PG] < gen[i, PMIN] + ctol) |
                             (gen[i, PG] > gen[i, PMAX] - ctol) |
                             (gen[i, MU_PMIN] > ptol) |
                             (gen[i, MU_PMAX] > ptol))):
                    fd.write('\n%3d%5d' % (i, gen[i, GEN_BUS]))
                    if (gen[i, PG] < gen[i, PMIN] + ctol) | (gen[i, MU_PMIN] > ptol):
                        fd.write('%8.3f' % gen[i, MU_PMIN])
                    else:
                        fd.write('     -  ')
                    if gen[i, PG]:
                        fd.write('%10.2f%10.2f%10.2f' % gen[i, [PMIN, PG, PMAX]])
                    else:
                        fd.write('%10.2f       -  %10.2f' % gen[i, [PMIN, PMAX]])

                    if (gen[i, PG] > gen[i, PMAX] - ctol) | (gen[i, MU_PMAX] > ptol):
                        fd.write('%9.3f' % gen[i, MU_PMAX])
                    else:
                        fd.write('      -  ')
            fd.write('\n')

        ## dispatchable load Q constraints
        if (not isDC) & ((OUT_QG_LIM == 2) | ((OUT_QG_LIM == 1) &
                                 (any(gen[onld, QG] < gen[onld, QMIN] + ctol) |
                                  any(gen[onld, QG] > gen[onld, QMAX] - ctol) |
                                  any(gen[onld, MU_QMIN] > ptol) |
                                  any(gen[onld, MU_QMAX] > ptol)))):
            fd.write('\nGen  Bus              Reactive Power Limits')
            fd.write('\n #    #   Qmin mu    Qmin       Qg       Qmax    Qmax mu')
            fd.write('\n---  ---  -------  --------  --------  --------  -------')
            for k in range(len(onld)):
                i = onld[k]
                if (OUT_QG_LIM == 2) | ((OUT_QG_LIM == 1) &
                            ((gen[i, QG] < gen[i, QMIN] + ctol) |
                             (gen[i, QG] > gen[i, QMAX] - ctol) |
                             (gen[i, MU_QMIN] > ptol) |
                             (gen[i, MU_QMAX] > ptol))):
                    fd.write('\n%3d%5d' % (i, gen(i, GEN_BUS)))
                    if (gen[i, QG] < gen[i, QMIN] + ctol) | (gen[i, MU_QMIN] > ptol):
                        fd.write('%8.3f' % gen[i, MU_QMIN])
                    else:
                        fd.write('     -  ')

                    if gen[i, QG]:
                        fd.write('%10.2f%10.2f%10.2f' % gen[i, [QMIN, QG, QMAX]])
                    else:
                        fd.write('%10.2f       -  %10.2f' % gen[i, [QMIN, QMAX]])

                    if (gen[i, QG] > gen[i, QMAX] - ctol) | (gen[i, MU_QMAX] > ptol):
                        fd.write('%9.3f' % gen[i, MU_QMAX])
                    else:
                        fd.write('      -  ')
            fd.write('\n')

        ## line flow constraints
        if (ppopt['OPF_FLOW_LIM'] == 1) | isDC:  ## P limit
            Ff = branch[:, PF]
            Ft = branch[:, PT]
            strg = '\n  #     Bus    Pf  mu     Pf      |Pmax|      Pt      Pt  mu   Bus'
        elif ppopt['OPF_FLOW_LIM'] == 2:   ## |I| limit
            Ff = abs( (branch[:, PF] + 1j * branch[:, QF]) / V[e2i[branch[:, F_BUS].astype(int)]] )
            Ft = abs( (branch[:, PT] + 1j * branch[:, QT]) / V[e2i[branch[:, T_BUS].astype(int)]] )
            strg = '\n  #     Bus   |If| mu    |If|     |Imax|     |It|    |It| mu   Bus'
        else:                ## |S| limit
            Ff = abs(branch[:, PF] + 1j * branch[:, QF])
            Ft = abs(branch[:, PT] + 1j * branch[:, QT])
            strg = '\n  #     Bus   |Sf| mu    |Sf|     |Smax|     |St|    |St| mu   Bus'

        if (OUT_LINE_LIM == 2) | ((OUT_LINE_LIM == 1) &
                            (any((branch[:, RATE_A] != 0) & (abs(Ff) > branch[:, RATE_A] - ctol)) |
                             any((branch[:, RATE_A] != 0) & (abs(Ft) > branch[:, RATE_A] - ctol)) |
                             any(branch[:, MU_SF] > ptol) |
                             any(branch[:, MU_ST] > ptol))):
            fd.write('\n================================================================================')
            fd.write('\n|     Branch Flow Constraints                                                  |')
            fd.write('\n================================================================================')
            fd.write('\nBrnch   From     "From" End        Limit       "To" End        To')
            fd.write(strg)
            fd.write('\n-----  -----  -------  --------  --------  --------  -------  -----')
            for i in range(nl):
                if (OUT_LINE_LIM == 2) | ((OUT_LINE_LIM == 1) &
                       (((branch[i, RATE_A] != 0) & (abs(Ff[i]) > branch[i, RATE_A] - ctol)) |
                        ((branch[i, RATE_A] != 0) & (abs(Ft[i]) > branch[i, RATE_A] - ctol)) |
                        (branch[i, MU_SF] > ptol) | (branch[i, MU_ST] > ptol))):
                    fd.write('\n%4d%7d' % (i, branch[i, F_BUS]))
                    if (Ff[i] > branch[i, RATE_A] - ctol) | (branch[i, MU_SF] > ptol):
                        fd.write('%10.3f' % branch[i, MU_SF])
                    else:
                        fd.write('      -   ')

                    fd.write('%9.2f%10.2f%10.2f' %
                        (Ff[i], branch[i, RATE_A], Ft[i]))
                    if (Ft[i] > branch[i, RATE_A] - ctol) | (branch[i, MU_ST] > ptol):
                        fd.write('%10.3f' % branch[i, MU_ST])
                    else:
                        fd.write('      -   ')
                    fd.write('%6d' % branch[i, T_BUS])
            fd.write('\n')

    ## execute userfcn callbacks for 'printpf' stage
    if have_results_struct and 'userfcn' in results:
        if not isOPF:  ## turn off option for all constraints if it isn't an OPF
            ppopt = ppoption(ppopt, 'OUT_ALL_LIM', 0)
        run_userfcn(results["userfcn"], 'printpf', results, fd, ppopt)
