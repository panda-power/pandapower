# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Defines constants for named column indices to bus matrix.

Some examples of usage, after defining the constants using the line above,
are::

    Pd = bus[3, PD]     # get the real power demand at bus 4
    bus[:, VMIN] = 0.95 # set the min voltage magnitude to 0.95 at all buses

The index, name and meaning of each column of the bus matrix is given
below:

columns 0-12 must be included in input matrix (in case file)
    0.  C{BUS_I}       bus number (1 to 29997)
    1.  C{BUS_TYPE}    bus type (1 = PQ, 2 = PV, 3 = ref, 4 = isolated)
    2.  C{PD}          real power demand (MW)
    3.  C{QD}          reactive power demand (MVAr)
    4.  C{GS}          shunt conductance (MW at V = 1.0 p.u.)
    5.  C{BS}          shunt susceptance (MVAr at V = 1.0 p.u.)
    6.  C{BUS_AREA}    area number, 1-100
    7.  C{VM}          voltage magnitude (p.u.)
    8.  C{VA}          voltage angle (degrees)
    9.  C{BASE_KV}     base voltage (kV)
    10. C{ZONE}        loss zone (1-999)
    11. C{VMAX}        maximum voltage magnitude (p.u.)
    12. C{VMIN}        minimum voltage magnitude (p.u.)

columns 13-16 are added to matrix after OPF solution
they are typically not present in the input matrix

(assume OPF objective function has units, u)
    13. C{LAM_P}       Lagrange multiplier on real power mismatch (u/MW)
    14. C{LAM_Q}       Lagrange multiplier on reactive power mismatch (u/MVAr)
    15. C{MU_VMAX}     Kuhn-Tucker multiplier on upper voltage limit (u/p.u.)
    16. C{MU_VMIN}     Kuhn-Tucker multiplier on lower voltage limit (u/p.u.)

additional constants, used to assign/compare values in the C{BUS_TYPE} column
    1.  C{PQ}    PQ bus
    2.  C{PV}    PV bus
    3.  C{REF}   reference bus
    4.  C{NONE}  isolated bus

@author: Ray Zimmerman (PSERC Cornell)
@author: Richard Lincoln
"""

# define bus types
PQ = 1
PV = 2
REF = 3
NONE = 4

# define the indices
BUS_I = 0    # bus number (1 to 29997)
BUS_TYPE = 1    # bus type
PD = 2    # Pd, real power demand (MW)
QD = 3    # Qd, reactive power demand (MVAr)
GS = 4    # Gs, shunt conductance (MW at V = 1.0 p.u.)
BS = 5    # Bs, shunt susceptance (MVAr at V = 1.0 p.u.)
BUS_AREA = 6    # area number, 1-100
VM = 7    # Vm, voltage magnitude (p.u.)
VA = 8    # Va, voltage angle (degrees)
BASE_KV = 9    # baseKV, base voltage (kV)
ZONE = 10   # zone, loss zone (1-999)
VMAX = 11   # maxVm, maximum voltage magnitude (p.u.)
VMIN = 12   # minVm, minimum voltage magnitude (p.u.)

# included in opf solution, not necessarily in input
# assume objective function has units, u
LAM_P = 13   # Lagrange multiplier on real power mismatch (u/MW)
LAM_Q = 14   # Lagrange multiplier on reactive power mismatch (u/MVAr)
MU_VMAX = 15   # Kuhn-Tucker multiplier on upper voltage limit (u/p.u.)
MU_VMIN = 16   # Kuhn-Tucker multiplier on lower voltage limit (u/p.u.)

# Additional pandapower extensions to ppc
CID = 13   # coefficient of constant current load at rated voltage in range [0,1]
CZD = 14   # coefficient of constant impedance load at rated voltage in range [0,1]

bus_cols = 15
