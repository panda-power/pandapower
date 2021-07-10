# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


# define the indices
from pandapower.pypower.idx_bus import bus_cols as start

R_EQUIV     = start + 0
X_EQUIV     = start + 1
KAPPA       = start + 2
M           = start + 3
C_MIN       = start + 4
C_MAX       = start + 5
IKSS1       = start + 6
IKSS2       = start + 7
IKCV        = start + 8
IP          = start + 9
IB          = start + 10
ITH         = start + 11
IK          = start + 12
R_EQUIV_OHM = start + 13
X_EQUIV_OHM = start + 14
K_G         = start + 15
K_SG        = start + 16
V_G         = start + 17
PS_TRAFO_IX = start + 18
GS_P        = start + 19
BS_P        = start + 20
SKSS        = start + 21

bus_cols_sc = 22