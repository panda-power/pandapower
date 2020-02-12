# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Builds index lists of each type of bus.
"""

from numpy import ones, flatnonzero as find
from pandapower.pypower.idx_bus import BUS_TYPE, REF, PV, PQ
from pandapower.pypower.idx_gen import GEN_BUS, GEN_STATUS
from scipy.sparse import csr_matrix as sparse


def bustypes(bus, gen):
    """Builds index lists of each type of bus (C{REF}, C{PV}, C{PQ}).

    Generators with "out-of-service" status are treated as L{PQ} buses with
    zero generation (regardless of C{Pg}/C{Qg} values in gen). Expects C{bus}
    and C{gen} have been converted to use internal consecutive bus numbering.

    @param bus: bus data
    @param gen: generator data
    @return: index lists of each bus type

    @author: Ray Zimmerman (PSERC Cornell)
    @author: Richard Lincoln

    changes by Uni Kassel (Florian Schaefer): If new ref bus is chosen -> Init as numpy array
    """
    # get generator status
#    nb = bus.shape[0]
#    ng = gen.shape[0]
    # gen connection matrix, element i, j is 1 if, generator j at bus i is ON
    #Cg = sparse((gen[:, GEN_STATUS] > 0,
#                 (gen[:, GEN_BUS], range(ng))), (nb, ng))
    # number of generators at each bus that are ON
    #bus_gen_status = (Cg * ones(ng, int)).astype(bool)

    # form index lists for slack, PV, and PQ buses
    ref = find((bus[:, BUS_TYPE] == REF)) # ref bus index
    pv  = find((bus[:, BUS_TYPE] == PV)) # PV bus indices
    pq  = find((bus[:, BUS_TYPE] == PQ)) # PQ bus indices
    return ref, pv, pq
