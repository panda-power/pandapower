# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


"""Solves a DC power flow.
"""

from numpy import copy, r_, transpose, real, array
from scipy.sparse.linalg import spsolve

def dcpf(B, Pbus, Va0, ref, pv, pq):
    """Solves a DC power flow.

    Solves for the bus voltage angles at all but the reference bus, given the
    full system C{B} matrix and the vector of bus real power injections, the
    initial vector of bus voltage angles (in radians), and column vectors with
    the lists of bus indices for the swing bus, PV buses, and PQ buses,
    respectively. Returns a vector of bus voltage angles in radians.

    @see: L{rundcpf}, L{runpf}

    @author: Carlos E. Murillo-Sanchez (PSERC Cornell & Universidad
    Autonoma de Manizales)
    @author: Ray Zimmerman (PSERC Cornell)

    # Version from pypower github (bugfix 'transpose')
    """
    pvpq = r_[pv, pq]

    ## initialize result vector
    Va = copy(Va0)

    ## update angles for non-reference buses
    if pvpq.shape == (1, 1): #workaround for bug in scipy <0.19
        pvpq = array(pvpq).flatten()
    pvpq_matrix = B[pvpq.T,:].tocsc()[:,pvpq]
    ref_matrix = transpose(Pbus[pvpq] - B[pvpq.T,:].tocsc()[:,ref] * Va0[ref])
    Va[pvpq] = real(spsolve(pvpq_matrix, ref_matrix))

    return Va
