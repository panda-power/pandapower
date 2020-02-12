# -*- coding: utf-8 -*-

# Copyright 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.



"""Computes partial derivatives of power injection w.r.t. voltage.
"""

from numpy import conj, diag, asarray
from scipy.sparse import issparse, csr_matrix as sparse


def dSbus_dV(Ybus, V):
    """Computes partial derivatives of power injection w.r.t. voltage.
    """

    if issparse(Ybus):
        return dSbus_dV_sparse(Ybus, V)
    else:
        return dSbus_dV_dense(Ybus, V)


def dSbus_dV_sparse(Ybus, V):
    Ibus = Ybus * V
    ib = range(len(V))
    diagV = sparse((V, (ib, ib)))
    diagIbus = sparse((Ibus, (ib, ib)))
    diagVnorm = sparse((V / abs(V), (ib, ib)))
    dS_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dS_dVa = 1j * diagV * conj(diagIbus - Ybus * diagV)
    return dS_dVm, dS_dVa


def dSbus_dV_dense(Ybus, V):
    # standard code from Pypower (slower than above)
    Ibus = Ybus * asarray(V).T

    diagV = asarray(diag(V))
    diagIbus = asarray(diag(asarray(Ibus).flatten()))
    diagVnorm = asarray(diag(V / abs(V)))

    dS_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
    dS_dVa = 1j * diagV * conj(diagIbus - Ybus * diagV)
    return dS_dVm, dS_dVa
