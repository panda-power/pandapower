# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from numba import jit
from numpy import conj, zeros, complex128
from scipy.sparse import issparse, csr_matrix as sparse
from pandapower.pypower.dSbus_dV import dSbus_dV_dense


# @jit(Tuple((c16[:], c16[:]))(c16[:], i4[:], i4[:], c16[:], c16[:]), nopython=True, cache=False)
@jit(nopython=True, cache=False)
def dSbus_dV_numba_sparse(Yx, Yp, Yj, V, Vnorm, Ibus): # pragma: no cover
    """Computes partial derivatives of power injection w.r.t. voltage.

    Calculates faster with numba and sparse matrices.

    Input: Ybus in CSR sparse form (Yx = data, Yp = indptr, Yj = indices), V and Vnorm (= V / abs(V))

    OUTPUT: data from CSR form of dS_dVm, dS_dVa
    (index pointer and indices are the same as the ones from Ybus)

    Translation of: dS_dVm = dS_dVm = diagV * conj(Ybus * diagVnorm) + conj(diagIbus) * diagVnorm
                             dS_dVa = 1j * diagV * conj(diagIbus - Ybus * diagV)
    """

    # transform input

    # init buffer vector
    buffer = zeros(len(V), dtype=complex128)
    dS_dVm = Yx.copy()
    dS_dVa = Yx.copy()

    # iterate through sparse matrix
    for r in range(len(Yp) - 1):
        for k in range(Yp[r], Yp[r + 1]):
            # Ibus = Ybus * V
            buffer[r] += Yx[k] * V[Yj[k]]
            # Ybus * diag(Vnorm)
            dS_dVm[k] *= Vnorm[Yj[k]]
            # Ybus * diag(V)
            dS_dVa[k] *= V[Yj[k]]

        Ibus[r] += buffer[r]

        # conj(diagIbus) * diagVnorm
        buffer[r] = conj(buffer[r]) * Vnorm[r]

    for r in range(len(Yp) - 1):
        for k in range(Yp[r], Yp[r + 1]):
            # diag(V) * conj(Ybus * diagVnorm)
            dS_dVm[k] = conj(dS_dVm[k]) * V[r]

            if r == Yj[k]:
                # diagonal elements
                dS_dVa[k] = -Ibus[r] + dS_dVa[k]
                dS_dVm[k] += buffer[r]

            # 1j * diagV * conj(diagIbus - Ybus * diagV)
            dS_dVa[k] = conj(-dS_dVa[k]) * (1j * V[r])

    return dS_dVm, dS_dVa


def dSbus_dV(Ybus, V, I=None):
    """
    Calls functions to calculate dS/dV depending on whether Ybus is sparse or not
    """

    if issparse(Ybus):
        # I is substracted from Y*V,
        # therefore it must be negative for numba version of dSbus_dV if it is not zeros anyways
        I = zeros(len(V), dtype=complex128) if I is None else -I
        # calculates sparse data
        dS_dVm, dS_dVa = dSbus_dV_numba_sparse(Ybus.data, Ybus.indptr, Ybus.indices, V, V / abs(V), I)
        # generate sparse CSR matrices with computed data and return them
        return sparse((dS_dVm, Ybus.indices, Ybus.indptr)), sparse((dS_dVa, Ybus.indices, Ybus.indptr))
    else:
        I = zeros(len(V), dtype=complex128) if I is None else I
        return dSbus_dV_dense(Ybus, V, I)
