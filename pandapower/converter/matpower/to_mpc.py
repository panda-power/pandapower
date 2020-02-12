# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy

import numpy as np
from scipy.io import savemat

from pandapower.converter.pypower import to_ppc

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def to_mpc(net, filename=None, **kwargs):
    """
    This function converts a pandapower net to a matpower case files (.mat) version 2.
    Note: python is 0-based while Matlab is 1-based.

    INPUT:
        **net** - The pandapower net.

    OPTIONAL:
        **filename** (str, None) - File path + name of the mat file which will be created. If None
            the mpc will only be returned

        ****kwargs** - please look at to_ppc() documentation

    EXAMPLE:
        import pandapower.converter as pc
        import pandapower.networks as pn
        net = pn.case9()
        pc.to_mpc(net, "case9.mat")

    """
    ppc = to_ppc(net, **kwargs)

    mpc = dict()
    mpc["mpc"] = _ppc2mpc(ppc)
    if filename is not None:
        # savemat
        savemat(filename, mpc)

    return mpc


def _ppc2mpc(ppc):
    """
    Convert network in Pypower/Matpower format
    Convert 0-based python to 1-based Matlab

    **INPUT**:
        * net - The pandapower format network
        * filename - File path + name of the mat file which is created
    """

    # convert to matpower
    # Matlab is one-based, so all entries (buses, lines, gens) have to start with 1 instead of 0
    mpc = copy.deepcopy(ppc)
    if len(np.where(mpc["bus"][:, 0] == 0)[0]):
        mpc["bus"][:, 0] = mpc["bus"][:, 0] + 1
        mpc["gen"][:, 0] = mpc["gen"][:, 0] + 1
        mpc["branch"][:, 0:2] = mpc["branch"][:, 0:2] + 1
    # adjust for the matpower converter -> taps should be 0 when there is no transformer, but are 1
    mpc["branch"][np.where(mpc["branch"][:, 8] == 1), 8] = 0
    # version is a string
    mpc["version"] = str(mpc["version"])
    # baseMVA has to be a float instead of int
    mpc["baseMVA"] = mpc["baseMVA"] * 1.0
    return mpc


if "__main__" == __name__:
    pass
