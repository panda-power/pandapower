# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import scipy.io

from pandapower.converter.pypower import from_ppc

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def from_mpc(mpc_file, f_hz=50, casename_mpc_file='mpc', validate_conversion=False):
    """
    This function converts a matpower case file version 2 to a pandapower net.

    Note: The input is a .mat file not an .m script. You need to save the mpc dict variable as .mat
    file. If the saved variable of the matlab workspace is not named 'mpc', you can adapt the value
    of 'casename_mpc_file' as needed.

    Note: python is 0-based while Matlab is 1-based.

    INPUT:

        **mpc_file** - path to a matpower case file (.mat format not .m script).

    OPTIONAL:

        **f_hz** (int, 50) - The frequency of the network.

        **casename_mpc_file** (str, 'mpc') - The name of the variable in .mat file which contain
        the matpower case structure, i.e. the arrays "gen", "branch" and "bus".

    OUTPUT:

        **net** - The pandapower network

    EXAMPLE:

        import pandapower.converter as pc

        pp_net = cv.from_mpc('case9.mat', f_hz=60)

    """
    ppc = _mpc2ppc(mpc_file, casename_mpc_file)
    net = from_ppc(ppc, f_hz, validate_conversion)

    return net


def _mpc2ppc(mpc_file, casename_mpc_file):
    # load mpc from file
    mpc = scipy.io.loadmat(mpc_file, squeeze_me=True, struct_as_record=False)

    # init empty ppc
    ppc = dict()

    _copy_data_from_mpc_to_ppc(ppc, mpc, casename_mpc_file)
    _adjust_ppc_indices(ppc)
    _change_ppc_TAP_value(ppc)

    return ppc


def _adjust_ppc_indices(ppc):
    # adjust indices of ppc, since ppc must start at 0 rather than 1 (matlab)
    ppc["bus"][:, 0] -= 1
    ppc["branch"][:, 0] -= 1
    ppc["branch"][:, 1] -= 1
    # if in ppc is only one gen -> numpy initially uses one dim array -> change to two dim array
    if len(ppc["gen"].shape) == 1:
        ppc["gen"] = np.array(ppc["gen"], ndmin=2)
    ppc["gen"][:, 0] -= 1


def _copy_data_from_mpc_to_ppc(ppc, mpc, casename_mpc_file):
    if casename_mpc_file in mpc:
        # if struct contains a field named mpc
        ppc['version'] = mpc[casename_mpc_file].version
        ppc["baseMVA"] = mpc[casename_mpc_file].baseMVA
        ppc["bus"] = mpc[casename_mpc_file].bus
        ppc["gen"] = mpc[casename_mpc_file].gen
        ppc["branch"] = mpc[casename_mpc_file].branch

        try:
            ppc['gencost'] = mpc[casename_mpc_file].gencost
        except:
            logger.info('gencost is not in mpc')

    else:
        logger.error('Matfile does not contain a valid mpc structure.')


def _change_ppc_TAP_value(ppc):
    # adjust for the matpower converter -> taps should be 0 when there is no transformer, but are 1
    ppc["branch"][np.where(ppc["branch"][:, 8] == 0), 8] = 1


if "__main__" == __name__:
    pass
