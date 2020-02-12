# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
from multiprocessing import cpu_count

import pytest

import pandapower as pp

try:
    import pplog as logging
except ImportError:
    import logging
test_dir = os.path.abspath(os.path.join(pp.pp_dir, "test"))


def _create_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    return logger


def _get_cpus():
    # returns of a string of all available CPUs - 1 or 1 if you only have one CPU
    return str(cpu_count() - 1) if cpu_count() > 1 else str(1)


def run_all_tests(parallel=False, n_cpu=None):
    """ function executing all tests

    Inputs:
    parallel (bool, False) - If true and pytest-xdist is installed, tests are run in parallel
    n_cpu (int, None) - number of CPUs to run the tests on in parallel. Only relevant for parallel runs.
    """
    logger = _create_logger()

    if parallel:
        if n_cpu is None:
            n_cpu = _get_cpus()
        pytest.main([test_dir, "-xs", "-n", n_cpu])
    else:
        pytest.main([test_dir, "-xs"])
    logger.setLevel(logging.INFO)


def run_fast_tests(parallel=False, n_cpu=None):
    """ function executing fast tests
    Only executes the tests which are **not** marked as slow with pytest.mark.slow

    parallel (bool, False) - If true and pytest-xdist is installed, tests are run in parallel
    n_cpu (int, None) - number of CPUs to run the tests on in parallel. Only relevant for parallel runs.

    """
    if parallel:
        if n_cpu is None:
            n_cpu = _get_cpus()
        pytest.main([test_dir, "-xs", "-m", "not slow", "-n", n_cpu])
    else:
        pytest.main([test_dir, "-xs", "-m", "not slow"])


def run_slow_tests(parallel=False, n_cpu=None):
    """ function executing slow tests
    Only executes the tests which are marked as slow with pytest.mark.slow

    parallel (bool, False) - If true and pytest-xdist is installed, tests are run in parallel
    n_cpu (int, None) - number of CPUs to run the tests on in parallel. Only relevant for parallel runs.
    """
    if parallel:
        if n_cpu is None:
            n_cpu = _get_cpus()
        pytest.main([test_dir, "-xs", "-m", "slow", "-n", n_cpu])
    else:
        pytest.main([test_dir, "-xs", "-m", "slow"])


if __name__ == "__main__":
    run_all_tests()
