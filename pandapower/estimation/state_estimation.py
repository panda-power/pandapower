# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from scipy.stats import chi2

from pandapower.estimation.util import set_bb_switch_impedance, reset_bb_switch_impedance
from pandapower.estimation.ppc_conversion import pp2eppci, _initialize_voltage
from pandapower.estimation.results import eppci2pp
from pandapower.estimation.algorithm.base import (WLSAlgorithm, 
                                                  WLSZeroInjectionConstraintsAlgorithm,
                                                  IRWLSAlgorithm)
from pandapower.estimation.algorithm.optimization import OptAlgorithm
from pandapower.estimation.algorithm.lp import LPAlgorithm


try:
    import pplog as logging
except ImportError:
    import logging
std_logger = logging.getLogger(__name__)


ALGORITHM_MAPPING = {'wls': WLSAlgorithm,
                     'wls_with_zero_constraint': WLSZeroInjectionConstraintsAlgorithm,
                     'opt': OptAlgorithm,
                     'irwls': IRWLSAlgorithm,
                     'lp': LPAlgorithm}
ALLOWED_OPT_VAR = {"a", "opt_method", "estimator"}


def estimate(net, algorithm='wls',
             init='flat', tolerance=1e-6, maximum_iterations=10,
             calculate_voltage_angles=True, 
             zero_injection='aux_bus', fuse_buses_with_bb_switch='all',
             **opt_vars):
    """
    Wrapper function for WLS state estimation.
    INPUT:
        **net** - The net within this line should be created

        **init** - (string) Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all
        buses, 'results' uses the values from *res_bus_est* if available and 'slack' considers the
        slack bus voltage (and optionally, angle) as the initial values. Default is 'flat'
    OPTIONAL:
        **tolerance** - (float) - When the maximum state change between iterations is less than
        tolerance, the process stops. Default is 1e-6

        **maximum_iterations** - (integer) - Maximum number of iterations. Default is 10

        **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
        shifts in transformers, if init is 'slack'. Default is True
        
        **zero_injection** - (str, iterable, None) - Defines which buses are zero injection bus or the method
        to identify zero injection bus, with 'wls_estimator' virtual measurements will be added, with 
        'wls_estimator with zero constraints' the buses will be handled as constraints
            "auto": all bus without p,q measurement, without p, q value (load, sgen...) and aux buses will be
                identified as zero injection bus  
            "aux_bus": only aux bus will be identified as zero injection bus
            None: no bus will be identified as zero injection bus
            iterable: the iterable should contain index of the zero injection bus and also aux bus will be identified
                as zero-injection bus

        **fuse_buses_with_bb_switch** - (str, iterable, None) - Defines how buses with closed bb switches should 
        be handled, if fuse buses will only fused to one for calculation, if not fuse, an auxiliary bus and 
        auxiliary line will be automatically added to the network to make the buses with different p,q injection
        measurements identifieble
            "all": all buses with bb-switches will be fused, the same as the default behaviour in load flow
            None: buses with bb-switches and individual p,q measurements will be reconfigurated
                by auxiliary elements
            iterable: the iterable should contain the index of buses to be fused, the behaviour is contigous e.g.
                if one of the bus among the buses connected through bb switch is given, then all of them will still
                be fused
    OUTPUT:
        **successful** (boolean) - Was the state estimation successful?
    """
    if algorithm not in ALGORITHM_MAPPING:
        raise UserWarning("Algorithm {} is not a valid estimator".format(algorithm))

    se = StateEstimation(net, tolerance, maximum_iterations, algorithm=algorithm)
    v_start, delta_start = _initialize_voltage(net, init, calculate_voltage_angles)
    return se.estimate(v_start=v_start, delta_start=delta_start, 
                       calculate_voltage_angles=calculate_voltage_angles, 
                       zero_injection=zero_injection,
                       fuse_buses_with_bb_switch=fuse_buses_with_bb_switch, **opt_vars)


def remove_bad_data(net, init='flat', tolerance=1e-6, maximum_iterations=10,
                    calculate_voltage_angles=True, rn_max_threshold=3.0):
    """
    Wrapper function for bad data removal.

    INPUT:
        **net** - The net within this line should be created

        **init** - (string) Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all
        buses, 'results' uses the values from *res_bus_est* if available and 'slack' considers the
        slack bus voltage (and optionally, angle) as the initial values. Default is 'flat'

    OPTIONAL:
        **tolerance** - (float) - When the maximum state change between iterations is less than
        tolerance, the process stops. Default is 1e-6

        **maximum_iterations** - (integer) - Maximum number of iterations. Default is 10

        **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
        shifts in transformers, if init is 'slack'. Default is True

        **rn_max_threshold** (float) - Identification threshold to determine
        if the largest normalized residual reflects a bad measurement
        (default value of 3.0)

    OUTPUT:
        **successful** (boolean) - Was the state estimation successful?
    """
    wls_se = StateEstimation(net, tolerance, maximum_iterations, algorithm="wls")
    v_start, delta_start = _initialize_voltage(net, init, calculate_voltage_angles)
    return wls_se.perform_rn_max_test(v_start, delta_start, calculate_voltage_angles,
                                   rn_max_threshold)


def chi2_analysis(net, init='flat', tolerance=1e-6, maximum_iterations=10,
                  calculate_voltage_angles=True, chi2_prob_false=0.05):
    """
    Wrapper function for the chi-squared test.

    INPUT:
        **net** - The net within this line should be created.

        **init** - (string) Initial voltage for the estimation. 'flat' sets 1.0 p.u. / 0° for all
        buses, 'results' uses the values from *res_bus_est* if available and 'slack' considers the
        slack bus voltage (and optionally, angle) as the initial values. Default is 'flat'

    OPTIONAL:
        **tolerance** - (float) - When the maximum state change between iterations is less than
        tolerance, the process stops. Default is 1e-6

        **maximum_iterations** - (integer) - Maximum number of iterations. Default is 10

        **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
        shifts in transformers, if init is 'slack'. Default is True

        **chi2_prob_false** (float) - probability of error / false alarms
        (default value: 0.05)

    OUTPUT:
        **bad_data_detected** (boolean) - Returns true if bad data has been detected
    """
    wls_se = StateEstimation(net, tolerance, maximum_iterations, algorithm="wls")
    v_start, delta_start = _initialize_voltage(net, init, calculate_voltage_angles)
    return wls_se.perform_chi2_test(v_start, delta_start, calculate_voltage_angles,
                                 chi2_prob_false)


class StateEstimation:
    """
    Any user of the estimation module only needs to use the class state_estimation. It contains all
    relevant functions to control and operator the module. Two functions are used to configure the
    system according to the users needs while one function is used for the actual estimation
    process.
    """
    def __init__(self, net, tolerance=1e-6, maximum_iterations=10, algorithm='wls', logger=None, recycle=False):
        self.logger = logger
        if self.logger is None:
            self.logger = std_logger
            # self.logger.setLevel(logging.DEBUG)
        self.net = net
        self.solver = ALGORITHM_MAPPING[algorithm](tolerance,
                                                   maximum_iterations, self.logger)
        self.ppc = None
        self.eppci = None
        self.recycle = recycle

        # variables for chi^2 / rn_max tests
        self.delta = None
        self.bad_data_present = None

    def estimate(self, v_start='flat', delta_start='flat', calculate_voltage_angles=True,
                 zero_injection=None, fuse_buses_with_bb_switch='all', **opt_vars):
        """
        The function estimate is the main function of the module. It takes up to three input
        arguments: v_start, delta_start and calculate_voltage_angles. The first two are the initial
        state variables for the estimation process. Usually they can be initialized in a
        "flat-start" condition: All voltages being 1.0 pu and all voltage angles being 0 degrees.
        In this case, the parameters can be left at their default values (None). If the estimation
        is applied continuously, using the results from the last estimation as the starting
        condition for the current estimation can decrease the  amount of iterations needed to
        estimate the current state. The third parameter defines whether all voltage angles are
        calculated absolutely, including phase shifts from transformers. If only the relative
        differences between buses are required, this parameter can be set to False. Returned is a
        boolean value, which is true after a successful estimation and false otherwise.
        The resulting complex voltage will be written into the pandapower network. The result
        fields are found res_bus_est of the pandapower network.
        INPUT:
            **net** - The net within this line should be created
            **v_start** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage magnitudes in p.u. (sorted by bus index)
            **delta_start** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage angles in degrees (sorted by bus index)
        OPTIONAL:
        **tolerance** - (float) - When the maximum state change between iterations is less than
        tolerance, the process stops. Default is 1e-6

        **maximum_iterations** - (integer) - Maximum number of iterations. Default is 10

        **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
        shifts in transformers, if init is 'slack'. Default is True
        
        **zero_injection** - (str, iterable, None) - Defines which buses are zero injection bus or the method
        to identify zero injection bus, with 'wls_estimator' virtual measurements will be added, with 
        'wls_estimator with zero constraints' the buses will be handled as constraints
            "auto": all bus without p,q measurement, without p, q value (load, sgen...) and aux buses will be
                identified as zero injection bus  
            "aux_bus": only aux bus will be identified as zero injection bus
            None: no bus will be identified as zero injection bus
            iterable: the iterable should contain index of the zero injection bus and also aux bus will be identified
                as zero-injection bus

        **fuse_buses_with_bb_switch** - (str, iterable, None) - Defines how buses with closed bb switches should 
        be handled, if fuse buses will only fused to one for calculation, if not fuse, an auxiliary bus and 
        auxiliary line will be automatically added to the network to make the buses with different p,q injection
        measurements identifieble
            "all": all buses with bb-switches will be fused, the same as the default behaviour in load flow
            None: buses with bb-switches and individual p,q measurements will be reconfigurated
                by auxiliary elements
            iterable: the iterable should contain the index of buses to be fused, the behaviour is contigous e.g.
                if one of the bus among the buses connected through bb switch is given, then all of them will still
                be fused
        OUTPUT:
            **successful** (boolean) - True if the estimation process was successful
        Optional estimation variables:
            The bus power injections can be accessed with *se.s_node_powers* and the estimated
            values corresponding to the (noisy) measurement values with *se.hx*. (*hx* denotes h(x))
        EXAMPLE:
            success = estimate(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]))
        """
        # check if all parameter are allowed
        for var_name in opt_vars.keys():
            if var_name not in ALLOWED_OPT_VAR:
                self.logger.warning("Caution! %s is not allowed as parameter"%var_name\
                                    + " for estimate and will be ignored!" )

        if self.net is None:
            raise UserWarning("SE Component was not initialized with a network.")

        # change the configuration of the pp net to avoid auto fusing of buses connected
        # through bb switch with elements on each bus if this feature enabled
        bus_to_be_fused = None
        if fuse_buses_with_bb_switch != 'all' and not self.net.switch.empty:
            if isinstance(fuse_buses_with_bb_switch, str):
                raise UserWarning("fuse_buses_with_bb_switch parameter is not correctly initialized")
            elif hasattr(fuse_buses_with_bb_switch, '__iter__'):
                bus_to_be_fused = fuse_buses_with_bb_switch
            set_bb_switch_impedance(self.net, bus_to_be_fused)

        self.net, self.ppc, self.eppci = pp2eppci(self.net, v_start=v_start, delta_start=delta_start, 
                                                  calculate_voltage_angles=calculate_voltage_angles, 
                                                  zero_injection=zero_injection, ppc=self.ppc, eppci=self.eppci)

        # Estimate voltage magnitude and angle with the given estimator
        self.eppci = self.solver.estimate(self.eppci, **opt_vars)

        if self.solver.successful:
            self.net = eppci2pp(self.net, self.ppc, self.eppci)
        else:
            self.logger.warning("Estimation failed! Pandapower network failed to update!")

        # clear the aux elements and calculation results created for the substitution of bb switches
        if fuse_buses_with_bb_switch != 'all' and not self.net.switch.empty:
            reset_bb_switch_impedance(self.net)
        
        # if recycle is not wished, reset ppc, ppci
        if not self.recycle:
            self.ppc, self.eppci = None, None
        return self.solver.successful

    def perform_chi2_test(self, v_in_out=None, delta_in_out=None,
                          calculate_voltage_angles=True, chi2_prob_false=0.05):
        """
        The function perform_chi2_test performs a Chi^2 test for bad data and topology error
        detection. The function can be called with the optional input arguments v_in_out and
        delta_in_out. Then, the Chi^2 test is performed after calling the function estimate using
        them as input arguments. It can also be called without these arguments if it is called
        from the same object with which estimate had been called beforehand. Then, the Chi^2 test is
        performed for the states estimated by the funtion estimate and the result, the existence of bad data,
        is given back as a boolean. As a optional argument the probability
        of a false measurement can be provided additionally. For bad data detection, the function
        perform_rn_max_test is more powerful and should be the function of choice. For topology
        error detection, however, perform_chi2_test should be used.

        INPUT:
            **v_in_out** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage magnitudes in p.u. (sorted by bus index)

            **delta_in_out** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage angles in degrees (sorted by bus index)

        OPTIONAL:
            **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
            shifts in transformers, if init is 'slack'. Default is True

            **chi2_prob_false** (float) - probability of error / false alarms (standard value: 0.05)

        OUTPUT:
            **successful** (boolean) - True if bad data has been detected

        EXAMPLE:
            perform_chi2_test(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]), 0.97)

        """
        # perform SE
        self.estimate(v_in_out, delta_in_out, calculate_voltage_angles)

        # Performance index J(hx)
        J = np.dot(self.solver.r.T, np.dot(self.solver.R_inv, self.solver.r))

        # Number of measurements
        m = len(self.net.measurement)

        # Number of state variables (the -1 is due to the reference bus)
        n = len(self.solver.eppci.v) + len(self.solver.eppci.delta) - 1

        # Chi^2 test threshold
        test_thresh = chi2.ppf(1 - chi2_prob_false, m - n)

        # Print results
        self.logger.debug("Result of Chi^2 test:")
        self.logger.debug("Number of measurements: %d" % m)
        self.logger.debug("Number of state variables: %d" % n)
        self.logger.debug("Performance index: %.2f" % J)
        self.logger.debug("Chi^2 test threshold: %.2f" % test_thresh)

        if J <= test_thresh:
            self.bad_data_present = False
            self.logger.debug("Chi^2 test passed. No bad data or topology error detected.")
        else:
            self.bad_data_present = True
            self.logger.debug("Chi^2 test failed. Bad data or topology error detected.")

        if self.solver.successful:
            return self.bad_data_present

    def perform_rn_max_test(self, v_in_out=None, delta_in_out=None,
                            calculate_voltage_angles=True, rn_max_threshold=3.0):
        """
        The function perform_rn_max_test performs a largest normalized residual test for bad data
        identification and removal. It takes two input arguments: v_in_out and delta_in_out.
        These are the initial state variables for the combined estimation and bad data
        identification and removal process. They can be initialized as described above, e.g.,
        using a "flat" start. In an iterative process, the function performs a state estimation,
        identifies a bad data measurement, removes it from the set of measurements
        (only if the rn_max threshold is violated by the largest residual of all measurements,
        which can be modified), performs the state estimation again,
        and so on and so forth until no further bad data measurements are detected.

        INPUT:
            **v_in_out** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage magnitudes in p.u. (sorted by bus index)

            **delta_in_out** (np.array, shape=(1,), optional) - Vector with initial values for all
            voltage angles in degrees (sorted by bus index)

        OPTIONAL:
            **calculate_voltage_angles** - (boolean) - Take into account absolute voltage angles and phase
            shifts in transformers, if init is 'slack'. Default is True

            **rn_max_threshold** (float) - Identification threshold to determine
            if the largest normalized residual reflects a bad measurement
            (standard value of 3.0)

        OUTPUT:
            **successful** (boolean) - True if all bad data could be removed

        EXAMPLE:
            perform_rn_max_test(np.array([1.0, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]), 5.0, 0.05)

        """
        num_iterations = 0

        while num_iterations <= 10:
            # Estimate the state with bad data identified in previous iteration
            # removed from set of measurements:
            self.estimate(v_in_out, delta_in_out, calculate_voltage_angles)

            # Try to remove the bad data
            try:
                # Error covariance matrix:
                R = np.linalg.inv(self.solver.R_inv)

                # for future debugging: this line's results have changed with the ppc
                # overhaul in April 2017 after commit 9ae5b8f42f69ae39f8c8cf (which still works)
                # there are differences of < 1e-10 for the Omega entries which cause
                # the function to work far worse. As of now it is unclear if it's just numerical
                # accuracy to blame or an error in the code. a sort in the ppc creation function
                # was removed which caused this issue
                # Covariance matrix of the residuals: \Omega = S*R = R - H*G^(-1)*H^T
                # (S is the sensitivity matrix: r = S*e):
                Omega = R - np.dot(self.solver.H, np.dot(np.linalg.inv(self.solver.Gm), self.solver.H.T))

                # Diagonalize \Omega:
                Omega = np.diag(np.diag(Omega))

                # Compute squareroot (|.| since some -0.0 produced nans):
                Omega = np.sqrt(np.absolute(Omega))

                OmegaInv = np.linalg.inv(Omega)

                # Compute normalized residuals (r^N_i = |r_i|/sqrt{Omega_ii}):
                rN = np.dot(OmegaInv, np.absolute(self.solver.r))

                if max(rN) <= rn_max_threshold:
                    self.logger.debug("Largest normalized residual test passed. "
                                      "No bad data detected.")
                    return True
                else:
                    self.logger.debug(
                        "Largest normalized residual test failed (%.1f > %.1f)."
                        % (max(rN), rn_max_threshold))

                    # Identify bad data: Determine index corresponding to max(rN):
                    idx_rN = np.argsort(rN, axis=0)[-1]

                    # Determine pandapower index of measurement to be removed:
                    meas_idx = self.solver.pp_meas_indices[idx_rN]

                    # Remove bad measurement:
                    self.logger.debug("Removing measurement: %s"
                                      % self.net.measurement.loc[meas_idx].values[0])
                    self.net.measurement.drop(meas_idx, inplace=True)
                    self.logger.debug("Bad data removed from the set of measurements.")

            except np.linalg.linalg.LinAlgError:
                self.logger.error("A problem appeared while using the linear algebra methods."
                                  "Check and change the measurement set.")
                return False

            self.logger.debug("rN_max identification threshold: %.2f" % rn_max_threshold)
            num_iterations += 1

        return False
