############################
State Estimation
############################

The module provides a state estimation for pandapower networks.

Theoretical Background
===========================
State Estimation is a process to estimate the electrical state of a network by eliminating inaccuracies and errors from measurement data. Various measurements are placed around the network and transferred to the operational control center via SCADA. Unfortunately measurements are not perfect: There are tolerances for each measurement device, which lead to an inherent inaccuracy in the measurement value. Analog transmission of data can change the measurement values through noise. Faulty devices can return completely wrong measurement values. To account for the measurement errors, the state estimation processes all available measurements and uses a regression method to identify the likely real state of the electrical network. 
The **output** of the state estimator is therefore a **set of voltage absolutes and voltage angles** for all buses in the grid. The **input** is the **network** in pandapower format and a number of **measurements**.

Amount of Measurements
-----------------------

There is a minimum amount of required measurements necessary for the regression to be mathematically possible.
Assuming the network contains :math:`n` buses, the network is then described by :math:`2n` variables, namely :math:`n` voltage absolute values and :math:`n` voltage angles.
A slack bus serves as the reference, its voltage angle is set to zero or the value provided in the corresponding *net.ext_grid.va_degree* entry (see *init* parameter) and is not altered in the estimation process.
The voltage angles of the other network buses are relative to the voltage angles of the connected slack bus. The state estimation therefore has to find :math:`2n-k` variables, where :math:`k` is the number of defined slack buses.
The minimum amount of measurements :math:`m_{min}` needed for the method to work is therefore:

:math:`m_{min} = 2n-k`

To perform well however, the number of redundant measurements should be higher. A value of :math:`m \approx 4n` is often considered reasonable for practical purposes.

Standard Deviation
---------------------
Since each measurement device may have a different tolerance and a different path length it has to travel to the control center, the accuracy of each measurement can be different.
Therefore each measurement is assigned an accuracy value in the form of a standard deviation. Typical measurement errors are 1 % for voltage measurements and 1-3 % for power measurements.

For a more in-depth explanation of the internals of the state estimation method, please see the following sources:  

.. seealso::
	- *Power System State Estimation: Theory and Implementation* by Ali Abur, Antonio Gómez Expósito, CRC Press, 2004.   
	- *State Estimation in Electric Power Systems - A Generalized Approach* by A. Monticelli, Springer, 1999.  


Defining Measurements
===========================

Measurements are defined via the pandapower *"create_measurement"* function. There are different physical properties, which can be measured at different elements. The following lists and table clarify the possible combinations. Contrary to the pandapower load flow results, bus power injection measurements are given in the producer system. Generated power is positive, consumed power is negative.

**Types of Measurements**

 - *"v"* for voltage measurements (in per-unit)
 - *"p"* for active power measurements (in MW)
 - *"q"* for reactive power measurements (in MVar)
 - *"i"* for electrical current measurements at a line (in kA)
 

**Element Types**

 - *"bus"* for bus measurements
 - *"line"* for line measurements
 - *"trafo"* for transformer measurements
 - *"trafo3w"* for three-winding-transformer measurements

   
**Available Measurements per Element**

+---------------+------------------------------+
| Element Type  |  Available Measurement Types |
+===============+==============================+
| bus           | v, p, q                      |
+---------------+------------------------------+
| line          | i, p, q                      |
+---------------+------------------------------+
| trafo         | i, p, q                      |
+---------------+------------------------------+
| trafo3w       | i, p, q                      |
+---------------+------------------------------+

The *"create_measurement"* function is defined as follows:

.. autofunction:: pandapower.create.create_measurement

Running the State Estimation
=============================

The state estimation can be used with the wrapper function *"estimate"*, which prevents the need to deal with the state_estimation class object and functions. It can be imported from *"estimation.state_estimation"*.

.. autofunction:: pandapower.estimation.estimate

Handling of bad data
=============================

.. note::  The bad data removal is not very robust at this time. Please treat the results with caution!

The state estimation class allows additionally the removal of bad data, especially single or non-interacting false measurements.
For detecting bad data the Chi-squared distribution is used to identify the presence of them.
Afterwards follows the largest normalized residual test that identifys the actual measurements which will be removed at the end.
Both methods are combined in the *perform_rn_max_test* function that is part of the state estimation class.
To access it, the following wrapper function *remove_bad_data* has been created.

.. autofunction:: pandapower.estimation.remove_bad_data

Nevertheless the Chi-squared test is available as well to allow a identification of topology errors or, as explained, false measurements.
It is named as *chi2_analysis*. The detection's result of present bad data of the Chi-squared test is stored internally as *bad_data_present* (boolean, class member variable) and returned by the function call.

.. autofunction:: pandapower.estimation.chi2_analysis

Background information about this topic can be sourced from the following literature:

.. seealso::
    - *Power System State Estimation: Theory and Implementation* by Ali Abur, Antonio Gómez Expósito, CRC Press, 2004.
    - *Power Generation, Operation, and Control* by Allen J. Wood, Bruce Wollenberg, Wiley Interscience Publication, 1996. 
 
Example
=============================

As an example, we will define measurements for a simple pandapower network *net* with 4 buses. Bus 4 is out-of-service. The external grid is connected at bus 1.

There are multiple measurements available, which have to be defined for the state estimator. There are two voltage measurements at buses 1 and 2. There are two power measurements (active and reactive power) at bus 2. There are also line power measurements at bus 1. The measurements are both for active and reactive power and are located on the line from bus 1 to bus 2 and from bus 1 to bus 3. This yields the following code: 

:: 

    pp.create_measurement(net, "v", "bus", 1.006, .004, bus1)  # V at bus 1
	pp.create_measurement(net, "v", "bus", 0.968, .004, bus2)  # V at bus 2

	pp.create_measurement(net, "p", "bus", -501, 10, bus2)     # P at bus 2
	pp.create_measurement(net, "q", "bus", -286, 10, bus2)     # Q at bus 2

	pp.create_measurement(net, "p", "line", 888, 8, element=line1, side="from")   # P_line (bus 1 -> bus 2) at bus 1
	pp.create_measurement(net, "p", "line", 1173, 8, element=line2, side="from")  # P_line (bus 1 -> bus 3) at bus 1
	# you can either define the side with a string ("from" / "to") or
	# using the bus index where the line ends and the measurement is located
	pp.create_measurement(net, "q", "line", 568, 8, element=line1, side=bus1)     # Q_line (bus 1 -> bus 2) at bus 1
	pp.create_measurement(net, "q", "line", 663, 8, element=line2, side=bus1)     # Q_line (bus 1 -> bus 3) at bus 1

Now that the data is ready, the state_estimation can be initialized and run. We want to use the flat start condition, in which all voltages are set to 1.0 p.u..

:: 

	success = estimate(net, init="flat")
	V, delta = net.res_bus_est.vm_pu, net.res_bus_est.va_degree 


The resulting variables now contain the voltage absolute values in *V*, the voltage angles in *delta*, an indication of success in *success*.
The bus power injections can be accessed similarly with *net.res_bus_est.p_mw* and *net.res_bus_est.q_mvar*. Line data is also available in the same format as defined in *res_line*.


If we like to check our data for fault measurements, and exclude them in in our state estimation, we use the following code:

::
    
    success_rn_max = remove_bad_data(net, init="flat")
    V_rn_max, delta_rn_max = net.res_bus_est.vm_pu, net.res_bus_est.va_degree

In the case that we only like to know if there is a likelihood of fault measurements (probabilty of fault can be adjusted), the Chi-squared test should be performed separatly.
If the test detects the possibility of fault data, the value of the added class member variable *bad_data_present* would be *true* as well as the boolean variable *success_chi2* that is used here:

::

    success_chi2 = chi2_analysis(net, init="flat")

Further Algorithms and Estimators
==================================
Since Pandapower 2.0.1 further algorithms and estimators (robust estimators) are available for the state estimation module, these include:

+-------------------------------------+----------------------+
| Algorithm                           | Available Estimators |
+=====================================+======================+
| wls (Newton-Gauss)                  |                      |
+-------------------------------------+----------------------+
| wls with zero injection constraints |                      |
+-------------------------------------+----------------------+
| lp                                  | lav                  |
+-------------------------------------+----------------------+
| irwls                               | wls, shgm            |
+-------------------------------------+----------------------+
| Scipy Optimization Tool             | wls, lav, ql, qc     |
+-------------------------------------+----------------------+

Most of the algorithms and estimators are implemented as explained in the *Power System State Estimation: Theory and Implementation* by Ali Abur, Antonio Gómez Expósito, CRC Press, 2004. While the QC and QL estimators are adjusted mathematically for a better convergence of scipy optimization tool. 

For SHGM: Please see *"Robust state estimation based on projection statistics," IEEE Trans. Power Syst, vol. 11, no. 2, pp. 1118--1127, 1996.* by L. Mili, M. Cheniae, N. Vichare, and P. Rousseeuw. The projection statistics was rewritten in Python based on the code published by original authors of the paper.  

Example of using extra estimators:
::

	# Using shgm 
	success = estimate(net, algorithm="irwls", estimator='shgm', a=5)

	# Using lav
	success = estimate(net, algorithm="lp")

	# Using ql
	success = estimate(net, algorithm="opt", estimator="ql", a=3)

Note that:
The state estimation with Scipy Optimization Tool could collapse in some cases with flat start, it's suggested to give the algorithm a warm start or try some other optimization's methods offered by scipy, which preserves the effects of the estimator while helps the convergence.

Example for chained estimation (warm start for SciPy Optimization Tool):
::

	# Initialize eppci for the algorithm which contains pypower-grid,
	# measurements and estimated grid state (initial value)
	net, ppc, eppci = pp2eppci(net)

	# Initialize algorithm
	estimation_wls = WLSAlgorithm(1e-3, 5)
    	estimation_opt = OptAlgorithm(1e-6, 1000)

	# Start Estimation with specified estimator
    	eppci = estimation_wls.estimate(eppci)
	# for some estimators extra parameters must be specified
    	eppci = estimation_opt.estimate(eppci, estimator="ql", a=3) 

	# Update the pandapower network with estimated results
    	net = eppci2pp(net, ppc, eppci)



.. Class state_estimation
   ======================

.. commentout automodule:: pandapower.estimation.state_estimation
    :members:
