
Optimization with PYPOWER
===========================

You can run an Optimal Power Flow using the PYPOWER OPF

AC OPF
-----------------------------------

.. autofunction:: pandapower.runopp

The internal solver uses the interior point method. By default, the initial state is the center of the operational constraints. Another option would be to initialize the optimisation with a valid loadflow solution. For optimiation of a timeseries, this warm start possibilty could imply a significant speedup. 
This is not yet provided in the actual version, but could be an useful extension in the future.
Another parametrisation for the AC OPF is, if voltage angles should be considered, which is the same option than for the loadflow calculation with pandapower.runpp: 


References:
      - "On the Computation and Application of Multi-period
        Security-Constrained Optimal Power Flow for Real-time
        Electricity Market Operations", Cornell University, May 2007.
      - H. Wang, C. E. Murillo-Sanchez, R. D. Zimmerman, R. J. Thomas,
        "On Computational Issues of Market-Based Optimal Power Flow",
        IEEE Transactions on Power Systems, Vol. 22, No. 3, Aug. 2007,
        pp. 1185-1193.
      - R. D. Zimmerman, C. E. Murillo-Sánchez, and R. J. Thomas, "MATPOWER: Steady-State 
        Operations, Planning and Analysis Tools for Power Systems Research and Education," 
        Power Systems, IEEE Transactions on, vol. 26, no. 1, pp. 12-19, Feb. 2011.
  

DC OPF
---------
The dc optimal power flow is a linearized optimization of the grid state. It offers two cost function options, that are fitting special use cases. 
To understand the usage, the `DC OPF tutorial <https://github.com/e2nIEE/pandapower/blob/develop/tutorials/opf_dcline.ipynb>`_ is recommended.
    
.. autofunction:: pandapower.rundcopp

Flexibilities, costs and constraints (except voltage constraints) are handled as in the :ref:`opf`.
Voltage constraints are not considered in the DC OPF, since voltage magnitutes are not part of the 
linearized power flow equations.

.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in net["_ppc_opf"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.


    
    
      