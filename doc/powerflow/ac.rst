Balanced AC Power Flow
=======================

pandapower uses PYPOWER to solve the power flow problem:

.. image:: /pics/flowcharts/pandapower_power flow.png
		:width: 40em
		:alt: alternate Text
		:align: center

.. autofunction:: pandapower.runpp

.. note::

    If you are interested in the pypower casefile that pandapower is using for power flow, you can find it in net["_ppc"].
    However all necessary informations are written into the pandpower format net, so the pandapower user should not usually have to deal with pypower.

