
==============
CIGRE Networks
==============

CIGRE-Networks were developed by the CIGRE Task Force C6.04.02 to "facilitate the analysis
and validation of new methods and techniques" that aim to "enable the economic, robust and
environmentally responsible integration of DER" (Distributed Energy Resources).
CIGRE-Networks are a set of comprehensive reference systems to allow the "analysis of DER
integration at high voltage, medium voltage and low voltage and at the desired degree of detail".

.. note::

    Source for this network is the final Report of Task Force C6.04.02 [1]: `"Benchmark Systems for Network Integration of Renewable and Distributed Energy Resources" <http://www.e-cigre.org/Order/select.asp?ID=729590>`_, 2014

    See also a correlating Paper with tiny changed network parameters [2]:
    `K. Rudion, A. Orths, Z. A. Styczynski and K. Strunz, Design of benchmark of medium voltage distribution network for investigation of DG integration <http://ieeexplore.ieee.org/document/1709447/?arnumber=1709447&tag=1>`_ 2006 IEEE Power Engineering Society General Meeting, Montreal, 2006

High voltage transmission network
---------------------------------

.. code:: python

 import pandapower.networks as pn

 # You may specify a length for the connection line between buses 6a and 6b
 net = pn.create_cigre_network_hv(length_km_6a_6b=0.1)

 '''
 This pandapower network includes the following parameter tables:
   - shunt (3 elements)
   - trafo (6 elements)
   - bus (13 elements)
   - line (9 elements)
   - load (5 elements)
   - ext_grid (1 elements)
   - gen (3 elements)
   - bus_geodata (13 elements)
 '''


.. image:: /pics/networks/cigre//cigre_network_hv.png
	:width: 42em
	:alt: alternate Text
	:align: center

`[Source: 1] <http://www.e-cigre.org/Order/select.asp?ID=729590>`_



Medium voltage distribution network
-----------------------------------

.. code:: python

 import pandapower.networks as pn

 net = pn.create_cigre_network_mv(with_der=False)

 '''
 This pandapower network includes the following parameter tables:
   - switch (8 elements)
   - load (18 elements)
   - ext_grid (1 elements)
   - line (15 elements)
   - trafo (2 elements)
   - bus (15 elements)
   - bus_geodata (15 elements)
 '''


.. image:: /pics/networks/cigre//cigre_network_mv.png
	:width: 42em
	:alt: alternate Text
	:align: center

`[Source: 1] <http://www.e-cigre.org/Order/select.asp?ID=729590>`_


---------------------------


Medium voltage distribution network with PV and Wind DER
--------------------------------------------------------

.. note:: This network contains additional 9 distributed energy resources compared to medium voltage distribution network:

			- 8 photovoltaic generators
			- 1 wind turbine

Compared to the case study of CIGRE Task Force C6.04.02 paper all pv and wind energy resources are
considered but 2 Batteries, 2 residential fuel cells, 1 CHP diesel and 1 CHP fuel cell are neglected.
Although the case study mentions the High Voltage as 220 kV, we assume 110 kV again because of no given 220 kV-Trafo data.

.. code:: python

    import pandapower.networks as pn

    net = pn.create_cigre_network_mv(with_der="pv_wind")

    '''
    This pandapower network includes the following parameter tables:
      - switch (8 elements)
      - load (18 elements)
      - ext_grid (1 elements)
      - sgen (9 elements)
      - line (15 elements)
      - trafo (2 elements)
      - bus (15 elements)
      - bus_geodata (15 elements)
    '''

.. image:: /pics/networks/cigre//cigre_network_mv_der.png
	:width: 42em
	:alt: alternate Text
	:align: center

`[Source: 1] <http://www.e-cigre.org/Order/select.asp?ID=729590>`_


---------------------------


Medium voltage distribution network with all DER
------------------------------------------------

.. note:: This network contains additional 15 distributed energy resources compared to medium voltage distribution network:

			- 8 photovoltaic generators
			- 1 wind turbine
			- 2 Batteries
			- 2 residential fuel cells
			- 1 CHP diesel
			- 1 CHP fuel cell

Compared to the case study of CIGRE Task Force C6.04.02 paper all distributed energy resources are
considered. Although the case study mentions the High Voltage as 220 kV, we assume 110 kV again because of no given 220 kV-Trafo data.

.. code:: python

    import pandapower.networks as pn

    net = pn.create_cigre_network_mv(with_der="all")

    '''
    This pandapower network includes the following parameter tables:
      - switch (8 elements)
      - load (18 elements)
      - ext_grid (1 elements)
      - sgen (15 elements)
      - line (15 elements)
      - trafo (2 elements)
      - bus (15 elements)
      - bus_geodata (15 elements)
    '''

.. image:: /pics/networks/cigre//cigre_network_mv_der_all.png
	:width: 42em
	:alt: alternate Text
	:align: center

`[Source: 1] <http://www.e-cigre.org/Order/select.asp?ID=729590>`_


---------------------------


Low voltage distribution network
---------------------------------

.. code:: python

 import pandapower.networks as pn

 net = pn.create_cigre_network_lv()

 '''
 This pandapower network includes the following parameter tables:
   - switch (3 elements)
   - load (15 elements)
   - ext_grid (1 elements)
   - line (37 elements)
   - trafo (3 elements)
   - bus (44 elements)
   - bus_geodata (44 elements)
 '''


.. image:: /pics/networks/cigre//cigre_network_lv.png
	:width: 42em
	:alt: alternate Text
	:align: center

`[Source: 1] <http://www.e-cigre.org/Order/select.asp?ID=729590>`_
