=============================
Generic Coordinates
=============================

If there are no geocoordinates in a network, generic coordinates can be created. There are two possibilities:

1. igraph (http:/igraph.org/python) (recommended) based on
    - python-igraph
    - pycairo

2. graphviz (http:/www.graphviz.org) based on 
    - networkx
    - graphviz

To avoid having to compile C libraries, precompiled wheels are available on https://www.lfd.uci.edu/%7Egohlke/pythonlibs/ (unofficial)


Generically created geocoordinates can then be plotted in the same way as real geocoordinates.


.. autofunction:: pandapower.plotting.create_generic_coordinates


Example plot with mv_oberrhein network from the pandapower.networks package as geographical plan (respect_switches=False):

.. image:: /pics/plotting/plotting_tutorial2.png
	:width: 30em
	:align: center
    
and as structural plan (respect_switches=True):

.. image:: /pics/plotting/plotting_tutorial3.png
	:width: 20em
	:align: center



