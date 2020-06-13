.. _elements:

###################################
Datastructure and Elements
###################################


A pandapower network consists of an element table for each electric element in the network. Each element table consists of a column for each
parameter and a row for each element.

pandapower provides electric models for 13 electric elements, for each of which you can find detailed 
information about the definition and interpretation of the parameters in the following documentation:


.. toctree:: 
    :maxdepth: 1

    elements/empty_network
    elements/bus
    elements/line
    elements/switch
    elements/load
    elements/asymmetric_load	
    elements/sgen
    elements/asymmetric_sgen
    elements/ext_grid	
    elements/trafo
    elements/trafo3w	
    elements/gen
    elements/shunt
    elements/impedance
    elements/ward	
    elements/xward
    elements/dcline
    elements/measurement
    elements/storage