###########
Toolbox
###########

The pandapower toolbox is a collection of helper functions that are implemented for the pandapower framework. It is
designed for functions of common application that fit nowhere else. Have a look at the available functions to save
yourself the effort of maybe implementing something twice. If you develop some functionality which could be
interesting to other users as well and do not fit into one of the specialized packages, feel welcome to add your
contribution. To improve overview functions are loosely grouped by functionality, please adhere to this notion when
adding your own functions and feel free to open new groups as needed.

.. note::
    If you implement a function that might be useful for others, it is mandatory to add a short docstring to make browsing
    the toolbox practical. Ideally further comments if appropriate and a reference of authorship should be added as well.

====================================
General Issues
====================================

.. autofunction:: pandapower.element_bus_tuples

.. autofunction:: pandapower.pp_elements

.. autofunction:: pandapower.pq_from_cosphi

.. autofunction:: pandapower.cosphi_from_pq

.. autofunction:: pandapower.dataframes_equal

.. autofunction:: pandapower.compare_arrays

.. autofunction:: pandapower.ensure_iterability

====================================
Result Information
====================================

.. autofunction:: pandapower.lf_info

.. autofunction:: pandapower.opf_task

.. autofunction:: pandapower.switch_info

.. autofunction:: pandapower.overloaded_lines

.. autofunction:: pandapower.violated_buses

.. autofunction:: pandapower.nets_equal

.. autofunction:: pandapower.clear_result_table

====================================
Simulation Setup and Preparation
====================================

.. autofunction:: pandapower.add_column_from_node_to_elements

.. autofunction:: pandapower.add_column_from_element_to_elements

.. autofunction:: pandapower.add_zones_to_elements

.. autofunction:: pandapower.reindex_buses

.. autofunction:: pandapower.create_continuous_bus_index

.. autofunction:: pandapower.reindex_elements

.. autofunction:: pandapower.create_continuous_elements_index

.. autofunction:: pandapower.set_scaling_by_type

.. autofunction:: pandapower.convert_format

.. autofunction:: pandapower.set_data_type_of_columns_to_default

====================================
Topology Modification
====================================

.. autofunction:: pandapower.close_switch_at_line_with_two_open_switches

.. autofunction:: pandapower.fuse_buses

.. autofunction:: pandapower.drop_buses

.. autofunction:: pandapower.drop_switches_at_buses

.. autofunction:: pandapower.drop_elements_at_buses

.. autofunction:: pandapower.drop_trafos

.. autofunction:: pandapower.drop_lines

.. autofunction:: pandapower.drop_duplicated_measurements

.. autofunction:: pandapower.set_element_status

.. autofunction:: pandapower.set_isolated_areas_out_of_service

.. autofunction:: pandapower.drop_out_of_service_elements

.. autofunction:: pandapower.drop_inactive_elements

.. autofunction:: pandapower.select_subnet

.. autofunction:: pandapower.merge_nets

.. autofunction:: pandapower.create_replacement_switch_for_branch

.. autofunction:: pandapower.replace_zero_branches_with_switches

.. autofunction:: pandapower.replace_impedance_by_line

.. autofunction:: pandapower.replace_line_by_impedance

.. autofunction:: pandapower.replace_ext_grid_by_gen

.. autofunction:: pandapower.replace_gen_by_ext_grid

.. autofunction:: pandapower.replace_gen_by_sgen

.. autofunction:: pandapower.replace_sgen_by_gen

====================================
Item/Element Selection
====================================

.. autofunction:: pandapower.get_element_index

.. autofunction:: pandapower.get_element_indices

.. autofunction:: pandapower.next_bus

.. autofunction:: pandapower.get_connected_elements

.. autofunction:: pandapower.get_connected_buses

.. autofunction:: pandapower.get_connected_buses_at_element

.. autofunction:: pandapower.get_connected_switches

.. autofunction:: pandapower.get_connected_elements_dict
