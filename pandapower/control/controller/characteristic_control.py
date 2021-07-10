# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from pandapower.control.basic_controller import Controller


class CharacteristicControl(Controller):
    """
    Controller that adjusts a certain parameter of an element in pandapower net based on a specified input parameter in pandapower net,
    according to a provided characteristic.
    Example: change the tap position of the transformers (net.trafo.tap_pos) based on transformer loading (net.res_trafo.loading_percent)
    according to a specified linear relationship. To this end, the input element is "res_trafo", the input variable is "loading_percent",
    the output element is "trafo" and the output variable is "tap_pos". The relationship between the values of the input and output
    variables is specified using the Characteristic class (or a scipy interpolator, e.g. scipy.interpolate.interp1d).

    INPUT:
        **net** (attrdict) - Pandapower net

        **output_element** (str) - name of the element table in pandapower net where the values are adjusted by the controller

        **output_variable** (str) - variable in the output element table, values of which are adjusted by the controller. Can also be an
                                        attribute of an object (e.g. parameter of a controller object), for this case it must start with
                                        "object." (e.g. "object.vm_set_pu")

        **output_element_index** (int or list or numpy array) - index of the elements, values fro which are adjusted

        **input_element** (str) - name of the element table or the element result table in pandapower net that provides input values for
                                    the controller

        **input_variable** (str) - name of the input variable in the input element table. Can also be an attribute of an object,
                                    similarly to output_variable

        **input_element_index** (int or list or numpy array) - index of elements in the input element table

        **characteristic** (object of class Characteristic, or a scipy interpolator object) - characteristic curve that describes the
                                                                                                relationship between the input and
                                                                                                output values

        **tol** (float) - tolerance for convergence

    OPTIONAL:

        **in_service** (bool, True) - Indicates if the controller is currently in_service

        **drop_same_existing_ctrl** (bool, False) - Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    """
    def __init__(self, net, output_element, output_variable, output_element_index, input_element, input_variable, input_element_index,
                 characteristic, tol=1e-3, in_service=True, order=0, level=0, drop_same_existing_ctrl=False, matching_params=None,
                 **kwargs):
        if matching_params is None:
            matching_params = {"element": output_element, "input_variable": input_variable, "output_variable": output_variable,
                               "element_index": input_element_index}
        super().__init__(net, in_service=in_service, order=order, level=level, drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, **kwargs)
        self.input_element = input_element
        self.input_variable = input_variable
        self.input_element_index = input_element_index
        self.output_element = output_element
        self.output_variable = output_variable
        self.output_element_index = output_element_index
        self.characteristic = characteristic
        self.tol = tol
        self.applied = False
        self.values = None
        self.input_object_attribute = None
        self.output_object_attribute = None
        # write functions faster, depending on type of self.output_element_index
        if self.output_variable.startswith('object'):
            # write to object attribute
            self.write = "object"
            self.output_object_attribute = self.output_variable.split(".")[1]
        elif isinstance(self.output_element_index, int):
            # use .at if element_index is integer for speedup
            self.write = "single_index"
        else:
            # use common .loc
            self.write = "loc"

        # read functions faster, depending on type of self.input_element_index
        if self.input_variable.startswith('object'):
            # write to object attribute
            self.read = "object"
            self.input_object_attribute = self.input_variable.split(".")[1]
        elif isinstance(self.input_element_index, int):
            # use .at if element_index is integer for speedup
            self.read = "single_index"
        else:
            # use common .loc
            self.read = "loc"

    def initialize_control(self, net):
        """
        At the beginning of each run_control call reset applied-flag
        """
        self.values = None
        self.applied = False

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        # read input values
        input_values = self.read_from_net(net)
        # calculate set values
        self.values = self.characteristic(input_values)
        # read previous set values
        output_values = self.read_from_net(net, output=True)
        # compare old and new set values
        diff = self.values - output_values
        # write new set values
        self.write_to_net(net)
        return self.applied and np.all(np.abs(diff) < self.tol)

    def control_step(self, net):
        """
        Set applied to true to make sure it runs at least once
        """
        self.applied = True

    def read_from_net(self, net, output=False):
        """
        Writes to self.element at index self.element_index in the column self.variable the data
        from self.values
        """
        # write functions faster, depending on type of self.element_index
        if output:
            element = self.output_element
            variable = self.output_variable
            object_attribute = self.output_object_attribute
            index = self.output_element_index
            flag = self.write
        else:
            element = self.input_element
            variable = self.input_variable
            object_attribute = self.input_object_attribute
            index = self.input_element_index
            flag = self.read

        if flag == "single_index":
            return self._read_from_single_index(net, element, variable, index)
        elif flag == "loc":
            return self._read_with_loc(net, element, variable, index)
        elif flag == "object":
            return self._read_from_object_attribute(net, element, object_attribute, index)
        else:
            raise NotImplementedError("CharacteristicControl: self.read must be one of "
                                      "['single_index', 'all_index', 'loc']")

    def write_to_net(self, net):
        """
        Writes to self.element at index self.element_index in the column self.variable the data
        from self.values
        """
        # write functions faster, depending on type of self.element_index
        if self.write == "single_index":
            self._write_to_single_index(net)
        elif self.write == "all_index":
            self._write_to_all_index(net)
        elif self.write == "loc":
            self._write_with_loc(net)
        elif self.write == "object":
            self._write_to_object_attribute(net)
        else:
            raise NotImplementedError("CharacteristicControl: self.write must be one of "
                                      "['single_index', 'all_index', 'loc']")

    def _write_to_single_index(self, net):
        net[self.output_element].at[self.output_element_index, self.output_variable] = self.values

    def _write_to_all_index(self, net):
        net[self.output_element].loc[:, self.output_variable] = self.values

    def _write_with_loc(self, net):
        net[self.output_element].loc[self.output_element_index, self.output_variable] = self.values

    def _write_to_object_attribute(self, net):
        if hasattr(self.output_element_index, '__iter__') and len(self.output_element_index) > 1:
            for idx, val in zip(self.output_element_index, self.values):
                setattr(net[self.output_element]["object"].at[idx], self.output_object_attribute, val)
        else:
            setattr(net[self.output_element]["object"].at[self.output_element_index], self.output_object_attribute, self.values)

    def _read_from_single_index(self, net, element, variable, index):
        return net[element].at[index, variable]

    def _read_from_all_index(self, net, element, variable):
        return net[element].loc[:, variable]

    def _read_with_loc(self, net, element, variable, index):
        return net[element].loc[index, variable]

    def _read_from_object_attribute(self, net, element, object_attribute, index):
        if hasattr(index, '__iter__') and len(index) > 1:
            values = np.array(shape=index.shape)
            for i, idx in enumerate(index):
                values[i] = setattr(net[element]["object"].at[idx], object_attribute)
        else:
            values = getattr(net[element]["object"].at[index], object_attribute)
        return values

    def __str__(self):
        return super().__str__() + " [%s.%s.%s.%s]" % (self.input_element, self.input_variable, self.output_element, self.output_variable)
