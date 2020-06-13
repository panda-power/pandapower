.. _timeseriesexample:

#############################
Timeseries Example
#############################

Common Use Case
=================
A common use case is to add two :ref:`const controller <ConstControl>` to the grid to update the P values of loads and static generators.
The p_mw values for each time step are stored in a data source, which is defined by the class ``DataSource``. This class
holds a pandas DataFrame which contains the values for each element and time step.
The :ref:`const controller <ConstControl>` gets the data source, reads p_mw values from it and writes the
corresponding ones to each sgen / load prior to the power flow calculation.
After running the power flow calculation, results for each time steps are written by the output writer class ``OutputWriter`` to
excel, csv, json or pickle files.


Simulating Time Series
=======================================
First you need a ``DataSource`` to store the p_mw values of sgens or loads. For example multiple CSV-files can be used
to provide the values for each element and time step. Each column contains a profile with a value
for each time step in the corresponding row. Here is a schematic example for a grid with N static generators and T time steps:

.. tabularcolumns:: |p{0.12\linewidth}|p{0.10\linewidth}|p{0.25\linewidth}|p{0.30\linewidth}|
.. csv-table::
   :file: datasource_example.csv
   :delim: ;

This structure is saved to a *.csv file and can be read by pandas and passed to the DataSource constructor like so:

::

    import numpy as np
    import pandas as pd

    import pandapower.control as control
    import pandapower.networks as nw
    import pandapower.timeseries as timeseries
    from pandapower.timeseries.data_sources.frame_data import DFData

    # load a pandapower network
    net = nw.mv_oberrhein(scenario='generation')
    # number of time steps
    n_ts = 95
    # load your timeseries from a file (here csv file)
    # df = pd.read_csv("sgen_timeseries.csv")
    # or create a DataFrame with some random time series as an example
    df = pd.DataFrame(np.random.normal(1., 0.1, size=(n_ts, len(net.sgen.index))),
                      index=list(range(n_ts)), columns=net.sgen.index) * net.sgen.p_mw.values
    # create the data source from it
    ds = DFData(df)

    # initialising ConstControl controller to update values of the regenerative generators ("sgen" elements)
    # the element_index specifies which elements to update (here all sgens in the net since net.sgen.index is passed)
    # the controlled variable is "p_mw"
    # the profile_name are the columns in the csv file (here this is also equal to the sgen indices 0-N )
    const_sgen = control.ConstControl(net, element='sgen', element_index=net.sgen.index,
                                      variable='p_mw', data_source=ds, profile_name=net.sgen.index)

    # do the same for loads
    # df = pd.read_csv("load_timeseries.csv")
    # create a DataFrame with some random time series as an example
    df = pd.DataFrame(np.random.normal(1., 0.1, size=(n_ts, len(net.load.index))),
                      index=list(range(n_ts)), columns=net.load.index) * net.load.p_mw.values
    ds = DFData(df)
    const_load = control.ConstControl(net, element='load', element_index=net.load.index,
                                      variable='p_mw', data_source=ds, profile_name=net.load.index)

    # starting the timeseries simulation for one day -> 96 15 min values.
    timeseries.run_timeseries(net)


We created a ``DataSource`` and passed it to the ``ConstControl``, while also providing the name of the
P-profile. To get the time series calculation results and save it to separate files we build an ``OutputWriter``.

::

    # initialising the outputwriter to save data to excel files in the current folder. You can change this to .json, .csv, or .pickle as well
    ow = timeseries.OutputWriter(net, output_path="./", output_file_type=".xlsx")
    # adding vm_pu of all buses and line_loading in percent of all lines as outputs to be stored
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')

    # starting the timeseries simulation for one day -> 96 15 min values.
    timeseries.run_timeseries(net)
    # now checkout the folders res_bus and res_line in your current working dir

We created an ``OutputWriter`` and stored the voltage magnitude **vm_pu** for each bus and the line loading in percent
**loading_percent** for every line to separate excel files.