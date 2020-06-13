# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np

import pandapower.topology as top


def plot_voltage_profile(net, plot_transformers=True, ax=None, xlabel="Distance from Slack [km]",
                         ylabel="Voltage [pu]", x0=0, trafocolor="r", bus_colors=None,
                         line_loading_weight=False, voltage_column=None, bus_size=3, lines=None,
                         **kwargs):
    if ax is None:
        plt.figure(facecolor="white", dpi=120)
        ax = plt.gca()
    if not net.converged:
        raise ValueError("no results in this pandapower network")
    if voltage_column is None:
        voltage_column = net.res_bus.vm_pu
    if lines is None:
        lines = net.line.index
    for eg in net.ext_grid[net.ext_grid.in_service == True].bus:
        d = top.calc_distance_to_bus(net, eg)
        for lix, line in net.line[(net.line.in_service == True) & net.line.index.isin(lines)].iterrows():
            if line.from_bus not in d.index:
                continue
            if not ((net.switch.element == line.name) & ~net.switch.closed & (
                    net.switch.et == 'l')).any():
                from_bus = line.from_bus
                to_bus = line.to_bus
                x = [x0 + d.at[from_bus], x0 + d.at[to_bus]]
                try:
                    y = [voltage_column.at[from_bus], voltage_column.at[to_bus]]
                except:
                    raise UserWarning
                if "linewidth" in kwargs or not line_loading_weight:
                    ax.plot(x, y, **kwargs)
                else:
                    ax.plot(x, y, linewidth=0.4 * np.sqrt(net.res_line.loading_percent.at[lix]),
                            **kwargs)
                if bus_colors is not None:
                    for bus, x, y in zip((from_bus, to_bus), x, y):
                        if bus in bus_colors:
                            ax.plot(x, y, 'or', color=bus_colors[bus], ms=bus_size)
                kwargs = {k: v for k, v in kwargs.items() if not k == "label"}
        # if plot_transformers:
        #     if hasattr(plot_transformers, "__iter__"):  # if is a list for example
        #         transformers = net.trafo.loc[list(plot_transformers)]
        #     else:
        #         transformers = net.trafo[net.trafo.in_service == True]
        #     for _, transformer in transformers.iterrows():
        #         if transformer.hv_bus not in d.index:
        #             continue
        #         ax.plot([x0 + d.loc[transformer.hv_bus],
        #                  x0 + d.loc[transformer.lv_bus]],
        #                 [voltage_column.loc[transformer.hv_bus],
        #                  voltage_column.loc[transformer.lv_bus]], color=trafocolor,
        #                 **{k: v for k, v in kwargs.items() if not k == "color"})

        # trafo geodata
        if plot_transformers:
            for trafo_table in ['trafo', 'trafo3w']:
                if trafo_table not in net.keys():
                    continue
                transformers = net[trafo_table].query('in_service')
                for tid, tr in transformers.iterrows():
                    t_buses = [tr[b_col] for b_col in ('lv_bus', 'mv_bus', 'hv_bus') if
                               b_col in tr.index]
                    if any([b not in d.index.values or b not in net.res_bus.index.values for b in
                            t_buses]):
                        # logger.info('cannot add trafo %d to plot' % tid)
                        continue

                    for bi, bj in combinations(t_buses, 2):
                        tr_coords = ([x0 + d.loc[bi], x0 + d.loc[bj]],
                                     [net.res_bus.at[bi, 'vm_pu'], net.res_bus.at[bj, 'vm_pu']])
                        ax.plot(*tr_coords, color=trafocolor,
                                **{k: v for k, v in kwargs.items() if not k == "color"})

        if xlabel:
            ax.set_xlabel(xlabel, fontweight="bold", color=(.4, .4, .4))
        if ylabel:
            ax.set_ylabel(ylabel, fontweight="bold", color=(.4, .4, .4))
    return ax


def plot_loading(net, element="line", boxcolor="b", mediancolor="r", whiskercolor="k", ax=None, index_subset=None):
    if ax is None:
        plt.figure(facecolor="white", dpi=80)
        ax = plt.gca()
    if index_subset is None:
        index_subset = net[element].index
    loadings = net["res_%s" % element].loading_percent.values[net["res_%s" % element].index.isin(index_subset)]
    boxplot = ax.boxplot(loadings[~np.isnan(loadings)], whis="range")
    for l in list(boxplot.keys()):
        plt.setp(boxplot[l], lw=3)
        if l == "medians":
            plt.setp(boxplot[l], color=mediancolor)
        elif l == "boxes" or l == "whiskers":
            plt.setp(boxplot[l], color=boxcolor)
        else:
            plt.setp(boxplot[l], color=whiskercolor)


def voltage_profile_to_bus_geodata(net, voltages=None):
    if voltages is None:
        if not net.converged:
            raise ValueError("no results in this pandapower network")
        voltages = net.res_bus.vm_pu

    mg = top.create_nxgraph(net, respect_switches=True)
    first_eg = net.ext_grid.bus.values[0]
    mg.add_edges_from([(first_eg, y, {"weight": 0}) for y in net.ext_grid.bus.values[1:]])
    dist = pd.Series(nx.single_source_dijkstra_path_length(mg, first_eg))

    bgd = pd.DataFrame({"x": dist.loc[net.bus.index.values].values,
                        "y": voltages.loc[net.bus.index.values].values},
                       index=net.bus.index)
    return bgd


if __name__ == "__main__":
    import pandapower as pp
    import pandapower.networks as nw
    import pandas as pd
    import networkx as nx
    import plotting

    net = nw.mv_oberrhein()
    pp.runpp(net)
    mg = top.create_nxgraph(net, respect_switches=True)
    feeders = list(top.connected_components(mg, notravbuses=set(net.trafo.lv_bus.values)))
    lines_with_open_switches = set(net.switch.query("not closed and et == 'l'").element.values)

    fig, axs = plt.subplots(2)
    for bgd, ax in zip([net.bus_geodata, voltage_profile_to_bus_geodata(net)], axs):
        for color, f in zip(["C0", "C1", "C2", "C3"], feeders):
            l = set(net.line.index[net.line.from_bus.isin(f)]) - lines_with_open_switches
            c = plotting.create_line_collection(net, lines=l, use_bus_geodata=True, color=color,
                                                bus_geodata=bgd)
            ax.add_collection(c)
            #            ax.scatter(bgd["x"], bgd["y"])
            ax.autoscale_view(True, True, True)

    h = 0.02
#    bc = plotting.create_bus_collection(net, buses=net.ext_grid.bus.values, width=h / ax.get_data_ratio(), height=h, bus_geodata=bgd)
#    ax.add_collection(bc)
