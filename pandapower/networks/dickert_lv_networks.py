import pandapower as pp
from numpy import append, ceil
from copy import deepcopy


def _change_to_ohl(net, idx_busbar, new_lines, n_cable):
    """
    This function changes line types from cable to ohl beginning at the end of the feeders in a \
    way, that the tapped line has the most portions of overhead-lines.
    """
    con_lines = pp.get_connected_elements(net, "line", idx_busbar) & new_lines
    cable_lines = con_lines
    last_con_lines = list(con_lines)

    while len(cable_lines) < n_cable:
        con_lines = sorted(pp.get_connected_elements(
                net, "line", net.line.to_bus.loc[last_con_lines]) & new_lines - cable_lines)
        last_con_lines = deepcopy(con_lines)
        while len(con_lines) > 0:
            cable_lines.add(con_lines.pop(0))
    for idx_line in list(new_lines - cable_lines):
        pp.change_std_type(net, idx_line, 'NFA2X 4x70', element="line")


def _create_loads_with_coincidence(net, buses):
    """
    This function creates loads to lv feeders with respect to coincidence factor c. c is \
    calculated by the number of feeder buses as described in Dickert, Schegner - \
    'Residential Load Models for Network Planning Purposes', Modern Electric Power Systems 2010, \
    Wroclaw, Poland.
    """
    # assumptions
    c_inf = 0.1
    P_max1 = 0.01  # MW
    powerfactor = 0.95  # in range of 0.9 to 1

    # calculations
    n_buses = len(buses)
    c = c_inf + (1 - c_inf) * n_buses**(-1/2)
    p_mw = c * P_max1
    p_mw, q_mvar = pp.pq_from_cosphi(p_mw, powerfactor, qmode='underexcited', pmode="load")

    # create loads
    for i in buses:
        pp.create_load(net, i, p_mw=p_mw, q_mvar=q_mvar, sn_mva=P_max1)


def _create_feeder(net, net_data, branching, idx_busbar, linetype, lv_vn_kv):
    """
    This function creates the dickert lv network feeders by creating the buses and lines. In case \
    of branching the right position of branching must be found with respect to sum of feeder nodes \
    'n_DP' and the number of 'branching'.
    """
    n_DP = net_data[1]
    d_DP = net_data[0]
    buses = pp.create_buses(net, int(n_DP), lv_vn_kv, zone='Feeder B' + str(branching),
                            type='m')
    from_bus = append(idx_busbar, buses[:-1])
    # branch consideration
    if branching == 1:
        n_LS = int(ceil(n_DP / 3))
        idx_B1 = 2*n_LS
        if n_LS*3 - n_DP == 2:
            idx_B1 -= 1
        from_bus[idx_B1] = buses[n_LS-1]
    elif branching == 2:
        n_LS = int(ceil(n_DP / 6))
        idx_B1 = 3*n_LS
        idx_B2 = 4*n_LS
        if n_LS*6 - n_DP >= 1:
            idx_B2 -= 1
        if n_LS*6 - n_DP >= 4:
            idx_B1 -= 1
            idx_B2 -= 1
        if n_LS*6 - n_DP == 5:
            idx_B2 -= 1
        from_bus[idx_B1] = buses[2*n_LS-1]
        from_bus[idx_B2] = buses[n_LS-1]
    elif branching == 3:
        n_LS = int(ceil(n_DP / 10))
        idx_B1 = 4*n_LS
        idx_B2 = 5*n_LS
        idx_B3 = 7*n_LS
        if n_LS*10 - n_DP >= 1:
            idx_B2 -= 1
            idx_B3 -= 1
        if n_LS*10 - n_DP >= 2:
            idx_B3 -= 1
        if n_LS*10 - n_DP >= 3:
            idx_B3 -= 1
        if n_LS*10 - n_DP >= 7:
            idx_B1 -= 1
            idx_B2 -= 1
            idx_B3 -= 1
        if n_LS*10 - n_DP >= 8:
            idx_B2 -= 1
            idx_B3 -= 1
        if n_LS*10 - n_DP == 9:
            idx_B3 -= 1
        from_bus[idx_B1] = buses[3*n_LS-1]
        from_bus[idx_B2] = buses[2*n_LS-1]
        from_bus[idx_B3] = buses[n_LS-1]
    elif branching != 0:
        raise ValueError("branching must be in (0, 1, 2, 3), but is %s" % str(branching))

    # create lines
    new_lines = set()
    for i, f_bus in enumerate(from_bus):
        new_lines.add(pp.create_line(net, f_bus, buses[i], length_km=d_DP*1e-3,
                                     std_type='NAYY 4x150 SE'))

    # line type consideration
    if linetype == 'C&OHL':
        _change_to_ohl(net, idx_busbar, new_lines, round(len(new_lines)*0.4))

    # create loads
    _create_loads_with_coincidence(net, buses)


def create_dickert_lv_feeders(net, busbar_index, feeders_range='short', linetype='cable',
                              customer='single', case='good'):
    """
    This function creates LV feeders from J. Dickert, M. Domagk and P. Schegner. "Benchmark \
    low voltage distribution networks based on cluster analysis of actual grid properties". \
    PowerTech, 2013 IEEE Grenoble.
    The number of this LV feeders will be in range of one to three, with respect to the optional \
    given parameters 'feeders_range', 'linetype', 'customer' and 'case'.
    The given 'preferred lines for feeders' are used, knowing that there are some other \
    standard types mentioned as well.

    Since the paper focusses on LV grids structure, load powers and MV connection are neglected, \
    so that the user should identify appropriate assumptions for trafo and load parameters.
    'trafo_type_name' and 'trafo_type_data' can be set directly by the user.
    By default, the load powers are calculated with coincidence factor, derived with normal \
    distributed peak system demand.

    INPUT:
        **net** (pandapowerNet) - The pandapower network to that the feeder will be connected to

        **busbar_index** (int) - The bus index of busbar, the feeders should be connected to

    OPTIONAL:

        **feeders_range** (str, 'short') - feeder length, which can be ('short', 'middle', 'long')

        **linetype** (str, 'cable') - the are different feeders provided for 'cable' or 'C&OHL'

        **customer** (str, 'single') - type of customers ('single' or 'multiple') supplied by the
            feeders

        **case** (str, 'good') - case of supply mission, which can be ('good', 'average', 'worse')

    EXAMPLE:

        import pandapower.networks as pn

        net = pn.create_dickert_lv_network()

        pn.create_dickert_lv_feeders(net, busbar_index=1, customer='multiple')
    """
    # --- paper data - TABLE III and IV
    parameters = {'short': {'cable': {'single': {'good': [60, 1, False, False, False],
                                                 'average': [120, 1, False, False, False],
                                                 'worse': [80, 2, False, False, False]},
                                      'multiple': {'good': [80, 3, True, False, False],
                                                   'average': [50, 6, True, False, False],
                                                   'worse': [40, 10, True, False, False]}}},
                  'middle': {'cable': {'multiple': {'good': [40, 15, True, True, False],
                                                    'average': [35, 20, True, True, False],
                                                    'worse': [30, 25, True, True, False]}},
                             'C&OHL': {'multiple': {'good': [50, 10, True, True, False],
                                                    'average': [45, 13, True, True, False],
                                                    'worse': [40, 16, True, True, False]}}},
                  'long': {'cable': {'multiple': {'good': [30, 30, False, True, True],
                                                  'average': [30, 40, False, True, True],
                                                  'worse': [30, 50, False, True, True]}},
                           'C&OHL': {'multiple': {'good': [40, 20, False, True, True],
                                                  'average': [40, 30, False, True, True],
                                                  'worse': [40, 40, False, True, True]}}}}
    # process network choosing input data
    try:
        case = case if case != "bad" else "worse"
        net_data = parameters[feeders_range][linetype][customer][case]
    except KeyError:
        raise ValueError("This combination of 'feeders_range', 'linetype', 'customer' and 'case' "
                         "is no dickert network.")

    # add missing line types
    if 'NFA2X 4x70' not in net.std_types['line'].keys():
        pp.create_std_type(net, {"c_nf_per_km": 12.8, "r_ohm_per_km": 0.443, "x_ohm_per_km": 0.07,
                                 "max_i_ka": 0.205, "type": "ol"}, name='NFA2X 4x70',
                           element="line")
    # determine low voltage vn_kv
    lv_vn_kv = net.bus.vn_kv.at[busbar_index]

    # feeder without branch line
    _create_feeder(net, net_data, 0, busbar_index, linetype, lv_vn_kv)
    # feeder with one branch line
    if net_data[2]:
        _create_feeder(net, net_data, 1, busbar_index, linetype, lv_vn_kv)
    # feeder with two branch lines
    if net_data[3]:
        _create_feeder(net, net_data, 2, busbar_index, linetype, lv_vn_kv)
    # feeder with three branch lines
    if net_data[4]:
        _create_feeder(net, net_data, 3, busbar_index, linetype, lv_vn_kv)


def create_dickert_lv_network(feeders_range='short', linetype='cable', customer='single',
                              case='good', trafo_type_name='0.4 MVA 20/0.4 kV',
                              trafo_type_data=None):
    """
    This function creates a LV network from J. Dickert, M. Domagk and P. Schegner. "Benchmark \
    low voltage distribution networks based on cluster analysis of actual grid properties". \
    PowerTech, 2013 IEEE Grenoble.
    This LV network will have one to three feeders connected to MV-LV-Trafo. To connect more feeders
    with respect to the optional given parameters 'feeders_range', 'linetype', 'customer' and
    'case', the 'create_dickert_lv_feeders' function can be executed.
    The given 'preferred lines for feeders' are used, knowing that there are some other \
    standard types mentioned as well.

    Since the paper focusses on LV grids structure, load powers and MV connection are neglected, \
    so that the user should identify appropriate assumptions for trafo and load parameters.
    'trafo_type_name' and 'trafo_type_data' can be set directly by the user.
    By default, the load powers are calculated with coincidence factor, derived with normal \
    distributed peak system demand, described in Dickert, Schegner - \
    'Residential Load Models for Network Planning Purposes', Modern Electric Power Systems 2010, \
    Wroclaw, Poland, with the given example assumptions:
        - c_inf = 0.1
        - P_max1 = 10 kW
        - powerfactor = 0.95 ind. (in range of 0.9 to 1)

    OPTIONAL:

        **feeders_range** (str, 'short') - feeder length, which can be ('short', 'middle', 'long')

        **linetype** (str, 'cable') - the are different feeders provided for 'cable' or 'C&OHL'

        **customer** (str, 'single') - type of customers ('single' or 'multiple') supplied by the \
            feeders

        **case** (str, 'good') - case of supply mission, which can be ('good', 'average', 'bad')

        **trafo_type_name** (str, '0.4 MVA 20/0.4 kV') - name of the HV-MV-Trafo standard type

        **trafo_type_data** (dict, None) - if 'trafo_type_name' is not in pandapower standard \
            types, the data of this new trafo types must be given here in pandapower trafo type way

    OUTPUT:

        **net** (pandapowerNet) - Returns the required dickert lv network

    EXAMPLE:

        import pandapower.networks as pn

        net = pn.create_dickert_lv_network()
    """
    # --- create network
    net = pp.create_empty_network(name='dickert_lv_network with' + feeders_range +
                                  '-range feeders, ' + linetype + 'and ' + customer +
                                  'customers in ' + case + 'case')
    # assumptions
    mv_vn_kv = 20
    lv_vn_kv = 0.4

    # create mv connection
    mv_bus = pp.create_bus(net, mv_vn_kv, name='mv bus')
    busbar_index = pp.create_bus(net, lv_vn_kv, name='busbar')
    pp.create_ext_grid(net, mv_bus)
    if trafo_type_name not in net.std_types['trafo'].keys():
        pp.create_std_type(net, trafo_type_data, name=trafo_type_name, element="trafo")
    pp.create_transformer(net, mv_bus, busbar_index, std_type=trafo_type_name)

    # create feeders
    create_dickert_lv_feeders(net=net, busbar_index=busbar_index, feeders_range=feeders_range,
                              linetype=linetype, customer=customer, case=case)

    return net

if __name__ == "__main__":
    if 0:
        feeders_range = 'middle'
        linetype = 'C&OHL'
        customer = 'multiple'
        case = 'bad'
        trafo_type_name = '0.4 MVA 20/0.4 kV'
        trafo_type_data = None
        net = create_dickert_lv_network(feeders_range=feeders_range, linetype=linetype,
                                        customer=customer, case=case,
                                        trafo_type_name=trafo_type_name,
                                        trafo_type_data=trafo_type_data)
        from pandapower.plotting import simple_plot
        simple_plot(net)
    else:
        pass
