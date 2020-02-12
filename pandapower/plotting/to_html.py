# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

# File created my Massimo Di Pierro

from itertools import combinations

try:
    from html import escape
except ImportError:
    # cgi is decprecated in python 3
    from cgi import escape
import json


def to_html(net, filename, respect_switches=True, include_lines=True, include_trafos=True, show_tables=True):
    """
    Saves a pandapower Network to an html file.

    INPUT:
        **net** (dict) - The pandapower format network

        **filename** (string) - The absolute or relative path to the input file.

    OPTIONAL:
        **respect_switches** (boolean, True) - True: open line switches are being considered (no edge between nodes)
                                               False: open line switches are being ignored

        **include_lines** (boolean, True) - determines, whether lines get converted to edges

        **include_trafos** (boolean, True) - determines, whether trafos get converted to edges

        **show_tables** (boolean, True) - shows pandapower element tables

    """
    if not filename.endswith(".html"):
        raise Exception("Please use .html to save pandapower networks!")
    with open(filename, "w") as f:
        html_str = _net_to_html(net, respect_switches, include_lines, include_trafos, show_tables)
        f.write(html_str)
        f.close()

class Raw(object):
    def __init__(self, html):
        self.html = html


class Tag(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        attr = ' ' + ' '.join('%s="%s"' % (k, escape(v)) for k, v in kwargs.items())
        contents = ''.join(a.html if isinstance(a, Raw) else escape(str(a)) for a in args)
        return Raw('<%s%s>%s</%s>' % (self.name, attr.rstrip(), contents, self.name))

def _net_to_graphjs(net, respect_switches=True, include_lines=True, include_trafos=True, images=None):
    """
     Converts a pandapower network into a graph for vis.js
     which contains a simplified representation
     of a network's topology, reduced to nodes and edges. Busses are being represented by nodes
     (Note: only buses with in_service = 1 appear in the graph), edges represent physical
     connections between buses (typically lines or trafos).

     INPUT:
        **net** (pandapowerNet) - variable that contains a pandapower network


     OPTIONAL:
        **respect_switches** (boolean, True) - True: open line switches are being considered
                                                     (no edge between nodes)
                                               False: open line switches are being ignored

        **include_lines** (boolean, True) - determines, whether lines get converted to edges

        **include_trafos** (boolean, True) - determines, whether trafos get converted to edges

        **show_tables** (boolean, True) - shows pandapower element tables

     EXAMPLE:

         from pandapower.html import _net_to_graphjs
         print _net_to_graph(net, respect_switches = False)
    """
    nodes = [{'id': int(id), 'label': str(id)} for id in net.bus[net.bus.in_service == 1].index]
    edges = []

    if images:
        for node in nodes:
            node['shape'] = 'image'
            node['image'] = images.get(str(node['id']), images['default'])

    if include_lines:
        # lines with open switches can be excluded
        nogolines = set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)]) \
            if respect_switches else set()

        edges += [{'from': int(fb),
                   'to': int(tb),
                   'label': '%.2f Mw %.2f Km' % (imax, length)}
                  for fb, tb, length, idx, inservice, imax in
                  list(zip(net.line.from_bus, net.line.to_bus, net.line.length_km,
                           net.line.index, net.line.in_service, net.line.max_i_ka))
                  if inservice == 1 and not idx in nogolines]

        edges += [{'from': int(fb), 'to': int(tb)}
                  for fb, tb, idx, inservice in
                  list(zip(net.impedance.from_bus, net.impedance.to_bus,
                           net.impedance.index, net.impedance.in_service))
                  if inservice == 1]

        edges += [{'from': int(fb),
                   'to': int(tb), 'dashes': 'true',
                   'label': '%.2f Mw %.2f Km' % (imax, length)}
                  for fb, tb, length, idx, inservice, imax in
                  list(zip(net.line.from_bus, net.line.to_bus, net.line.length_km,
                           net.line.index, net.line.in_service, net.line.max_i_ka))
                  if inservice != 1 or idx in nogolines]

        edges += [{'from': int(fb), 'to': int(tb), 'dashes': 'true'}
                  for fb, tb, idx, inservice in
                  list(zip(net.impedance.from_bus, net.impedance.to_bus,
                           net.impedance.index, net.impedance.in_service))
                  if inservice != 1]

    if include_trafos:
        nogotrafos = set(net.switch.element[(net.switch.et == "t") & (net.switch.closed == 0)])
        edges += [{'from': int(hvb), 'to': int(lvb), 'label': 'trafo'}
                  for hvb, lvb, idx, inservice in
                  list(zip(net.trafo.hv_bus, net.trafo.lv_bus,
                           net.trafo.index, net.trafo.in_service))
                  if inservice == 1 and not idx in nogotrafos]
        for _, t3tab in net.trafo3w.iterrows():
            edges += [{'from': int(bus1),'to': int(bus2),'label': 'trafo3'}
                      for bus1, bus2 in combinations([t3tab.hv_bus, t3tab.mv_bus, t3tab.lv_bus], 2)
                      if t3tab.in_service]

    # add bus-bus switches
    bs = net.switch[(net.switch.et == "b") &
                    ((net.switch.closed == 1) | (not respect_switches))]
    edges += [{'from': int(b), 'to': int(e), 'label': 'switch'}
              for b, e, i in list(zip(bs.bus, bs.element, bs.index))]

    script = "var data = {nodes: new vis.DataSet(%s), edges: new vis.DataSet(%s)};" % (
        json.dumps(nodes), json.dumps(edges))
    script += "var container = document.getElementById('net');"
    script += "var network = new vis.Network(container, data);"
    script += "network.setOptions({interaction: {zoomView: true}});"

    return script


def _net_to_html(net, respect_switches=True, include_lines=True, include_trafos=True, show_tables=True, images=None):
    """
     Converts a pandapower network into an html page which contains a simplified representation
     of a network's topology, reduced to nodes and edges. Busses are being represented by nodes
     (Note: only buses with in_service = 1 appear in the graph), edges represent physical
     connections between buses (typically lines or trafos).

     INPUT:
        **net** (pandapowerNet) - variable that contains a pandapower network


     OPTIONAL:
        **respect_switches** (boolean, True) - True: open line switches are being considered
                                                     (no edge between nodes)
                                               False: open line switches are being ignored

        **include_lines** (boolean, True) - determines, whether lines get converted to edges

        **include_trafos** (boolean, True) - determines, whether trafos get converted to edges

        **show_tables** (boolean, True) - shows pandapower element tables

        **images** (dict) - an optional dict of images to map a node id into an image

     EXAMPLE:

         from pandapower.html import to_html
         html = to_html(net, respect_switches = False)
         open('C:\\index.html', "w").write(html)

    """

    HTML, HEAD, STYLE, BODY, DIV = Tag('html'), Tag('head'), Tag('style'), Tag('body'), Tag('div')
    TABLE, TR, TH, TD, SCRIPT = Tag('table'), Tag('tr'), Tag('th'), Tag('td'), Tag('script')
    H2 = Tag('h2')

    style = 'tr:first {background:#e1e1e1;} th,td {text-align:center; border:1px solid #e1e1e1;}'

    script = _net_to_graphjs(net, respect_switches, include_lines, include_trafos,  images)

    tables = []
    if show_tables:
        for name in ['bus', 'trafo', 'line', 'load', 'ext_grid', 'gen', 'sgen',
                     'res_bus', 'res_trafo', 'res_line', 'res_load', 'res_ext_grid', 'res_gen', 'res_sgen']:
            item = getattr(net, name)
            table = TABLE(TR(*map(TH, item.columns)),
                          *[TR(*map(TD, row)) for row in item.values])
            tables.append(DIV(H2(name), table))

    page = HTML(
        HEAD(STYLE(style)),
        BODY(DIV(*tables), DIV(id='net', style="border:1px solid #f1f1f1;max-width:90%")),
        SCRIPT(src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.18.1/vis.min.js"),
        SCRIPT(Raw(script))
    )

    return page.html
