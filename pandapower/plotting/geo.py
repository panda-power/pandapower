try:
    from fiona.crs import from_epsg
    from shapely.geometry import Point, LineString
    from geopandas import GeoDataFrame, GeoSeries
    from pyproj import Proj, transform
except ImportError:
    pass


def _node_geometries_from_geodata(node_geo, epsg=31467):
    """
    Creates a geopandas geodataframe from a given dataframe of with node coordinates as x and y
    values.

    :param node_geo: The dataframe containing the node coordinates (x and y values)
    :type node_geo: pandas.dataframe
    :param epsg: The epsg projection of the node coordinates
    :type epsg: int, default 31467 (= Gauss-Krüger Zone 3)
    :return: node_geodata - a geodataframe containing the node_geo and Points in the geometry column
    """
    geoms = [Point(x, y) for x, y in node_geo[["x", "y"]].values]
    return GeoDataFrame(node_geo, crs=from_epsg(epsg), geometry=geoms, index=node_geo.index)


def _branch_geometries_from_geodata(branch_geo, epsg=31467):
    geoms = GeoSeries([LineString(x) for x in branch_geo.coords.values], index=branch_geo.index,
                      crs=from_epsg(epsg))
    return GeoDataFrame(branch_geo, crs=from_epsg(epsg), geometry=geoms, index=branch_geo.index)


def _transform_node_geometry_to_geodata(node_geo):
    """
    Create x and y values from geodataframe

    :param node_geo: The dataframe containing the node geometries (as shapely points)
    :type node_geo: geopandas.GeoDataFrame
    :return: bus_geo - The given geodataframe with x and y values
    """
    node_geo["x"] = [p.x for p in node_geo.geometry]
    node_geo["y"] = [p.y for p in node_geo.geometry]
    return node_geo


def _transform_branch_geometry_to_coords(branch_geo):
    """
    Create coords entries from geodataframe geometries

    :param branch_geo: The dataframe containing the branch geometries (as shapely LineStrings)
    :type branch_geo: geopandas.GeoDataFrame
    :return: branch_geo - The given geodataframe with coords
    """
    branch_geo["coords"] = branch_geo["coords"].geometry.apply(lambda x: list(x.coords))
    return branch_geo


def _convert_xy_epsg(x, y, epsg_in=4326, epsg_out=31467):
    """
    Converts the given x and y coordinates according to the defined epsg projections.

    :param x: x-values of coordinates
    :type x: iterable
    :param y: y-values of coordinates
    :type y: iterable
    :param epsg_in: current epsg projection
    :type epsg_in: int, default 4326 (= WGS84)
    :param epsg_out: epsg projection to be transformed to
    :type epsg_out: int, default 31467 (= Gauss-Krüger Zone 3)
    :return: transformed_coords - x and y values in new coordinate system
    """
    in_proj = Proj(init='epsg:%i' % epsg_in)
    out_proj = Proj(init='epsg:%i' % epsg_out)
    return transform(in_proj, out_proj, x, y)


def convert_gis_to_geodata(net, node_geodata=True, branch_geodata=True):
    """
    Extracts information on bus and line geodata from the geometries of a geopandas geodataframe.

    :param net: The net for which to convert the geodata
    :type net: pandapowerNet
    :param node_geodata: flag if to extract x and y values for bus geodata
    :type node_geodata: bool, default True
    :param branch_geodata: flag if to extract coordinates values for line geodata
    :type branch_geodata: bool, default True
    :return: No output.
    """
    if node_geodata:
        _transform_node_geometry_to_geodata(net.bus_geodata)
    if branch_geodata:
        _transform_branch_geometry_to_coords(net.line_geodata)


def convert_geodata_to_gis(net, epsg=31467, node_geodata=True, branch_geodata=True):
    """
    Transforms the bus and line geodata of a net into a geopandaas geodataframe with the respective
    geometries.

    :param net: The net for which to convert the geodata
    :type net: pandapowerNet
    :param epsg: current epsg projection
    :type epsg: int, default 4326 (= WGS84)
    :param node_geodata: flag if to transform the bus geodata table
    :type node_geodata: bool, default True
    :param branch_geodata: flag if to transform the line geodata table
    :type branch_geodata: bool, default True
    :return: No output.
    """
    if node_geodata:
        net["bus_geodata"] = _node_geometries_from_geodata(net["bus_geodata"], epsg)
    if branch_geodata:
        net["line_geodata"] = _branch_geometries_from_geodata(net["line_geodata"], epsg)
    net["gis_epsg_code"] = epsg


def convert_epsg_bus_geodata(net, epsg_in=4326, epsg_out=31467):
    """
    Converts bus geodata in net from epsg_in to epsg_out

    :param net: The pandapower network
    :type net: pandapowerNet
    :param epsg_in: current epsg projection
    :type epsg_in: int, default 4326 (= WGS84)
    :param epsg_out: epsg projection to be transformed to
    :type epsg_out: int, default 31467 (= Gauss-Krüger Zone 3)
    :return: net - the given pandapower network (no copy!)
    """
    net['bus_geodata'].loc[:, "x"], net['bus_geodata'].loc[:, "y"] = _convert_xy_epsg(
        net['bus_geodata'].loc[:, "x"], net['bus_geodata'].loc[:, "y"], epsg_in, epsg_out)
    return net
