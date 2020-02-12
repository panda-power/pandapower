# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
import numpy as np

from pandapower.auxiliary import ppException

try:
    import pplog as logging
except ImportError:
    import logging
logger = logging.getLogger(__name__)

class MapboxTokenMissing(ppException):
    """
    Exception being raised in case loadflow did not converge.
    """
    pass

def _on_map_test(x, y):
    """
    checks if bus_geodata can be located on a map using geopy
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        geolocator = Nominatim(user_agent="pandapower_user_mapboxplot")

    except ImportError:
        # if geopy is not available there will be no geo-coordinates check
        # therefore if geo-coordinates are not real and user sets on_map=True, an empty map will be plot!
        logger.warning('Geo-coordinates check cannot be peformed because geopy package not available \n\t--> '
                       'if geo-coordinates are not in lat/lon format an empty plot may appear...')
        return True
    try:
        location = geolocator.reverse("{0}, {1}".format(x, y), language='en-US')
    except GeocoderTimedOut:
        logger.Error("Existing net geodata cannot be geo-located: possible reason: geo-data not in lat/long ->"
                     "try geo_data_to_latlong(net, projection) to transform geodata to lat/long!")

    if location.address is None:
        return False
    else:
        return True


def geo_data_to_latlong(net, projection):
    """
    Transforms network's geodata (in `net.bus_geodata` and `net.line_geodata`) from specified projection to lat/long (WGS84).

    INPUT:
        **net** (pandapowerNet) - The pandapower network

        **projection** (String) - projection from which geodata are transformed to lat/long. some examples

                - "epsg:31467" - 3-degree Gauss-Kruger zone 3
                - "epsg:2032" - NAD27(CGQ77) / UTM zone 18N
                - "epsg:2190" - Azores Oriental 1940 / UTM zone 26N
    """
    try:
        from pyproj import Proj, transform
    except ImportError:
        logger.warning('Geo-coordinates check cannot be peformed because pyproj package not available \n\t--> '
                       'if geo-coordinates are not in lat/lon format an empty plot may appear...')
        return
    if projection == 'epsg:4326':
        return
    wgs84 = Proj(init='epsg:4326')  # lat/long

    try:
        projection = Proj(init=projection)
    except:
        logger.warning("Transformation of geodata to lat/long failed! because of:]\n"
                       "Unknown projection provided "
                         "(format 'epsg:<number>' required as available at http://spatialreference.org/ref/epsg/ )")
        return

    # transform all geodata to long/lat using set or found projection
    try:
        lon, lat = transform(projection, wgs84, net.bus_geodata.loc[:, 'x'].values, net.bus_geodata.loc[:, 'y'].values)
        net.bus_geodata.loc[:, 'x'], net.bus_geodata.loc[:, 'y'] = lat, lon

        if net.line_geodata.shape[0] > 0:
            for idx in net.line_geodata.index:
                line_coo = np.array(net.line_geodata.loc[idx, 'coords'])
                lon, lat = transform(projection, wgs84, line_coo[:, 0], line_coo[:, 1])
                net.line_geodata.loc[idx, 'coords'] = np.array([lat,lon]).T.tolist()
        return
    except:
        logger.warning('Transformation of geodata to lat/long failed!')
        return

def set_mapbox_token(token):
    from pandapower import pp_dir
    path = os.path.join(pp_dir, "plotting", "plotly")
    filename = os.path.join(path, 'mapbox_token.txt')
    with open(filename, "w") as mapbox_file:
        mapbox_file.write(token)

def _get_mapbox_token():
    from pandapower import pp_dir
    path = os.path.join(pp_dir, "plotting", "plotly")
    filename = os.path.join(path, 'mapbox_token.txt')
    with open(filename, "r") as mapbox_file:
        return mapbox_file.read()

