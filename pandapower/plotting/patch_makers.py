from matplotlib.patches import RegularPolygon, Arc, Circle, Rectangle, Ellipse
import numpy as np
from pandapower.plotting.plotting_toolbox import _rotate_dim2, get_color_list, get_angle_list, \
    get_linewidth_list

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def node_patches(node_coords, size, patch_type, colors=None, **kwargs):
    """
    Creates node patches from coordinates translating the patch type into patches.

    :param node_coords: coordinates of the nodes to draw
    :type node_coords: iterable
    :param size: size of the patch (can be interpreted differently, depending on the patch type)
    :type size: float
    :param patch_type: type of patches to create  - can be one of
        - "circle" or "ellipse" for an ellipse (cirlces are just ellipses with the same width \
            + height)\
        - "rect" or "rectangle" for a rectangle\
        - "poly<n>" for a polygon with n edges
    :type patch_type: str
    :param colors: colors or color of the patches
    :type colors: iterable, float
    :param kwargs: additional keyword arguments to pass to the patch initialization \
        (might contain "width", "height", "angle" depending on the patch type)
    :type kwargs: dict
    :return: patches - list of rectangle patches for the nodes
    """
    if patch_type.lower() == 'ellipse' or patch_type.lower() == 'circle':
        # circles are just ellipses
        if patch_type.lower() == "circle" and len(set(kwargs.keys()) & {"width", "height"}) == 1:
            wh = kwargs["width"] if "width" in kwargs else kwargs["height"]
            width = wh
            height = wh
        else:
            width = kwargs.pop("width", 2 * size)
            height = kwargs.pop("height", 2 * size)
        angle = kwargs.pop('angle', 0)
        return ellipse_patches(node_coords, width, height, angle, color=colors, **kwargs)
    elif patch_type.lower() == "rect" or patch_type.lower() == "rectangle":
        width = kwargs.pop("width", 2 * size)
        height = kwargs.pop("height", 2 * size)
        return rectangle_patches(node_coords, width, height, color=colors, **kwargs)
    elif patch_type.lower().startswith("poly"):
        edges = int(patch_type[4:])
        return polygon_patches(node_coords, size, edges, color=colors, **kwargs)
    else:
        logger.error("Wrong patchtype. Please choose a correct patch type.")
        raise ValueError("Wrong patchtype")


def ellipse_patches(node_coords, width, height, angle=0, color=None, **kwargs):
    """
    Function to create a list of ellipse patches from node coordinates.

    :param node_coords: coordinates of the nodes to draw
    :type node_coords: iterable
    :param width: width of the ellipse (described by an exterior rectangle)
    :type width: float
    :param height: height of the ellipse (described by an exterior rectangle)
    :type height: float
    :param angle: angle by which to rotate the ellipse
    :type angle: float
    :param color: color or colors of the patches
    :type color: iterable, float
    :param kwargs: additional keyword arguments to pass to the Ellipse initialization
    :type kwargs: dict
    :return: patches - list of ellipse patches for the nodes
    """
    patches = list()
    angles = get_angle_list(angle, len(node_coords))
    if color is not None:
        colors = get_color_list(color, len(node_coords))
        for (x, y), col, ang in zip(node_coords, colors, angles):
            patches.append(Ellipse((x, y), width, height, angle=ang, color=col, **kwargs))
    else:
        for (x, y), ang in zip(node_coords, angles):
            patches.append(Ellipse((x, y), width, height, angle=ang, **kwargs))
    return patches


def rectangle_patches(node_coords, width, height, color=None, **kwargs):
    """
    Function to create a list of rectangle patches from node coordinates.

    :param node_coords: coordinates of the nodes to draw
    :type node_coords: iterable
    :param width: width of the rectangle
    :type width: float
    :param height: height of the rectangle
    :type height: float
    :param color: color or colors of the patches
    :type color: iterable, float
    :param kwargs: additional keyword arguments to pass to the Rectangle initialization
    :type kwargs: dict
    :return: patches - list of rectangle patches for the nodes
    """
    patches = list()
    if color is not None:
        colors = get_color_list(color, len(node_coords))
        for (x, y), col in zip(node_coords, colors):
            patches.append(Rectangle((x - width / 2, y - height / 2), width, height, color=color,
                                     **kwargs))
    else:
        for x, y in node_coords:
            patches.append(Rectangle((x - width / 2, y - height / 2), width, height, **kwargs))
    return patches


def polygon_patches(node_coords, radius, num_edges, color=None, **kwargs):
    """
    Function to create a list of polygon patches from node coordinates. The number of edges for the
    polygon can be defined.

    :param node_coords: coordinates of the nodes to draw
    :type node_coords: iterable
    :param radius: radius for the polygon (from centroid to edges)
    :type radius: float
    :param num_edges: number of edges of the polygon
    :type num_edges: int
    :param color: color or colors of the patches
    :type color: iterable, float
    :param kwargs: additional keyword arguments to pass to the Polygon initialization
    :type kwargs: dict
    :return: patches - list of rectangle patches for the nodes
    """
    patches = list()
    if color is not None:
        colors = get_color_list(color, len(node_coords))
        for (x, y), col in zip(node_coords, colors):
            patches.append(RegularPolygon([x, y], numVertices=num_edges, radius=radius, color=color,
                                          **kwargs))
    else:
        for x, y in node_coords:
            patches.append(RegularPolygon([x, y], numVertices=num_edges, radius=radius, **kwargs))
    return patches


def load_patches(node_coords, size, angles, **kwargs):
    """
    Creation function of patches for loads.

    :param node_coords: coordinates of the nodes that the loads belong to.
    :type node_coords: iterable
    :param size: size of the patch
    :type size: float
    :param angles: angles by which to rotate the patches (in radians)
    :type angles: iterable(float), float
    :param kwargs: additional keyword arguments (might contain parameters "offset",\
        "patch_edgecolor" and "patch_facecolor")
    :type kwargs:
    :return: Return values are: \
        - lines (list) - list of coordinates for lines leading to load patches\
        - polys (list of RegularPolygon) - list containing the load patches\
        - keywords (set) - set of keywords removed from kwargs
    """
    offset = kwargs.get("offset", 1.2 * size)
    all_angles = get_angle_list(angles, len(node_coords))
    edgecolor = kwargs.get("patch_edgecolor", "w")
    facecolor = kwargs.get("patch_facecolor", "w")
    edgecolors = get_color_list(edgecolor, len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))
    polys, lines = list(), list()
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, offset + size]), all_angles[i])
        p3 = node_geo + _rotate_dim2(np.array([0, offset + size / 2]), all_angles[i])
        polys.append(RegularPolygon(p2, numVertices=3, radius=size, orientation=-all_angles[i],
                                    fc=facecolors[i], ec=edgecolors[i]))
        lines.append((node_geo, p3))
    return lines, polys, {"offset", "patch_edgecolor", "patch_facecolor"}


def gen_patches(node_coords, size, angles, **kwargs):
    """
    Creation function of patches for generators.

    :param node_coords: coordinates of the nodes that the generators belong to.
    :type node_coords: iterable
    :param size: size of the patch
    :type size: float
    :param angles: angles by which to rotate the patches (in radians)
    :type angles: iterable(float), float
    :param kwargs: additional keyword arguments (might contain parameters "offset",\
        "patch_edgecolor" and "patch_facecolor")
    :type kwargs:
    :return: Return values are: \
        - lines (list) - list of coordinates for lines leading to generator patches\
        - polys (list of RegularPolygon) - list containing the generator patches\
        - keywords (set) - set of keywords removed from kwargs
    """
    polys, lines = list(), list()
    offset = kwargs.get("offset", 2. * size)
    all_angles = get_angle_list(angles, len(node_coords))
    edgecolor = kwargs.get("patch_edgecolor", "k")
    facecolor = kwargs.get("patch_facecolor", (1, 0, 0, 0))
    edgecolors = get_color_list(edgecolor, len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, size + offset]), all_angles[i])
        polys.append(Circle(p2, size, fc=facecolors[i], ec=edgecolors[i]))
        polys.append(
            Arc(p2 + np.array([-size / 6.2, -size / 2.6]), size / 2, size, theta1=65, theta2=120,
                ec=edgecolors[i]))
        polys.append(
            Arc(p2 + np.array([size / 6.2, size / 2.6]), size / 2, size, theta1=245, theta2=300,
                ec=edgecolors[i]))
        print("Arc:", polys[-1])
        lines.append((node_geo, p2 + np.array([0, size])))
    return lines, polys, {"offset", "patch_edgecolor", "patch_facecolor"}


def sgen_patches(node_coords, size, angles, **kwargs):
    """
    Creation function of patches for static generators.

    :param node_coords: coordinates of the nodes that the static generators belong to.
    :type node_coords: iterable
    :param size: size of the patch
    :type size: float
    :param angles: angles by which to rotate the patches (in radians)
    :type angles: iterable(float), float
    :param kwargs: additional keyword arguments (might contain parameters "offset", "r_triangle",\
        "patch_edgecolor" and "patch_facecolor")
    :type kwargs:
    :return: Return values are: \
        - lines (list) - list of coordinates for lines leading to static generator patches\
        - polys (list of RegularPolygon) - list containing the static generator patches\
        - keywords (set) - set of keywords removed from kwargs
    """
    polys, lines = list(), list()
    offset = kwargs.get("offset", 2 * size)
    r_triangle = kwargs.get("r_triangles", size * 0.4)
    edgecolor = kwargs.get("patch_edgecolor", "w")
    facecolor = kwargs.get("patch_facecolor", "w")
    edgecolors = get_color_list(edgecolor, len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))
    for i, node_geo in enumerate(node_coords):
        mid_circ = node_geo + _rotate_dim2(np.array([0, offset + size]), angles[i])
        circ_edge = node_geo + _rotate_dim2(np.array([0, offset]), angles[i])
        mid_tri1 = mid_circ + _rotate_dim2(np.array([r_triangle, -r_triangle / 4]), angles[i])
        mid_tri2 = mid_circ + _rotate_dim2(np.array([-r_triangle, r_triangle / 4]), angles[i])
        # dropped perpendicular foot of triangle1
        perp_foot1 = mid_tri1 + _rotate_dim2(np.array([0, -r_triangle / 2]), angles[i])
        line_end1 = perp_foot1 + + _rotate_dim2(np.array([-2.5 * r_triangle, 0]), angles[i])
        perp_foot2 = mid_tri2 + _rotate_dim2(np.array([0, r_triangle / 2]), angles[i])
        line_end2 = perp_foot2 + + _rotate_dim2(np.array([2.5 * r_triangle, 0]), angles[i])
        polys.append(Circle(mid_circ, size, fc=facecolors[i], ec=edgecolors[i]))
        polys.append(RegularPolygon(mid_tri1, numVertices=3, radius=r_triangle,
                                    orientation=-angles[i], fc=facecolors[i], ec=edgecolors[i]))
        polys.append(RegularPolygon(mid_tri2, numVertices=3, radius=r_triangle,
                                    orientation=np.pi - angles[i], fc=facecolors[i],
                                    ec=edgecolors[i]))
        lines.append((node_geo, circ_edge))
        lines.append((perp_foot1, line_end1))
        lines.append((perp_foot2, line_end2))
    return lines, polys, {"offset", "r_triangle", "patch_edgecolor", "patch_facecolor"}


def storage_patches(node_coords, size, angles, **kwargs):
    """
    Creation function of patches for storage systems.

    :param node_coords: coordinates of the nodes that the storage system belong to.
    :type node_coords: iterable
    :param size: size of the patch
    :type size: float
    :param angles: angles by which to rotate the patches (in radians)
    :type angles: iterable(float), float
    :param kwargs: additional keyword arguments (might contain parameters "offset", "r_triangle",\
        "patch_edgecolor" and "patch_facecolor")
    :type kwargs:
    :return: Return values are: \
        - lines (list) - list of coordinates for lines leading to storage patches\
        - polys (list of RegularPolygon) - list containing the storage patches\
        - keywords (set) - set of keywords removed from kwargs
    """
    polys, lines = list(), list()
    offset = kwargs.get("offset", 1 * size)
    r_triangle = kwargs.get("r_triangles", size * 0.4)
    for i, node_geo in enumerate(node_coords):
        mid_circ = node_geo + _rotate_dim2(np.array([0, offset + r_triangle * 2.]), angles[i])
        circ_edge = node_geo + _rotate_dim2(np.array([0, offset]), angles[i])
        mid_tri1 = mid_circ + _rotate_dim2(np.array([-r_triangle, -r_triangle]), angles[i])

        # dropped perpendicular foot of triangle1
        perp_foot1 = mid_tri1 + _rotate_dim2(np.array([r_triangle * 0.5, -r_triangle/4]), angles[i])
        line_end1 = perp_foot1 + _rotate_dim2(np.array([1 * r_triangle, 0]), angles[i])

        perp_foot2 = mid_tri1 + _rotate_dim2(np.array([0, -r_triangle]), angles[i])
        line_end2 = perp_foot2 + _rotate_dim2(np.array([2. * r_triangle, 0]), angles[i])

        lines.append((node_geo, circ_edge))
        lines.append((perp_foot1, line_end1))
        lines.append((perp_foot2, line_end2))
    return lines, polys, {"offset", "r_triangle", "patch_edgecolor", "patch_facecolor"}


def ext_grid_patches(node_coords, size, angles, **kwargs):
    """
    Creation function of patches for external grids.

    :param node_coords: coordinates of the nodes that the external grids belong to.
    :type node_coords: iterable
    :param size: size of the patch
    :type size: float
    :param angles: angles by which to rotate the patches (in radians)
    :type angles: iterable(float), float
    :param kwargs: additional keyword arguments (might contain parameters "offset",\
        "patch_edgecolor" and "patch_facecolor")
    :type kwargs:
    :return: Return values are: \
        - lines (list) - list of coordinates for lines leading to external grid patches\
        - polys (list of RegularPolygon) - list containing the external grid patches\
        - keywords (set) - set of keywords removed from kwargs (empty
    """
    offset = kwargs.get("offset", 2 * size)
    all_angles = get_angle_list(angles, len(node_coords))
    edgecolor = kwargs.get("patch_edgecolor", "w")
    facecolor = kwargs.get("patch_facecolor", "w")
    edgecolors = get_color_list(edgecolor, len(node_coords))
    facecolors = get_color_list(facecolor, len(node_coords))
    polys, lines = list(), list()
    for i, node_geo in enumerate(node_coords):
        p2 = node_geo + _rotate_dim2(np.array([0, offset]), all_angles[i])
        p_ll = p2 + _rotate_dim2(np.array([-size, 0]), all_angles[i])
        polys.append(Rectangle(p_ll, 2 * size, 2 * size, angle=(-all_angles[i] / np.pi * 180),
                               fc=facecolors[i], ec=edgecolors[i], hatch="XXX"))
        lines.append((node_geo, p2))
    return lines, polys, {"offset", "patch_edgecolor", "patch_facecolor"}


def trafo_patches(coords, size, **kwargs):
    """
    Creates a list of patches and line coordinates representing transformers each connecting two
    nodes.

    :param coords: list of connecting node coordinates (usually should be \
        `[((x11, y11), (x12, y12)), ((x21, y21), (x22, y22)), ...]`)
    :type coords: (N, (2, 2)) shaped iterable
    :param size: size of the trafo patches
    :type size: float
    :param kwargs: additional keyword arguments (might contain parameters "patch_edgecolor" and\
        "patch_facecolor")
    :type kwargs:
    :return: Return values are: \
        - lines (list) - list of coordinates for lines connecting nodes and transformer patches\
        - circles (list of Circle) - list containing the transformer patches (rings)
    """
    edgecolor = kwargs.get("patch_edgecolor", "w")
    facecolor = kwargs.get("patch_facecolor", (1, 0, 0, 0))
    edgecolors = get_color_list(edgecolor, len(coords))
    facecolors = get_color_list(facecolor, len(coords))
    linewidths = kwargs.get("linewidths", 2.)
    linewidths = get_linewidth_list(linewidths, len(coords), name_entries="trafos")
    circles, lines = list(), list()
    for i, (p1, p2) in enumerate(coords):
        p1 = np.array(p1)
        p2 = np.array(p2)
        if np.all(p1 == p2):
            continue
        d = np.sqrt(np.sum((p1 - p2) ** 2))
        if size is None:
            size_this = np.sqrt(d) / 5
        else:
            size_this = size
        off = size_this * 0.35
        circ1 = (0.5 - off / d) * (p1 - p2) + p2
        circ2 = (0.5 + off / d) * (p1 - p2) + p2
        circles.append(Circle(circ1, size_this, fc=facecolors[i], ec=edgecolors[i],
                              lw=linewidths[i]))
        circles.append(Circle(circ2, size_this, fc=facecolors[i], ec=edgecolors[i],
                              lw=linewidths[i]))

        lp1 = (0.5 - off / d - size_this / d) * (p2 - p1) + p1
        lp2 = (0.5 - off / d - size_this / d) * (p1 - p2) + p2
        lines.append([p1, lp1])
        lines.append([p2, lp2])
    return lines, circles, {"patch_edgecolor", "patch_facecolor"}
