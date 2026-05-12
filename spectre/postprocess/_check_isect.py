# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np

from spectre.postprocess.config import run_decorator


@run_decorator
def on_segment(px, py, qx, qy, rx, ry):
    return bool(qx <= max(px, rx) and qx >= min(px, rx) and qy <= max(py, ry) and qy >= min(py, ry))


@run_decorator
def triplet_orientation(px, py, qx, qy, rx, ry):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise

    val = (float(qy - py) * (rx - qx)) - (float(qx - px) * (ry - qy))
    if val > 0:
        return 1
    elif val < 0:
        return 2
    else:
        return 0


@run_decorator
def intersect_segment(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y):
    # Find the 4 orientations required for the general and special cases
    o1 = triplet_orientation(p1x, p1y, q1x, q1y, p2x, p2y)
    o2 = triplet_orientation(p1x, p1y, q1x, q1y, q2x, q2y)
    o3 = triplet_orientation(p2x, p2y, q2x, q2y, p1x, p1y)
    o4 = triplet_orientation(p2x, p2y, q2x, q2y, q1x, q1y)

    # General case
    if (o1 != o2) and (o3 != o4):
        return True

    # Special Cases

    # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
    if (o1 == 0) and on_segment(p1x, p1y, p2x, p2y, q1x, q1y):
        return True

    # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
    if (o2 == 0) and on_segment(p1x, p1y, q2x, q2y, q1x, q1y):
        return True

    # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
    if (o3 == 0) and on_segment(p2x, p2y, p1x, p1y, q2x, q2y):
        return True

    # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
    return bool(o4 == 0 and on_segment(p2x, p2y, q1x, q1y, q2x, q2y))


@run_decorator
def intersect_curve(x1, y1, x2, y2):
    for seg1 in range(len(x1) - 1):
        p1x = x1[seg1]
        p1y = y1[seg1]
        q1x = x1[seg1 + 1]
        q1y = y1[seg1 + 1]

        for seg2 in range(len(x2) - 1):
            p2x = x2[seg2]
            p2y = y2[seg2]
            q2x = x2[seg2 + 1]
            q2y = y2[seg2 + 1]

            isect = intersect_segment(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y)
            if isect:
                return True
    return False


def check_intersection_interfaces(
    self, surfaces=None, zetas=None, thetas=None, verbose=False, verbose_plot=False
):
    """Checks whether any pair of adjacent interface surfaces intersect each other.

    Args:
        surfaces (array-like, optional): Indices of surfaces to check. Defaults to all volumes.
        zetas (array-like, optional): Toroidal cross-sections to check.
            Defaults to 8 evenly spaced values in [0, 2π).
        thetas (array-like, optional): Poloidal grid used to trace each surface.
            Defaults to 92 points in [0, 2π].
        verbose (bool, optional): If ``True``, prints a message when an intersection is found.
            Defaults to ``False``.
        verbose_plot (bool, optional): If ``True``, plots the intersecting two surfaces when found.
            Defaults to ``False``.

    Returns:
        bool: True if any intersection is found, False otherwise.
    """

    nvol = self.input.physics.Nvol
    geometry = self.input.physics.Igeometry

    if surfaces is None:
        surfaces = np.arange(nvol)
    if zetas is None:
        zetas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 92)

    for zeta in zetas:
        rarr, zarr = self.get_coord_transform(surfaces[0], 1.0, thetas, zeta)
        if geometry == 3:
            pts_x_current = rarr[0, 0, :, 0]
            pts_y_current = zarr[0, 0, :, 0]
        elif geometry == 2:
            r = rarr[0, 0, :, 0]
            pts_x_current = r * np.cos(thetas)
            pts_y_current = r * np.sin(thetas)
        elif geometry == 1:
            r = rarr[0, 0, :, 0]
            pts_x_current = thetas
            pts_y_current = r

        for s in range(len(surfaces) - 1):
            rarr, zarr = self.get_coord_transform(surfaces[s + 1], 1.0, thetas, zeta)
            if geometry == 3:
                pts_x_next = rarr[0, 0, :, 0]
                pts_y_next = zarr[0, 0, :, 0]
            elif geometry == 2:
                r = rarr[0, 0, :, 0]
                pts_x_next = r * np.cos(thetas)
                pts_y_next = r * np.sin(thetas)
            elif geometry == 1:
                r = rarr[0, 0, :, 0]
                pts_x_next = thetas
                pts_y_next = r

            isect = intersect_curve(pts_x_current, pts_y_current, pts_x_next, pts_y_next)
            if isect:
                if verbose:
                    print(f"Intersection for {s} and {s + 1} surfaces")
                if verbose_plot:
                    import matplotlib.pyplot as plt

                    plt.figure()
                    plt.plot(pts_x_current, pts_y_current, "b.")
                    plt.plot(pts_x_next, pts_y_next, "r.")
                    plt.title(f"Intersection for {s} and {s + 1} surfaces")
                return True

            pts_x_current = pts_x_next
            pts_y_current = pts_y_next

    return False
