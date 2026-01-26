## Written by Erol Balkovic 8/9/2024
## erol.balkovic@epfl.ch

import numpy as np

from spectre.geometry import get_coord_transform
from spectre.config import run_decorator


@run_decorator
def on_segment(px, py, qx, qy, rx, ry):
    if (qx <= max(px, rx)) and (qx >= min(px, rx)) and (qy <= max(py, ry)) and (qy >= min(py, ry)):
        return True
    return False


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
        # a bit sketchy
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
    if (o4 == 0) and on_segment(p2x, p2y, q1x, q1y, q2x, q2y):
        return True

    return False


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


@run_decorator
def intersect_curve_self(x1, y1):
    for seg1 in range(len(x1) - 1):
        p1x = x1[seg1]
        p1y = y1[seg1]
        q1x = x1[seg1 + 1]
        q1y = y1[seg1 + 1]

        for seg2 in range(seg1 + 2, len(x1) - 1):
            p2x = x1[seg2]
            p2y = y1[seg2]
            q2x = x1[seg2 + 1]
            q2y = y1[seg2 + 1]

            isect = intersect_segment(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y)
            if isect:
                return True
    return False


def check_intersection_wrapper(
    xin, test, nt=None, nz=None, surfaces=None, zetas=None, thetas=None, verbose=False, verbose_plot=False
):
    nvol = test.allglobal_mod.mvol
    if nvol == 1:
        return False

    rank = test.comm.Get_rank()

    geometry = test.input_list_mod.igeometry
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    ngdof = test.allglobal_mod.ngdof
    mn = test.allglobal_mod.mn
    nt = test.allglobal_mod.nt if nt is None else nt
    nz = test.allglobal_mod.nz if nz is None else nz

    if len(xin) - 1 != ngdof:
        raise ValueError(
            f"The size of xin does not match ngdof ({ngdof=} {xin.shape=})",
        )

    if surfaces is None:
        surfaces = np.arange(nvol)
    if zetas is None:
        zetas = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 184)

    cos_theta = np.cos(thetas)
    sin_theta = np.sin(thetas)

    irbc = np.zeros((mn, nvol + 1), dtype=np.float64, order="F")
    irbs = np.zeros((mn, nvol + 1), dtype=np.float64, order="F")
    izbc = np.zeros((mn, nvol + 1), dtype=np.float64, order="F")
    izbs = np.zeros((mn, nvol + 1), dtype=np.float64, order="F")

    test.packing_mod.packxi(ngdof, xin, nvol, mn, nt, nz, irbc, izbs, irbs, izbc, "U", False, False)

    if np.allclose(irbc, np.zeros((mn, nvol + 1))):
        raise ValueError("irbc is all zeros", irbc[:5, :2])

    if irbc.shape != (mn, nvol + 1):
        raise ValueError(f"irbc has the wrong shape, got {irbc.shape}, expected {(mn, nvol + 1)}")

    if izbs.shape != (mn, nvol + 1):
        raise ValueError(f"izbs has the wrong shape, got {izbs.shape}, expected {(mn, nvol + 1)}")

    if im.shape != (mn,):
        raise ValueError(f"im has the wrong shape, got {im.shape}, expected {(mn,)}")

    if in_.shape != (mn,):
        raise ValueError(f"in_ has the wrong shape, got {in_.shape}, expected {(mn,)}")

    shape_tgt = (4, 1, len(thetas), 1)

    for zeta in zetas:
        ind = surfaces[0]

        rarr, zarr = get_coord_transform(
            geometry,
            irbc[:, ind],
            irbc[:, ind + 1],
            izbs[:, ind],
            izbs[:, ind + 1],
            im,
            in_,
            1.0,
            thetas,
            zeta,
            ind,
        )

        if rarr.shape != shape_tgt:
            raise ValueError("rarr has the wrong shape ", rarr.shape)
        if zarr.shape != shape_tgt:
            raise ValueError("zarr has the wrong shape ", zarr.shape)

        if geometry == 3:
            pts_x_current = rarr[0, 0, :, 0]
            pts_y_current = zarr[0, 0, :, 0]
        elif geometry == 2:
            r = rarr[0, 0, :, 0]
            pts_x_current = r * cos_theta
            pts_y_current = r * sin_theta
        elif geometry == 1:
            r = rarr[0, 0, :, 0]
            pts_x_current = thetas
            pts_y_current = r

        isect = intersect_curve_self(pts_x_current[:-1], pts_y_current[:-1])

        if isect:
            if verbose:
                if rank == 0:
                    print(f"Self-Intersection for {surfaces[0]} surface")
            if verbose_plot:
                import matplotlib.pyplot as plt

                plt.figure()
                plt.plot(pts_x_current, pts_y_current, "g.")
                plt.title(f"Self-Intersection for {surfaces[0]} surface")
            return True

        for s in range(len(surfaces) - 1):
            ind = surfaces[s + 1]
            rarr, zarr = get_coord_transform(
                geometry,
                irbc[:, ind],
                irbc[:, ind + 1],
                izbs[:, ind],
                izbs[:, ind + 1],
                im,
                in_,
                1.0,
                thetas,
                zeta,
                ind,
            )

            if rarr.shape != shape_tgt:
                raise ValueError("rarr has the wrong shape ", rarr.shape)
            if zarr.shape != shape_tgt:
                raise ValueError("zarr has the wrong shape ", zarr.shape)

            if geometry == 3:
                pts_x_next = rarr[0, 0, :, 0]
                pts_y_next = zarr[0, 0, :, 0]
            elif geometry == 2:
                r = rarr[0, 0, :, 0]
                pts_x_next = r * cos_theta
                pts_y_next = r * sin_theta
            elif geometry == 1:
                r = rarr[0, 0, :, 0]
                pts_x_next = thetas
                pts_y_next = r

            isect = intersect_curve(pts_x_current, pts_y_current, pts_x_next, pts_y_next)

            if isect:
                if verbose:
                    if rank == 0:
                        print(f"Intersection for {s} and {s + 1} surfaces")
                if verbose_plot:
                    import matplotlib.pyplot as plt

                    plt.figure()
                    plt.plot(pts_x_current, pts_y_current, "b.")
                    plt.plot(pts_x_next, pts_y_next, "r.")
                    plt.title(f"Intersection for {s} and {s + 1} surfaces")
                return True

            isect = intersect_curve_self(pts_x_current[:-1], pts_y_current[:-1])
            if isect:
                if verbose:
                    if rank == 0:
                        print(f"Self-Intersection for {s} surface")
                if verbose_plot:
                    import matplotlib.pyplot as plt

                    plt.figure()
                    plt.plot(pts_x_current, pts_y_current, "g.")
                    plt.title(f"Self-Intersection for {s} surface")

                return True

            pts_x_current = pts_x_next
            pts_y_current = pts_y_next

    return False
