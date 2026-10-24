"""Obstacles: everything inside the room that is not floor, ceiling or wall, as axis-aligned boxes.

Leftover points are rasterised top-down at 2 cm, opened with a small kernel and split with
cv2.connectedComponentsWithStats; each component becomes a robust (percentile) axis-aligned box in
the room frame plus an oriented footprint from cv2.minAreaRect.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import points_in_polygon
from .openings import wall_coords, wall_frame


def structure_mask(P, walls, H, band, floor_tol=0.03, ceil_tol=0.06):
    """Points explained by the floor, the ceiling or a wall band. The ceiling band is wider: real
    ceilings are not perfectly flat or level (3 cm over a 5 m room in the iPhone test scan) and
    furniture does not come within 6 cm of them."""
    m = (P[:, 1] < floor_tol) | (P[:, 1] > H - ceil_tol)
    for w in walls:
        a, u, n, L = wall_frame(w["a"], w["b"])
        t, _, d = wall_coords(P, a, u, n)
        m |= (np.abs(d) < band[w["id"]]) & (t > -0.05) & (t < L + 0.05)
    return m


def find_obstacles(P, walls, polygon, H, band, cell=0.02, min_area=0.02, min_points=40):
    struct = structure_mask(P, walls, H, band)
    inside = points_in_polygon(P[:, [0, 2]], polygon)
    # ignore the floor-noise tail (< 5 cm): stray floor points would bridge separate furniture
    cand = np.flatnonzero(~struct & inside & (P[:, 1] > 0.05) & (P[:, 1] < H))
    if len(cand) == 0:
        return [], struct
    q = P[cand][:, [0, 2]]
    lo = q.min(0) - cell
    ij = ((q - lo) / cell).astype(int)
    img = np.zeros((ij[:, 1].max() + 2, ij[:, 0].max() + 2), np.int32)
    np.add.at(img, (ij[:, 1], ij[:, 0]), 1)
    binimg = (img >= 1).astype(np.uint8)
    binimg = cv2.morphologyEx(binimg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # drop isolated noise cells
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binimg, connectivity=8)
    plab = lab[ij[:, 1], ij[:, 0]]
    out = []
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] * cell * cell < min_area:
            continue
        sel = cand[plab == c]
        if len(sel) < min_points:
            continue
        pts = P[sel]
        lo3 = np.percentile(pts, 0.5, axis=0)
        hi3 = np.percentile(pts, 99.5, axis=0)
        if hi3[1] - lo3[1] < 0.05:
            continue
        if lo3[1] < 0.1:
            lo3[1] = 0.0                       # resting on the floor
        (cx, cz), (w, d), ang = cv2.minAreaRect(pts[:, [0, 2]].astype(np.float32))
        out.append(dict(min=lo3, max=hi3, points=pts, n=len(sel),
                        footprint=dict(center=[float(cx), float(cz)], size=[float(w), float(d)], angle_deg=float(ang))))
    out.sort(key=lambda o: -o["n"])
    return out, struct
