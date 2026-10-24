"""Doors, windows and coverage gaps from each wall's occupancy image.

Every wall gets a 2 cm occupancy grid over its own plane coordinates (t along the wall, y up),
built from the points within the wall's inlier band. Empty regions are found with OpenCV image ops
(morphological close/open, connectedComponentsWithStats); rectangular empty regions are openings:

  door    : touches the floor, top >= 1.9 m, has a lintel below the ceiling, 0.55-1.8 m wide
  window  : sill >= 0.3 m, top below the ceiling, at least 0.25 m x 0.25 m
  opening : touches the floor and rectangular, but not door-shaped (pass-through, full-height gap)

Everything else that is empty is a coverage gap ("no points", or "occluded by oK" when an obstacle
stands between the wall and the room). Opening edges are refined below the grid resolution with a
counting estimator: if wall points have a locally uniform density rho per metre along t, the edge
sits at  t_start + count(points in [t_start, t_mid]) / rho, which is unbiased under symmetric noise.
"""
from __future__ import annotations

import cv2
import numpy as np

CELL = 0.02
EDGE_BAND = 0.04       # rows within 4 cm of floor / ceiling are junction rows and are ignored


def wall_frame(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    u = (b - a) / L
    n = np.array([-u[1], u[0]])            # interior normal for a counter-clockwise polygon
    return a, u, n, L


def wall_coords(P, a, u, n):
    q = P[:, [0, 2]] - a
    return q @ u, P[:, 1], q @ n           # t, y, signed distance (+ inside)


def occupancy_image(t, y, L, H, cell=CELL):
    cols, rows = int(np.ceil(L / cell)), int(np.ceil(H / cell))
    ok = (t >= 0) & (t < L) & (y >= 0) & (y < H)
    img = np.zeros((rows, cols), np.float32)
    np.add.at(img, ((y[ok] / cell).astype(int), (t[ok] / cell).astype(int)), 1)
    return img                              # row 0 = floor level


def _edge(coord, sel, coarse, side, lo, hi, mid, rho_expected, binw=0.02):
    """Sub-cell edge position by counting.

    side = -1: wall material lies below the coarse edge (left jamb, sill); +1: above (right jamb, lintel).
    rho (points per metre of the 1D profile) is the mean of the well-filled 2 cm bins in the 15 cm
    next to the edge, so a partly occluded reference strip does not bias it; if that strip is
    mostly empty (e.g. the jamb is hidden behind an open door leaf) the coarse edge is kept."""
    if side < 0:
        r0, r1 = max(coarse - 0.15, lo), coarse - 0.02
    else:
        r0, r1 = coarse + 0.02, min(coarse + 0.15, hi)
    if r1 - r0 < 0.04:
        return coarse, False
    nb = int((r1 - r0) / binw)
    c = coord[sel]
    hist, _ = np.histogram(c, bins=nb, range=(r0, r0 + nb * binw))
    full = hist[hist >= 0.5 * hist.max()] if hist.max() > 0 else hist
    rho = float(np.mean(full)) / binw            # density of the bins that are actually wall
    if rho < 0.3 * rho_expected or rho <= 0:
        return coarse, False
    if side < 0:
        start = max(coarse - 0.04, lo)
        cnt = np.count_nonzero((c >= start) & (c < mid))
        return start + cnt / rho, True
    stop = min(coarse + 0.04, hi)
    cnt = np.count_nonzero((c > mid) & (c <= stop))
    return stop - cnt / rho, True


def _split_components(empty, near_shadow):
    """Connected empty regions, each split into column runs that are / are not mostly covered by
    an obstacle standing within 35 cm of the wall. An open door and the wall hidden behind its leaf
    form one empty blob; the leaf (an obstacle hugging the wall) separates them at the hinge."""
    ncomp, lab, stats, _ = cv2.connectedComponentsWithStats(empty, connectivity=8)
    parts = []
    for c in range(1, ncomp):
        x, y, w, h, _ = stats[c]
        comp = lab[y:y + h, x:x + w] == c
        csum = np.maximum(comp.sum(0), 1)
        cls = (near_shadow[y:y + h, x:x + w] & comp).sum(0) / csum > 0.5
        if w >= 5:                                   # majority filter: no slivers narrower than 6 cm
            cls = np.convolve(cls.astype(float), np.ones(5) / 5, "same") > 0.5
        cuts = np.flatnonzero(np.diff(cls.astype(int))) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, w]):
            sub = np.zeros_like(empty, dtype=bool)
            sub[y:y + h, x + a:x + b] = comp[:, a:b]
            rows = np.flatnonzero(sub.any(1))
            if len(rows) == 0:
                continue
            parts.append((x + a, rows[0], b - a, rows[-1] - rows[0] + 1, int(sub.sum()), sub))
    return parts


def _merge_gaps(gaps, touch=0.05, sliver=0.1):
    """Merge side-by-side gaps of one wall; slivers (< 10 cm) are absorbed by their neighbour."""
    gaps = sorted(gaps, key=lambda g: g["from"])
    out = []
    for g in gaps:
        if out:
            p = out[-1]
            adjacent = g["from"] - p["to"] <= touch and min(g["y_to"], p["y_to"]) > max(g["y_from"], p["y_from"])
            narrow = (g["to"] - g["from"] < sliver) or (p["to"] - p["from"] < sliver)
            if adjacent and (g["reason"] == p["reason"] or narrow):
                if g["reason"] == p["reason"]:
                    out[-1] = dict(p, to=max(p["to"], g["to"]), y_from=min(p["y_from"], g["y_from"]),
                                   y_to=max(p["y_to"], g["y_to"]), area_m2=round(p["area_m2"] + g["area_m2"], 3))
                else:                                # absorb the sliver into the bigger gap
                    big = p if (p["to"] - p["from"]) >= (g["to"] - g["from"]) else g
                    out[-1] = dict(big, **{"from": p["from"], "to": max(p["to"], g["to"])})
                continue
        out.append(g)
    return out


def analyze_wall(wid, a, b, H, P, band, obstacles, wall_rms):
    """Occupancy, openings and gaps for one wall. P: measurement cloud in the room frame."""
    a, u, n, L = wall_frame(a, b)
    t, y, d = wall_coords(P, a, u, n)
    on = (np.abs(d) < band) & (t > -0.05) & (t < L + 0.05) & (y > EDGE_BAND) & (y < H - EDGE_BAND)
    tw, yw = t[on], y[on]
    img = occupancy_image(tw, yw, L, H)
    rows, cols = img.shape
    r_lo, r_hi = int(np.ceil(EDGE_BAND / CELL)), rows - int(np.ceil(EDGE_BAND / CELL))
    occ = (img > 0).astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    valid = np.zeros_like(occ)
    valid[r_lo:r_hi, :] = 1
    empty = ((occ == 0) & (valid == 1)).astype(np.uint8)
    empty = cv2.morphologyEx(empty, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # see-through evidence: points behind the wall plane (hallway through a door, outside a window)
    beyond = (d < -0.08) & (t > 0) & (t < L) & (y > 0) & (y < H)
    beyond_img = occupancy_image(t[beyond], y[beyond], L, H) > 0
    recess = (d < -band) & (d >= -0.08) & (t > 0) & (t < L) & (y > 0) & (y < H)
    recess_img = occupancy_image(t[recess], y[recess], L, H) > 0   # a closed door sits 2-6 cm back
    # obstacle shadow: obstacle points within 1 m in front of the wall, projected onto it
    shadow = np.full(occ.shape, -1, np.int32)
    near_shadow = np.zeros(occ.shape, bool)
    hug_t, hug_y = [], []                   # obstacle points hugging the wall (door leaf near its hinge)
    for k, ob in enumerate(obstacles):
        tt, yy, dd = wall_coords(ob["points"], a, u, n)
        s = (dd > 0.02) & (dd < 1.0) & (tt > 0) & (tt < L)
        if s.sum() < 20:
            continue
        sh = (occupancy_image(tt[s], yy[s], L, H) > 0).astype(np.uint8)
        sh = cv2.dilate(sh, np.ones((5, 5), np.uint8))
        shadow[(sh > 0) & (shadow < 0)] = k
        s2 = s & (dd < 0.35)
        if s2.sum() >= 20:
            near_shadow |= occupancy_image(tt[s2], yy[s2], L, H) > 0
            s3 = s & (dd < 0.12)
            hug_t.append(tt[s3])
            hug_y.append(yy[s3])
    hug_t = np.concatenate(hug_t) if hug_t else np.zeros(0)
    hug_y = np.concatenate(hug_y) if hug_y else np.zeros(0)
    near_shadow = cv2.dilate(near_shadow.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    parts = _split_components(empty, near_shadow)
    occupied = img > 0
    med_count = float(np.median(img[occupied])) if occupied.any() else 0.0
    dens_area = len(tw) / max(occupied.sum() * CELL * CELL, 1e-6)   # wall points per m^2
    openings, gaps = [], []
    spacing = 1.0 / np.sqrt(max(len(tw) / max(L * (H - 2 * EDGE_BAND), 1e-6), 1.0))
    for x, yr, w, h, area, comp in parts:
        if area * CELL * CELL < 0.02:
            continue
        t0, t1 = x * CELL, (x + w) * CELL
        y0, y1 = yr * CELL, (yr + h) * CELL
        fill = area / float(w * h)
        sh_ids = shadow[comp]
        sh_frac = float((sh_ids >= 0).mean())
        see = float(beyond_img[comp].mean())
        rec = float(recess_img[comp].mean())
        touches_floor = yr <= r_lo
        touches_ceil = yr + h >= r_hi
        width, height = t1 - t0, y1 - y0
        kind = None
        evidence = see + rec                      # something seen through or set back in the gap
        if fill >= 0.8 and (sh_frac < 0.6 or evidence > 0.5) and x > 0 and x + w < cols:
            if touches_floor and not touches_ceil and y1 >= 1.9 and 0.55 <= width <= 1.8:
                kind = "door"
            elif not touches_floor and not touches_ceil and y0 >= 0.3 and width >= 0.25 and height >= 0.25:
                kind = "window"
            elif touches_floor and width >= 0.5 and height >= 1.0:
                kind = "opening"
        if kind is None:
            reason = "no points"
            if fill >= 0.5 and (img[yr:yr + h, x:x + w] > 0).mean() >= 0.15:   # sparse, not empty
                reason = "low density"
            elif sh_frac >= 0.4:
                vals, cnts = np.unique(sh_ids[sh_ids >= 0], return_counts=True)
                reason = f"occluded by o{int(vals[np.argmax(cnts)])}"
            gaps.append(dict(wall=wid, **{"from": round(t0, 3), "to": round(t1, 3)},
                             y_from=round(y0, 3), y_to=round(y1, 3), reason=reason,
                             area_m2=round(area * CELL * CELL, 3)))
            continue
        state = "closed" if rec > 0.5 and see < 0.2 else ("open" if see > 0.2 else "unknown")
        # sub-cell refinement of the four edges on the raw points of this wall
        mid_y = (yw > y0 + 0.2 * height) & (yw < y1 - 0.2 * height)
        mid_t = (tw > t0 + 0.15 * width) & (tw < t1 - 0.15 * width)
        tc, yc = 0.5 * (t0 + t1), 0.5 * (y0 + y1)
        h_sel, w_sel = 0.6 * height, 0.7 * width          # extent of the selection bands
        lt, okl = _edge(tw, mid_y, t0, -1, 0.0, L, tc, dens_area * h_sel)
        rt, okr = _edge(tw, mid_y, t1, +1, 0.0, L, tc, dens_area * h_sel)
        # a jamb hidden by an open leaf: the leaf's hinge end (closest to the wall) marks the edge
        hy = (hug_y > y0 + 0.2 * height) & (hug_y < y1 - 0.2 * height)
        hinge = None
        if not okl and x > 0 and near_shadow[yr:yr + h, x - 1].mean() > 0.5:
            cand = hug_t[hy & (hug_t < t0 + 0.06) & (hug_t > t0 - 0.3)]
            if len(cand) >= 10:
                lt, okl, hinge = float(np.percentile(cand, 98)), True, "a"
        if not okr and x + w < cols and near_shadow[yr:yr + h, x + w].mean() > 0.5:
            cand = hug_t[hy & (hug_t > t1 - 0.06) & (hug_t < t1 + 0.3)]
            if len(cand) >= 10:
                rt, okr, hinge = float(np.percentile(cand, 2)), True, "b"
        top, okt = _edge(yw, mid_t, y1, +1, 0.0, H - EDGE_BAND, yc, dens_area * w_sel) if not touches_ceil else (H, False)
        sill, oks = 0.0, True
        if kind == "window":
            sill, oks = _edge(yw, mid_t, y0, -1, EDGE_BAND, H, yc, dens_area * w_sel)
        # edge sigma: lattice quantisation of the sampling plus along-surface noise (~ wall RMS)
        s_edge = float(np.sqrt(spacing ** 2 / 12 + wall_rms ** 2))
        n_unref = sum(not f for f in (okl, okr, okt, oks))
        s_open = float(np.sqrt(2 * s_edge ** 2 + n_unref * (CELL ** 2 / 3)))   # unrefined edge: +/- one cell
        if hinge:                    # jamb hidden behind the open leaf: edge from the leaf's hinge end
            s_open = float(np.sqrt(s_open ** 2 + 0.02 ** 2))
        openings.append(dict(type=kind, wall=wid, offset=float(lt), width=float(rt - lt),
                             sill=float(sill), height=float(top - sill), sigma=round(s_open, 4),
                             see_through=round(see, 3), refined=bool(okl and okr and okt and oks), state=state, hinge_side=hinge))
    # coverage + low-density regions (exclude openings)
    open_mask = np.zeros_like(occ)
    for o in openings:
        c0, c1 = int(o["offset"] / CELL), int(np.ceil((o["offset"] + o["width"]) / CELL))
        r0, r1 = int(o["sill"] / CELL), int(np.ceil((o["sill"] + o["height"]) / CELL))
        open_mask[max(r0, 0):r1, max(c0, 0):c1] = 1
    area_mask = (valid == 1) & (open_mask == 0)
    coverage = float(occ[area_mask].mean()) if area_mask.any() else 0.0
    dens = cv2.boxFilter(img, -1, (5, 5))
    med = float(np.median(img[img > 0])) if (img > 0).any() else 0.0
    low = ((dens < 0.25 * med) & (occ > 0) & area_mask).astype(np.uint8)
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nl, llab, lstats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    for c in range(1, nl):
        x, yr, w, h, area = lstats[c]
        if area * CELL * CELL >= 0.08:
            gaps.append(dict(wall=wid, **{"from": round(x * CELL, 3), "to": round((x + w) * CELL, 3)},
                             y_from=round(yr * CELL, 3), y_to=round((yr + h) * CELL, 3),
                             reason="low density", area_m2=round(area * CELL * CELL, 3)))
    gaps = _merge_gaps(gaps)
    return dict(openings=openings, gaps=gaps, coverage=coverage, image=img, points=int(on.sum()),
                spacing=spacing)
