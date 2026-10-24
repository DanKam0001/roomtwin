"""Walls: iterative vertical-plane segmentation, merging, footprint polygon, per-wall uncertainty.

Because gravity is known, a vertical plane is a 2D line in (x, z). Lines are found with one-point
MSAC on oriented points (position + horizontal normal), polished with cv2.fitLine (Huber), merged
when near-coplanar, filtered to those that lie on the room boundary (furniture fronts do not), and
ordered along the room outline from cv2.findContours. Adjacent lines are intersected to give the
corners, so L-shapes, angled walls and alcoves come out as the true polygon.

Uncertainty (documented in pipeline/README.md):
  A line fitted to N_eff independent points with perpendicular residual RMS s and along-wall spread
  var_t has a perpendicular position error at along-wall coordinate t of
        sigma_perp(t)^2 = s^2 / N_eff * (1 + (t - t_mean)^2 / var_t)
  (the standard prediction variance of a least-squares line). N_eff is the number of occupied
  2 cm x 2 cm cells on the wall (points closer than that are not independent samples).
  A corner moves along wall i by sigma_perp_j / sin(phi) when the neighbour j moves, so
        sigma_L^2 = (sigma_perp_prev(a) / sin phi_a)^2 + (sigma_perp_next(b) / sin phi_b)^2
                    + (sigma_perp_i * (|cot phi_a| + |cot phi_b|))^2 + (k_scale * L)^2
  where k_scale is the device scale uncertainty (default 0.3 %, an assumption that the ArUco check
  can measure).
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import fit_line_2d, msac_lines_oriented, polygon_area


class Line:
    """Vertical plane as a 2D line: point p, unit direction d, unit normal n = (-d_z, d_x)."""

    def __init__(self, p, d):
        self.p = np.asarray(p, float)
        self.d = np.asarray(d, float) / np.linalg.norm(d)

    @property
    def n(self):
        return np.array([-self.d[1], self.d[0]])

    def dist(self, q):
        return (q - self.p) @ self.n

    def t(self, q):
        return (q - self.p) @ self.d

    def at(self, t):
        return self.p + np.outer(np.atleast_1d(t), self.d).squeeze()


def cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _intersect(l1: Line, l2: Line):
    A = np.array([l1.d, -l2.d]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    s = np.linalg.solve(A, l2.p - l1.p)
    return l1.p + s[0] * l1.d


def _vertical_coverage(y, H, bin_size=0.1):
    if len(y) == 0:
        return 0.0
    nb = max(int(H / bin_size), 1)
    h, _ = np.histogram(y, bins=nb, range=(0, H))
    return float((h > 0).mean())


def _runs(t, max_gap):
    """Sorted along-wall coordinates -> list of (start, end) runs split at gaps > max_gap."""
    t = np.sort(t)
    if len(t) == 0:
        return []
    cut = np.flatnonzero(np.diff(t) > max_gap)
    starts = np.r_[t[0], t[cut + 1]]
    ends = np.r_[t[cut], t[-1]]
    return list(zip(starts, ends))


def extract_wall_lines(P, N, H, rng, thresh=0.02, min_len=0.25, max_walls=24):
    """Iterative MSAC on horizontal-normal points. Returns list of dict(line, idx) (idx into P)."""
    band = (P[:, 1] > 0.08) & (P[:, 1] < H - 0.08)
    vert = ~np.isnan(N[:, 0]) & (np.abs(np.nan_to_num(N[:, 1])) < np.sin(np.radians(15)))
    cand = np.flatnonzero(band & vert)
    p2 = P[cand][:, [0, 2]]
    n2 = N[cand][:, [0, 2]]
    n2 /= np.linalg.norm(n2, axis=1, keepdims=True)
    remaining = np.ones(len(cand), bool)
    walls = []
    min_support = max(60, int(0.004 * len(cand)))
    fails = 0
    while len(walls) < max_walls and remaining.sum() > min_support and fails < 4:
        ridx = np.flatnonzero(remaining)
        hyp = msac_lines_oriented(p2[ridx], n2[ridx], thresh, 500, rng)
        if hyp is None:
            break
        line = Line(*hyp)
        for _ in range(3):                                  # local optimisation with cv2.fitLine
            r = np.abs(line.dist(p2[ridx]))
            agree = np.abs(n2[ridx] @ line.n) > np.cos(np.radians(25))
            inl = ridx[(r < thresh) & agree]
            if len(inl) < min_support:
                break
            line = Line(*fit_line_2d(p2[inl]))
        r = np.abs(line.dist(p2[ridx]))
        agree = np.abs(n2[ridx] @ line.n) > np.cos(np.radians(25))
        inl = ridx[(r < thresh) & agree]
        if len(inl) < min_support:
            fails += 1
            remaining[ridx[(r < thresh) & agree]] = False
            continue
        remaining[inl] = False
        # keep the contiguous runs that look like wall: long enough and tall enough
        t = line.t(p2[inl])
        y = P[cand[inl], 1]
        runs = []
        for t0, t1 in _runs(t, 0.3):        # full-height gaps (alcove mouths) split the line
            sel = (t >= t0) & (t <= t1)
            if t1 - t0 < min_len:
                continue
            vcov = _vertical_coverage(y[sel], H)
            # walls reach the ceiling: some of the run must have points in the top band
            nb = max(int((t1 - t0) / 0.1), 1)
            top = sel & (y > H - 0.5)
            hist, _ = np.histogram(t[top], bins=nb, range=(t0, t1))
            top_cov = float((hist > 0).mean())
            if ((top_cov >= 0.25 and vcov >= 0.15) or (vcov >= 0.6 and t1 - t0 >= 1.2)) and sel.sum() >= 30:
                runs.append(sel)
        if not runs or sum(r.sum() for r in runs) < min_support:
            fails += 1
            continue
        fails = 0
        for sel in runs:
            walls.append(dict(line=line, idx=cand[inl[sel]]))
    return walls


def _interior_side(line, lo, hi, F):
    tf, df = line.t(F), line.dist(F)
    near = (tf > lo) & (tf < hi) & (np.abs(df) < 4.0)
    return 1.0 if (df[near] > 0).sum() >= (df[near] < 0).sum() else -1.0


def merge_coplanar(walls, P, floor_mask, angle_deg=4.0, offset=0.05, small_gap=0.3, max_gap=2.0):
    """Union near-coplanar candidates and refit with cv2.fitLine.

    Pieces are merged when their extents overlap or the gap is < 30 cm, or when the gap is < 2 m and
    nothing is behind it (a tall cupboard hiding the wall). A gap with structure behind it is an
    alcove or passage mouth and stays open."""
    F = P[floor_mask][:, [0, 2]]
    struct = P[(P[:, 1] > 0.1)][:, [0, 2]]
    changed = True
    while changed:
        changed = False
        for i in range(len(walls)):
            for j in range(i + 1, len(walls)):
                li, lj = walls[i]["line"], walls[j]["line"]
                if abs(li.d @ lj.d) < np.cos(np.radians(angle_deg)):
                    continue
                qj = P[walls[j]["idx"]][:, [0, 2]]
                if abs(np.median(li.dist(qj))) > offset:
                    continue
                ti = li.t(P[walls[i]["idx"]][:, [0, 2]])
                tj = li.t(qj)
                (a0, a1), (b0, b1) = np.percentile(ti, [1, 99]), np.percentile(tj, [1, 99])
                gap = max(b0 - a1, a0 - b1)
                if gap > small_gap:
                    if gap > max_gap:
                        continue
                    g0, g1 = (a1, b0) if b0 > a1 else (b1, a0)
                    side = _interior_side(li, min(a0, b0), max(a1, b1), F)
                    ts, ds = li.t(struct), li.dist(struct) * side
                    behind = (ts > g0 + 0.05) & (ts < g1 - 0.05) & (ds < -0.1) & (ds > -3.0)
                    if behind.sum() > 100:
                        continue
                idx = np.union1d(walls[i]["idx"], walls[j]["idx"])
                walls[i] = dict(line=Line(*fit_line_2d(P[idx][:, [0, 2]])), idx=idx)
                walls.pop(j)
                changed = True
                break
            if changed:
                break
    return walls


def _extent(w, P):
    t = w["line"].t(P[w["idx"]][:, [0, 2]])
    return np.percentile(t, 1), np.percentile(t, 99)


def prune_outside(walls, P, floor_mask):
    """Remove candidates that are not room walls.

    For each candidate, the interior side is the side with more floor points in front of it.
      * outside surface: a candidate lying behind another wall (> 10 cm beyond its plane) and within
        that wall's extent, e.g. the hallway wall seen through an open door;
      * clutter: a parallel candidate standing in front of a longer wall (< 1.2 m) and within its
        extent, e.g. a wardrobe front.
    Returns (kept, rejected)."""
    F = P[floor_mask][:, [0, 2]]
    info = []
    for w in walls:
        lo, hi = _extent(w, P)
        tf, df = w["line"].t(F), w["line"].dist(F)
        near = (tf > lo) & (tf < hi) & (np.abs(df) < 4.0)
        side = 1.0 if (df[near] > 0).sum() >= (df[near] < 0).sum() else -1.0
        info.append((lo, hi, side))
    bad = set()
    for i, wi in enumerate(walls):
        lo, hi, side = info[i]
        for j, wj in enumerate(walls):
            if i == j or j in bad:
                continue
            q = P[wj["idx"]][:, [0, 2]]
            dist = wi["line"].dist(q) * side
            t = wi["line"].t(q)
            within = np.mean((t > lo - 0.1) & (t < hi + 0.1))
            if within < 0.8:
                continue
            # strictly inside the extent (not touching an end): a reflex corner's neighbour is also
            # behind the plane but starts at the wall's end
            inside = np.mean((t > lo + 0.15) & (t < hi - 0.15))
            if inside > 0.8 and np.mean(dist < -0.10) > 0.8:
                bad.add(j)                                  # behind wall i: outside the room
            elif (np.mean((dist > 0.05) & (dist < 1.2)) > 0.8 and abs(wi["line"].d @ wj["line"].d) > np.cos(np.radians(10))
                  and (info[j][1] - info[j][0]) < 0.9 * (hi - lo)
                  and (wi["line"].n * side) @ (wj["line"].n * info[j][2]) > 0):
                # in front of a longer wall and facing the same way: a furniture front. A surface facing
                # the other way (towards wall i) is the back of a step in the outline, i.e. a real wall.
                bad.add(j)
    kept = [w for k, w in enumerate(walls) if k not in bad]
    return kept, [w for k, w in enumerate(walls) if k in bad], info


def behind_walls_mask(walls, P, info, margin=0.10):
    """Points behind any kept wall (within its extent): seen through doors/windows, not room."""
    q = P[:, [0, 2]]
    m = np.zeros(len(P), bool)
    for w, (lo, hi, side) in zip(walls, info):
        t = w["line"].t(q)
        m |= (t > lo) & (t < hi) & (w["line"].dist(q) * side < -margin)
    return m


def room_outline(P, H, floor_mask, cell=0.02, segments=()):
    """Outer boundary of the scanned room: rasterise structure points in (x, z), close, fill.

    Returns (contour Kx2 metres in walk order, distance-to-outline image, grid origin, cell)."""
    sel = floor_mask | ((P[:, 1] > 0.05) & (P[:, 1] < H - 0.05))
    q = P[sel][:, [0, 2]]
    lo = q.min(0) - 0.3
    size = np.ceil((q.max(0) + 0.3 - lo) / cell).astype(int)
    img = np.zeros((size[1], size[0]), np.uint8)
    ij = ((q - lo) / cell).astype(int)
    img[ij[:, 1], ij[:, 0]] = 255
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))       # drop speckle
    # draw the wall candidates across their full extent: door/window gaps are bridged, so an
    # unscanned patch of floor in the middle stays an interior hole instead of leaking outside
    for a, b in segments:
        pa = tuple(int(v) for v in np.round((a - lo) / cell))
        pb = tuple(int(v) for v in np.round((b - lo) / cell))
        cv2.line(img, pa, pb, 255, 3)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    big = max(cnts, key=cv2.contourArea)
    outline_img = np.full_like(img, 255)
    cv2.drawContours(outline_img, [big], -1, 0, 1)
    dist = cv2.distanceTransform(outline_img, cv2.DIST_L2, 5) * cell                # metres to outline
    contour = big[:, 0, :].astype(float) * cell + lo + cell / 2
    return contour, dist, lo, cell


def boundary_fraction(seg_a, seg_b, dist, lo, cell, tol=0.15):
    """Fraction of a wall segment that lies within tol of the room outline (distanceTransform)."""
    ts = np.linspace(0, 1, 40)[:, None]
    pts = seg_a + ts * (seg_b - seg_a)
    ij = np.clip(((pts - lo) / cell).astype(int), 0, [dist.shape[1] - 1, dist.shape[0] - 1])
    return float((dist[ij[:, 1], ij[:, 0]] < tol).mean())


def line_stats(line: Line, P2, Y, cell=0.02):
    """Residual RMS, effective sample count and along-wall spread for the uncertainty model."""
    r = line.dist(P2)
    t = line.t(P2)
    rms = float(np.sqrt(np.mean(r ** 2)))
    cells = np.unique(np.stack([np.floor(t / cell), np.floor(Y / cell)], 1), axis=0)
    return dict(rms=rms, n_eff=int(len(cells)), t_mean=float(t.mean()), var_t=float(t.var()),
                n=int(len(t)), t_min=float(t.min()), t_max=float(t.max()))


def refine_on_measurement_cloud(w, Pm, H, gate=0.03):
    """Re-fit a wall line on the dense (1 cm) cloud: points near the line, inside its extent."""
    ln = w["line"]
    q = Pm[:, [0, 2]]
    t = ln.t(q)
    a, b = w["seg"]
    base = (t > ln.t(a) - 0.05) & (t < ln.t(b) + 0.05) & (Pm[:, 1] > 0.08) & (Pm[:, 1] < H - 0.08)
    sel = base & (np.abs(ln.dist(q)) < gate)
    if sel.sum() < 50:
        return
    for _ in range(2):
        new = Line(*fit_line_2d(q[sel]))
        if new.d @ ln.d < 0:
            new = Line(new.p, -new.d)
        r = new.dist(q)
        rms = float(np.sqrt(np.mean(r[sel] ** 2)))
        sel = base & (np.abs(r) < max(2.5 * rms, 0.012))
        ln = new
    w["line"] = ln
    w["stats"] = line_stats(ln, q[sel], Pm[sel, 1])
    w["meas_idx"] = np.flatnonzero(sel)


def sigma_perp(st, t):
    return st["rms"] / np.sqrt(st["n_eff"]) * np.sqrt(1.0 + (t - st["t_mean"]) ** 2 / max(st["var_t"], 1e-6))


def build_footprint(walls, P, H, floor_mask, k_scale=0.003, P_meas=None, min_short_wall=0.15):
    """Order boundary walls along the room outline, intersect neighbours, compute lengths + sigma.

    Returns (wall dicts, polygon Kx2 CCW, rejected wall candidates)."""
    walls, rejected, _ = prune_outside(walls, P, floor_mask)
    _, _, info = prune_outside(walls, P, floor_mask)
    outside = behind_walls_mask(walls, P, info)
    segs = [tuple(w["line"].at(np.array(_extent(w, P)))) for w in walls]
    contour, dist, lo, cell = room_outline(P[~outside], H, floor_mask[~outside], segments=segs)
    ordered = []
    for w in walls:
        q = P[w["idx"]][:, [0, 2]]
        t = w["line"].t(q)
        a, b = w["line"].at(np.percentile(t, 1)), w["line"].at(np.percentile(t, 99))
        frac = boundary_fraction(a, b, dist, lo, cell)
        if frac < 0.4:
            rejected.append(w)            # e.g. a wardrobe front standing in front of the real wall
            continue
        mid = 0.5 * (a + b)
        k = int(np.argmin(np.linalg.norm(contour - mid, axis=1)))
        ordered.append(dict(w, seg=(a, b), order=k, boundary=frac))
    ordered.sort(key=lambda w: w["order"])
    # neighbours along the outline that are nearly parallel and overlap are competing explanations
    # of the same stretch of boundary (e.g. an open door leaf next to its wall): keep the stronger
    i = 0
    while len(ordered) > 3 and i < len(ordered):
        j = (i + 1) % len(ordered)
        li, lj = ordered[i]["line"], ordered[j]["line"]
        if abs(li.d @ lj.d) > np.cos(np.radians(25)):
            ta = sorted(li.t(np.array(ordered[i]["seg"])))
            tb = sorted(li.t(np.array(ordered[j]["seg"])))
            overlap = min(ta[1], tb[1]) - max(ta[0], tb[0])
            off = abs(np.median(li.dist(P[ordered[j]["idx"]][:, [0, 2]])))
            if overlap > 0.1 and off > 0.08:
                weak = i if len(ordered[i]["idx"]) < len(ordered[j]["idx"]) else j
                rejected.append(ordered.pop(weak))
                continue
        i += 1
    _merge_collinear_neighbours(ordered, P)
    n = len(ordered)
    if n < 3:
        raise RuntimeError(f"only {n} boundary walls found")
    K = len(contour)

    def cidx(p):
        return int(np.argmin(np.linalg.norm(contour - p, axis=1)))

    for w in ordered:
        a, b = w["seg"]
        if (cidx(b) - cidx(a)) % K > K / 2:          # orient every wall along the outline walk
            w["seg"] = (b, a)
            w["line"] = Line(w["line"].p, -w["line"].d)
        q = P[w["idx"]]
        w["stats"] = line_stats(w["line"], q[:, [0, 2]], q[:, 1])
        if P_meas is not None:
            refine_on_measurement_cloud(w, P_meas, H)
    # corners; an observed wall that ends up shorter than 15 cm (or reversed) between its corners is
    # clutter (door-frame return, cupboard side): drop it, re-merge collinear neighbours, repeat
    while True:
        corners, inferred = _corners(ordered)
        k = _short_wall(ordered, corners, min_short_wall) if len(ordered) > 3 else None
        if k is None and len(ordered) > 3:
            k = _shallow_step(ordered, corners, inferred)
        if k is None:
            break
        rejected.append(ordered.pop(k))
        _merge_collinear_neighbours(ordered, P)
        for w in ordered:
            if "meas_idx" not in w:
                q = P[w["idx"]]
                w["stats"] = line_stats(w["line"], q[:, [0, 2]], q[:, 1])
                if P_meas is not None:
                    refine_on_measurement_cloud(w, P_meas, H)
    n = len(ordered)
    # walls between corners (plus inferred connector walls)
    out = []
    for i in range(n):
        a = corners[i - 1][1]
        b = corners[i][0]
        out.append(dict(wall=ordered[i], a=a, b=b, inferred=False))
        if i in inferred:
            out.append(dict(wall=None, a=corners[i][0], b=corners[i][1], inferred=True))
    poly = np.array([o["a"] for o in out])
    if polygon_area(poly) < 0:                     # enforce counter-clockwise in (x, z)
        out = out[::-1]
        for o in out:
            o["a"], o["b"] = o["b"], o["a"]
        poly = np.array([o["a"] for o in out])
    _wall_sigmas(out, k_scale)
    return out, poly, rejected


def _merge_collinear_neighbours(ordered, P):
    """Adjacent collinear pieces along the outline (one wall split by a large opening) -> one wall."""
    i = 0
    while len(ordered) > 2 and i < len(ordered):
        j = (i + 1) % len(ordered)
        li, lj = ordered[i]["line"], ordered[j]["line"]
        if abs(li.d @ lj.d) > np.cos(np.radians(6)) and abs(np.median(li.dist(P[ordered[j]["idx"]][:, [0, 2]]))) < 0.08:
            idx = np.union1d(ordered[i]["idx"], ordered[j]["idx"])
            ln = Line(*fit_line_2d(P[idx][:, [0, 2]]))
            if ln.d @ li.d < 0:
                ln = Line(ln.p, -ln.d)
            q = P[idx][:, [0, 2]]
            t = ln.t(q)
            seg = (ln.at(np.percentile(t, 1)), ln.at(np.percentile(t, 99)))
            ordered[i] = dict(ordered[i], line=ln, idx=idx, seg=seg)
            ordered[i].pop("meas_idx", None)
            ordered.pop(j)
            if j < i:
                i -= 1
            continue
        i += 1


def _corners(ordered):
    n = len(ordered)
    corners, inferred = [], []
    for i in range(n):
        wi, wj = ordered[i], ordered[(i + 1) % n]
        sin_phi = abs(cross2(wi["line"].d, wj["line"].d))
        c = _intersect(wi["line"], wj["line"]) if sin_phi > np.sin(np.radians(20)) else None
        end_i, start_j = wi["seg"][1], wj["seg"][0]
        if c is not None and min(np.linalg.norm(c - end_i), np.linalg.norm(c - start_j)) > 1.5:
            c = None                       # implausible far intersection: a wall is missing here
        if c is None:
            # parallel step (missing short wall): connect end of i to the foot point on line j
            p1 = end_i
            p2 = wj["line"].at(wj["line"].t(p1))
            corners.append((p1, p2))
            inferred.append(i)
        else:
            corners.append((c, c))
    return corners, inferred


def _short_wall(ordered, corners, min_len):
    """Index of the weakest observed wall that came out shorter than min_len or reversed between its
    corners (clutter: a door-frame return, a cupboard side), or None."""
    n = len(ordered)
    bad = []
    for i in range(n):
        a, b = corners[i - 1][1], corners[i][0]
        v = b - a
        L = float(np.linalg.norm(v))
        if L < min_len or v @ ordered[i]["line"].d < 0:
            bad.append(i)
    if not bad:
        return None
    return min(bad, key=lambda i: len(ordered[i]["idx"]))


def _shallow_step(ordered, corners, inferred, max_depth=0.2, max_len=1.2):
    """Two parallel walls joined by an unobserved step shallower than 20 cm: the shorter one (if
    under 1.2 m) is a recess or a protrusion of clutter (a door leaf in its frame seen from the
    other side, a radiator, a cupboard side), not a wall of the room. Returns its index or None."""
    n = len(ordered)
    for i in inferred:
        p1, p2 = corners[i]
        if np.linalg.norm(p2 - p1) >= max_depth:
            continue
        j = (i + 1) % n
        lens = [float(np.linalg.norm(ordered[k]["seg"][1] - ordered[k]["seg"][0])) for k in (i, j)]
        k = (i, j)[int(np.argmin(lens))]
        if min(lens) < max_len:
            return k
    return None


def _wall_sigmas(out, k_scale):
    n = len(out)
    for i, o in enumerate(out):
        L = float(np.linalg.norm(o["b"] - o["a"]))
        o["length"] = L
        if o["inferred"]:
            o["sigma"] = 0.05 + k_scale * L
            continue
        prev, nxt = out[(i - 1) % n], out[(i + 1) % n]
        var = (k_scale * L) ** 2
        cot_sum = 0.0
        for nb, corner in ((prev, o["a"]), (nxt, o["b"])):
            if nb["inferred"] or nb["wall"] is None:
                var += 0.05 ** 2
                continue
            sphi = abs(cross2(o["wall"]["line"].d, nb["wall"]["line"].d))
            sphi = max(sphi, 0.2)
            sp = sigma_perp(nb["wall"]["stats"], nb["wall"]["line"].t(corner))
            var += (sp / sphi) ** 2
            cot_sum += np.sqrt(max(1 - sphi ** 2, 0)) / sphi
        own = sigma_perp(o["wall"]["stats"], o["wall"]["line"].t(0.5 * (o["a"] + o["b"])))
        var += (own * cot_sum) ** 2
        o["sigma"] = float(np.sqrt(var))
