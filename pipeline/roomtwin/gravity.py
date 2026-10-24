"""Gravity (up) detection and the floor / ceiling planes.

No up axis is assumed. Dominant surface orientations are clustered from per-point normals; every
cluster axis (both signs) is a gravity candidate and is scored with physical cues:

  * perpendicularity: the walls are perpendicular to the true up axis (all other clusters are),
  * clutter skew:     furniture rests on the floor, so off-plane points sit in the lower half,
  * furniture tops:   interior horizontal surfaces (bed, desk, table tops) lie in the lower half,
  * floor density:    the floor is scanned more completely than the ceiling,
  * span:             floor-to-ceiling distance is near 2.55 m (soft Gaussian prior, sigma 0.35 m),
  * axis prior:       exports are y-up or z-up; a soft bonus only, so the cues above can override it,
  * device hint:      for RGB-D frames without poses, the mean camera "down" axis (phones are held
                      roughly upright) is a soft hint.

The winner is then refined: MSAC on the lowest horizontal-plane peak gives the floor plane, and its
PCA normal (cv2.PCACompute2) becomes the exact gravity direction.
"""
from __future__ import annotations

import numpy as np

from .geometry import msac_plane, refine_plane, rotation_between


def dominant_directions(normals: np.ndarray, rng, max_dirs: int = 5, min_frac: float = 0.04):
    """Greedy axial clustering of unit normals (sign-agnostic). Returns [(axis, weight_fraction)]."""
    n = normals[~np.isnan(normals[:, 0])]
    if len(n) > 30000:
        n = n[rng.choice(len(n), 30000, replace=False)]
    total = len(n)
    dirs = []
    cos_in, cos_rm = np.cos(np.radians(10)), np.cos(np.radians(20))
    for _ in range(max_dirs):
        if len(n) < 50:
            break
        cand = n[rng.choice(len(n), min(400, len(n)), replace=False)]
        support = (np.abs(n @ cand.T) > cos_in).sum(0)
        c = cand[int(np.argmax(support))]
        sup = np.abs(n @ c) > cos_in
        sel = n[sup] * np.sign(n[sup] @ c)[:, None]
        w, v = np.linalg.eigh(sel.T @ sel)
        c = v[:, 2] * np.sign(v[:, 2] @ c)
        frac = sup.sum() / total
        if frac < min_frac:
            break
        dirs.append((c, float(frac)))
        n = n[np.abs(n @ c) < cos_rm]
    return dirs


def _height_peaks(h: np.ndarray, bin_size: float = 0.02):
    """Histogram peaks (centre, count) of heights, smoothed over 3 bins."""
    if len(h) == 0:
        return []
    lo, hi = h.min(), h.max()
    nb = max(int((hi - lo) / bin_size) + 1, 1)
    cnt, edges = np.histogram(h, bins=nb, range=(lo, lo + nb * bin_size))
    sm = np.convolve(cnt, [1, 2, 1], "same") / 4
    peaks = [(0.5 * (edges[i] + edges[i + 1]), sm[i]) for i in range(len(sm))
             if sm[i] > 0 and sm[i] >= sm[max(i - 1, 0)] and sm[i] >= sm[min(i + 1, len(sm) - 1)]]
    return peaks


def score_up_candidates(points, normals, dirs, up_hint=None):
    """Score each (axis, sign). Returns sorted list of dicts with the individual cues."""
    valid = ~np.isnan(normals[:, 0])
    out = []
    for i, (d, _) in enumerate(dirs):
        others = [(c, w) for j, (c, w) in enumerate(dirs) if j != i and abs(c @ d) < np.cos(np.radians(20))]
        wsum = sum(w for _, w in others)
        perp = sum(w for c, w in others if abs(c @ d) < np.sin(np.radians(8))) / wsum if wsum > 0 else 0.5
        for sign in (1.0, -1.0):
            u = d * sign
            h = points @ u
            horiz = valid & (np.abs(np.nan_to_num(normals) @ u) > np.cos(np.radians(15)))
            peaks = _height_peaks(h[horiz])
            if not peaks:
                continue
            pmax = max(c for _, c in peaks)
            bottom = min(p[0] for p in peaks if p[1] >= 0.1 * pmax)
            top = max(p[0] for p in peaks if p[1] >= 0.03 * pmax)
            span = top - bottom
            nb = np.sum(horiz & (np.abs(h - bottom) < 0.04))
            nt = np.sum(horiz & (np.abs(h - top) < 0.04))
            density = (nb - nt) / max(nb + nt, 1)
            mid = (h > bottom + 0.06) & (h < top - 0.06)
            skew = 0.5 - float(np.mean((h[mid] - bottom) / max(span, 1e-6))) if mid.any() else 0.0
            # furniture tops (beds, desks, tables) are horizontal surfaces in the lower half
            fm = horiz & (h > bottom + 0.15) & (h < top - 0.15)
            conf = min(1.0, fm.sum() / 500.0)
            skew_h = (0.5 - float(np.mean((h[fm] - bottom) / max(span, 1e-6)))) * conf if fm.any() else 0.0
            axis = max(abs(u[1]), abs(u[2]))
            axis_prior = 1.0 if axis > np.cos(np.radians(5)) else 0.0
            sign_prior = 0.5 if (u[1] > 0.95 or u[2] > 0.95) else 0.0
            # habitable ceiling heights cluster around 2.4-2.7 m; wall-to-wall spans are usually longer
            span_ok = 2.0 * np.exp(-0.5 * ((span - 2.55) / 0.35) ** 2) - (1.0 if not 1.9 <= span <= 4.5 else 0.0)
            hint = 2.0 * float(u @ up_hint) if up_hint is not None else 0.0
            score = 4.0 * skew + 6.0 * skew_h + 1.0 * density + 1.5 * perp + axis_prior + sign_prior + span_ok + hint
            out.append(dict(up=u, score=float(score), skew=skew, skew_h=skew_h, density=float(density), perp=float(perp),
                            axis_prior=axis_prior + sign_prior, span=float(span)))
    out.sort(key=lambda r: -r["score"])
    return out


def fit_horizontal_plane(points, normals, y_peak, rng, thresh=0.015, window=0.06):
    """MSAC + PCA refine of the horizontal plane near height y_peak (points already roughly y-up)."""
    sel = np.abs(points[:, 1] - y_peak) < window
    horiz = sel & ~np.isnan(normals[:, 0]) & (np.abs(np.nan_to_num(normals[:, 1])) > np.cos(np.radians(15)))
    P = points[horiz]
    if len(P) < 50:
        return None
    res = msac_plane(P, thresh, iters=200, rng=rng)
    if res is None:
        return None
    n, d, _ = res
    if n[1] < 0:
        n, d = -n, -d
    # final polish on every point (any normal) within the threshold of the plane
    near = sel & (np.abs(points @ n + d) < thresh)
    n, d, rms = refine_plane(points[near])
    if n[1] < 0:
        n, d = -n, -d
    return dict(normal=n, d=d, rms=rms, mask=np.abs(points @ n + d) < max(thresh, 2.5 * rms))


def detect_gravity(points, normals, rng, up_hint=None):
    """Returns (R, info): R rotates scan coordinates so that +y is up (floor not yet at 0)."""
    dirs = dominant_directions(normals, rng)
    if not dirs:
        raise RuntimeError("no planar structure found; is this a room scan?")
    cands = score_up_candidates(points, normals, dirs, up_hint)
    best = cands[0]
    R = rotation_between(best["up"], np.array([0.0, 1.0, 0.0]))
    info = dict(candidates=[{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
                            for c in cands[:4]],
                margin=float(best["score"] - cands[1]["score"]) if len(cands) > 1 else 99.0)
    return R, info


def floor_and_ceiling(points, normals, rng):
    """Floor = lowest significant horizontal peak, ceiling = highest one >= 1.8 m above the floor."""
    valid = ~np.isnan(normals[:, 0])
    horiz = valid & (np.abs(np.nan_to_num(normals[:, 1])) > np.cos(np.radians(15)))
    peaks = _height_peaks(points[horiz, 1])
    pmax = max(c for _, c in peaks)
    floor_y = min(p[0] for p in peaks if p[1] >= 0.15 * pmax)
    floor = fit_horizontal_plane(points, normals, floor_y, rng)
    if floor is None:
        raise RuntimeError("floor plane not found")
    # level the cloud exactly on the floor normal, floor at y = 0
    R = rotation_between(floor["normal"], np.array([0.0, 1.0, 0.0]))
    ceiling = None
    cands = [p for p in peaks if p[1] >= 0.03 * pmax and p[0] - floor_y > 1.8]
    if cands:
        ceil_y = max(p[0] for p in cands)
        ceiling = fit_horizontal_plane(points, normals, ceil_y, rng)
    return floor, ceiling, R
