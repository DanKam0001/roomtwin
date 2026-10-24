"""Point-cloud primitives: voxel downsampling, outlier removal, normals, robust plane/line fits.

OpenCV 5.0.0 has cv::SACSegmentation and cv::voxelGridSampling in the C++ 3d module, but neither is
wrapped in the Python bindings of opencv-python 5.0.0.93 (checked with dir(cv2) and cv2/__init__.pyi).
So the sample-consensus loop below is an MSAC implementation in numpy that uses the same plane model
as cv::SAC_MODEL_PLANE (n.x + d = 0, |n| = 1), and every model is polished with OpenCV least squares:
cv2.PCACompute2 for planes and cv2.fitLine (Huber M-estimator) for wall lines.
"""
from __future__ import annotations

import cv2
import numpy as np

cv2.ocl.setUseOpenCL(False)  # deterministic CPU path; OpenCL TSDF kernels fail on some laptop GPUs


# ----------------------------------------------------------------------------------- voxel grid
def _voxel_keys(points: np.ndarray, size: float, offset: float = 0.0):
    q = np.floor((points - points.min(0) + offset) / size).astype(np.int64)
    dims = q.max(0) + 2
    keys = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    return keys, q, dims


def voxel_downsample(points: np.ndarray, size: float, colors: np.ndarray | None = None):
    """Centroid per occupied voxel (unbiased, unlike snapping to voxel centres).

    Returns (points, colors, counts) where counts is the number of raw points per voxel."""
    keys, _, _ = _voxel_keys(points, size)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    out = np.empty((len(uniq), 3), np.float64)
    for k in range(3):
        out[:, k] = np.bincount(inv, weights=points[:, k], minlength=len(uniq)) / counts
    col = None
    if colors is not None:
        col = np.empty((len(uniq), 3), np.float64)
        for k in range(3):
            col[:, k] = np.bincount(inv, weights=colors[:, k], minlength=len(uniq)) / counts
    return out, col, counts


def grid_neighbour_counts(points: np.ndarray, cell: float) -> np.ndarray:
    """Number of points in the 3x3x3 block of grid cells around each point (radius-outlier proxy)."""
    keys, q, dims = _voxel_keys(points, cell)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    total = np.zeros(len(points), np.int64)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nk = keys + (dx * dims[1] + dy) * dims[2] + dz
                pos = np.searchsorted(uniq, nk)
                pos = np.clip(pos, 0, len(uniq) - 1)
                hit = uniq[pos] == nk
                total[hit] += counts[pos[hit]]
    return total


def remove_outliers(points: np.ndarray, cell: float = 0.05, min_neighbours: int = 8) -> np.ndarray:
    """Boolean keep-mask: drop isolated points (flying pixels, reflections, sparse junk)."""
    return grid_neighbour_counts(points, cell) >= min_neighbours


# ----------------------------------------------------------------------------------- normals
def _cell_normals(points, cell, offset):
    keys, _, _ = _voxel_keys(points, cell, offset)
    uniq, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    m = len(uniq)
    s = np.stack([np.bincount(inv, weights=points[:, k], minlength=m) for k in range(3)], 1)
    mean = s / cnt[:, None]
    cov = np.empty((m, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            v = np.bincount(inv, weights=points[:, i] * points[:, j], minlength=m) / cnt
            cov[:, i, j] = cov[:, j, i] = v - mean[:, i] * mean[:, j]
    w, v = np.linalg.eigh(cov)                      # ascending eigenvalues
    normal = v[:, :, 0]
    # planarity score: off-plane spread relative to in-plane spread; a cell is planar when the
    # off-plane std is below max_noise and clearly smaller than the in-plane extent
    w0 = np.maximum(w[:, 0], 0)
    score = np.sqrt(w0) / np.sqrt(np.maximum(w[:, 1], 1e-12))
    ok = cnt >= 6
    return normal[inv], np.where(ok, score, 9.0)[inv], np.sqrt(w0)[inv]


def estimate_normals(points: np.ndarray, cell: float = 0.10, max_noise: float = 0.015, max_ratio: float = 0.5):
    """Per-point normals from local PCA on two staggered grids; keep the flatter estimate.

    A cell counts as planar when its off-plane std is < max_noise (metres) and below max_ratio times
    the in-plane std; non-planar points (corners, clutter) get NaN normals. Vectorised (~1 s / 1M pts)."""
    n0, r0, s0 = _cell_normals(points, cell, 0.0)
    n1, r1, s1 = _cell_normals(points, cell, cell / 2)
    pick = r1 < r0
    normals = np.where(pick[:, None], n1, n0)
    ratio = np.minimum(r0, r1)
    noise = np.where(pick, s1, s0)
    normals[(ratio > max_ratio) | (noise > max_noise)] = np.nan
    return normals, ratio


# ----------------------------------------------------------------------------------- planes
def refine_plane(points: np.ndarray):
    """Least-squares plane via cv2.PCACompute2: returns (normal, d, rms) with n.x + d = 0."""
    mean, vecs, vals = cv2.PCACompute2(np.ascontiguousarray(points, np.float64), mean=None)
    n = vecs[2].astype(np.float64)
    n /= np.linalg.norm(n)
    d = -float(n @ mean[0])
    rms = float(np.sqrt(max(vals[2, 0], 0.0)))
    return n, d, rms


def msac_plane(points: np.ndarray, thresh: float, iters: int = 300, rng=None,
               sample_eval: int = 4000, normals: np.ndarray | None = None, max_angle_deg: float = 25.0):
    """MSAC plane segmentation (SAC_MODEL_PLANE) with local refinement.

    Hypotheses are minimal 3-point samples; they are scored with the truncated quadratic MSAC cost on a
    random evaluation subset, the winner is re-estimated with PCA on all inliers (twice).
    If normals are given, an inlier must also have a compatible normal (prevents bridging surfaces).
    Returns (normal, d, inlier_mask) or None."""
    rng = rng or np.random.default_rng(0)
    n_pts = len(points)
    if n_pts < 3:
        return None
    ev = points if n_pts <= sample_eval else points[rng.choice(n_pts, sample_eval, replace=False)]
    idx = rng.integers(0, n_pts, size=(iters, 3))
    p0, p1, p2 = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(nrm, axis=1)
    good = ln > 1e-9
    if not good.any():
        return None
    nrm = nrm[good] / ln[good, None]
    d = -np.einsum("ij,ij->i", nrm, p0[good])
    res = np.abs(ev @ nrm.T + d)                    # (eval, hyps)
    cost = np.minimum(res, thresh) ** 2
    best = int(np.argmin(cost.sum(0)))
    n, dd = nrm[best], d[best]
    cos_t = np.cos(np.radians(max_angle_deg))
    for _ in range(3):                              # local optimisation
        r = np.abs(points @ n + dd)
        inl = r < thresh
        if normals is not None:
            ok = np.isnan(normals[:, 0]) | (np.abs(np.nan_to_num(normals) @ n) > cos_t)
            inl &= ok
        if inl.sum() < 3:
            return None
        n, dd, _ = refine_plane(points[inl])
    inl = np.abs(points @ n + dd) < thresh
    return n, dd, inl


# ----------------------------------------------------------------------------------- 2D lines
def fit_line_2d(p2: np.ndarray):
    """cv2.fitLine with the Huber M-estimator. Returns (point_on_line, unit_direction)."""
    vx, vy, x0, y0 = cv2.fitLine(np.ascontiguousarray(p2, np.float32), cv2.DIST_HUBER, 0, 0.001, 0.001).ravel()
    d = np.array([vx, vy], np.float64)
    return np.array([x0, y0], np.float64), d / np.linalg.norm(d)


def msac_lines_oriented(p2: np.ndarray, n2: np.ndarray, thresh: float, iters: int, rng,
                        sample_eval: int = 6000, cos_normal: float = 0.94):
    """One-point MSAC for 2D lines: each hypothesis is an oriented point (position + normal).

    Returns (point, direction) of the best hypothesis (unrefined) or None."""
    n_pts = len(p2)
    if n_pts < 10:
        return None
    ev_idx = np.arange(n_pts) if n_pts <= sample_eval else rng.choice(n_pts, sample_eval, replace=False)
    pe, ne = p2[ev_idx], n2[ev_idx]
    hyp = rng.integers(0, n_pts, size=iters)
    hn = n2[hyp]
    hc = np.einsum("ij,ij->i", hn, p2[hyp])
    res = np.abs(pe @ hn.T - hc)
    agree = np.abs(ne @ hn.T) > cos_normal
    cost = np.where(agree, np.minimum(res, thresh) ** 2, thresh ** 2)
    best = int(np.argmin(cost.sum(0)))
    nb = hn[best]
    return p2[hyp[best]], np.array([-nb[1], nb[0]])


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b (axis-angle through cv2.Rodrigues)."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.clip(a @ b, -1, 1))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        perp = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])
        axis, s = perp / np.linalg.norm(perp), 1.0
        return cv2.Rodrigues(axis * np.pi)[0]
    return cv2.Rodrigues(axis / s * np.arctan2(s, c))[0]


def points_in_polygon(pts2: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd point-in-polygon test (pts2: Nx2, poly: Kx2)."""
    x, y = pts2[:, 0], pts2[:, 1]
    inside = np.zeros(len(pts2), bool)
    k = len(poly)
    for i in range(k):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % k]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cond & (x < xi)
    return inside


def polygon_area(poly: np.ndarray) -> float:
    """Signed shoelace area in the (x, z) plane (positive = counter-clockwise)."""
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))
