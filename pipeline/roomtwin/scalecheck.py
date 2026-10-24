"""Independent scale check with an ArUco marker of known size.

For every photo (ARKit-style frame: RGB + depth + intrinsics JSON, see rgbd.py layout A) the marker
is detected with cv2.aruco.ArucoDetector (sub-pixel corners) and measured twice:

  * camera-only: cv2.solvePnP(SOLVEPNP_IPPE_SQUARE) with the known side length gives the marker's
    distance from the calibrated RGB camera alone;
  * LiDAR: a plane is fitted (cv2.PCACompute2) to the depth points inside the marker, the four
    corner rays are intersected with it, and the 3D side lengths are measured.

scale = LiDAR side length / true side length (1.000 = perfect). depth_ratio = LiDAR distance / PnP
distance cross-checks the depth map against the camera model. The result goes into room.json
"scale_check" and becomes a QA action when the scan scale is off by more than its uncertainty.
"""
from __future__ import annotations

import cv2
import numpy as np

from .rgbd import _frame_K, _load_depth, discover

DICTS = ["DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11"]


def _detector(name):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, p)


def measure_marker(gray, depth, K_rgb, K_depth, size):
    """Return list of per-marker dicts for one frame."""
    out = []
    for name in DICTS:
        corners, ids, _ = _detector(name).detectMarkers(gray)
        if ids is not None and len(ids):
            break
    else:
        return out
    h, w = depth.shape
    sx, sy = w / gray.shape[1], h / gray.shape[0]
    obj = np.array([[-size / 2, size / 2, 0], [size / 2, size / 2, 0], [size / 2, -size / 2, 0],
                    [-size / 2, -size / 2, 0]], np.float64)
    for c, mid in zip(corners, ids.ravel()):
        c = c.reshape(4, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, c, K_rgb, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        # depth points inside the marker quad (shrunk 10 % to stay off its edges)
        cd = (c + 0.5) * [sx, sy] - 0.5                # RGB pixel -> depth pixel (centre convention)
        ctr = cd.mean(0)
        quad = np.round(ctr + 0.9 * (cd - ctr)).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, quad, 1)
        v, u = np.nonzero((mask > 0) & (depth > 0))
        if len(u) < 12:
            continue
        z = depth[v, u]
        X = np.stack([(u - K_depth[0, 2]) / K_depth[0, 0] * z, (v - K_depth[1, 2]) / K_depth[1, 1] * z, z], 1)
        mean, vecs, _ = cv2.PCACompute2(X, mean=None)
        n = vecs[2]
        d = -float(n @ mean[0])
        # intersect the RGB corner rays (in depth-camera pixels) with the LiDAR plane
        rays = np.stack([(cd[:, 0] - K_depth[0, 2]) / K_depth[0, 0], (cd[:, 1] - K_depth[1, 2]) / K_depth[1, 1],
                         np.ones(4)], 1)
        tt = -d / (rays @ n)
        P = rays * tt[:, None]
        sides = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
        centre_lidar = float(np.linalg.norm(P.mean(0)))
        centre_pnp = float(np.linalg.norm(tvec))
        out.append(dict(id=int(mid), dictionary=name, side_m=float(sides.mean()), scale=float(sides.mean() / size),
                        depth_ratio=centre_lidar / centre_pnp, distance_m=centre_pnp, depth_points=int(len(u))))
    return out


def check_folder(folder, marker_size):
    info = discover(folder)
    rows = []
    for fr in info["frames"]:
        if not fr["rgb"]:
            continue
        rgb = cv2.imread(fr["rgb"], cv2.IMREAD_GRAYSCALE)
        depth = _load_depth(fr["depth"], fr["depth_scale"])
        K_rgb = fr["K"].copy()
        W, H = fr["K_size"]
        if W and H and (W != rgb.shape[1] or H != rgb.shape[0]):
            K_rgb[0] *= rgb.shape[1] / W
            K_rgb[1] *= rgb.shape[0] / H
        Kd = _frame_K(fr, depth.shape)
        for r in measure_marker(rgb, depth, K_rgb, Kd, marker_size):
            r["frame"] = fr["index"]
            rows.append(r)
    if not rows:
        return dict(status="no marker found", marker_size_m=marker_size, observations=[])
    s = np.array([r["scale"] for r in rows])
    dr = np.array([r["depth_ratio"] for r in rows])
    sig = float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.01
    scale = float(np.median(s))
    res = dict(status="ok", marker_size_m=marker_size, scale=round(scale, 4), scale_sigma=round(sig, 4),
               depth_ratio=round(float(np.median(dr)), 4), observations=rows)
    off = abs(scale - 1)
    res["verdict"] = ("scan scale matches the marker" if off < max(2 * sig, 0.005)
                      else f"scan scale is off by {100 * (scale - 1):+.1f}%: rescan or apply the correction")
    return res
