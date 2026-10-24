"""Synthetic LiDAR-like room scans with exact ground truth.

Each room is a set of planar surfaces (walls with door/window holes, floor, partial ceiling,
furniture boxes, an open door leaf, a patch of hallway seen through the door). Surfaces are sampled
at a phone-LiDAR-like density with normal noise (~7 mm), some tangential jitter, missing regions
(the wall behind the open door), a low-density patch, random outliers, and then rotated into a
random gravity convention (y-up or z-up) with a random yaw and offset.

The same scene can be ray-cast into RGB-D frames (ARKit-style JSON per frame) for the TSDF path.

    python -m roomtwin.synth --out D:/roomtwin_data/synth          # all rooms, point clouds
    python -m roomtwin.synth --out ... --rgbd rect                  # plus RGB-D frames of one room
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import points_in_polygon, polygon_area
from .io_ply import write_ply


@dataclass
class Surface:
    origin: np.ndarray
    U: np.ndarray
    V: np.ndarray
    poly: np.ndarray              # Kx2 in (u, v)
    label: str
    color: tuple
    holes: list = field(default_factory=list)   # (u0, v0, u1, v1)

    @property
    def N(self):
        return np.cross(self.U, self.V)


@dataclass
class RoomSpec:
    name: str
    corners: list                 # CCW (x, z)
    height: float
    openings: list                # dicts: wall, type, offset, width, sill, height
    boxes: list                   # dicts: min, max, label
    door_leaf: int | None = 0     # index into openings of the door that stands open
    low_density: list = field(default_factory=list)   # (wall, t0, t1, y0, y1, keep_fraction)
    ceiling_keep: float = 0.6


ROOMS = {
    "rect": RoomSpec(
        "rect", [(0, 0), (4.2, 0), (4.2, 3.6), (0, 3.6)], 2.50,
        [dict(wall=0, type="door", offset=2.9, width=0.82, sill=0.0, height=2.04),
         dict(wall=2, type="window", offset=1.2, width=1.40, sill=0.90, height=1.20)],
        [dict(min=(0.1, 0, 0.9), max=(1.6, 0.55, 2.9), label="bed"),
         dict(min=(3.55, 0, 1.2), max=(4.15, 2.0, 2.4), label="wardrobe"),
         dict(min=(1.5, 0, 3.0), max=(2.7, 0.75, 3.55), label="desk")],
        low_density=[(1, 0.4, 1.1, 1.6, 2.4, 0.08)]),
    "lshape": RoomSpec(
        "lshape", [(0, 0), (5.0, 0), (5.0, 2.4), (2.6, 2.4), (2.6, 4.4), (0, 4.4)], 2.60,
        [dict(wall=0, type="door", offset=1.2, width=0.90, sill=0.0, height=2.10),
         dict(wall=4, type="window", offset=0.8, width=1.00, sill=1.00, height=1.10),
         dict(wall=5, type="window", offset=1.6, width=1.20, sill=0.80, height=1.30)],
        [dict(min=(3.2, 0, 0.1), max=(4.9, 0.6, 2.1), label="bed"),
         dict(min=(0.1, 0, 3.6), max=(1.3, 0.8, 4.3), label="desk")]),
    "angled": RoomSpec(
        "angled", [(0, 0), (4.5, 0), (4.5, 2.2), (2.5, 2.2 + 2.0 * np.tan(np.radians(30))),
                   (0, 2.2 + 2.0 * np.tan(np.radians(30)))], 2.45,
        [dict(wall=0, type="door", offset=0.4, width=0.82, sill=0.0, height=2.04),
         dict(wall=2, type="window", offset=0.55, width=1.20, sill=0.85, height=1.10)],
        [dict(min=(0.1, 0, 0.9), max=(1.1, 0.45, 3.0), label="sofa"),
         dict(min=(3.0, 0, 0.6), max=(3.8, 0.72, 1.4), label="table")]),
    "alcove": RoomSpec(
        "alcove", [(0, 0), (4.0, 0), (4.0, 4.0), (2.6, 4.0), (2.6, 4.6), (1.4, 4.6), (1.4, 4.0), (0, 4.0)],
        2.40,
        [dict(wall=0, type="door", offset=0.3, width=0.80, sill=0.0, height=2.00),
         dict(wall=1, type="window", offset=1.4, width=1.20, sill=0.95, height=1.10)],
        [dict(min=(2.2, 0, 1.4), max=(3.9, 0.5, 3.4), label="bed"),
         dict(min=(1.5, 0, 4.1), max=(2.5, 0.9, 4.55), label="drawers")]),
}


def _rect(u1, v1):
    return np.array([[0, 0], [u1, 0], [u1, v1], [0, v1]], np.float64)


def _box_surfaces(bmin, bmax, label, color, yaw_axes=None):
    """Five visible faces of an axis-aligned box (no bottom)."""
    x0, y0, z0 = bmin
    x1, y1, z1 = bmax
    ex, ey, ez = np.eye(3)
    s = [Surface(np.array([x0, y1, z1]), ex, -ez, _rect(x1 - x0, z1 - z0), label, color)]   # top
    s.append(Surface(np.array([x0, y0, z0]), ex, ey, _rect(x1 - x0, y1 - y0), label, color))
    s.append(Surface(np.array([x1, y0, z1]), -ex, ey, _rect(x1 - x0, y1 - y0), label, color))
    s.append(Surface(np.array([x0, y0, z1]), -ez, ey, _rect(z1 - z0, y1 - y0), label, color))
    s.append(Surface(np.array([x1, y0, z0]), ez, ey, _rect(z1 - z0, y1 - y0), label, color))
    return s


def build_scene(spec: RoomSpec):
    """Surfaces of the room in its ground-truth frame (y up, floor y = 0, interior normals)."""
    C = np.asarray(spec.corners, np.float64)
    H = spec.height
    surfaces, walls = [], []
    for i in range(len(C)):
        a, b = C[i], C[(i + 1) % len(C)]
        L = float(np.linalg.norm(b - a))
        U = np.array([(b - a)[0] / L, 0, (b - a)[1] / L])
        V = np.array([0.0, 1.0, 0.0])
        # interior is to the left of a->b for a CCW polygon: N = U x V must equal (-Uz, 0, Ux)
        holes = [(o["offset"], o["sill"], o["offset"] + o["width"], o["sill"] + o["height"])
                 for o in spec.openings if o["wall"] == i]
        # U x V = (-Uz, 0, Ux): the left normal of a->b, i.e. pointing into a CCW room
        s = Surface(np.array([a[0], 0.0, a[1]]), U, V, _rect(L, H), f"wall{i}", (0.85, 0.83, 0.78), holes)
        walls.append(dict(a=a, b=b, length=L, U=U))
        surfaces.append(s)
    floor = Surface(np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), C[:, ::-1].copy(), "floor", (0.55, 0.45, 0.35))
    ceil = Surface(np.array([0, H, 0.0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), C.copy(), "ceiling", (0.95, 0.95, 0.95))
    surfaces += [floor, ceil]
    palette = [(0.3, 0.4, 0.7), (0.6, 0.3, 0.3), (0.3, 0.6, 0.4), (0.6, 0.6, 0.3)]
    for k, bx in enumerate(spec.boxes):
        surfaces += _box_surfaces(bx["min"], bx["max"], "furniture:" + bx["label"], palette[k % 4])
    # open door leaf: hinged at the door's 'a'-side jamb, swung ~165 deg so it lies close to the wall
    leaf_gap = None
    if spec.door_leaf is not None:
        o = spec.openings[spec.door_leaf]
        w = walls[o["wall"]]
        Ud = np.array([w["U"][0], w["U"][2]])
        Nin = np.array([-Ud[1], Ud[0]])
        ang = np.radians(15)
        lw = o["width"] - 0.04
        a2 = w["a"] + Ud * o["offset"]
        side = -1.0 if o["offset"] > lw + 0.1 else 1.0          # swing towards the side with room
        t_hinge = o["offset"] if side < 0 else o["offset"] + o["width"]
        leaf_dir = side * Ud * np.cos(ang) + Nin * np.sin(ang)
        hinge = w["a"] + Ud * t_hinge + Nin * 0.03
        U3 = np.array([leaf_dir[0], 0, leaf_dir[1]])
        Vy = np.array([0, 1.0, 0])
        n_leaf = np.cross(U3, Vy)
        h0 = np.array([hinge[0], 0.0, hinge[1]])
        surfaces.append(Surface(h0, U3, Vy, _rect(lw, o["height"] - 0.02), "door_leaf", (0.7, 0.55, 0.4)))
        surfaces.append(Surface(h0 + U3 * lw + n_leaf * 0.04, -U3, Vy, _rect(lw, o["height"] - 0.02),
                                "door_leaf", (0.7, 0.55, 0.4)))
        # wall region hidden behind the leaf (t measured from wall a)
        span = sorted([t_hinge, t_hinge + side * (lw * np.cos(ang) + 0.05)])
        leaf_gap = (o["wall"], span[0], span[1], 0.0, o["height"])
        # hallway outside the door: floor patch and the opposite hallway wall (points beyond the wall plane)
        outN = -Nin
        c0 = a2 + outN * 0.02
        Uo = np.array([Ud[0], 0, Ud[1]])
        Wo = np.array([outN[0], 0, outN[1]])
        surfaces.append(Surface(np.array([c0[0], 0.0, c0[1]]) - Uo * 0.3, Wo, Uo, _rect(1.2, o["width"] + 0.6), "outside", (0.4, 0.4, 0.4)))
        far = c0 + outN * 1.2
        surfaces.append(Surface(np.array([far[0], 0.0, far[1]]) - Uo * 0.3, Uo, np.array([0, 1.0, 0]),
                                _rect(o["width"] + 0.6, H), "outside", (0.5, 0.5, 0.5)))
    return surfaces, walls, leaf_gap


def sample_surface(s: Surface, density: float, rng):
    """Uniform samples on a planar polygon minus rectangular holes. Returns Nx3 points and (u, v)."""
    lo, hi = s.poly.min(0), s.poly.max(0)
    area = float(np.prod(hi - lo))
    n = rng.poisson(density * area)
    uv = lo + rng.random((n, 2)) * (hi - lo)
    keep = points_in_polygon(uv, s.poly)
    for (u0, v0, u1, v1) in s.holes:
        keep &= ~((uv[:, 0] > u0) & (uv[:, 0] < u1) & (uv[:, 1] > v0) & (uv[:, 1] < v1))
    uv = uv[keep]
    p = s.origin + uv[:, :1] * s.U + uv[:, 1:2] * s.V
    return p, uv


def generate_cloud(spec: RoomSpec, seed: int = 0, spacing: float = 0.015, noise: float = 0.007,
                   tangential: float = 0.002, outlier_frac: float = 0.002):
    """Point cloud of a room in its ground-truth frame + per-point colours."""
    rng = np.random.default_rng(seed)
    surfaces, walls, leaf_gap = build_scene(spec)
    density = 1.0 / spacing ** 2
    pts, cols = [], []
    for s in surfaces:
        p, uv = sample_surface(s, density, rng)
        keep = np.ones(len(p), bool)
        if s.label.startswith("wall"):
            wi = int(s.label[4:])
            t = (p - np.array([walls[wi]["a"][0], 0, walls[wi]["a"][1]])) @ walls[wi]["U"]
            y = p[:, 1]
            if leaf_gap is not None and leaf_gap[0] == wi:
                keep &= ~((t > leaf_gap[1]) & (t < leaf_gap[2]) & (y < leaf_gap[4]))
            for (w, t0, t1, y0, y1, frac) in spec.low_density:
                if w == wi:
                    region = (t > t0) & (t < t1) & (y > y0) & (y < y1)
                    keep &= ~region | (rng.random(len(p)) < frac)
        if s.label == "ceiling":
            # partial ceiling coverage: a few covered blobs, like a phone scan that rarely looks up
            centres = np.asarray(spec.corners)[rng.choice(len(spec.corners), 3, replace=False)]
            d = np.min(np.linalg.norm(p[:, [0, 2]][:, None, :] - centres[None], axis=2), 1)
            keep &= d < np.sqrt(spec.ceiling_keep * abs(polygon_area(np.asarray(spec.corners))) / 3 / np.pi) * 1.3
        if s.label.startswith("floor"):
            for bx in spec.boxes:             # floor under furniture is never seen
                keep &= ~((p[:, 0] > bx["min"][0]) & (p[:, 0] < bx["max"][0]) &
                          (p[:, 2] > bx["min"][2]) & (p[:, 2] < bx["max"][2]))
        if s.label.startswith("wall"):        # wall behind tall furniture touching it is hidden
            for bx in spec.boxes:
                inside = np.all((p > np.asarray(bx["min"]) - 0.08) & (p < np.asarray(bx["max"]) + 0.08), 1)
                keep &= ~inside
        p = p[keep]
        p = p + rng.normal(0, noise, (len(p), 1)) * s.N + rng.normal(0, tangential, (len(p), 1)) * s.U \
            + rng.normal(0, tangential, (len(p), 1)) * s.V
        pts.append(p)
        cols.append(np.tile(s.color, (len(p), 1)) * rng.uniform(0.9, 1.1, (len(p), 1)))
    P = np.concatenate(pts)
    Cc = np.clip(np.concatenate(cols), 0, 1)
    n_out = int(outlier_frac * len(P))
    lo, hi = P.min(0) - 0.3, P.max(0) + 0.3
    P = np.concatenate([P, lo + rng.random((n_out, 3)) * (hi - lo)])
    Cc = np.concatenate([Cc, rng.random((n_out, 3))])
    return P, Cc


def ground_truth(spec: RoomSpec):
    C = np.asarray(spec.corners, np.float64)
    walls = []
    for i in range(len(C)):
        a, b = C[i], C[(i + 1) % len(C)]
        walls.append({"id": f"w{i}", "a": a.tolist(), "b": b.tolist(), "length": float(np.linalg.norm(b - a))})
    return {"name": spec.name, "ceiling_height": spec.height, "walls": walls,
            "openings": [dict(o) for o in spec.openings],
            "floor_area_m2": abs(polygon_area(C)),
            "boxes": [dict(min=list(b["min"]), max=list(b["max"]), label=b["label"]) for b in spec.boxes]}


def random_pose(rng, up: str):
    """Ground-truth frame -> scan frame: yaw about +y, then the y-up -> z-up swap if requested."""
    yaw = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    if up == "z":
        R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) @ R     # y -> z
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.uniform(-3, 3, 3)
    return T


def make_room(spec: RoomSpec, out_dir: str, seed: int, up: str, spacing: float = 0.015, suffix: str = "",
              noise: float = 0.007):
    rng = np.random.default_rng(seed + 1000)
    P, C = generate_cloud(spec, seed=seed, spacing=spacing, noise=noise)
    T = random_pose(rng, up)
    Pw = P @ T[:3, :3].T + T[:3, 3]
    os.makedirs(out_dir, exist_ok=True)
    name = spec.name + suffix
    ply = os.path.join(out_dir, f"{name}.ply")
    write_ply(ply, Pw, C)
    gt = ground_truth(spec)
    gt.update({"gt_to_scan": T.tolist(), "up": up, "points": int(len(Pw)), "file": os.path.basename(ply)})
    with open(os.path.join(out_dir, f"{name}_gt.json"), "w") as f:
        json.dump(gt, f, indent=1)
    return ply, gt


# ------------------------------------------------------------------------------- RGB-D rendering
def raycast_depth(surfaces, K, cam_to_world, w, h):
    """Depth (z in camera frame, metres) and surface index per pixel, OpenCV camera convention."""
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    rays_c = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    R, c = cam_to_world[:3, :3], cam_to_world[:3, 3]
    rays = rays_c @ R.T
    depth = np.full(len(rays), np.inf)
    sid = np.full(len(rays), -1)
    uvs = np.zeros((len(rays), 2))
    for k, s in enumerate(surfaces):
        n = s.N
        den = rays @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((s.origin - c) @ n) / den
        ok = np.isfinite(t) & (t > 0.05) & (t < depth)
        if not ok.any():
            continue
        hit = c + rays[ok] * t[ok, None]
        rel = hit - s.origin
        uv = np.stack([rel @ s.U, rel @ s.V], 1)
        inside = points_in_polygon(uv, s.poly)
        for (u0, v0, u1, v1) in s.holes:
            inside &= ~((uv[:, 0] > u0) & (uv[:, 0] < u1) & (uv[:, 1] > v0) & (uv[:, 1] < v1))
        idx = np.flatnonzero(ok)[inside]
        depth[idx] = t[idx]
        sid[idx] = k
        uvs[idx] = uv[inside]
    return depth.reshape(h, w), sid.reshape(h, w), uvs.reshape(h, w, 2)


def surface_texture(uv, sid):
    """Texture fixed to the surfaces (so photometric odometry has consistent gradients):
    5 cm tiles with a hashed grey level plus a smooth pattern."""
    iu = np.floor(uv[..., 0] / 0.05).astype(np.int64)
    iv = np.floor(uv[..., 1] / 0.05).astype(np.int64)
    hsh = ((iu * 73856093) ^ (iv * 19349663) ^ (sid.astype(np.int64) * 83492791)) % 1000 / 1000.0
    return 0.75 + 0.18 * hsh + 0.07 * np.sin(9 * uv[..., 0]) * np.sin(7 * uv[..., 1])


def look_at(eye, target, up=(0, 1, 0)):
    """OpenCV camera-to-world: +z forward, +y down."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))          # right = forward x up
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = eye
    return T


def make_rgbd(spec: RoomSpec, out_dir: str, seed: int = 0, n_frames: int = 80, up: str = "y",
              depth_noise: float = 0.004, with_poses: bool = True):
    """Render ARKit-style frames: frame_XXXXX.jpg (512x384), depth_XXXXX.png (uint16 mm, 256x192),
    frame_XXXXX.json with 'intrinsics' (for the RGB image) and row-major 'cameraPoseARFrame'."""
    rng = np.random.default_rng(seed)
    surfaces, _, _ = build_scene(spec)
    C = np.asarray(spec.corners, float)
    centre = C.mean(0)
    T = random_pose(np.random.default_rng(seed + 1000), up)
    os.makedirs(out_dir, exist_ok=True)
    W, H = 256, 192
    K = np.array([[212.0, 0, 127.5], [0, 212.0, 95.5], [0, 0, 1]])
    flip = np.diag([1.0, -1, -1, 1])                 # OpenCV camera <-> ARKit camera axes
    for i in range(n_frames):
        # two loops around the room: first looking down at the floor/wall feet, then up
        half = n_frames // 2
        a = 2 * np.pi * (i % half) / half
        s_ = np.clip((i - half + 10) / 20.0, 0, 1)
        pitch = -0.4 + 0.8 * s_ * s_ * (3 - 2 * s_)   # smooth transition between the loops
        eye = np.array([centre[0] + 0.3 * np.cos(a), 1.4 + 0.1 * np.sin(3 * a), centre[1] + 0.3 * np.sin(a)])
        tgt = eye + np.array([np.cos(a + 1.2), pitch, np.sin(a + 1.2)])
        pose = look_at(eye, tgt)
        depth, sid, uv = raycast_depth(surfaces, K, pose, W, H)
        col = np.zeros((H, W, 3))
        for k, s in enumerate(surfaces):
            col[sid == k] = s.color
        col = np.clip(col * surface_texture(uv, sid)[..., None], 0, 1)
        col[sid < 0] = 0
        d = depth + rng.normal(0, depth_noise, depth.shape)
        d[~np.isfinite(depth)] = 0
        d_mm = np.clip(d * 1000, 0, 65535).astype(np.uint16)
        rgb = cv2.resize((col[..., ::-1] * 255).astype(np.uint8), (2 * W, 2 * H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:05d}.jpg"), rgb, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(out_dir, f"depth_{i:05d}.png"), d_mm)
        pose_scan = T @ pose @ flip                       # ARKit camera-to-world in the scan frame
        if i == 0:
            first_cv = T @ pose                           # OpenCV camera 0 -> scan frame
        Krgb = K.copy()
        Krgb[:2] *= 2
        Krgb[0, 2], Krgb[1, 2] = (K[0, 2] + 0.5) * 2 - 0.5, (K[1, 2] + 0.5) * 2 - 0.5
        meta = {"intrinsics": Krgb.ravel().tolist(), "imageWidth": 2 * W, "imageHeight": 2 * H,
                "depthScale": 1000.0}
        if with_poses:
            meta["cameraPoseARFrame"] = pose_scan.ravel().tolist()
        with open(os.path.join(out_dir, f"frame_{i:05d}.json"), "w") as f:
            json.dump(meta, f)
    gt = ground_truth(spec)
    # without poses the fused cloud lives in camera 0's (OpenCV) frame
    to_scan = T if with_poses else np.linalg.inv(first_cv) @ T
    gt.update({"gt_to_scan": to_scan.tolist(), "up": up, "frames": n_frames, "poses": with_poses})
    with open(os.path.join(out_dir, "gt.json"), "w") as f:
        json.dump(gt, f, indent=1)
    return out_dir, gt


def make_aruco_frames(out_dir, size=0.15, n=4, depth_scale=1.0, seed=0, depth_noise=0.002):
    """Photos of a printed ArUco marker (DICT_4X4_50 id 7) on a wall, with LiDAR-like depth.

    depth_scale != 1 simulates a mis-scaled depth sensor, which the scale check must report."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    W, H = 1024, 768
    K = np.array([[860.0, 0, (W - 1) / 2], [0, 860.0, (H - 1) / 2], [0, 0, 1]])
    dw, dh = 256, 192
    Kd = K.copy()
    Kd[:2] *= dw / W
    Kd[0, 2], Kd[1, 2] = (K[0, 2] + 0.5) * dw / W - 0.5, (K[1, 2] + 0.5) * dh / H - 0.5
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 600, borderBits=1)
    pad = 120                                          # white quiet zone around the marker
    tex = cv2.copyMakeBorder(marker, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    half = size / 2 * (600 + 2 * pad) / 600
    for i in range(n):
        dist = rng.uniform(0.8, 1.6)
        yaw, pitch = np.radians(rng.uniform(-30, 30)), np.radians(rng.uniform(-20, 20))
        c = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), dist])
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
        R = Ry @ Rx
        ex, ey, nz = R[:, 0], R[:, 1], R[:, 2]          # marker x right, y down (image-like)
        quad3 = np.array([c - half * ex - half * ey, c + half * ex - half * ey, c + half * ex + half * ey,
                          c - half * ex + half * ey])
        q2 = (quad3 @ K.T)[:, :2] / quad3[:, 2:]
        tw, th = tex.shape[1], tex.shape[0]
        src = np.array([[-0.5, -0.5], [tw - 0.5, -0.5], [tw - 0.5, th - 0.5], [-0.5, th - 0.5]], np.float32)
        Hm = cv2.getPerspectiveTransform(src, q2.astype(np.float32))
        img = np.full((H, W), 128, np.uint8)
        warped = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=128)
        cover = cv2.warpPerspective(np.full(tex.shape, 255, np.uint8), Hm, (W, H)) > 0
        img[cover] = warped[cover]
        img = np.clip(img + rng.normal(0, 2, img.shape), 0, 255).astype(np.uint8)
        u, v = np.meshgrid(np.arange(dw), np.arange(dh))
        rays = np.stack([(u - Kd[0, 2]) / Kd[0, 0], (v - Kd[1, 2]) / Kd[1, 1], np.ones_like(u, float)], -1)
        z = (c @ nz) / (rays @ nz)                       # wall plane through the marker
        d = z * depth_scale + rng.normal(0, depth_noise, z.shape)
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(os.path.join(out_dir, f"depth_{i:05d}.png"), np.clip(d * 1000, 0, 65535).astype(np.uint16))
        with open(os.path.join(out_dir, f"frame_{i:05d}.json"), "w") as f:
            json.dump({"intrinsics": K.ravel().tolist(), "imageWidth": W, "imageHeight": H, "depthScale": 1000.0}, f)
    return out_dir


# Default dataset: one per room type, alternating gravity conventions
DATASET = [("rect", 1, "y"), ("lshape", 2, "z"), ("angled", 3, "y"), ("alcove", 4, "z")]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate synthetic room scans with ground truth.")
    ap.add_argument("--out", default="D:/roomtwin_data/synth")
    ap.add_argument("--spacing", type=float, default=0.015, help="sample spacing in metres")
    ap.add_argument("--big", action="store_true", help="also write a ~2M point version of 'rect'")
    ap.add_argument("--rgbd", nargs="*", default=[], help="rooms to render as RGB-D frame folders")
    a = ap.parse_args(argv)
    for name, seed, up in DATASET:
        ply, gt = make_room(ROOMS[name], a.out, seed, up, a.spacing)
        print(f"{ply}: {gt['points']} points, {up}-up")
    if a.big:
        ply, gt = make_room(ROOMS["rect"], a.out, 11, "z", 0.0058, suffix="_big")
        print(f"{ply}: {gt['points']} points")
    for name in a.rgbd:
        d, _ = make_rgbd(ROOMS[name], os.path.join(a.out, f"{name}_rgbd"), seed=5)
        print(f"{d}: RGB-D frames")


if __name__ == "__main__":
    main()
