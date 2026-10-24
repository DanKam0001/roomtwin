"""RGB-D frame folders -> fused point cloud with OpenCV 5 cv2.Volume (HashTSDF) and cv2.Odometry.

Accepted layouts (auto-detected, first match wins):

  A. ARKit-style, one JSON per frame (3D Scanner App / Record3D-like exports, roomtwin.synth):
       frame_00012.jpg   depth_00012.png|.exr   frame_00012.json      (files may also sit in
       rgb/, depth/ subfolders; frames are matched by the last integer in the file name)
     JSON keys: "intrinsics" (9 values or 3x3, for the RGB image; or {"fx","fy","cx","cy"}),
       optional "imageWidth"/"imageHeight" (size the intrinsics refer to), camera-to-world pose in
       one of "cameraPoseARFrame" | "pose" | "transform" | "camera_to_world" (16 values row-major,
       column-major is detected, or 4x4), optional "depthScale" (uint16 units per metre).
     Poses use the ARKit camera convention (x right, y up, looking down -z).
  B. ScanNet-style: color/*.jpg, depth/*.png, pose/*.txt (4x4 camera-to-world, OpenCV camera
     convention), intrinsic/intrinsic_depth.txt (4x4 or 3x3).
  C. TUM RGB-D: rgb.txt, depth.txt (+ groundtruth.txt "t tx ty tz qx qy qz qw"), depth scale 5000.

  An optional roomtwin_frames.json overrides {"convention": "arkit"|"opencv", "depth_scale": 1000,
  "intrinsics": [fx, fy, cx, cy] (depth image)}.

If any pose is missing, all poses are estimated with cv2.Odometry (RGB_DEPTH) frame-to-model:
each new frame is registered against a raycast of the TSDF built so far (KinectFusion style).
If poses are given, cv2.Odometry is still run between consecutive frames as a tracking check and
frames whose given motion disagrees with it are reported.

Notes on OpenCV 5.0.0 (all verified with synthetic planes / rooms, scripts in tests/):
  * the dense VolumeType_TSDF CPU integrator applies the camera's world-z translation twice;
  * HashTSDF.fetchPointsNormals returns every voxel of the truncation band, not the zero crossing;
  * HashTSDF.raycast surfaces are offset by half a voxel along (+x, +y, -z) of the volume frame.
So we use HashTSDF, extract the surface by raycasting from the input poses, and remove the offset
(it would otherwise also bias frame-to-model odometry by half a voxel per frame).
"""
from __future__ import annotations

import glob
import json
import os

import cv2
import numpy as np

from .geometry import voxel_downsample
from .io_ply import trailing_int

ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])     # camera axes flip (right-multiplied)
# Measured on synthetic scenes (voxel 1 cm and 2 cm, see README): HashTSDF raycast surfaces are
# offset by half a voxel along (+x, +y, -z) of the volume (world) frame. We subtract it.
RAYCAST_OFFSET_VOX = np.array([0.5, 0.5, -0.5])
IMG_EXT = (".jpg", ".jpeg", ".png")
DEPTH_EXT = (".png", ".exr", ".tif", ".tiff")


def _as_K(v):
    if isinstance(v, dict):
        return np.array([[v["fx"], 0, v["cx"]], [0, v["fy"], v["cy"]], [0, 0, 1.0]])
    a = np.asarray(v, float).ravel()
    if a.size == 4:
        return np.array([[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1.0]])
    if a.size == 9:
        K = a.reshape(3, 3)
        return K.T if abs(K[2, 0]) + abs(K[2, 1]) > 1 else K     # column-major export
    if a.size == 16:
        return a.reshape(4, 4)[:3, :3]
    raise ValueError("unrecognised intrinsics")


def _as_pose(v):
    a = np.asarray(v, float).ravel()
    if a.size != 16:
        raise ValueError("pose must have 16 values")
    T = a.reshape(4, 4)
    if np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        return T
    if np.allclose(T[:, 3], [0, 0, 0, 1], atol=1e-6):
        return T.T                                   # column-major (simd_float4x4 dump)
    raise ValueError("pose is not a rigid 4x4")


def _load_depth(path, scale):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise IOError(path)
    if d.ndim == 3:
        d = d[..., 0]
    if d.dtype == np.uint16 or d.dtype == np.int32:
        d = d.astype(np.float32) / scale
    d = d.astype(np.float32)
    d[~np.isfinite(d)] = 0
    return d


def _layout_arkit(folder, cfg):
    jsons = [p for p in glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)
             if os.path.basename(p) != "roomtwin_frames.json" and trailing_int(p) is not None]
    if not jsons:
        return None
    files = glob.glob(os.path.join(folder, "**", "*.*"), recursive=True)
    depth = {trailing_int(p): p for p in files if "depth" in p.lower().replace(folder.lower(), "")
             and p.lower().endswith(DEPTH_EXT) and "conf" not in os.path.basename(p).lower()}
    rgb = {trailing_int(p): p for p in files if p.lower().endswith(IMG_EXT)
           and "depth" not in p.lower().replace(folder.lower(), "") and "conf" not in os.path.basename(p).lower()}
    frames = []
    for jp in sorted(jsons, key=trailing_int):
        i = trailing_int(jp)
        if i not in depth:
            continue
        meta = json.load(open(jp))
        K = _as_K(meta.get("intrinsics") or meta.get("K") or meta["camera_intrinsics"])
        pose = None
        for key in ("cameraPoseARFrame", "pose", "transform", "camera_to_world", "cameraTransform"):
            if key in meta:
                pose = _as_pose(meta[key])
                break
        size = (meta.get("imageWidth") or meta.get("width"), meta.get("imageHeight") or meta.get("height"))
        frames.append(dict(index=i, rgb=rgb.get(i), depth=depth[i], K=K, K_size=size, pose=pose,
                           depth_scale=meta.get("depthScale", cfg.get("depth_scale", 1000.0))))
    return dict(frames=frames, convention=cfg.get("convention", "arkit"), layout="arkit-json") if frames else None


def _layout_scannet(folder, cfg):
    if not (os.path.isdir(os.path.join(folder, "depth")) and os.path.isdir(os.path.join(folder, "pose"))):
        return None
    Kf = os.path.join(folder, "intrinsic", "intrinsic_depth.txt")
    K = _as_K(np.loadtxt(Kf)) if os.path.exists(Kf) else _as_K(cfg["intrinsics"])
    color = {trailing_int(p): p for p in glob.glob(os.path.join(folder, "color", "*")) if p.lower().endswith(IMG_EXT)}
    frames = []
    for dp in sorted(glob.glob(os.path.join(folder, "depth", "*")), key=lambda p: trailing_int(p) or 0):
        i = trailing_int(dp)
        pp = os.path.join(folder, "pose", f"{os.path.splitext(os.path.basename(dp))[0]}.txt")
        pose = np.loadtxt(pp) if os.path.exists(pp) else None
        if pose is not None and not np.all(np.isfinite(pose)):
            pose = None
        frames.append(dict(index=i, rgb=color.get(i), depth=dp, K=K, K_size=(None, None), pose=pose,
                           depth_scale=cfg.get("depth_scale", 1000.0)))
    return dict(frames=frames, convention=cfg.get("convention", "opencv"), layout="scannet") if frames else None


def _read_tum_list(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        rows.append(line.split())
    return rows


def _quat_pose(t, q):
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _layout_tum(folder, cfg):
    if not (os.path.exists(os.path.join(folder, "rgb.txt")) and os.path.exists(os.path.join(folder, "depth.txt"))):
        return None
    rgb = [(float(r[0]), os.path.join(folder, r[1])) for r in _read_tum_list(os.path.join(folder, "rgb.txt"))]
    dep = [(float(r[0]), os.path.join(folder, r[1])) for r in _read_tum_list(os.path.join(folder, "depth.txt"))]
    gt = []
    gp = os.path.join(folder, "groundtruth.txt")
    if os.path.exists(gp):
        gt = [(float(r[0]), np.array(r[1:4], float), np.array(r[4:8], float)) for r in _read_tum_list(gp)]
    name = folder.lower()
    if "freiburg2" in name:
        K = [520.9, 521.0, 325.1, 249.7]
    elif "freiburg3" in name:
        K = [535.4, 539.2, 320.1, 247.6]
    else:
        K = [517.3, 516.5, 318.6, 255.3]
    K = _as_K(cfg.get("intrinsics", K))
    rt = np.array([r[0] for r in rgb])
    gtt = np.array([g[0] for g in gt]) if gt else None
    frames = []
    for k, (t, dp) in enumerate(dep):
        j = int(np.argmin(np.abs(rt - t)))
        pose = None
        if gt:
            g = int(np.argmin(np.abs(gtt - t)))
            if abs(gtt[g] - t) < 0.02:
                pose = _quat_pose(gt[g][1], gt[g][2])
        frames.append(dict(index=k, rgb=rgb[j][1] if abs(rt[j] - t) < 0.02 else None, depth=dp, K=K,
                           K_size=(None, None), pose=pose, depth_scale=cfg.get("depth_scale", 5000.0)))
    return dict(frames=frames, convention="opencv", layout="tum")


def discover(folder):
    cfg = {}
    mp = os.path.join(folder, "roomtwin_frames.json")
    if os.path.exists(mp):
        cfg = json.load(open(mp))
    for fn in (_layout_tum, _layout_scannet, _layout_arkit):
        res = fn(folder, cfg)
        if res:
            return res
    raise ValueError(f"no supported RGB-D layout found in {folder}")


def _frame_K(fr, depth_shape):
    """Intrinsics scaled to the depth image (ARKit gives them for the full-size RGB image)."""
    K = fr["K"].copy()
    h, w = depth_shape
    W, H = fr["K_size"]
    if (W is None or H is None) and fr["rgb"]:
        img = cv2.imread(fr["rgb"], cv2.IMREAD_REDUCED_COLOR_2)
        if img is not None:
            H, W = img.shape[0] * 2, img.shape[1] * 2
    if W and H and (W != w or H != h):
        sx, sy = w / W, h / H
        K[0, 0] *= sx
        K[1, 1] *= sy
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K


def _make_volume(K, w, h, voxel, max_depth):
    s = cv2.VolumeSettings(cv2.VolumeType_HashTSDF)
    s.setIntegrateWidth(w)
    s.setIntegrateHeight(h)
    s.setRaycastWidth(w)
    s.setRaycastHeight(h)
    s.setCameraIntegrateIntrinsics(K)
    s.setCameraRaycastIntrinsics(K)
    s.setVoxelSize(voxel)
    s.setTsdfTruncateDistance(4 * voxel)
    s.setMaxDepth(max_depth)
    s.setDepthFactor(1.0)                             # we pass float metres
    return cv2.Volume(cv2.VolumeType_HashTSDF, s)


def _odometry(K):
    st = cv2.OdometrySettings()
    st.setCameraMatrix(K.astype(np.float32))       # must be CV_32F
    st.setMaxDepth(4.5)
    return cv2.Odometry(cv2.OdometryType_RGB_DEPTH, st, cv2.OdometryAlgoType_COMMON)


def _odo_frame(odo, depth, gray):
    f = cv2.OdometryFrame(depth=depth, image=gray)
    odo.prepareFrame(f)
    return f


def _model_depth(vol, pose, K, w, h, voxel):
    """Depth image of the fused model seen from `pose`: raycast, remove the half-voxel offset,
    re-project to pixels (nearest)."""
    pts, _ = vol.raycast(pose.astype(np.float32))
    Q = np.asarray(pts)[..., :3].reshape(-1, 3)
    ok = np.all(np.isfinite(Q), 1) & (Q[:, 2] > 0.1)
    Q = Q[ok] - pose[:3, :3].T @ (RAYCAST_OFFSET_VOX * voxel)
    u = np.round(K[0, 0] * Q[:, 0] / Q[:, 2] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * Q[:, 1] / Q[:, 2] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    depth = np.zeros((h, w), np.float32)
    depth[v[inside], u[inside]] = Q[inside, 2]
    return depth


def _plausible(Rt, last, max_dt=0.04, max_ddeg=4.0):
    """Reject odometry results that a hand-held phone cannot produce: the inter-frame motion may
    differ from the previous inter-frame motion (constant-velocity prediction) only a little."""
    D = np.linalg.inv(last) @ Rt
    ang = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
    return np.linalg.norm(D[:3, 3]) < max_dt and ang < max_ddeg


def fuse_folder(folder, voxel=0.01, max_depth=4.0, raycast_every=None, max_frames=600, log=print):
    """Returns (points Nx3, colors Nx3 in [0,1], source dict)."""
    info = discover(folder)
    frames = info["frames"]
    if len(frames) > max_frames:                      # subsample very long captures
        step = int(np.ceil(len(frames) / max_frames))
        frames = frames[::step]
    d0 = _load_depth(frames[0]["depth"], frames[0]["depth_scale"])
    h, w = d0.shape
    K = _frame_K(frames[0], (h, w))
    have_poses = all(f["pose"] is not None for f in frames)
    vol = _make_volume(K, w, h, voxel, max_depth)
    odo = _odometry(K)
    poses, colors_img, depth_imgs, track_err = [], [], [], []
    prev = None
    last_motion, n_lost = np.eye(4), 0
    for k, fr in enumerate(frames):
        depth = _load_depth(fr["depth"], fr["depth_scale"])
        depth[(depth < 0.1) | (depth > max_depth)] = 0
        rgb = cv2.imread(fr["rgb"]) if fr["rgb"] else None
        if rgb is None:
            rgb = np.full((h, w, 3), 160, np.uint8)
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        cur = _odo_frame(odo, depth, gray)
        if have_poses:
            T = fr["pose"] @ ARKIT_TO_CV if info["convention"] == "arkit" else fr["pose"]
            if prev is not None:                      # tracking check against visual odometry
                ok, Rt = odo.compute(cur, prev[1])
                if ok:
                    rel_given = np.linalg.inv(prev[0]) @ T
                    track_err.append(float(np.linalg.norm(rel_given[:3, 3] - Rt[:3, 3])))
        else:
            if prev is None:
                T = np.eye(4)
            else:
                # frame-to-model: register against the TSDF raycast at the previous pose
                model = _odo_frame(odo, _model_depth(vol, prev[0], K, w, h, voxel), prev[2])
                ok, Rt = odo.compute(cur, model)
                if not (ok and _plausible(Rt, last_motion)):
                    ok, Rt = odo.compute(cur, prev[1])      # fall back to frame-to-frame
                if not (ok and _plausible(Rt, last_motion)):
                    Rt, n_lost = last_motion, n_lost + 1    # constant-velocity guess
                T = prev[0] @ Rt
                last_motion = Rt
        vol.integrate(depth, T.astype(np.float32))
        prev = (T, cur, gray)
        poses.append(T)
        colors_img.append(rgb)
    # surface extraction: raycast the fused TSDF from (a subset of) the input viewpoints
    every = raycast_every or max(1, len(poses) // 40)
    pts_all, col_all = [], []
    for k in range(0, len(poses), every):
        P, _ = vol.raycast(poses[k].astype(np.float32))
        P = np.asarray(P)[..., :3].reshape(-1, 3)
        valid = np.all(np.isfinite(P), 1) & (P[:, 2] > 0.1)
        Pw = P[valid] @ poses[k][:3, :3].T + poses[k][:3, 3] - RAYCAST_OFFSET_VOX * voxel
        pts_all.append(Pw)
        col_all.append(colors_img[k].reshape(-1, 3)[valid][:, ::-1] / 255.0)
    P = np.concatenate(pts_all)
    C = np.concatenate(col_all)
    P, C, _ = voxel_downsample(P, voxel / 2, C)
    src = dict(type="rgbd", file=os.path.basename(os.path.normpath(folder)), layout=info["layout"],
               frames=len(frames), poses="given" if have_poses else "cv2.Odometry frame-to-model",
               tsdf=dict(type="HashTSDF", voxel=voxel, truncation=4 * voxel))
    if not have_poses:
        src["odometry_lost_frames"] = n_lost
        # the frame is the first camera's: use the mean camera 'up' (-y in OpenCV) as a gravity hint
        up = -np.mean([T[:3, 1] for T in poses], axis=0)
        src["up_hint"] = (up / np.linalg.norm(up)).round(4).tolist()
    if track_err:
        e = np.asarray(track_err)
        src["tracking_check"] = dict(median_m=round(float(np.median(e)), 4), worst_frame=int(np.argmax(e)) + 1,
                                     worst_m=round(float(e.max()), 4), frames_over_5cm=int((e > 0.05).sum()))
    log(f"[roomtwin] fused {len(frames)} frames ({src['poses']}) into {len(P)} points")
    return P, C, src
