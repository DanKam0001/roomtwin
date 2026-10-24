"""Scan -> room.json (primary room) + house.json (every room of the scan, shared frame, doorways).

    python -m roomtwin.process scan.ply -o out/
    python -m roomtwin.process frames_dir/ -o out/ [--aruco photos_dir --marker-size 0.15]

Stages (all in metres):
  1. load (cv2.loadPointCloud / PLY fallback, or RGB-D frames fused with cv2.Volume HashTSDF)
  2. clean: 1 cm voxel centroids (measurement cloud), grid radius-outlier removal, 2 cm working cloud
  3. normals (local PCA), gravity from plane-normal clusters + floor MSAC, level so floor is y = 0
  4. walls: vertical-plane MSAC -> cv2.fitLine, merge, outline ordering, corners, sigma
  5. obstacles, 6. openings + coverage per wall, 7. QA, 8. exports
"""
from __future__ import annotations

import argparse
import json
import os
import time

import cv2
import numpy as np

from . import __version__
from .geometry import estimate_normals, polygon_area, remove_outliers, voxel_downsample
from .gravity import detect_gravity, floor_and_ceiling
from .io_ply import load_cloud
from .obstacles import find_obstacles
from .openings import analyze_wall
from .qa import build_qa, surface_coverage
from .walls import Line, build_footprint, extract_wall_lines, merge_coplanar


def prepare_cloud(P_raw, C_raw=None, source=None, k_scale=0.003, seed=0, log=print):
    """Clean, normals, gravity, floor at y = 0 and the global ceiling estimate (shared by all rooms)."""
    rng = np.random.default_rng(seed)
    timings = {}
    t0 = time.perf_counter()

    def tick(name):
        nonlocal t0
        t = time.perf_counter()
        timings[name] = round(t - t0, 3)
        t0 = t

    ok = np.all(np.isfinite(P_raw), 1)
    P_raw = P_raw[ok]
    C_raw = C_raw[ok] if C_raw is not None else None
    # 1 cm measurement cloud (voxel centroids) and outlier removal
    P1, C1, _ = voxel_downsample(P_raw, 0.01, C_raw)
    keep = remove_outliers(P1, cell=0.05, min_neighbours=8)
    P1 = P1[keep]
    C1 = C1[keep] if C1 is not None else None
    # 2 cm working cloud for segmentation
    P2, _, _ = voxel_downsample(P1, 0.02)
    tick("clean")
    N2, _ = estimate_normals(P2, cell=0.08)
    tick("normals")

    # gravity: coarse from normal clusters, exact from the floor plane
    hint = (source or {}).get("up_hint")
    R1, ginfo = detect_gravity(P2, N2, rng, np.asarray(hint) if hint is not None else None)
    P2r, N2r = P2 @ R1.T, N2 @ R1.T
    floor, ceiling, R2 = floor_and_ceiling(P2r, N2r, rng)
    R = R2 @ R1
    P2 = P2 @ R.T
    N2 = N2 @ R.T
    P1 = P1 @ R.T
    horiz = ~np.isnan(N2[:, 0]) & (np.abs(np.nan_to_num(N2[:, 1])) > np.cos(np.radians(15)))
    y0, _ = level_height(P2, horiz, float(np.median(P2[floor["mask"] & horiz, 1])))
    P2[:, 1] -= y0
    P1[:, 1] -= y0
    floor_mask2 = np.abs(P2[:, 1]) < 0.03
    fsel = floor_mask2 & horiz
    fy = P2[fsel, 1]
    sig_f = float(np.std(fy) / np.sqrt(max(len(np.unique(np.floor(P2[fsel][:, [0, 2]] / 0.02), axis=0)), 1)))
    H_global = None
    if ceiling is not None:
        H_global, _ = level_height(P2, horiz, float(np.median(P2[ceiling["mask"] & horiz, 1])))
    tick("gravity_floor")
    return dict(P1=P1, C1=C1, P2=P2, N2=N2, horiz=horiz, floor_mask2=floor_mask2, R=R, y0=y0, sig_f=sig_f,
                H_global=H_global, ginfo=ginfo, floor=floor, n_raw=int(len(P_raw)), timings=timings,
                k_scale=k_scale, seed=seed, source=source)


def level_height(P2, horiz, y_init, window=0.04):
    """Median of horizontal-surface points in a wide symmetric window, iterated: unbiased by the
    MSAC plane's initial offset (a narrow truncation window would keep part of that offset)."""
    y = y_init
    sel = horiz & (np.abs(P2[:, 1] - y) < window)
    for _ in range(3):
        sel = horiz & (np.abs(P2[:, 1] - y) < window)
        if not sel.any():
            break
        y = float(np.median(P2[sel, 1]))
    return y, sel


def room_ceiling(prep, sel2):
    """Ceiling height of one room: the highest well-supported horizontal layer above 1.8 m among
    this room's points (rooms of one flat can differ), else the global ceiling, else the 99.5th
    percentile of the room's points. Returns (H, sigma_H, mask over P2 or None)."""
    P2, horiz, k_scale = prep["P2"], prep["horiz"], prep["k_scale"]
    hs = sel2 & horiz & (P2[:, 1] > 1.8)
    H = None
    if hs.sum() >= 50:
        y = P2[hs, 1]
        hist, edges = np.histogram(y, bins=np.arange(1.8, y.max() + 0.03, 0.02))
        if hist.max() >= 50:
            strong = np.flatnonzero(hist >= max(0.2 * hist.max(), 50))
            H = 0.5 * (edges[strong[-1]] + edges[strong[-1] + 1])
    if prep["H_global"] is not None and (H is None or abs(H - prep["H_global"]) < 0.06):
        H = prep["H_global"]                          # same ceiling as the whole scan
    if H is None:
        H = float(np.percentile(P2[sel2, 1], 99.5)) if sel2.any() else 2.4
        return H, float(np.sqrt(prep["sig_f"] ** 2 + 0.05 ** 2 + (k_scale * H) ** 2)), None
    H, _ = level_height(P2, horiz & sel2, H)
    cmask = sel2 & horiz & (np.abs(P2[:, 1] - H) < 0.04)
    if cmask.sum() < 30:
        return H, float(np.sqrt(prep["sig_f"] ** 2 + 0.05 ** 2 + (k_scale * H) ** 2)), None
    cy = P2[cmask, 1]
    sig_c = float(np.std(cy) / np.sqrt(max(len(np.unique(np.floor(P2[cmask][:, [0, 2]] / 0.02), axis=0)), 1)))
    return H, float(np.sqrt(prep["sig_f"] ** 2 + sig_c ** 2 + (k_scale * H) ** 2)), cmask


def fit_room(prep, sel2=None, frame=None, log=print, P1_room=None):
    """Walls, openings, obstacles, coverage and QA for one room.

    sel2: mask over prep["P2"] (the room's working points, walls included); None = everything.
    frame: (yaw, shift) of a shared house frame; None = this room defines its own frame (best wall
    along +x, footprint starting at the origin). Returns (room dict, extra) in that frame."""
    P2a, N2a = prep["P2"], prep["N2"]
    if sel2 is None:
        sel2 = np.ones(len(P2a), bool)
    k_scale = prep["k_scale"]
    rng = np.random.default_rng(prep["seed"])
    timings = {}
    t0 = time.perf_counter()

    def tick(name):
        nonlocal t0
        t = time.perf_counter()
        timings[name] = round(t - t0, 3)
        t0 = t

    H, sigma_H, cmask_all = room_ceiling(prep, sel2)
    ceiling = cmask_all is not None
    P2, N2 = P2a[sel2], N2a[sel2]
    floor_mask2 = np.abs(P2[:, 1]) < 0.03
    P1 = prep["P1"] if P1_room is None else P1_room
    R, ginfo = prep["R"], prep["ginfo"]
    t_pre = np.asarray(prep.get("t_pre", (0.0, -prep["y0"], 0.0)), float)
    log(f"[roomtwin] ceiling {H:.3f} m, {int(sel2.sum())} pts (2 cm) in this room")

    # walls
    cand = extract_wall_lines(P2, N2, H, rng)
    cand = merge_coplanar(cand, P2, floor_mask2)
    if not cand:
        raise RuntimeError("no walls found")
    if frame is None:
        # yaw: align the best-supported wall with +x so furniture boxes are axis-aligned
        main = max(cand, key=lambda w: len(w["idx"]))
        yaw = -np.arctan2(main["line"].d[1], main["line"].d[0])
    else:
        yaw = frame[0]
    c, s = np.cos(yaw), np.sin(yaw)
    R2d = np.array([[c, -s], [s, c]])                 # acts on (x, z)
    R3 = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])  # same rotation lifted to 3D (x, z rows)
    P1, P2, N2 = P1 @ R3.T, P2 @ R3.T, N2 @ R3.T
    for w in cand:
        w["line"] = Line(R2d @ w["line"].p, R2d @ w["line"].d)
    out, poly, rejected = build_footprint(cand, P2, H, floor_mask2, k_scale=k_scale, P_meas=P1)
    # translate so the footprint starts at the origin (or by the shared house frame's shift)
    shift = poly.min(0) if frame is None else np.asarray(frame[1], float)
    poly = poly - shift
    P1[:, [0, 2]] -= shift
    P2[:, [0, 2]] -= shift
    for o in out:
        o["a"], o["b"] = o["a"] - shift, o["b"] - shift
        if o["wall"] is not None:
            o["wall"]["line"] = Line(o["wall"]["line"].p - shift, o["wall"]["line"].d)
    # scan -> room: levelling rotation R and offset t_pre (floor at y = 0), then yaw R3, then shift
    T = np.eye(4)
    T[:3, :3] = R3 @ R
    T[:3, 3] = R3 @ t_pre - np.array([shift[0], 0.0, shift[1]])
    tick("walls")

    walls = []
    cpts = (P2a[cmask_all] @ R3.T - np.array([shift[0], 0, shift[1]])) if ceiling else np.zeros((0, 3))
    for i, o in enumerate(out):
        a, b = o["a"], o["b"]
        u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        st = o["wall"]["stats"] if o["wall"] is not None else dict(rms=0.0, n_eff=0)
        # local wall height: ceiling points within 60 cm of this wall (ceilings are not always level)
        hw = H
        if len(cpts):
            q = cpts[:, [0, 2]] - a
            t, dn = q @ u, q @ np.array([-u[1], u[0]])
            near = (t > 0) & (t < o["length"]) & (dn > 0) & (dn < 0.6)
            if near.sum() >= 50:
                hw = float(np.median(cpts[near, 1]))
        walls.append(dict(id=f"w{i}", a=a, b=b, length=o["length"], sigma=o["sigma"], height=hw,
                          normal=np.array([-u[1], u[0]]), inlier_rms=st["rms"], n_eff=st["n_eff"],
                          inferred=o["inferred"]))
    band = {w["id"]: max(3.0 * w["inlier_rms"], 0.025) for w in walls}
    obstacles, struct = find_obstacles(P1, walls, poly, H, band)
    tick("obstacles")
    wall_results, openings = {}, []
    counters = {"door": 0, "window": 0, "opening": 0}
    prefix = {"door": "d", "window": "win", "opening": "op"}
    for w in walls:
        res = analyze_wall(w["id"], w["a"], w["b"], H, P1, band[w["id"]], obstacles, w["inlier_rms"] or 0.01)
        for op in res["openings"]:
            op["id"] = f"{prefix[op['type']]}{counters[op['type']]}"
            counters[op["type"]] += 1
            openings.append(op)
        wall_results[w["id"]] = res
    tick("openings")
    floor_cov = surface_coverage(P1, poly, np.abs(P1[:, 1]) < 0.03, obstacles)
    ceil_cov = surface_coverage(P1, poly, np.abs(P1[:, 1] - H) < 0.04) if ceiling else 0.0
    wall_area = sum(w["length"] * H for w in walls)
    wall_cov = sum(wall_results[w["id"]]["coverage"] * w["length"] * H for w in walls) / max(wall_area, 1e-9)
    score = 0.6 * wall_cov + 0.25 * floor_cov + 0.15 * ceil_cov
    qa, gaps = build_qa(walls, wall_results, openings, ceiling or None, floor_cov, ceil_cov, ginfo)
    tick("qa")

    r4 = lambda v: [round(float(x), 4) for x in v]
    room = {
        "version": 1,
        "generator": f"roomtwin {__version__} / OpenCV {cv2.__version__}",
        "source": dict(prep.get("source") or {}, points=prep["n_raw"]),
        "frame": "metres; y up, floor at y = 0; polygon counter-clockwise in (x, z); wall normals point into the room",
        "transform": [r4(row) for row in T],
        "floor": {"polygon": [r4(p) for p in poly], "area_m2": round(abs(polygon_area(poly)), 3)},
        "ceiling_height": {"value": round(H, 4), "sigma": round(sigma_H, 4), "observed": bool(ceiling)},
        "walls": [{"id": w["id"], "a": r4(w["a"]), "b": r4(w["b"]), "length": round(w["length"], 4),
                   "sigma": round(w["sigma"], 4), "height": round(w["height"], 4), "normal": r4(w["normal"]),
                   "inlier_rms": round(w["inlier_rms"], 4), "n_eff": w["n_eff"], "inferred": w["inferred"],
                   "coverage": round(wall_results[w["id"]]["coverage"], 3)} for w in walls],
        "openings": [{"id": o["id"], "type": o["type"], "wall": o["wall"], "offset": round(o["offset"], 4),
                      "width": round(o["width"], 4), "sill": round(o["sill"], 4), "height": round(o["height"], 4),
                      "sigma": o["sigma"], "see_through": o["see_through"], "state": o["state"],
                      "refined": o["refined"], "hinge_side": o["hinge_side"],
                      **({"flush": True} if o.get("flush") else {})} for o in openings],
        "obstacles": [{"id": f"o{k}", "label": "unknown", "min": r4(ob["min"]), "max": r4(ob["max"]),
                       "points": ob["n"], "footprint": ob["footprint"]} for k, ob in enumerate(obstacles)],
        "coverage": {"score": round(score, 3), "walls": round(wall_cov, 3), "floor": round(floor_cov, 3),
                     "ceiling": round(ceil_cov, 3), "gaps": gaps},
        "qa": qa,
        "gravity": {"candidates": [{k: (r4(v) if isinstance(v, list) else round(v, 3)) for k, v in c.items()}
                                   for c in ginfo["candidates"]], "margin": round(ginfo["margin"], 3),
                    "floor_rms": round(prep["floor"]["rms"], 4)},
        "timings_s": timings,
    }
    labels = np.zeros(len(P1), np.uint8)               # 0 other, 1 floor, 2 ceiling, 3 wall
    labels[np.abs(P1[:, 1]) < 0.03] = 1
    labels[np.abs(P1[:, 1] - H) < 0.04] = 2
    labels[struct & (labels == 0)] = 3
    return room, dict(P=P1, C=prep["C1"], labels=labels, obstacles=obstacles, wall_results=wall_results,
                      T=T, yaw=yaw, shift=shift, polygon=poly, H=H, struct=struct)


def process_cloud(P_raw, C_raw=None, source=None, k_scale=0.003, seed=0, log=print):
    """Single-room entry point (kept for existing callers): the primary room of the scan.

    Returns (room dict, extra). extra["house"] holds the multi-room result when the scan contains
    more than one room (see roomtwin.rooms)."""
    from .rooms import process_house
    house, primary, extras = process_house(P_raw, C_raw, source, k_scale=k_scale, seed=seed, log=log)
    room, extra = primary
    extra = dict(extra, house=house, room_extras=extras)
    return room, extra


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m roomtwin.process", description="Room scan to room.json")
    ap.add_argument("input", help=".ply/.obj point cloud or mesh, or a folder of RGB-D frames")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--scale-sigma", type=float, default=0.003, help="relative scale uncertainty of the device")
    ap.add_argument("--aruco", help="folder with ArUco marker photos + depth for an independent scale check")
    ap.add_argument("--marker-size", type=float, default=0.15, help="ArUco marker side length in metres")
    ap.add_argument("--voxel", type=float, default=0.01, help="TSDF voxel size for RGB-D input")
    ap.add_argument("--no-mesh", action="store_true", help="skip room_mesh.glb")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    t = time.perf_counter()
    if os.path.isdir(a.input):
        from .rgbd import fuse_folder
        P, C, src = fuse_folder(a.input, voxel=a.voxel, log=print)
    else:
        P, C, loader = load_cloud(a.input)
        src = {"type": "pointcloud", "file": os.path.basename(a.input), "loader": loader}
    load_s = time.perf_counter() - t
    try:
        from .rooms import process_house
        house, (room, extra), results = process_house(P, C, src, k_scale=a.scale_sigma)
    except RuntimeError as e:
        # still hand the agent a machine-readable answer: what failed and what to do next
        room = {"version": 1, "source": dict(src, points=int(len(P))), "error": str(e), "walls": [], "openings": [],
                "obstacles": [], "coverage": {"score": 0.0, "gaps": []},
                "qa": [dict(severity="error", code="PIPELINE_FAILED", message=f"Could not build the room: {e}. "
                            "Rescan the whole room slowly, keeping walls, floor and corners in view.",
                            action="rescan", target={})]}
        with open(os.path.join(a.out, "room.json"), "w") as f:
            json.dump(room, f, indent=1)
        print(f"[roomtwin] failed: {e}")
        raise SystemExit(2)
    for r, _ in results.values():
        r["timings_s"] = dict(load=round(load_s, 3), **r["timings_s"])
    if a.aruco:
        from .scalecheck import check_folder
        sc = check_folder(a.aruco, a.marker_size)
        room["scale_check"] = sc
        if sc.get("status") == "ok" and "off by" in sc.get("verdict", ""):
            room["qa"].insert(0, dict(severity="warn", code="SCALE_MISMATCH", action="marker",
                                      message=f"ArUco check: {sc['verdict']} (scale {sc['scale']:.4f} +/- {sc['scale_sigma']:.4f}).",
                                      target=dict(scale=sc["scale"])))
        elif sc.get("status") != "ok":
            room["qa"].append(dict(severity="info", code="MARKER_NOT_FOUND", action="marker",
                                   message="No ArUco marker found in the scale-check photos.", target={}))
    from .export import write_house_outputs
    write_house_outputs(a.out, house, results, room["id"], mesh=not a.no_mesh)
    total = time.perf_counter() - t
    room["timings_s"]["total"] = round(total, 3)
    room["house"] = "house.json"
    with open(os.path.join(a.out, "room.json"), "w") as f:
        json.dump(room, f, indent=1, default=_json_default)
    with open(os.path.join(a.out, "house.json"), "w") as f:
        json.dump(house, f, indent=1, default=_json_default)
    for r in house["rooms"]:
        rm = r["room"]
        tag = " (primary -> room.json)" if r["id"] == room["id"] else ""
        print(f"[roomtwin] {r['id']} {r['name']}{tag}: {len(rm['walls'])} walls, {len(rm['openings'])} openings, "
              f"{len(rm['obstacles'])} obstacles, area {rm['floor']['area_m2']:.2f} m2, coverage {rm['coverage']['score']:.2f}")
    print(f"[roomtwin] house.json: {len(house['rooms'])} rooms, {len(house['doorways'])} doorways, {total:.1f} s -> {a.out}")
    for q in room["qa"]:
        print(f"  [{q['severity']}] {q['message']}")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")

if __name__ == "__main__":
    main()
