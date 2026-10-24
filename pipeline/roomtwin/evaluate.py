"""Measured vs ground truth on the synthetic rooms (evaluation evidence for the report).

    python -m roomtwin.evaluate                      # generate (if needed), process, print the table
    python -m roomtwin.evaluate --rooms rect lshape --md eval.md

Ground truth is mapped into the pipeline's room frame with room.json "transform" (scan -> room)
composed with the generator's gt_to_scan, so a wrong gravity or yaw would show up as large errors.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .io_ply import load_cloud
from .process import process_cloud
from .synth import DATASET, ROOMS, make_room


def _to_room(M, xz, y=0.0):
    p = M @ np.array([xz[0], y, xz[1], 1.0])
    return p[[0, 2]], p[1]


def compare(room, gt):
    M = np.asarray(room["transform"]) @ np.asarray(gt["gt_to_scan"])
    up = M[:3, :3] @ np.array([0, 1.0, 0])
    rows = [dict(item="gravity", truth=0.0, measured=float(np.degrees(np.arccos(np.clip(up[1], -1, 1)))),
                 unit="deg", sigma=None)]
    rows[-1]["error"] = rows[-1]["measured"]
    H = room["ceiling_height"]
    rows.append(dict(item="ceiling height", truth=gt["ceiling_height"], measured=H["value"],
                     error=H["value"] - gt["ceiling_height"], sigma=H["sigma"], unit="m"))
    rows.append(dict(item="floor area", truth=gt["floor_area_m2"], measured=room["floor"]["area_m2"],
                     error=room["floor"]["area_m2"] - gt["floor_area_m2"], sigma=None, unit="m2"))
    walls = room["walls"]
    used = set()
    wall_map = {}
    for gw in gt["walls"]:
        a, _ = _to_room(M, gw["a"])
        b, _ = _to_room(M, gw["b"])
        d = (b - a) / np.linalg.norm(b - a)
        best, bc = None, 1e9
        for w in walls:
            if w["id"] in used:
                continue
            wa, wb = np.asarray(w["a"]), np.asarray(w["b"])
            wd = (wb - wa) / max(np.linalg.norm(wb - wa), 1e-9)
            if abs(wd @ d) < np.cos(np.radians(10)):
                continue
            c = np.linalg.norm(0.5 * (wa + wb) - 0.5 * (a + b))
            if c < bc:
                best, bc = w, c
        if best is None or bc > 0.5:
            rows.append(dict(item=f"wall {gw['id']}", truth=gw["length"], measured=None, error=None, sigma=None, unit="m"))
            continue
        used.add(best["id"])
        wall_map[gw["id"]] = (best, a, b)
        rows.append(dict(item=f"wall {gw['id']}->{best['id']}", truth=gw["length"], measured=best["length"],
                         error=best["length"] - gw["length"], sigma=best["sigma"], unit="m"))
    extra = [w["id"] for w in walls if w["id"] not in used]
    for go in gt["openings"]:
        gid = f"w{go['wall']}"
        label = f"{go['type']} on {gid}"
        if gid not in wall_map:
            rows.append(dict(item=label, truth=go["width"], measured=None, error=None, sigma=None, unit="m"))
            continue
        w, a, b = wall_map[gid]
        u = (b - a) / np.linalg.norm(b - a)
        p0, p1 = a + u * go["offset"], a + u * (go["offset"] + go["width"])
        wa, wb = np.asarray(w["a"]), np.asarray(w["b"])
        wu = (wb - wa) / np.linalg.norm(wb - wa)
        t0, t1 = sorted([(p0 - wa) @ wu, (p1 - wa) @ wu])
        cands = [o for o in room["openings"] if o["wall"] == w["id"] and o["type"] == go["type"]]
        if not cands:
            rows.append(dict(item=label, truth=go["width"], measured=None, error=None, sigma=None, unit="m"))
            continue
        o = min(cands, key=lambda o: abs(o["offset"] + o["width"] / 2 - (t0 + t1) / 2))
        for key, truth, meas in (("offset", t0, o["offset"]), ("width", go["width"], o["width"]),
                                 ("sill", go["sill"], o["sill"]), ("height", go["height"], o["height"])):
            if go["type"] == "door" and key == "sill":
                continue
            rows.append(dict(item=f"{o['id']} ({go['type']}) {key}", truth=truth, measured=meas,
                             error=meas - truth, sigma=o["sigma"], unit="m"))
    # obstacles: top-view IoU of each furniture box with the best-matching obstacle
    for bx in gt["boxes"]:
        corners = [(x, z) for x in (bx["min"][0], bx["max"][0]) for z in (bx["min"][2], bx["max"][2])]
        pr = np.array([_to_room(M, c)[0] for c in corners])
        g0, g1 = pr.min(0), pr.max(0)
        best = 0.0
        for ob in room["obstacles"]:
            o0, o1 = np.asarray(ob["min"])[[0, 2]], np.asarray(ob["max"])[[0, 2]]
            inter = np.prod(np.clip(np.minimum(g1, o1) - np.maximum(g0, o0), 0, None))
            union = np.prod(g1 - g0) + np.prod(o1 - o0) - inter
            best = max(best, inter / union if union > 0 else 0)
        rows.append(dict(item=f"obstacle {bx['label']} IoU", truth=1.0, measured=round(best, 3), error=None, sigma=None, unit=""))
    return rows, extra


def run(data_dir, rooms=None, regen=False, log=print):
    os.makedirs(data_dir, exist_ok=True)
    results = []
    for name, seed, up in DATASET:
        if rooms and name not in rooms:
            continue
        ply = os.path.join(data_dir, f"{name}.ply")
        gt_path = os.path.join(data_dir, f"{name}_gt.json")
        if regen or not (os.path.exists(ply) and os.path.exists(gt_path)):
            make_room(ROOMS[name], data_dir, seed, up)
        gt = json.load(open(gt_path))
        P, C, _ = load_cloud(ply)
        t = time.perf_counter()
        room, _ = process_cloud(P, C, {"type": "pointcloud", "file": os.path.basename(ply)}, log=lambda *a: None)
        dt = time.perf_counter() - t
        rows, extra = compare(room, gt)
        results.append(dict(room=name, up=up, points=len(P), seconds=round(dt, 2), rows=rows, extra_walls=extra,
                            qa=[q["code"] for q in room["qa"]]))
    return results


def sweep(out_dir, seeds=range(1, 7), noises=(0.005, 0.010), log=print):
    """Robustness sweep: every room x seed (random yaw, gravity convention, sampling) x noise level."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, _, _ in DATASET:
        for seed in seeds:
            for noise in noises:
                up = "y" if seed % 2 else "z"
                tag = f"_s{seed}_n{int(noise * 1000)}"
                ply, gt = make_room(ROOMS[name], out_dir, 100 + seed, up, suffix=tag, noise=noise)
                P, C, _ = load_cloud(ply)
                try:
                    room, _ = process_cloud(P, C, log=lambda *a: None)
                    res, extra = compare(room, gt)
                    err = None
                except Exception as e:           # a failure is a result too
                    res, extra, err = [], [], str(e)
                we = [abs(r["error"]) for r in res if r["item"].startswith("wall") and r["error"] is not None]
                oe = [abs(r["error"]) for r in res if "(" in r["item"] and r["error"] is not None]
                miss = [r["item"] for r in res if r["measured"] is None]
                row = dict(room=name, seed=seed, noise=noise, up=up, error=err,
                           max_wall_cm=round(100 * max(we), 2) if we else None,
                           max_open_cm=round(100 * max(oe), 2) if oe else None, missing=miss, extra=extra)
                row["pass"] = bool(err is None and not miss and not extra and we and max(we) <= 0.02
                                   and (not oe or max(oe) <= 0.03))
                log(row)
                rows.append(row)
    n_pass = sum(r["pass"] for r in rows)
    log(f"sweep: {n_pass}/{len(rows)} scans pass (all walls <= 2 cm, all opening dims <= 3 cm, none missing)")
    return rows


def fmt(v, unit):
    if v is None:
        return "-"
    return f"{v:.3f}" if unit in ("m", "m2", "deg") else f"{v}"


def table(results):
    lines = ["| room | item | truth | measured | error | sigma |", "|---|---|---|---|---|---|"]
    for r in results:
        for row in r["rows"]:
            err = "-" if row["error"] is None else (f"{row['error'] * 100:+.1f} cm" if row["unit"] == "m" else f"{row['error']:+.3f} {row['unit']}")
            sig = "-" if row["sigma"] is None else f"{row['sigma'] * 100:.1f} cm"
            lines.append(f"| {r['room']} ({r['up']}-up) | {row['item']} | {fmt(row['truth'], row['unit'])} | "
                         f"{fmt(row['measured'], row['unit'])} | {err} | {sig} |")
    return "\n".join(lines)


def summary(results):
    out = []
    for r in results:
        we = [abs(x["error"]) for x in r["rows"] if x["item"].startswith("wall") and x["error"] is not None]
        oe = [abs(x["error"]) for x in r["rows"] if "(" in x["item"] and x["error"] is not None]
        miss = [x["item"] for x in r["rows"] if x["measured"] is None]
        z = [abs(x["error"]) / x["sigma"] for x in r["rows"] if x["sigma"] and x["error"] is not None and x["unit"] == "m"]
        out.append(dict(room=r["room"], points=r["points"], seconds=r["seconds"],
                        max_wall_err_cm=round(100 * max(we), 2) if we else None,
                        max_opening_err_cm=round(100 * max(oe), 2) if oe else None,
                        median_abs_z=round(float(np.median(z)), 2) if z else None,
                        missing=miss, extra_walls=r["extra_walls"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m roomtwin.evaluate")
    ap.add_argument("--data", default="D:/roomtwin_data/synth")
    ap.add_argument("--rooms", nargs="*")
    ap.add_argument("--regen", action="store_true")
    ap.add_argument("--md", help="write the markdown table here")
    ap.add_argument("--json", help="write raw results here")
    ap.add_argument("--sweep", action="store_true", help="robustness sweep over seeds and noise levels")
    a = ap.parse_args(argv)
    if a.sweep:
        rows = sweep(os.path.join(os.path.dirname(a.data.rstrip("/\\")), "synth_sweep"))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(rows, f, indent=1)
        return
    res = run(a.data, a.rooms, a.regen)
    tab = table(res)
    print(tab)
    print()
    for s in summary(res):
        print(s)
    if a.md:
        with open(a.md, "w") as f:
            f.write(tab + "\n")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(results=res, summary=summary(res)), f, indent=1, default=float)


if __name__ == "__main__":
    main()
