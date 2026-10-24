"""Coverage score and scan-QA actions for the agent.

Every QA entry is machine-readable: severity (info|warn|error), a stable code, a human message,
an action (rescan|marker|ok) and a target that says exactly where to point the phone.

coverage.score = 0.6 * wall coverage + 0.25 * floor coverage + 0.15 * ceiling coverage, where each
term is the observed fraction of 2 cm (walls) or 5 cm (floor, ceiling) cells; openings are excluded
from walls and furniture footprints count as explained floor.
"""
from __future__ import annotations

import cv2
import numpy as np


def surface_coverage(P, polygon, y_sel, obstacles=None, cell=0.05):
    lo = polygon.min(0)
    size = np.ceil((polygon.max(0) - lo) / cell).astype(int) + 1
    mask = np.zeros((size[1], size[0]), np.uint8)
    cv2.fillPoly(mask, [np.round((polygon - lo) / cell).astype(np.int32)], 1)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))           # ignore the wall junction ring
    seen = np.zeros_like(mask)
    q = P[y_sel][:, [0, 2]]
    ij = ((q - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < size[0]) & (ij[:, 1] < size[1])
    seen[ij[ok, 1], ij[ok, 0]] = 1
    if obstacles:
        for ob in obstacles:
            p0 = np.floor((np.asarray(ob["min"])[[0, 2]] - lo) / cell).astype(int)
            p1 = np.ceil((np.asarray(ob["max"])[[0, 2]] - lo) / cell).astype(int)
            seen[max(p0[1], 0):p1[1], max(p0[0], 0):p1[0]] = 1
    seen = cv2.morphologyEx(seen, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return float(seen[mask > 0].mean()) if mask.any() else 0.0


def _near_opening(gap, openings, tol=0.35):
    for o in openings:
        if o["wall"] != gap["wall"]:
            continue
        if gap["to"] >= o["offset"] - tol and gap["from"] <= o["offset"] + o["width"] + tol:
            return o
    return None


def build_qa(walls, wall_results, openings, ceiling, floor_cov, ceil_cov, gravity_info, sigma_goal=0.02):
    qa, gaps = [], []
    for w in walls:
        res = wall_results[w["id"]]
        for g in res["gaps"]:
            gaps.append(g)
            o = _near_opening(g, openings)
            where = f"from {g['from']:.2f} m to {g['to']:.2f} m along {w['id']}, {g['y_from']:.2f}-{g['y_to']:.2f} m high"
            if o is not None and o["type"] == "door" and g["reason"] != "low density":
                msg = (f"Rescan the corner behind the door {o['id']} on {w['id']} ({where}): "
                       f"close the door or step past it so the wall behind the leaf is visible.")
                sev, act, code = "warn", "rescan", "GAP_BEHIND_DOOR"
            elif g["reason"].startswith("occluded"):
                ob = g["reason"].split()[-1]
                msg = f"Wall {w['id']} is hidden behind {ob} {where}; scan it from the side or accept furniture occlusion."
                sev, act, code = "info", "ok", "WALL_OCCLUDED"
            elif g["reason"] == "low density":
                msg = f"Wall {w['id']} is thinly sampled {where}; sweep that patch again slowly from 1-2 m."
                sev, act, code = "warn", "rescan", "WALL_LOW_DENSITY"
            else:
                msg = f"Wall {w['id']} has no points {where}; rescan that region."
                sev, act, code = ("warn" if g["area_m2"] > 0.1 else "info"), "rescan", "WALL_GAP"
            qa.append(dict(severity=sev, code=code, message=msg, action=act,
                           target=dict(wall=w["id"], **{"from": g["from"], "to": g["to"]},
                                       y_from=g["y_from"], y_to=g["y_to"])))
        if w.get("inferred"):
            qa.append(dict(severity="error", code="WALL_INFERRED",
                           message=f"Wall {w['id']} was not observed; its position is inferred from its neighbours. "
                                   f"Scan the corner between them.", action="rescan",
                           target=dict(wall=w["id"])))
        elif w["sigma"] > sigma_goal:
            qa.append(dict(severity="warn", code="WALL_FEW_POINTS",
                           message=f"Wall {w['id']} has too few points ({w['n_eff']} cells) for +/-{sigma_goal * 100:.0f} cm: "
                                   f"length sigma is {w['sigma'] * 100:.1f} cm. Rescan it and both corners from 1-2 m away.",
                           action="rescan", target=dict(wall=w["id"])))
        elif res["coverage"] < 0.6:
            qa.append(dict(severity="warn", code="WALL_LOW_COVERAGE",
                           message=f"Only {res['coverage'] * 100:.0f}% of wall {w['id']} was seen; sweep it top to bottom.",
                           action="rescan", target=dict(wall=w["id"])))
    # corners: each wall end should have points in its last 30 cm
    n = len(walls)
    ends = []
    for w in walls:
        img = wall_results[w["id"]]["image"]
        r0, r1 = 2, max(img.shape[0] - 2, 3)
        k = max(int(0.3 / 0.02), 1)
        ends.append(((img[r0:r1, :k] > 0).mean() if img.shape[1] else 0.0,
                     (img[r0:r1, -k:] > 0).mean() if img.shape[1] else 0.0))
    for i in range(n):
        j = (i + 1) % n
        seen = min(ends[i][1], ends[j][0])
        if seen < 0.15 and not (walls[i].get("inferred") or walls[j].get("inferred")):
            c = [round(float(v), 2) for v in walls[i]["b"]]
            qa.append(dict(severity="warn", code="CORNER_UNSEEN",
                           message=f"The corner between {walls[i]['id']} and {walls[j]['id']} (x={c[0]}, z={c[1]}) is barely "
                                   f"seen ({seen * 100:.0f}% of the last 30 cm); point the phone into that corner and rescan it.",
                           action="rescan", target=dict(corner=[walls[i]["id"], walls[j]["id"]], xz=c)))
    if ceiling is None:
        qa.append(dict(severity="warn", code="NO_CEILING",
                       message="No ceiling plane found; ceiling height comes from the tops of the walls. Tilt the phone up and sweep the ceiling once.",
                       action="rescan", target=dict(surface="ceiling")))
    elif ceil_cov < 0.3:
        qa.append(dict(severity="info", code="CEILING_PARTIAL",
                       message=f"Ceiling coverage is {ceil_cov * 100:.0f}%; height is still measured from the seen part.",
                       action="ok", target=dict(surface="ceiling")))
    if floor_cov < 0.6:
        qa.append(dict(severity="warn", code="FLOOR_LOW_COVERAGE",
                       message=f"Only {floor_cov * 100:.0f}% of the floor was seen; sweep the floor, including under the door.",
                       action="rescan", target=dict(surface="floor")))
    if gravity_info.get("margin", 99) < 1.0:
        qa.append(dict(severity="warn", code="GRAVITY_AMBIGUOUS",
                       message="Up direction was hard to decide (box-like room with little furniture). "
                               "Place an ArUco marker on the floor or check that the floor plan is not sideways.",
                       action="marker", target=dict(surface="floor")))
    if not any(q["action"] != "ok" for q in qa):
        qa.append(dict(severity="info", code="OK", message="Scan is complete enough for +/-2 cm walls.", action="ok", target={}))
    return qa, gaps
