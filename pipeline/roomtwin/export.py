"""Outputs: room_preview.ply (decimated, labelled cloud), room_mesh.glb (walls with openings cut out,
floor, ceiling) and floorplan.png (top view drawn with cv2 drawing functions)."""
from __future__ import annotations

import json
import os
import struct

import cv2
import numpy as np

from .io_ply import write_ply

LABEL_RGB = np.array([[0.85, 0.45, 0.2], [0.55, 0.55, 0.6], [0.9, 0.9, 0.95], [0.75, 0.78, 0.85]])


# ------------------------------------------------------------------------------- triangulation
def _ear_clip(poly):
    """Triangulate a simple CCW polygon (list of 2D points) by ear clipping. Returns index triples."""
    idx = list(range(len(poly)))
    tris = []
    P = np.asarray(poly, float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for k in range(n):
            i0, i1, i2 = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            a, b, c = P[i0], P[i1], P[i2]
            if cross(a, b, c) <= 1e-12:
                continue                       # reflex vertex
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = P[j]
                if cross(a, b, p) >= 0 and cross(b, c, p) >= 0 and cross(c, a, p) >= 0:
                    inside = True
                    break
            if not inside:
                tris.append((i0, i1, i2))
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def room_mesh(room):
    """Three primitives: floor, walls (with openings cut out), ceiling. Returns dict of arrays."""
    H = room["ceiling_height"]["value"]
    poly = np.asarray(room["floor"]["polygon"], float)
    tris = _ear_clip(poly)
    prims = {}
    v = np.array([[p[0], 0.0, p[1]] for p in poly])
    # CCW in (x, z) seen from +y is clockwise in a right-handed view from above -> flip for +y normals
    prims["floor"] = (v, np.array([(a, c, b) for a, b, c in tris]), np.tile([0, 1.0, 0], (len(v), 1)))
    vc = v.copy()
    vc[:, 1] = H
    prims["ceiling"] = (vc, np.array(tris), np.tile([0, -1.0, 0], (len(v), 1)))
    verts, faces, norms = [], [], []
    for w in room["walls"]:
        a, b = np.asarray(w["a"]), np.asarray(w["b"])
        L = w["length"]
        u = (b - a) / max(L, 1e-9)
        n = np.array([-u[1], 0, u[0]])
        ops = [o for o in room["openings"] if o["wall"] == w["id"]]
        ts = sorted({0.0, L, *[min(max(o["offset"], 0), L) for o in ops], *[min(max(o["offset"] + o["width"], 0), L) for o in ops]})
        ys = sorted({0.0, H, *[min(o["sill"], H) for o in ops], *[min(o["sill"] + o["height"], H) for o in ops]})
        for i in range(len(ts) - 1):
            for j in range(len(ys) - 1):
                tc, yc = 0.5 * (ts[i] + ts[i + 1]), 0.5 * (ys[j] + ys[j + 1])
                if any(o["offset"] < tc < o["offset"] + o["width"] and o["sill"] < yc < o["sill"] + o["height"] for o in ops):
                    continue
                base = len(verts)
                for t, y in ((ts[i], ys[j]), (ts[i + 1], ys[j]), (ts[i + 1], ys[j + 1]), (ts[i], ys[j + 1])):
                    p = a + u * t
                    verts.append([p[0], y, p[1]])
                    norms.append(n)
                faces += [(base, base + 2, base + 1), (base, base + 3, base + 2)]
    prims["walls"] = (np.array(verts), np.array(faces), np.array(norms))
    return prims


def write_glb(path, prims):
    """Minimal glTF 2.0 binary writer (positions, normals, uint32 indices, one material each)."""
    colors = {"floor": [0.62, 0.5, 0.38, 1], "walls": [0.92, 0.91, 0.88, 1], "ceiling": [0.97, 0.97, 0.97, 1]}
    bin_parts, views, accessors, meshes_prims, materials = [], [], [], [], []
    offset = 0

    def add(arr, target, comp, typ, minmax=False):
        nonlocal offset
        data = arr.tobytes()
        pad = (-len(data)) % 4
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(data), "target": target})
        bin_parts.append(data + b"\0" * pad)
        offset += len(data) + pad
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": int(arr.shape[0]) if typ != "SCALAR" else int(arr.size), "type": typ}
        if minmax:
            acc["min"] = arr.min(0).tolist()
            acc["max"] = arr.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    for name, (v, f, n) in prims.items():
        if len(v) == 0 or len(f) == 0:
            continue
        pos = add(v.astype(np.float32), 34962, 5126, "VEC3", True)
        nor = add(n.astype(np.float32), 34962, 5126, "VEC3")
        ind = add(f.astype(np.uint32).ravel(), 34963, 5125, "SCALAR")
        materials.append({"name": name, "doubleSided": True,
                          "pbrMetallicRoughness": {"baseColorFactor": colors[name], "metallicFactor": 0, "roughnessFactor": 0.9}})
        meshes_prims.append({"attributes": {"POSITION": pos, "NORMAL": nor}, "indices": ind, "material": len(materials) - 1})
    gltf = {"asset": {"version": "2.0", "generator": "roomtwin"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0, "name": "room"}], "meshes": [{"primitives": meshes_prims, "name": "room"}],
            "materials": materials, "accessors": accessors, "bufferViews": views,
            "buffers": [{"byteLength": offset}]}
    js = json.dumps(gltf).encode()
    js += b" " * ((-len(js)) % 4)
    binblob = b"".join(bin_parts)
    total = 12 + 8 + len(js) + 8 + len(binblob)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binblob), 0x004E4942) + binblob)


# ------------------------------------------------------------------------------- floor plan
def draw_floorplan(room, path, px_per_m=140, margin=110):
    poly = np.asarray(room["floor"]["polygon"], float)
    size = poly.max(0) * px_per_m + 2 * margin
    W, Hh = int(size[0]), int(size[1]) + 40
    img = np.full((Hh, W, 3), 255, np.uint8)

    def P(xz):
        xz = np.asarray(xz, float)
        return (int(round(margin + xz[0] * px_per_m)), int(round(margin + 40 + xz[1] * px_per_m)))

    cv2.fillPoly(img, [np.array([P(p) for p in poly], np.int32)], (238, 240, 242), cv2.LINE_AA)
    for k, ob in enumerate(room["obstacles"]):
        p0, p1 = P([ob["min"][0], ob["min"][2]]), P([ob["max"][0], ob["max"][2]])
        cv2.rectangle(img, p0, p1, (200, 200, 205), -1)
        cv2.rectangle(img, p0, p1, (120, 120, 130), 1, cv2.LINE_AA)
        dims = f"{ob['id']} {ob['max'][0] - ob['min'][0]:.2f}x{ob['max'][2] - ob['min'][2]:.2f}x{ob['max'][1]:.2f}"
        cv2.putText(img, dims, (p0[0] + 4, p0[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 70, 80), 1, cv2.LINE_AA)
    walls = {w["id"]: w for w in room["walls"]}
    centroid = poly.mean(0)
    for w in room["walls"]:
        col = (40, 40, 40) if not w.get("inferred") else (0, 140, 255)
        cv2.line(img, P(w["a"]), P(w["b"]), col, 5, cv2.LINE_AA)
    for g in room["coverage"]["gaps"]:
        w = walls[g["wall"]]
        a, b = np.asarray(w["a"]), np.asarray(w["b"])
        u = (b - a) / w["length"]
        n = np.asarray(w["normal"])
        p0, p1 = a + u * g["from"] + n * 0.06, a + u * g["to"] + n * 0.06
        col = (0, 0, 220) if g["reason"] == "no points" else (0, 150, 255)
        cv2.line(img, P(p0), P(p1), col, 3, cv2.LINE_AA)
    for o in room["openings"]:
        w = walls[o["wall"]]
        a, b = np.asarray(w["a"]), np.asarray(w["b"])
        u = (b - a) / w["length"]
        n = np.asarray(w["normal"])
        p0, p1 = a + u * o["offset"], a + u * (o["offset"] + o["width"])
        cv2.line(img, P(p0), P(p1), (255, 255, 255), 7)
        if o["type"] == "door":
            r = int(o["width"] * px_per_m)
            ang = np.degrees(np.arctan2(n[1], n[0]))
            ang_u = np.degrees(np.arctan2(u[1], u[0]))
            # quarter swing from the wall direction into the room
            a0, a1 = sorted([ang_u, ang])
            if a1 - a0 > 180:
                a0, a1 = a1, a0 + 360
            cv2.ellipse(img, P(p0), (r, r), 0, a0, a1, (150, 110, 60), 1, cv2.LINE_AA)
            cv2.line(img, P(p0), P(p0 + n * o["width"]), (150, 110, 60), 2, cv2.LINE_AA)
        else:
            cv2.line(img, P(p0), P(p1), (200, 120, 0), 2, cv2.LINE_AA)
            cv2.line(img, P(p0 + n * 0.04), P(p1 + n * 0.04), (200, 120, 0), 1, cv2.LINE_AA)
        mid = 0.5 * (p0 + p1) - n * 0.22
        cv2.putText(img, f"{o['id']} {o['width']:.2f}", P(mid), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 120, 0), 1, cv2.LINE_AA)
    for w in room["walls"]:
        a, b = np.asarray(w["a"]), np.asarray(w["b"])
        n = np.asarray(w["normal"])
        mid = 0.5 * (a + b) + n * 0.28
        txt = f"{w['id']} {w['length']:.3f}+/-{w['sigma'] * 100:.1f}cm"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        pm = P(mid)
        cv2.putText(img, txt, (pm[0] - tw // 2, pm[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.circle(img, P(a), 4, (0, 0, 0), -1, cv2.LINE_AA)
    title = (f"RoomTwin  area {room['floor']['area_m2']:.2f} m2  ceiling {room['ceiling_height']['value']:.3f} m"
             f"  coverage {room['coverage']['score'] * 100:.0f}%")
    cv2.putText(img, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    x0, y0 = 12, Hh - 18
    cv2.line(img, (x0, y0), (x0 + px_per_m, y0), (0, 0, 0), 2)
    cv2.putText(img, "1 m", (x0 + px_per_m + 6, y0 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def draw_wall_elevations(room, wall_results, out_dir, px=3):
    """walls/<id>.png: the occupancy grid each wall was analysed on (floor at the bottom), with
    detected openings (green) and coverage gaps (red = no points, orange = occluded/low density)."""
    from .openings import CELL
    os.makedirs(os.path.join(out_dir, "walls"), exist_ok=True)
    files = {}
    for w in room["walls"]:
        res = wall_results.get(w["id"])
        if res is None:
            continue
        img = res["image"]
        occ = np.where(img > 0, 90, 245).astype(np.uint8)
        vis = cv2.cvtColor(occ[::-1], cv2.COLOR_GRAY2BGR)
        vis = cv2.resize(vis, (vis.shape[1] * px, vis.shape[0] * px), interpolation=cv2.INTER_NEAREST)
        rows = img.shape[0]

        def rect(t0, t1, y0, y1, col, th):
            p0 = (int(t0 / CELL * px), int((rows - y1 / CELL) * px))
            p1 = (int(t1 / CELL * px), int((rows - y0 / CELL) * px))
            cv2.rectangle(vis, p0, p1, col, th)

        for g in room["coverage"]["gaps"]:
            if g["wall"] == w["id"]:
                rect(g["from"], g["to"], g["y_from"], g["y_to"], (0, 0, 220) if g["reason"] == "no points" else (0, 150, 255), 2)
        for o in room["openings"]:
            if o["wall"] == w["id"]:
                rect(o["offset"], o["offset"] + o["width"], o["sill"], o["sill"] + o["height"], (0, 170, 0), 2)
                cv2.putText(vis, o["id"], (int(o["offset"] / CELL * px) + 4, int((rows - (o["sill"] + o["height"]) / CELL) * px) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 120, 0), 1, cv2.LINE_AA)
        cv2.putText(vis, f"{w['id']}  {w['length']:.3f} m  coverage {w['coverage'] * 100:.0f}%", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        path = os.path.join("walls", f"{w['id']}.png")
        cv2.imwrite(os.path.join(out_dir, path), vis)
        files[w["id"]] = path.replace(os.sep, "/")
    return files


def write_outputs(out_dir, room, extra, mesh=True, max_preview=250000, seed=0):
    P, C, lab = extra["P"], extra["C"], extra["labels"]
    rng = np.random.default_rng(seed)
    sel = np.arange(len(P)) if len(P) <= max_preview else rng.choice(len(P), max_preview, replace=False)
    col = C[sel] if C is not None else LABEL_RGB[lab[sel]]
    write_ply(os.path.join(out_dir, "room_preview.ply"), P[sel], col)
    room["preview"] = "room_preview.ply"
    draw_floorplan(room, os.path.join(out_dir, "floorplan.png"))
    room["floorplan"] = "floorplan.png"
    room["wall_elevations"] = draw_wall_elevations(room, extra["wall_results"], out_dir)
    if mesh:
        write_glb(os.path.join(out_dir, "room_mesh.glb"), room_mesh(room))
        room["mesh"] = "room_mesh.glb"
