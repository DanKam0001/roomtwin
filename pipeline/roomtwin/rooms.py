"""Room segmentation for multi-room scans, per-room fitting and house.json assembly.

A phone scan often covers more than the room the user cares about: the hallway behind the door,
the next room through an archway. Fitting one polygon to everything mixes their walls, so rooms
are separated first, on a 2 cm top-view grid in the gravity-levelled frame:

  barrier   vertical-surface points between 1.1 m and 2.0 m (walls; beds, desks and chairs are lower)
            plus vertical surfaces above 2.08 m (the lintel: the wall above a door closes the doorway)
  interior  observed floor (horizontal points near y = 0), ceiling (horizontal points > 1.95 m) or
            anything below 1.1 m (the floor under a bed is never seen, its top is), closed with a
            5x5 kernel, minus the dilated barrier

Connected components of the interior are the rooms (a doorway is open below 2 m but closed by its
lintel, so the two sides fall apart; a corridor bend has no lintel and stays one room). Regions
enclosed by another region (the floor inside a bed frame, a desk's knee space) are merged into it.
When a doorway's lintel was never scanned, a distance-transform watershed (cv2.distanceTransform,
cv2.watershed) splits a region at a door-wide constriction (<= 1.0 m) between two basins that are
both room-sized (>= 2 m2 floor, >= 1.2 m wide); corridors are too narrow to seed and stay whole.

Points are assigned to the room of the nearest interior cell (cv2.distanceTransformWithLabels), so
the two faces of a shared wall go to the rooms they face. Each room is then re-levelled on its own
floor plane: long scans drift by a degree or so between rooms (the real test scan: 1.2 deg), which
is 5 cm over a ceiling height and would bias wall lines and opening heights. The rotation pivots on
the room's floor, so plan (x, z) positions stay in one shared frame.

The existing single-room fit (walls, openings, obstacles, coverage, QA) runs per room in the frame
of the primary room (largest well-bounded floor area). Doorways are matched between rooms: two
openings on parallel, opposite-facing walls less than 45 cm apart (one wall thickness plus a door
recess) that overlap along the wall. A door seen from one side only gets an inferred partner on the
other room's wall.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .geometry import rotation_between
from .gravity import fit_horizontal_plane

CELL = 0.02
MIN_ROOM_AREA = 2.0          # m2 of observed interior; smaller regions are views through doors
MIN_REGION_AREA = 0.3        # m2; smaller interior specks are not regions at all
MIN_BOUNDED = 0.3            # fraction of a region's rim that must be wall (else: a view into the unknown)
RAY_WIDTH = 0.5              # m; floor strips narrower than this are grazing views, not room
ASSIGN_MAX = 0.60            # m; points farther than this from any room's interior are dropped


# ------------------------------------------------------------------------------- grids
def _grid(q, lo, shape, mask):
    g = np.zeros(shape, np.uint8)
    ij = ((q[mask] - lo) / CELL).astype(int)
    g[ij[:, 1], ij[:, 0]] = 1
    return g


def build_grids(P2, horiz, H):
    """Top-view grids (rows = z, cols = x) of barrier, lintel and interior evidence."""
    q = P2[:, [0, 2]]
    lo = q.min(0) - 0.2
    shape = tuple(np.ceil((q.max(0) + 0.2 - lo) / CELL).astype(int)[::-1] + 1)
    y = P2[:, 1]
    vert = ~horiz
    wall = _grid(q, lo, shape, vert & (y > 1.1) & (y < 2.0))
    # lintel band: any vertical surface above 2.08 m (ceilings are horizontal, so no upper limit is
    # needed; rooms of one scan differ in ceiling height and drift by a few cm)
    lintel = _grid(q, lo, shape, vert & (y > 2.08) & (y < 3.2))
    floor = _grid(q, lo, shape, horiz & (y > -0.3) & (y < 0.25))
    ceil = _grid(q, lo, shape, horiz & (y > 1.95))
    low = _grid(q, lo, shape, (y > -0.3) & (y < 1.1))        # furniture: the floor under a bed is never seen
    return dict(lo=lo, shape=shape, wall=wall, lintel=lintel, floor=floor, ceil=ceil, low=low)


def _clean_barrier(b, min_extent=0.3):
    """Drop barrier specks (lamps, a curtain rail end, noise) shorter than min_extent; dilate one cell."""
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))    # join a sparsely sampled wall
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    small = np.zeros(n, bool)
    ext = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) * CELL
    small[1:] = ext[1:] < min_extent
    b = np.where(small[lab], 0, b).astype(np.uint8)
    return cv2.dilate(b, np.ones((3, 3), np.uint8))


# ------------------------------------------------------------------------------- regions
def _merge_enclosed(lab, n):
    """Regions lying inside another region's filled outline (floor inside a bed frame) join it."""
    changed = True
    while changed:
        changed = False
        ids = [i for i in range(1, n + 1) if (lab == i).any()]
        areas = {i: int((lab == i).sum()) for i in ids}
        for i in sorted(ids, key=lambda k: -areas[k]):
            m = (lab == i).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filled = np.zeros_like(m)
            cv2.drawContours(filled, cnts, -1, 1, -1)
            for j in ids:
                if j == i or areas[j] > areas[i]:
                    continue
                inside = filled[lab == j].mean()
                if inside > 0.9:
                    lab[lab == j] = i
                    changed = True
            if changed:
                break
    return lab


def _watershed_split(region, barrier, min_area=MIN_ROOM_AREA, max_door=1.0, min_peak=0.6):
    """Split one region at a door-wide constriction between two room-sized basins.

    Returns a label image over the region (1..k). Seeds are the components of the distance
    transform above min_peak (half of a 1.2 m room width); corridors never reach it and are
    flooded by their neighbour, which is why a split also needs the constriction to be narrow."""
    free = region.astype(np.uint8)
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * CELL
    seeds = (dist > min_peak).astype(np.uint8)
    ns, slab = cv2.connectedComponents(seeds, connectivity=8)
    if ns <= 2:
        return free.astype(np.int32)
    markers = slab.astype(np.int32)
    markers[free == 0] = -1                     # outside: not flooded
    markers[(free == 1) & (seeds == 0)] = 0
    vis = cv2.cvtColor((np.clip(1 - dist / max(dist.max(), 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    markers[markers == -1] = ns                 # background marker
    cv2.watershed(vis, markers)
    lab = np.where((markers > 0) & (markers < ns) & (free == 1), markers, 0)
    # watershed lines (-1) inside the region go to the neighbouring basin with the larger label count
    line = (markers == -1) & (free == 1)
    if line.any():
        near = cv2.dilate(lab.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        lab[line] = near[line]
    # merge basins unless both are room-sized and their shared boundary is door-wide
    k = ns - 1
    parent = list(range(k + 1))

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a

    for a in range(1, k + 1):
        for b in range(a + 1, k + 1):
            ma, mb = lab == a, lab == b
            if not ma.any() or not mb.any():
                continue
            touch = cv2.dilate(ma.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & mb
            if not touch.any():
                continue
            ys, xs = np.nonzero(touch)
            width = CELL * max(np.ptp(xs), np.ptp(ys)) + CELL
            big = ma.sum() * CELL ** 2 >= min_area and mb.sum() * CELL ** 2 >= min_area
            if not (big and width <= max_door):
                parent[find(b)] = find(a)
    out = np.zeros_like(lab)
    for a in range(1, k + 1):
        out[lab == a] = find(a)
    _, out = np.unique(out, return_inverse=True)
    return out.reshape(lab.shape).astype(np.int32)


def segment_rooms(P2, horiz, H, log=print):
    """Label image of rooms plus per-room stats. P2/horiz: levelled working cloud (floor y = 0)."""
    g = build_grids(P2, horiz, H)
    barrier = _clean_barrier(g["wall"] | g["lintel"])
    evidence = cv2.morphologyEx(g["floor"] | g["ceil"] | g["low"], cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    interior = (evidence & (1 - barrier)).astype(np.uint8)
    n, lab = cv2.connectedComponents(interior, connectivity=4)
    lab = lab.astype(np.int32)
    min_cells = MIN_REGION_AREA / CELL ** 2
    for i in range(1, n):
        if (lab == i).sum() < min_cells:
            lab[lab == i] = 0
    lab = _merge_enclosed(lab, n - 1)
    # fallback split at unlintelled doorways
    out = np.zeros_like(lab)
    nxt = 1
    for i in np.unique(lab):
        if i == 0:
            continue
        sub = _watershed_split(lab == i, barrier)
        for j in np.unique(sub):
            if j == 0:
                continue
            out[sub == j] = nxt
            nxt += 1
    lab = out
    # strip grazing-view "rays" (long thin floor strips seen through a doorway at a low angle)
    k = int(round(RAY_WIDTH / CELL)) | 1
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for i in range(1, nxt):
        m = (lab == i).astype(np.uint8)
        if not m.any():
            continue
        core = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk)
        keep = cv2.dilate(core, np.ones((7, 7), np.uint8)) & m
        lab[(m == 1) & (keep == 0)] = 0
    rooms = []
    for i in range(1, nxt):
        m = lab == i
        area = float(m.sum() * CELL ** 2)
        if area < MIN_REGION_AREA:
            continue
        # boundedness: fraction of the region's rim that touches barrier (walls) rather than unknown
        rim = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~m
        bounded = float(barrier[rim].mean()) if rim.any() else 0.0
        dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) * CELL
        half_w = float(np.percentile(dist[m], 95)) if m.any() else 0.0
        rooms.append(dict(label=i, area=area, bounded=bounded, half_width=half_w,
                          valid=area >= MIN_ROOM_AREA and bounded >= MIN_BOUNDED))
    log(f"[roomtwin] segmentation: {len(rooms)} regions " +
        ", ".join(f"#{r['label']} {r['area']:.1f} m2 bounded {r['bounded']:.2f}{'' if r['valid'] else ' (dropped)'}"
                  for r in rooms))
    return lab, rooms, g, barrier


def assign_points(lab, lo, q, max_dist=ASSIGN_MAX):
    """Room label of every point (x, z) = label of its nearest interior cell, 0 if farther than max_dist."""
    zero = (lab == 0).astype(np.uint8)
    dist, idx = cv2.distanceTransformWithLabels(zero, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # map each Voronoi label back to the room label of its seed pixel
    seed_room = np.zeros(int(idx.max()) + 1, np.int32)
    seeds = zero == 0
    seed_room[idx[seeds]] = lab[seeds]
    ij = np.clip(((q - lo) / CELL).astype(int), 0, [lab.shape[1] - 1, lab.shape[0] - 1])
    room = seed_room[idx[ij[:, 1], ij[:, 0]]]
    d = dist[ij[:, 1], ij[:, 0]] * CELL
    room[d > max_dist] = 0
    return room


# ------------------------------------------------------------------------------- per-room levelling
def level_room(prep, core2, rng, max_tilt_deg=3.0):
    """Rigid transform (R, t) that puts this room's own floor plane at y = 0 with +y normal.

    The rotation pivots on the floor under the room's centroid, so plan positions do not move."""
    P2, N2, horiz = prep["P2"], prep["N2"], prep["horiz"]
    sel = core2 & horiz & (P2[:, 1] > -0.3) & (P2[:, 1] < 0.25)
    if sel.sum() < 200:
        return np.eye(3), np.zeros(3), 0.0
    y = P2[sel, 1]
    hist, edges = np.histogram(y, bins=np.arange(-0.3, 0.26, 0.01))
    peak = float(0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]))
    pl = fit_horizontal_plane(P2[core2], N2[core2], peak, rng)
    if pl is None:
        return np.eye(3), np.array([0.0, -peak, 0.0]), 0.0
    n = pl["normal"]
    tilt = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
    R = rotation_between(n, np.array([0.0, 1.0, 0.0])) if tilt <= max_tilt_deg else np.eye(3)
    fl = P2[core2][pl["mask"]]
    c = np.array([np.median(fl[:, 0]), 0.0, np.median(fl[:, 2])])
    c[1] = -(pl["d"] + n[0] * c[0] + n[2] * c[2]) / n[1]       # floor height under the centroid
    t = np.array([c[0], 0.0, c[2]]) - R @ c
    return R, t, tilt


def room_prep(prep, R, t, core2):
    """A prep dict (see process.prepare_cloud) for one room, re-levelled by (R, t)."""
    from .process import level_height
    P2 = prep["P2"] @ R.T + t
    N2 = prep["N2"] @ R.T
    P1 = prep["P1"] @ R.T + t
    horiz = ~np.isnan(N2[:, 0]) & (np.abs(np.nan_to_num(N2[:, 1])) > np.cos(np.radians(15)))
    sel = core2 & horiz & (np.abs(P2[:, 1]) < 0.06)
    if sel.sum() >= 50:
        dy, fsel = level_height(P2, horiz & core2, float(np.median(P2[sel, 1])))
        P2[:, 1] -= dy
        P1[:, 1] -= dy
        t = t - np.array([0.0, dy, 0.0])
        fy = P2[fsel, 1]
        sig_f = float(np.std(fy) / np.sqrt(max(len(np.unique(np.floor(P2[fsel][:, [0, 2]] / 0.02), axis=0)), 1)))
    else:
        sig_f = prep["sig_f"]
    t_pre0 = np.asarray(prep.get("t_pre", (0.0, -prep["y0"], 0.0)), float)
    sub = dict(prep, P1=P1, P2=P2, N2=N2, horiz=horiz, floor_mask2=np.abs(P2[:, 1]) < 0.03, sig_f=sig_f,
               R=R @ prep["R"], t_pre=R @ t_pre0 + t, y0=0.0)
    # transform bookkeeping: scan -> levelled-global is (prep R, -y0); then (R, t) on top
    sub["R_level"], sub["t_level"] = R, t
    if prep["H_global"] is not None:
        sub["H_global"] = None              # each room measures its own ceiling after re-levelling
    return sub


# ------------------------------------------------------------------------------- doorways
def _opening_world(room, o):
    w = next(w for w in room["walls"] if w["id"] == o["wall"])
    a, b = np.asarray(w["a"], float), np.asarray(w["b"], float)
    u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    return a + u * o["offset"], a + u * (o["offset"] + o["width"]), u, np.asarray(w["normal"], float)


def _passable(o):
    return o["type"] in ("door", "opening") and o["sill"] <= 0.05


def match_doorways(rooms, max_gap=0.45, angle_deg=15.0):
    """Pair openings across rooms; add an inferred partner when a door is seen from one side only.

    rooms: list of (id, room dict) in the shared frame. Returns the house.json doorways list."""
    doorways, used = [], set()
    cos_t = np.cos(np.radians(angle_deg))
    for ia, (ida, ra) in enumerate(rooms):
        for oa in ra["openings"]:
            if not _passable(oa) or (ida, oa["id"]) in used:
                continue
            a0, a1, ua, na = _opening_world(ra, oa)
            best = None
            for ib, (idb, rb) in enumerate(rooms):
                if ib == ia:
                    continue
                for ob in rb["openings"]:
                    if not _passable(ob) or (idb, ob["id"]) in used:
                        continue
                    b0, b1, ub, nb = _opening_world(rb, ob)
                    if abs(ua @ ub) < cos_t or na @ nb > -cos_t:
                        continue
                    gap = abs((0.5 * (b0 + b1) - 0.5 * (a0 + a1)) @ na)
                    ta = sorted([0.0, oa["width"]])
                    tb = sorted([(b0 - a0) @ ua, (b1 - a0) @ ua])
                    ov = min(ta[1], tb[1]) - max(ta[0], tb[0])
                    if gap <= max_gap and ov >= 0.5 * min(oa["width"], ob["width"]):
                        score = ov - gap
                        if best is None or score > best[0]:
                            best = (score, idb, ob)
            if best is not None:
                _, idb, ob = best
                used |= {(ida, oa["id"]), (idb, ob["id"])}
                doorways.append({"a": {"room": ida, "opening": oa["id"]}, "b": {"room": idb, "opening": ob["id"]}})
    # one-sided doors: project onto a parallel, opposite-facing wall of another room just behind it
    for ida, ra in rooms:
        for oa in list(ra["openings"]):
            if not _passable(oa) or (ida, oa["id"]) in used:
                continue
            a0, a1, ua, na = _opening_world(ra, oa)
            mid = 0.5 * (a0 + a1)
            for idb, rb in rooms:
                if idb == ida:
                    continue
                hit = None
                for w in rb["walls"]:
                    wa, wb = np.asarray(w["a"], float), np.asarray(w["b"], float)
                    L = float(np.linalg.norm(wb - wa))
                    uw = (wb - wa) / max(L, 1e-9)
                    nw = np.asarray(w["normal"], float)
                    if abs(uw @ ua) < cos_t or nw @ na > -cos_t:
                        continue
                    gap = (mid - wa) @ nw                  # behind wall b's room side = negative
                    if not (-max_gap <= gap <= 0.02):
                        continue
                    t0, t1 = sorted([(a0 - wa) @ uw, (a1 - wa) @ uw])
                    if t0 < -0.05 or t1 > L + 0.05:
                        continue
                    hit = (w, max(t0, 0.0), min(t1, L), abs(gap))
                    break
                if hit is None:
                    continue
                w, t0, t1, gap = hit
                pid = f"x{ida[:1]}{oa['id']}"
                rb["openings"].append(dict(id=pid, type=oa["type"], wall=w["id"], offset=round(t0, 4),
                                           width=round(t1 - t0, 4), sill=oa["sill"], height=oa["height"],
                                           sigma=round(float(np.hypot(oa["sigma"], 0.02)), 4), see_through=0.0,
                                           state=oa.get("state", "unknown"), refined=False, hinge_side=None,
                                           inferred=True, inferred_from=f"{ida}/{oa['id']}"))
                used |= {(ida, oa["id"]), (idb, pid)}
                doorways.append({"a": {"room": ida, "opening": oa["id"]}, "b": {"room": idb, "opening": pid},
                                 "inferred_side": "b"})
                break
    return doorways


# ------------------------------------------------------------------------------- driver
def process_house(P_raw, C_raw=None, source=None, k_scale=0.003, seed=0, log=print, prep=None):
    """Scan -> (house dict, (primary room, primary extra), {room_id: (room, extra)}).

    prep: output of process.prepare_cloud, to skip cleaning/gravity (P_raw is then ignored)."""
    from .process import fit_room, prepare_cloud
    t0 = time.perf_counter()
    if prep is None:
        prep = prepare_cloud(P_raw, C_raw, source, k_scale=k_scale, seed=seed, log=log)
    rng = np.random.default_rng(seed)
    P2, horiz = prep["P2"], prep["horiz"]
    H = prep["H_global"]
    log(f"[roomtwin] up found (margin {prep['ginfo']['margin']:.2f}), global ceiling "
        f"{'none' if H is None else f'{H:.3f} m'}, {len(prep['P1'])} pts (1 cm), {len(P2)} pts (2 cm)")
    t = time.perf_counter()
    lab, regions, grids, barrier = segment_rooms(P2, horiz, H, log=log)
    seg_s = time.perf_counter() - t
    valid = [r for r in regions if r["valid"]]
    q2 = P2[:, [0, 2]]
    if len(valid) <= 1:
        # one room: classic path on the whole cloud (views through doors are pruned by the wall logic)
        room, extra = fit_room(prep, None, None, log=log)
        room["timings_s"] = dict(prep["timings"], segment=round(seg_s, 3), **room["timings_s"])
        rid = "room"
        room["id"], room["name"] = rid, "Room"
        house = _house_dict([(rid, room)], [], None, prep, regions)
        extra = dict(extra, segmentation=dict(labels=lab, lo=grids["lo"], regions=regions, assign=None))
        return house, (room, extra), {rid: (room, extra)}
    seeds = lab.copy()
    for r in regions:
        if r["area"] < MIN_ROOM_AREA:
            seeds[seeds == r["label"]] = 0     # fragments: their points go to the nearest real room
    plab = assign_points(seeds, grids["lo"], q2)
    # primary = largest well-bounded floor area; corridors (narrow) are never primary if a room exists
    def rank(r):
        return r["area"] * (0.5 + r["bounded"]) * (1.0 if r["half_width"] >= 0.75 else 0.3)
    valid.sort(key=rank, reverse=True)
    results, frame = {}, None
    names = []
    n_hall, n_room = 0, 0
    for k, reg in enumerate(valid):
        core2 = plab == reg["label"]
        if core2.sum() < 200:
            continue
        hallway = reg["half_width"] < 0.75
        if k == 0:
            rid, name = "room1", "Main room"
            n_room = 1
        elif hallway:
            n_hall += 1
            rid, name = f"hall{n_hall}", "Hallway" if n_hall == 1 else f"Hallway {n_hall}"
        else:
            n_room += 1
            rid, name = f"room{n_room}", f"Room {n_room}"
        R, tt, tilt = level_room(prep, core2, np.random.default_rng(seed + 7919 * int(reg["label"])))
        sub = room_prep(prep, R, tt, core2)
        log(f"[roomtwin] {rid} ({name}): {reg['area']:.1f} m2 interior, bounded {reg['bounded']:.2f}, "
            f"re-levelled by {tilt:.2f} deg")
        try:
            room, extra = fit_room(sub, core2, frame, log=log)
        except RuntimeError as e:
            log(f"[roomtwin] {rid}: {e}; skipped")
            if k == 0:
                raise
            continue
        if frame is None:
            frame = (extra["yaw"], extra["shift"])
        room["id"], room["name"] = rid, name
        room["segmentation"] = {"interior_area_m2": round(reg["area"], 2), "bounded": round(reg["bounded"], 3),
                                "half_width_p95": round(reg["half_width"], 3), "kind": "hallway" if hallway else "room",
                                "relevel_deg": round(tilt, 3)}
        if hallway and reg["bounded"] < 0.8:
            room["qa"].append(dict(severity="info", code="ROOM_PARTIAL",
                                   message=f"{name} is only partly scanned ({reg['bounded'] * 100:.0f}% of its rim is wall); "
                                           "its open ends are closed with inferred walls. Walk its full length to measure it.",
                                   action="rescan", target=dict(room=rid)))
        extra["room_id"] = rid
        results[rid] = (room, extra)
        names.append(rid)
    order = [(rid, results[rid][0]) for rid in names]
    doorways = match_doorways(order)
    for d in doorways:
        for side in ("a", "b"):
            rm = results[d[side]["room"]][0]
            other = d["b" if side == "a" else "a"]
            for o in rm["openings"]:
                if o["id"] == d[side]["opening"]:
                    o["leads_to"] = other["room"]
    entry = _pick_entry(order, doorways)
    house = _house_dict(order, doorways, entry, prep, regions)
    for rid, (room, _) in results.items():
        room["timings_s"] = dict(prep["timings"], segment=round(seg_s, 3), **room["timings_s"])
    primary = results[names[0]]
    primary[1]["segmentation"] = dict(labels=lab, lo=grids["lo"], regions=regions, assign=plab)
    log(f"[roomtwin] house: {len(order)} rooms, {len(doorways)} doorways, {time.perf_counter() - t0:.1f} s")
    return house, primary, results


def _pick_entry(order, doorways):
    paired = {(d[s]["room"], d[s]["opening"]) for d in doorways for s in ("a", "b")}
    best = None
    for k, (rid, room) in enumerate(order):
        for o in room["openings"]:
            if not _passable(o) or (rid, o["id"]) in paired:
                continue
            score = (k > 0, o["type"] == "door", o["width"])     # prefer a door outside the main room
            if best is None or score > best[0]:
                best = (score, rid, o["id"])
    return {"room": best[1], "opening": best[2]} if best else None


def _house_dict(order, doorways, entry, prep, regions):
    house = {
        "version": 1,
        "type": "house",
        "name": "Scanned home",
        "frame": "shared frame of the primary room (rooms[0]); metres, y up, floor at y = 0; "
                 "every room is already in it, so each transform is identity",
        "rooms": [{"id": rid, "name": room.get("name", rid), "transform": {"x": 0, "z": 0, "rot": 0}, "room": room}
                  for rid, room in order],
        "doorways": doorways,
        "segmentation": {"cell_m": CELL, "regions": [{k: (round(v, 3) if isinstance(v, float) else v)
                                                      for k, v in r.items()} for r in regions]},
    }
    if entry:
        house["entry"] = entry
    return house
