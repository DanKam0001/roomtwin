// Builds the 2D collision environment used by the route search and walk mode.
// Walls become line segments (inner faces) with passable openings cut out;
// windows stay solid. Pure, structured-clone safe.
import { add, scale, dist } from '../core/geom.js';
import { isPassable, openingByKey } from '../core/world.js';

export function yawFromDir(dir) {
  // axisX(yaw) == dir  ->  yaw = atan2(-dir.z, dir.x)
  return Math.atan2(-dir[1], dir[0]);
}

export function itemObb(item, inflate = 0) {
  return {
    cx: item.pose.x, cz: item.pose.z, yaw: item.pose.yaw,
    hx: item.assembled.w / 2 + inflate, hz: item.assembled.d / 2 + inflate,
    ref: item.id, label: item.name,
  };
}

export function obstacleObb(o) {
  return { cx: o.cx, cz: o.cz, hx: o.hx, hz: o.hz, yaw: o.yaw, ref: o.key, label: o.label };
}

// opts: closed: Set of opening keys to treat as shut; widen: {key: width} to widen
// the cut (used when an item clears the opening only by tilting, which the 2D
// search cannot model); boxes: extra OBBs (placed items); entryKey + porchDepth
// define the outdoor apron the load starts from.
export function buildEnv(world, opts = {}) {
  const closed = new Set(opts.closed || []);
  const widen = { ...(opts.widen || {}) };
  for (const op of world.openings) {
    if (op.partner) {
      if (closed.has(op.key)) closed.add(op.partner);
      if (widen[op.key] != null) widen[op.partner] = Math.max(widen[op.partner] || 0, widen[op.key]);
    }
  }
  const segs = [];
  const tags = [];
  const cuts = {}; // opening key -> [a, b] actual cut endpoints
  for (const wall of world.walls) {
    const L = dist(wall.a, wall.b);
    const ivs = [];
    for (const key of wall.openings) {
      const op = openingByKey(world, key);
      if (!op || !isPassable(op) || closed.has(op.key)) continue;
      const w = Math.max(op.width, widen[op.key] || 0);
      const extra = (w - op.width) / 2;
      const s = Math.max(0, op.offset - extra), e = Math.min(L, op.offset + op.width + extra);
      ivs.push([s, e]);
      cuts[op.key] = [add(wall.a, scale(wall.dir, s)), add(wall.a, scale(wall.dir, e))];
    }
    ivs.sort((x, y) => x[0] - y[0]);
    let t = 0;
    for (const [s, e] of ivs) {
      if (s > t + 1e-6) { segs.push([...add(wall.a, scale(wall.dir, t)), ...add(wall.a, scale(wall.dir, s))]); tags.push(wall.key); }
      t = Math.max(t, e);
    }
    if (L > t + 1e-6) { segs.push([...add(wall.a, scale(wall.dir, t)), ...wall.b]); tags.push(wall.key); }
  }
  // Jambs: close the wall cavity between the two halves of a doorway.
  for (const d of world.doorways) {
    const ca = cuts[d.a], cb = cuts[d.b];
    if (!ca || !cb) continue;
    const [a0, a1] = ca;
    const pairDirect = dist(a0, cb[0]) + dist(a1, cb[1]) <= dist(a0, cb[1]) + dist(a1, cb[0]);
    const m0 = pairDirect ? cb[0] : cb[1], m1 = pairDirect ? cb[1] : cb[0];
    if (dist(a0, m0) > 1e-4) { segs.push([...a0, ...m0]); tags.push(`jamb:${d.a}`); }
    if (dist(a1, m1) > 1e-4) { segs.push([...a1, ...m1]); tags.push(`jamb:${d.a}`); }
  }

  const allowedBoxes = [];
  for (const op of world.openings) {
    if (!cuts[op.key]) continue;
    const [a, b] = cuts[op.key];
    const c = scale(add(a, b), 0.5);
    allowedBoxes.push({ cx: c[0], cz: c[1], hx: dist(a, b) / 2, hz: 0.4, yaw: yawFromDir(op.dir) });
  }
  let porch = null;
  const entry = opts.entryKey ? openingByKey(world, opts.entryKey) : null;
  const porchDepth = opts.porchDepth ?? 3;
  if (entry && cuts[entry.key]) {
    const c = add(entry.center, scale(entry.normalIn, -porchDepth / 2 - 0.02));
    porch = { cx: c[0], cz: c[1], hx: Math.max(1.6, entry.width), hz: porchDepth / 2, yaw: yawFromDir(entry.dir) };
    allowedBoxes.push(porch);
  }

  const [x0, z0, x1, z1] = world.bounds;
  const m = porchDepth + 0.6;
  return {
    segments: segs,
    segTags: tags,
    boxes: (opts.boxes || []).map((b) => ({ ...b })),
    allowedPolys: world.rooms.map((r) => r.polygon),
    allowedBoxes,
    porch,
    entry: entry ? { key: entry.key, center: entry.center, normalIn: entry.normalIn, dir: entry.dir, width: entry.width } : null,
    bounds: [x0 - m, z0 - m, x1 + m, z1 + m],
  };
}

