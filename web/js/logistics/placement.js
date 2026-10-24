// Placement checks for furniture in the plan: walls, other items, existing
// obstacles, door swings / doorway clear zones, and being inside a room.
import { segmentIntersectsObb, obbOverlap, convexPolyOverlapsObb, pointInPolygon, dot, sub, closestPointOnSegment } from '../core/geom.js';
import { isPassable, doorSwingPolygon, openingClearZone } from '../core/world.js';
import { itemObb, obstacleObb } from './env.js';

const FLUSH = 0.003; // items may sit this close to walls without counting as a hit

export function evaluatePlacement(world, items) {
  const out = {};
  const obbs = items.map((it) => ({ it, o: itemObb(it) }));
  const swings = world.openings.filter((op) => isPassable(op)).map((op) => ({
    op,
    poly: op.type === 'door' && op.swing !== 'none' ? doorSwingPolygon(op) : openingClearZone(op),
    kind: op.type === 'door' && op.swing !== 'none' ? 'swing' : 'doorway',
  }));
  for (const { it, o } of obbs) {
    const problems = [];
    const shrunk = { ...o, hx: Math.max(0.001, o.hx - FLUSH), hz: Math.max(0.001, o.hz - FLUSH) };
    if (!world.rooms.some((r) => pointInPolygon([o.cx, o.cz], r.polygon))) problems.push({ kind: 'outside', text: 'Outside the rooms' });
    for (const w of world.walls) {
      if (segmentIntersectsObb(w.a, w.b, shrunk)) { problems.push({ kind: 'wall', ref: w.key, text: `Goes through wall ${w.id}` }); break; }
    }
    for (const ob of world.obstacles) {
      if (ob.ymax <= 0.05) continue;
      if (obbOverlap(shrunk, obstacleObb(ob), 1e-4)) problems.push({ kind: 'obstacle', ref: ob.key, text: `Overlaps the existing ${/^(unknown|obstacle|object)?$/i.test(ob.label || '') ? 'furniture' : ob.label}` });
    }
    for (const { it: other, o: oo } of obbs) {
      if (other.id === it.id) continue;
      if (obbOverlap(shrunk, oo, 1e-4)) problems.push({ kind: 'item', ref: other.id, text: `Overlaps the ${other.name}` });
    }
    for (const s of swings) {
      if (convexPolyOverlapsObb(s.poly, shrunk, 1e-4)) {
        problems.push({ kind: s.kind, ref: s.op.key, text: s.kind === 'swing' ? `Blocks the swing of door ${s.op.id}` : `Blocks the doorway ${s.op.id}` });
      }
    }
    for (const r of world.rooms) {
      if (pointInPolygon([o.cx, o.cz], r.polygon) && it.assembled.h > r.ceiling.value) {
        problems.push({ kind: 'ceiling', text: `Taller than the ceiling (${r.ceiling.value.toFixed(2)} m)` });
      }
    }
    out[it.id] = { ok: problems.length === 0, problems };
  }
  return out;
}

// Snap an item's back to the nearest wall when close and roughly parallel.
// Returns a new pose or null when nothing is near.
export function snapToWall(world, item, pose, threshold = 0.18) {
  const hd = item.assembled.d / 2, hw = item.assembled.w / 2;
  let best = null;
  for (const w of world.walls) {
    const c = closestPointOnSegment([pose.x, pose.z], w.a, w.b);
    const t = c[2];
    if (t <= 0 || t >= 1) continue;
    const off = dot(sub([pose.x, pose.z], [c[0], c[1]]), w.normalIn);
    if (off < 0) continue;
    // candidate: back against the wall, front facing the room
    const yaw = Math.atan2(w.normalIn[0], w.normalIn[1]);
    const gap = off - hd;
    if (Math.abs(gap) > threshold) continue;
    if (!best || Math.abs(gap) < Math.abs(best.gap)) {
      // keep the item within the wall's length
      const L = w.geomLength;
      const along = Math.min(Math.max(t * L, hw), L - hw);
      const base = [w.a[0] + w.dir[0] * along, w.a[1] + w.dir[1] * along];
      best = { gap, pose: { x: base[0] + w.normalIn[0] * (hd + 0.005), z: base[1] + w.normalIn[1] * (hd + 0.005), yaw }, wall: w.key };
    }
  }
  return best;
}

