// Normalises room.json / house.json (docs/SPEC.md) into one "world" in a shared 2D
// frame. Pure data, structured-clone safe, no DOM, no three.js.
import {
  sub, add, scale, norm, len, pointInPolygon, polygonArea, polygonCentroid, axisX, axisZ,
} from './geom.js';

const PASSABLE_SILL = 0.05; // openings with a sill above this are not walk/carry paths

function makeTransform(t) {
  const yaw = ((t && (t.rot ?? t.rotation ?? t.yaw_deg)) || 0) * Math.PI / 180;
  const tx = (t && (t.x ?? t.tx)) || 0, tz = (t && (t.z ?? t.tz)) || 0;
  const ux = axisX(yaw), uz = axisZ(yaw);
  return {
    yaw, tx, tz,
    p: (q) => [q[0] * ux[0] + q[1] * uz[0] + tx, q[0] * ux[1] + q[1] * uz[1] + tz],
    v: (q) => [q[0] * ux[0] + q[1] * uz[0], q[0] * ux[1] + q[1] * uz[1]],
  };
}

function wallsFromPolygon(poly, height) {
  return poly.map((a, i) => {
    const b = poly[(i + 1) % poly.length];
    return { id: `w${i}`, a, b, length: len(sub(b, a)), sigma: null, height };
  });
}

function polygonFromWalls(walls) {
  return walls.map((w) => w.a);
}

export function isPassable(op) {
  return (op.type === 'door' || op.type === 'opening') && (op.sill || 0) <= PASSABLE_SILL;
}

// Adds one room (in its own frame) to the world under a transform.
function addRoom(world, roomId, room, transform, extra = {}) {
  const T = makeTransform(transform);
  const ceilingValue = room.ceiling_height?.value ?? 2.4;
  const rawWalls = room.walls?.length ? room.walls : wallsFromPolygon(room.floor?.polygon || [], ceilingValue);
  const rawPoly = room.floor?.polygon?.length >= 3 ? room.floor.polygon : polygonFromWalls(rawWalls);
  const polygon = rawPoly.map(T.p);
  const r = {
    id: roomId,
    name: extra.name || room.name || roomId,
    polygon,
    area: room.floor?.area_m2 ?? Math.abs(polygonArea(rawPoly)),
    ceiling: { value: ceilingValue, sigma: room.ceiling_height?.sigma ?? null },
    centroid: polygonCentroid(polygon),
    source: room.source || null,
    coverage: room.coverage || null,
    qa: room.qa || [],
    mesh: room.mesh || null,
    transform: { x: T.tx, z: T.tz, yaw: T.yaw },
    wallKeys: [],
  };
  world.rooms.push(r);

  const wallByKey = {};
  for (const w of rawWalls) {
    const a = T.p(w.a), b = T.p(w.b);
    const dir = norm(sub(b, a));
    let n = [-dir[1], dir[0]];
    const mid = scale(add(a, b), 0.5);
    if (!pointInPolygon(add(mid, scale(n, 0.05)), polygon)) n = [-n[0], -n[1]];
    const key = `${roomId}/${w.id}`;
    const wall = {
      key, id: w.id, roomId, a, b, dir, normalIn: n,
      length: w.length ?? len(sub(b, a)),
      geomLength: len(sub(b, a)),
      sigma: w.sigma ?? null,
      height: w.height ?? ceilingValue,
      inlier_rms: w.inlier_rms ?? null,
      openings: [],
    };
    world.walls.push(wall);
    wallByKey[w.id] = wall;
    r.wallKeys.push(key);
  }

  for (const o of room.openings || []) {
    const wall = wallByKey[o.wall];
    if (!wall) continue;
    const a = add(wall.a, scale(wall.dir, o.offset));
    const b = add(wall.a, scale(wall.dir, o.offset + o.width));
    const key = `${roomId}/${o.id}`;
    const op = {
      key, id: o.id, roomId, wallKey: wall.key, type: o.type || 'door', name: o.name || null,
      offset: o.offset, width: o.width, sill: o.sill || 0, height: o.height ?? 2.04, sigma: o.sigma ?? null,
      a, b, center: scale(add(a, b), 0.5), dir: wall.dir, normalIn: wall.normalIn,
      swing: o.swing || (o.type === 'door' ? 'in' : 'none'),
      hinge: o.hinge || 'a',
      partner: null,
    };
    wall.openings.push(op.key);
    world.openings.push(op);
  }

  for (const ob of room.obstacles || []) {
    const c0 = [(ob.min[0] + ob.max[0]) / 2, (ob.min[2] + ob.max[2]) / 2];
    const c = T.p(c0);
    world.obstacles.push({
      key: `${roomId}/${ob.id}`, id: ob.id, roomId, label: ob.label || 'obstacle',
      cx: c[0], cz: c[1],
      hx: Math.abs(ob.max[0] - ob.min[0]) / 2, hz: Math.abs(ob.max[2] - ob.min[2]) / 2,
      yaw: T.yaw, ymin: Math.min(ob.min[1], ob.max[1]), ymax: Math.max(ob.min[1], ob.max[1]),
    });
  }
  return r;
}

function emptyWorld(kind, name) {
  return { kind, name, rooms: [], walls: [], openings: [], obstacles: [], doorways: [], entryKey: null, bounds: null };
}

function finish(world, entry) {
  // Doorways: link partner openings; avoid a double door swing on a shared doorway.
  for (const d of world.doorways) {
    const a = world.openings.find((o) => o.key === d.a);
    const b = world.openings.find((o) => o.key === d.b);
    if (!a || !b) continue;
    a.partner = b.key; b.partner = a.key;
    if (a.swing !== 'none' && b.swing !== 'none') b.swing = 'none';
  }
  const exterior = world.openings.filter((o) => isPassable(o) && !o.partner);
  world.entryCandidates = exterior.map((o) => o.key);
  if (entry && world.openings.some((o) => o.key === entry)) world.entryKey = entry;
  else world.entryKey = (exterior.find((o) => o.type === 'door') || exterior[0] || {}).key || null;

  let minx = Infinity, minz = Infinity, maxx = -Infinity, maxz = -Infinity;
  for (const r of world.rooms) for (const p of r.polygon) {
    minx = Math.min(minx, p[0]); maxx = Math.max(maxx, p[0]);
    minz = Math.min(minz, p[1]); maxz = Math.max(maxz, p[1]);
  }
  world.bounds = [minx, minz, maxx, maxz];
  world.maxCeiling = Math.max(...world.rooms.map((r) => r.ceiling.value));
  return world;
}

export function worldFromRoom(room, opts = {}) {
  const w = emptyWorld('room', opts.name || room.name || 'Room');
  addRoom(w, opts.roomId || 'room', room, null, { name: opts.name || room.name || 'Room' });
  return finish(w, opts.entry ? `${opts.roomId || 'room'}/${opts.entry}` : null);
}

// house: {rooms:[{id, name, transform:{x,z,rot}, room:{...} | file}], doorways:[{a:{room,opening}, b:{...}}], entry:{room,opening}}
// roomsById: resolved room.json objects for entries that used "file".
export function worldFromHouse(house, roomsById = {}) {
  const w = emptyWorld('house', house.name || 'House');
  for (const entry of house.rooms) {
    const room = entry.room || roomsById[entry.id];
    if (!room) throw new Error(`house.json: room "${entry.id}" has no data (missing file?)`);
    addRoom(w, entry.id, room, entry.transform, { name: entry.name });
  }
  for (const d of house.doorways || []) {
    w.doorways.push({ a: `${d.a.room}/${d.a.opening}`, b: `${d.b.room}/${d.b.opening}` });
  }
  const entry = house.entry ? `${house.entry.room}/${house.entry.opening}` : null;
  return finish(w, entry);
}

export function roomAt(world, p) {
  return world.rooms.find((r) => pointInPolygon(p, r.polygon)) || null;
}

export function openingByKey(world, key) {
  return world.openings.find((o) => o.key === key) || null;
}

export function openingLabel(world, op) {
  if (!op) return 'door';
  const room = world.rooms.find((r) => r.id === op.roomId);
  if (op.partner) {
    const p = openingByKey(world, op.partner);
    const other = world.rooms.find((r) => r.id === p?.roomId);
    return `the ${room?.name || op.roomId} / ${other?.name || ''} doorway`;
  }
  if (op.name) return `the ${op.name}`;
  if (op.key === world.entryKey) return 'the entry door';
  return `${op.type} ${op.id}${world.kind === 'house' ? ` (${room?.name || op.roomId})` : ''}`;
}

// Breadth-first route over the room graph; returns the list of opening keys
// (entry first) an item must pass to reach targetRoomId.
export function doorsOnRoomPath(world, entryKey, targetRoomId) {
  const entry = openingByKey(world, entryKey);
  if (!entry) return null;
  const start = entry.roomId;
  if (!targetRoomId) return null;
  const prev = { [start]: null };
  const q = [start];
  while (q.length) {
    const r = q.shift();
    if (r === targetRoomId) break;
    for (const d of world.doorways) {
      const a = openingByKey(world, d.a), b = openingByKey(world, d.b);
      if (!a || !b) continue;
      for (const [x, y] of [[a, b], [b, a]]) {
        if (x.roomId === r && !(y.roomId in prev)) { prev[y.roomId] = { via: x.key, other: y.key, from: r }; q.push(y.roomId); }
      }
    }
  }
  if (!(targetRoomId in prev)) return null;
  const doors = [];
  let cur = targetRoomId;
  while (prev[cur]) { doors.unshift(prev[cur].via); cur = prev[cur].from; }
  return [entryKey, ...doors];
}

// Effective clear size of a passable opening; doorways use the tighter of the pair.
export function openingClearSize(world, op) {
  let width = op.width, height = op.height;
  if (op.partner) {
    const p = openingByKey(world, op.partner);
    if (p) { width = Math.min(width, p.width); height = Math.min(height, p.height); }
  }
  return { width, height };
}

// Quarter-disc swept by a door leaf, as a convex polygon (hinge first).
export function doorSwingPolygon(op, steps = 10) {
  const hinge = op.hinge === 'b' ? op.b : op.a;
  const along = op.hinge === 'b' ? scale(op.dir, -1) : op.dir;
  const side = op.swing === 'out' ? scale(op.normalIn, -1) : op.normalIn;
  const pts = [hinge];
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * Math.PI / 2;
    const d = add(scale(along, Math.cos(t)), scale(side, Math.sin(t)));
    pts.push(add(hinge, scale(d, op.width)));
  }
  return pts;
}

// Clear zone in front of a doorless opening (or behind a door) that must stay free.
export function openingClearZone(op, depth = 0.45) {
  const n = op.normalIn;
  return [op.a, op.b, add(op.b, scale(n, depth)), add(op.a, scale(n, depth))];
}
