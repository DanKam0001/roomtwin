// High-level logistics: which doors an item passes, how it is carried, the
// configuration-space route, why it gets stuck, and the build order.
// Pure (runs in the browser main thread, a Web Worker, or node tests).
import { roomAt, doorsOnRoomPath, openingByKey, openingLabel, openingClearSize, isPassable } from '../core/world.js';
import { polygonArea, sub, cross, dist, raySegment, norm, obbOverlap, add, scale, pointInPolygon } from '../core/geom.js';
import { assessItemDoors, evaluateOrientations, orientationKey, orientationPhrase, rankCandidates, DEFAULT_CLEARANCE } from './doorfit.js';
import { buildEnv, itemObb, obstacleObb } from './env.js';
import { searchRoute, cornerMaxLength, densify } from './route.js';

const SQUEEZE_TOL = 0.02; // route search shrinks the footprint by this per side (grid tolerance)

const m2 = (v) => `${v.toFixed(2)} m`;
const cm = (v) => `${Math.max(0, Math.round(v * 100))} cm`;
const dimsCm = (b) => `${Math.round(b.w * 100)} x ${Math.round(b.d * 100)} x ${Math.round(b.h * 100)} cm`;

export function doorsForItem(world, item, entryKey) {
  const room = roomAt(world, [item.pose.x, item.pose.z]);
  if (!room) return { room: null, doors: null };
  const keys = doorsOnRoomPath(world, entryKey, room.id);
  if (!keys) return { room, doors: null };
  const doors = keys.map((k) => {
    const op = openingByKey(world, k);
    const size = openingClearSize(world, op);
    return { key: k, label: openingLabel(world, op), width: size.width, height: size.height };
  });
  return { room, doors };
}

export function checkDoors(world, item, entryKey, clearance = DEFAULT_CLEARANCE) {
  const { room, doors } = doorsForItem(world, item, entryKey);
  if (!room) return { verdict: 'outside', headline: 'Place it inside a room first.', detail: '' };
  if (!doors) return { verdict: 'unreachable', headline: 'No doorway connects the entry to this room.', detail: '' };
  const res = { ...assessItemDoors(item, doors, clearance), doors, roomId: room.id };
  if (res.verdict === 'no') {
    // Would another exterior door do? (e.g. a wide garden slider)
    for (const k of (world.entryCandidates || []).filter((x) => x !== entryKey)) {
      const alt = doorsForItem(world, item, k).doors;
      if (!alt) continue;
      const a = assessItemDoors(item, alt, clearance);
      if (a.verdict === 'fits' || a.verdict === 'flatpack') {
        res.alternative = { entryKey: k, label: openingLabel(world, openingByKey(world, k)), verdict: a.verdict };
        res.detail += ` It ${a.verdict === 'fits' ? 'fits' : 'goes in as boxes'} through ${res.alternative.label}, if you can bring it in that way.`;
        break;
      }
    }
  }
  return res;
}

function largestPackage(item) {
  const pk = (item.packages || []).filter((p) => p && p.w && p.d && p.h);
  if (!pk.length) return null;
  return [...pk].sort((a, b) => Math.max(b.w, b.d, b.h) - Math.max(a.w, a.d, a.h) || b.w * b.d * b.h - a.w * a.d * a.h)[0];
}

function passablePortals(world) {
  return world.openings.filter((o) => isPassable(o));
}

function itemVolume(item) { const a = item.assembled; return a.w * a.d * a.h; }

// Corner diagnosis near a pose: nearest reflex (inside) corner of any room,
// the two corridor widths meeting there, and the tightest-corner length limit.
export function analyseCorner(world, pose, acrossWidth, maxDist = 1.8) {
  let best = null;
  for (const room of world.rooms) {
    const P = room.polygon, n = P.length, s = Math.sign(polygonArea(P));
    for (let i = 0; i < n; i++) {
      const prev = P[(i - 1 + n) % n], cur = P[i], next = P[(i + 1) % n];
      const turn = cross(sub(cur, prev), sub(next, cur));
      if (Math.sign(turn) === s || Math.abs(turn) < 1e-9) continue; // convex corner
      const d = dist(cur, [pose.x, pose.z]);
      if (d > maxDist || (best && d >= best.d)) continue;
      const width = (from, to) => {
        const dir = norm(sub(to, from));
        const along = Math.min(0.3, dist(from, to) / 2);
        const p = add(from, scale(dir, along));
        let nrm = [-dir[1], dir[0]];
        if (!pointInPolygon(add(p, scale(nrm, 0.01)), P)) nrm = [-nrm[0], -nrm[1]];
        let t = Infinity;
        for (const w of world.walls) t = Math.min(t, raySegment(add(p, scale(nrm, 1e-4)), nrm, w.a, w.b));
        return t + 1e-4;
      };
      const a = width(cur, prev), b = width(cur, next);
      if (!isFinite(a) || !isFinite(b)) continue;
      best = { d, corner: cur, a, b, roomId: room.id };
    }
  }
  // Turning into a door from a narrow corridor is a corner too: one leg is the
  // corridor in front of the door, the other is the door opening itself.
  for (const op of world.openings) {
    if (!isPassable(op)) continue;
    const d = dist(op.center, [pose.x, pose.z]);
    if (d > maxDist || (best && d >= best.d)) continue;
    const side = Math.sign((pose.x - op.center[0]) * op.normalIn[0] + (pose.z - op.center[1]) * op.normalIn[1]) || 1;
    const nrm = scale(op.normalIn, side);
    let a = Infinity;
    const p0 = add(op.center, scale(nrm, 1e-4));
    for (const w of world.walls) if (w.key !== op.wallKey) a = Math.min(a, raySegment(p0, nrm, w.a, w.b) + 1e-4);
    if (!isFinite(a) || a > 1.5) continue; // a door into an open room is not a tight turn
    best = { d, corner: op.center, a, b: op.width, roomId: op.roomId, door: op.key };
  }
  if (!best) return null;
  best.maxLength = cornerMaxLength(best.a, best.b, acrossWidth);
  return best;
}

function describeRef(world, items, ref) {
  if (!ref) return 'a wall';
  if (ref.startsWith('jamb:')) return 'a door frame';
  const wall = world.walls.find((w) => w.key === ref);
  if (wall) return `wall ${wall.id}${world.kind === 'house' ? ` (${world.rooms.find((r) => r.id === wall.roomId)?.name})` : ''}`;
  const it = items.find((i) => i.id === ref);
  if (it) return `the ${it.name}`;
  const ob = world.obstacles.find((o) => o.key === ref);
  if (ob) return /^(unknown|obstacle|object)?$/i.test(ob.label || '') ? 'the existing furniture' : `the ${ob.label}`;
  return 'something';
}

// Route for one item. obstacleItems: items already in place (excluding this one).
export function planRoute(world, item, obstacleItems, entryKey, opts = {}) {
  const clearance = opts.clearance ?? DEFAULT_CLEARANCE;
  const fit = checkDoors(world, item, entryKey, clearance);
  const base = { itemId: item.id, ok: false, fit, warnings: [] };
  if (!fit.doors) return { ...base, reason: fit.verdict, explanation: fit.headline };

  const pkg = item.flatpack ? largestPackage(item) : null;
  const mode = pkg ? 'package' : 'assembled';
  const load = pkg ? { ...pkg } : { ...item.assembled, diagonal_depth: item.diagonal_depth };
  if (mode === 'assembled' && !fit.assembledFits) return { ...base, reason: 'door', explanation: `${fit.headline}. ${fit.detail}` };
  if (mode === 'package' && fit.packages && !fit.packages.fits) return { ...base, reason: 'door', explanation: `${fit.headline}. ${fit.detail}` };

  // Room checks that do not depend on the route.
  const room = world.rooms.find((r) => r.id === fit.roomId);
  const ceil = room?.ceiling.value ?? 2.4;
  const a = item.assembled;
  if (a.h > ceil - 0.01) base.warnings.push(`It is ${cm(a.h - ceil + 0.01)} taller than the ceiling (${m2(ceil)}).`);
  if (item.flatpack && a.h > 1.5) {
    const tip = Math.hypot(a.h, a.d);
    if (tip > ceil - 0.01) base.warnings.push(`Built lying down, it needs ${m2(tip)} of height to stand up but the ceiling is ${m2(ceil)}: build it upright where it stands.`);
  }

  // Carry candidates: orientations that pass every door on the room path.
  const portals = passablePortals(world);
  const evalByPortal = {};
  for (const op of portals) {
    const size = openingClearSize(world, op);
    const W = size.width - clearance, H = size.height - clearance;
    const map = {};
    for (const c of evaluateOrientations(load, W, H)) map[orientationKey(c)] = c;
    evalByPortal[op.key] = { map, W, width: size.width };
  }
  const pathKeys = fit.doors.map((d) => d.key);
  const tightKey = pathKeys.reduce((m, k) => (evalByPortal[k].width < evalByPortal[m].width ? k : m), pathKeys[0]);
  const ranked = rankCandidates(Object.values(evalByPortal[tightKey].map));
  const candidates = [];
  const seen = new Set();
  for (const c of ranked) {
    const key = orientationKey(c);
    if (!pathKeys.every((k) => evalByPortal[k].map[key]?.fits)) continue;
    const closed = [], widen = {};
    for (const op of portals) {
      const e = evalByPortal[op.key].map[key];
      if (!e || !e.fits) closed.push(op.key);
      else if (c.footprint.across > op.width - 2 * SQUEEZE_TOL - 0.005) widen[op.key] = c.footprint.across + 2 * SQUEEZE_TOL;
    }
    const fpKey = [Math.round(c.footprint.travel * 100), Math.round(c.footprint.across * 100)].sort().join('x');
    const dedupe = `${fpKey}|${closed.sort().join(',')}|${Object.keys(widen).sort().join(',')}|${c.vertical === 'h' && !c.special}`;
    if (seen.has(dedupe)) continue;
    seen.add(dedupe);
    candidates.push({ c, closed, widen });
  }
  if (!candidates.length) return { ...base, reason: 'door', explanation: 'No single carrying orientation clears every door on the way.' };

  const boxes = [
    ...world.obstacles.filter((o) => o.ymax > 0.05).map(obstacleObb),
    ...obstacleItems.map((it) => itemObb(it)),
  ];
  const entry = openingByKey(world, entryKey);
  const n = entry.normalIn;
  const maxTries = opts.maxTries ?? 4;
  const attempts = [];
  for (const cand of candidates.slice(0, maxTries)) {
    const { c } = cand;
    const env = buildEnv(world, { closed: cand.closed, widen: cand.widen, boxes, entryKey, porchDepth: Math.max(2.5, c.footprint.travel + 1.2) });
    const fp = { hx: c.footprint.across / 2, hz: c.footprint.travel / 2 };
    const startC = add(entry.center, scale(n, -(c.footprint.travel / 2 + 0.3)));
    const start = { x: startC[0], z: startC[1], yaw: Math.atan2(n[0], n[1]) };
    const exact = mode === 'assembled' && c.vertical === 'h' && !c.special;
    const goal = exact
      ? { x: item.pose.x, z: item.pose.z, yaw: c.horizontal === 'w' ? item.pose.yaw : item.pose.yaw + Math.PI / 2, yawFree: false }
      : { x: item.pose.x, z: item.pose.z, yaw: item.pose.yaw, yawFree: true, tol: mode === 'package' ? 0.8 : 0.6 };
    const r = searchRoute(env, fp, start, goal, { res: opts.res ?? 0.04, yawSteps: opts.yawSteps ?? 24, tol: SQUEEZE_TOL, maxExpand: opts.maxExpand });
    attempts.push({ cand, r, fp, exact });
    if (r.ok) break;
  }
  const win = attempts.find((x) => x.r.ok);
  const carry = (x) => ({
    orientation: orientationKey(x.cand.c), phrase: orientationPhrase(x.cand.c),
    travel: x.cand.c.footprint.travel, across: x.cand.c.footprint.across,
    vertical: load[x.cand.c.vertical], box: load, mode,
  });
  const loadName = mode === 'package' ? `the biggest box (${dimsCm(load)})` : `the ${item.name}`;

  if (win) {
    const r = win.r;
    const t = r.tightest;
    let expl = `Route found: ${m2(r.length)} from the entry, carrying ${loadName} ${orientationPhrase(win.cand.c)}.`;
    if (t && isFinite(t.clearance)) {
      expl += t.clearance < SQUEEZE_TOL
        ? ` Tightest squeeze: under ${cm(SQUEEZE_TOL)} to spare by ${describeRef(world, obstacleItems, t.ref)}, so go slowly there.`
        : ` Tightest squeeze: ${cm(t.clearance)} to spare by ${describeRef(world, obstacleItems, t.ref)}.`;
    }
    if (!win.exact) expl += mode === 'package' ? ' Unbox and build it where it goes.' : ' Set it down beside its spot and stand it up there.';
    if (mode === 'package' && item.needs_assembly_space) expl += ` It needs about ${m2(item.needs_assembly_space)} of clear floor around it to build.`;
    return {
      ...base, ok: true, mode, carry: carry(win), path: r.path, length: r.length, tightest: t,
      explanation: expl, attempts: attempts.length,
    };
  }

  // Failure: explain using the most natural carrying orientation.
  const first = attempts[0];
  const r = first.r;
  const fc = first.cand.c;
  let explanation;
  let corner = null;
  const stuckWhat = r.stuckHit ? describeRef(world, obstacleItems, r.stuckHit.ref) : null;
  if (r.reason === 'goal-blocked') {
    explanation = `Its spot is blocked by ${describeRef(world, obstacleItems, r.blocker?.ref)}: move it before checking the route.`;
  } else if (r.reason === 'start-blocked') {
    explanation = 'Something is blocking the entry door itself.';
  } else {
    corner = r.stuck ? analyseCorner(world, r.stuck, fc.footprint.across) : null;
    const size = `${m2(fc.footprint.travel)} long and ${m2(fc.footprint.across)} wide`;
    if (corner && corner.maxLength < fc.footprint.travel + 0.05) {
      const where = corner.door ? `Gets stuck turning into ${openingLabel(world, openingByKey(world, corner.door))} (marked in red): a ${m2(corner.a)} corridor and a ${m2(corner.b)} opening` : `Gets stuck at the corner marked in red: corridors of ${m2(corner.a)} and ${m2(corner.b)}`;
      explanation = `${where} let a ${m2(fc.footprint.across)}-wide load turn only if it is under ${m2(corner.maxLength)} long, and ${loadName} is ${m2(fc.footprint.travel)} carried ${orientationPhrase(fc)}.`;
    } else if (r.reason === 'too-narrow') {
      explanation = `Gets stuck where marked: no gap on the way is wide enough for a load ${size}${stuckWhat ? ` (${stuckWhat} is in the way)` : ''}.`;
    } else {
      explanation = `Gets stuck where marked${stuckWhat ? `, against ${stuckWhat}` : ''}: there is no room to turn ${loadName} (${size}, carried ${orientationPhrase(fc)}).`;
    }
    if (attempts.length > 1) explanation += ` Tried ${attempts.length} ways of carrying it.`;
    if (item.flatpack && mode === 'assembled') explanation += ' It is flat-pack: route the boxes instead.';
  }
  return {
    ...base, ok: false, mode, carry: carry(first), reason: r.reason, stuck: r.stuck, partial: r.partial || null,
    corner, explanation, attempts: attempts.length,
  };
}

function pathCollides(path, fp, box) {
  if (!path) return true;
  for (const p of densify(path, 0.1)) {
    if (obbOverlap({ cx: p.x, cz: p.z, hx: fp.hx - SQUEEZE_TOL, hz: fp.hz - SQUEEZE_TOL, yaw: p.yaw }, box, 1e-4)) return true;
  }
  return false;
}

function assemblyZone(item) {
  const s = item.flatpack ? (item.needs_assembly_space ?? 0.5) : 0;
  return s > 0 ? itemObb(item, s) : null;
}

// SPEC build order: farthest from the entry along the route first; never block a
// later item's route or assembly space; big before small; explain each step.
// Items that cannot come in through the chosen entry try the other exterior doors.
export function planBuildOrder(world, items, entryKey, opts = {}, onProgress = () => {}) {
  const routeOpts = { ...opts, maxTries: opts.maxTries ?? 3 };
  const fpOf = (r) => ({ hx: r.carry.across / 2, hz: r.carry.travel / 2 });
  const cache = {};
  const entryOf = {};
  let done = 0;
  const total = items.length;
  const route = (it, placed) => {
    const r = planRoute(world, it, placed, entryOf[it.id] || entryKey, routeOpts);
    done++; onProgress({ done, total: Math.max(total, done), item: it.name });
    return r;
  };
  for (const it of items) {
    cache[it.id] = route(it, []);
    if (!cache[it.id].ok && cache[it.id].reason !== 'outside' && cache[it.id].reason !== 'goal-blocked') {
      for (const k of (world.entryCandidates || []).filter((x) => x !== entryKey)) {
        entryOf[it.id] = k;
        const alt = route(it, []);
        if (alt.ok) { cache[it.id] = alt; break; }
        delete entryOf[it.id];
      }
    }
  }

  const unreachable = items.filter((it) => !cache[it.id].ok);
  let remaining = items.filter((it) => cache[it.id].ok);
  const placed = [];
  const steps = [];
  const warnings = [];
  const waited = {}; // itemId -> reason it was delayed
  const sharedSpace = new Set(); // mutual assembly-space overlaps (reported once)

  const routeLen = (it) => cache[it.id].length;
  const order = (list) => [...list].sort((x, y) => {
    const dx = routeLen(x), dy = routeLen(y);
    if (Math.abs(dx - dy) > 0.5) return dy - dx;
    return itemVolume(y) - itemVolume(x);
  });
  const inZoneOf = (a, b) => { const z = assemblyZone(b); return !!z && obbOverlap(z, itemObb(a), 1e-4); };

  while (remaining.length) {
    const sorted = order(remaining);
    let chosen = null, chosenRoute = null, tentative = {};
    for (const c of sorted) {
      let rc = cache[c.id];
      if (placed.some((p) => pathCollides(rc.path, fpOf(rc), itemObb(p)))) {
        rc = route(c, placed);
        if (rc.ok) cache[c.id] = rc;
      }
      if (!rc.ok) { waited[c.id] = waited[c.id] || 'the way in was blocked by what came before'; continue; }
      const cObb = itemObb(c);
      let blocks = null;
      const tent = {};
      for (const j of remaining) {
        if (j.id === c.id) continue;
        // Sitting in a later item's building space only forces a wait when the
        // reverse is not also true (a mutual overlap cannot be ordered away).
        if (inZoneOf(c, j)) {
          if (!inZoneOf(j, c)) { blocks = { item: j, why: 'assembly space' }; break; }
          sharedSpace.add([c.name, j.name].sort().join('|'));
        }
        const rj = cache[j.id];
        if (pathCollides(rj.path, fpOf(rj), cObb)) {
          const nr = route(j, [...placed, c]);
          if (!nr.ok) { blocks = { item: j, why: 'route' }; break; }
          tent[j.id] = nr;
        }
      }
      if (blocks) {
        waited[c.id] = blocks.why === 'route'
          ? `putting it in earlier would block the way to the ${blocks.item.name}`
          : `it would sit in the space needed to build the ${blocks.item.name}`;
        continue;
      }
      chosen = c; chosenRoute = rc; tentative = tent; break;
    }
    if (!chosen) {
      chosen = sorted[0];
      chosenRoute = cache[chosen.id];
      warnings.push(`No order keeps everything clear: the ${chosen.name} goes in now, but ${waited[chosen.id] || 'it blocks another item'}.`);
    }
    Object.assign(cache, tentative);
    const idx = steps.length;
    const reasons = [];
    const dist0 = routeLen(chosen);
    const viaOther = entryOf[chosen.id] ? openingLabel(world, openingByKey(world, entryOf[chosen.id])) : null;
    const farthest = remaining.every((o) => routeLen(o) <= dist0 + 1e-6);
    if (idx === 0 && farthest) reasons.push(`farthest from the entry (${m2(dist0)} along the route), so it goes in while the way is clear`);
    else if (farthest) reasons.push(`farthest of what is left (${m2(dist0)} along the route)`);
    else reasons.push(`${m2(dist0)} along the route`);
    if (viaOther) reasons.push(`comes in through ${viaOther}, because it cannot turn in from the main entry`);
    const peers = remaining.filter((o) => o.id !== chosen.id && Math.abs(routeLen(o) - dist0) <= 0.5);
    if (peers.length && peers.every((o) => itemVolume(o) <= itemVolume(chosen))) reasons.push(`bigger than the ${peers.map((p) => p.name).join(' and ')} at a similar distance, so it goes first`);
    if (waited[chosen.id]) reasons.push(`waited because ${waited[chosen.id]}`);
    if (chosen.flatpack) reasons.push(`build it in place (about ${m2(chosen.needs_assembly_space ?? 0.5)} of space around it)`);
    if (remaining.length === 1 && idx > 0 && items.every((o) => !cache[o.id].ok || routeLen(o) >= dist0 - 1e-6)) reasons.push('closest to the entry, so it goes in last');
    steps.push({
      n: idx + 1, itemId: chosen.id, name: chosen.name, distance: dist0, entryKey: entryOf[chosen.id] || entryKey,
      reason: reasons.join('; ').replace(/^./, (ch) => ch.toUpperCase()) + '.',
      path: chosenRoute.path, carry: chosenRoute.carry, mode: chosenRoute.mode,
    });
    placed.push(chosen);
    remaining = remaining.filter((o) => o.id !== chosen.id);
  }
  for (const pair of sharedSpace) {
    const [a, b] = pair.split('|');
    warnings.push(`The ${a} and the ${b} sit inside each other's building space: build each one in the open middle of the room, then slide it into place.`);
  }
  for (const u of unreachable) warnings.push(`The ${u.name} cannot be brought in: ${cache[u.id].explanation}`);
  return { ok: !unreachable.length && !warnings.length, steps, warnings, unreachable: unreachable.map((u) => u.id) };
}

// Single-item check for the UI: if the chosen entry fails, see whether another
// exterior door works and say so.
export function planRouteWithAlternatives(world, item, obstacleItems, entryKey, opts = {}) {
  const r = planRoute(world, item, obstacleItems, entryKey, opts);
  if (r.ok || r.reason === 'outside' || r.reason === 'goal-blocked') return r;
  for (const k of (world.entryCandidates || []).filter((x) => x !== entryKey).slice(0, 3)) {
    const alt = planRoute(world, item, obstacleItems, k, opts);
    if (alt.ok) {
      const label = openingLabel(world, openingByKey(world, k));
      r.alternative = { entryKey: k, label, result: alt };
      r.explanation += ` It does go in through ${label} (${m2(alt.length)}).`;
      break;
    }
  }
  return r;
}
