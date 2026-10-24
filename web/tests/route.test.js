import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { worldFromRoom, worldFromHouse } from '../js/core/world.js';
import { cornerMaxLength } from '../js/logistics/route.js';
import { planRoute, planBuildOrder, analyseCorner } from '../js/logistics/plan.js';

const here = dirname(fileURLToPath(import.meta.url));
const sample = (f) => JSON.parse(readFileSync(join(here, '..', 'samples', f), 'utf8'));
const house = () => worldFromHouse(sample('house.json'), { living: sample('living_l.json'), bedroom: sample('bedroom.json') });
const item = (id, name, a, pose, extra = {}) => ({ id, name, assembled: a, pose: { yaw: 0, ...pose }, ...extra });

test('tightest-corner formula matches the ladder limit and a known value', () => {
  // Zero-width ladder in two 1 m corridors: 2*sqrt(2).
  assert.ok(Math.abs(cornerMaxLength(1, 1, 0) - 2 * Math.SQRT2) < 1e-3);
  // 0.9 m corridors, 0.5 m wide load: (0.9 - 0.5/sqrt2) * 2*sqrt2.
  const expect = (0.9 - 0.5 / Math.SQRT2) * 2 * Math.SQRT2;
  assert.ok(Math.abs(cornerMaxLength(0.9, 0.9, 0.5) - expect) < 1e-3);
  assert.equal(cornerMaxLength(0.9, 0.9, 0.95), 0);
});

// An L-shaped hallway on its own: 0.9 m legs, door at the far end of leg 1,
// targets in leg 2.
const lHall = () => worldFromRoom({
  floor: { polygon: [[0, 0], [5, 0], [5, 4], [4.1, 4], [4.1, 0.9], [0, 0.9]] },
  ceiling_height: { value: 2.4 },
  walls: [
    { id: 'w0', a: [0, 0], b: [5, 0] }, { id: 'w1', a: [5, 0], b: [5, 4] }, { id: 'w2', a: [5, 4], b: [4.1, 4] },
    { id: 'w3', a: [4.1, 4], b: [4.1, 0.9] }, { id: 'w4', a: [4.1, 0.9], b: [0, 0.9] }, { id: 'w5', a: [0, 0.9], b: [0, 0] },
  ],
  openings: [{ id: 'd0', type: 'door', wall: 'w5', offset: 0.04, width: 0.82, height: 2.04 }],
}, { name: 'L hall' });

test('a short item turns the L-shaped hallway corner', () => {
  const w = lHall();
  // 1.2 m bench, 0.45 m wide: the corner limit is about 1.65 m.
  const bench = item('b', 'bench', { w: 1.2, d: 0.45, h: 0.45 }, { x: 4.55, z: 3.2, yaw: Math.PI / 2 });
  const r = planRoute(w, bench, [], w.entryKey);
  assert.equal(r.ok, true, r.explanation);
  assert.ok(r.length > 4 && r.length < 9, `length ${r.length}`);
  const last = r.path[r.path.length - 1];
  assert.ok(Math.hypot(last.x - 4.55, last.z - 3.2) < 0.08);
});

test('a long item cannot turn the L-shaped hallway corner, and we say where', () => {
  const w = lHall();
  // 2.4 m bench: too long to turn flat, too tall to stand on end through a 2.04 m door.
  const bench = item('b', 'long bench', { w: 2.4, d: 0.45, h: 0.45 }, { x: 4.55, z: 2.6, yaw: Math.PI / 2 });
  const r = planRoute(w, bench, [], w.entryKey);
  assert.equal(r.ok, false);
  assert.ok(r.corner, 'expected a corner diagnosis');
  assert.deepEqual(r.corner.corner, [4.1, 0.9]);
  assert.ok(Math.abs(r.corner.a - 0.9) < 1e-6 && Math.abs(r.corner.b - 0.9) < 1e-6);
  assert.ok(r.corner.maxLength < 2.4);
  assert.match(r.explanation, /corner marked in red/);
});

test('route avoids already-placed items and fails when one blocks the way', () => {
  const w = lHall();
  const target = item('t', 'stool', { w: 0.4, d: 0.4, h: 0.45 }, { x: 4.55, z: 3.5 });
  assert.equal(planRoute(w, target, [], w.entryKey).ok, true);
  // A cabinet filling leg 2 blocks it completely.
  const cabinet = item('c', 'cabinet', { w: 0.9, d: 0.5, h: 0.9 }, { x: 4.55, z: 2.0 });
  assert.equal(planRoute(w, target, [cabinet], w.entryKey).ok, false);
});

test('house: a tall wardrobe cannot reach the bedroom assembled, its flat-pack boxes can', () => {
  const w = house();
  const base = { w: 1.5, d: 0.66, h: 2.36 };
  const pose = { x: 4.4, z: 6.3, yaw: Math.PI };
  const assembled = planRoute(w, item('wa', 'wardrobe', base, pose, { flatpack: false }), [], w.entryKey);
  assert.equal(assembled.ok, false, assembled.explanation);
  const flat = planRoute(w, item('wf', 'wardrobe', base, pose, {
    flatpack: true, needs_assembly_space: 0.6,
    packages: [{ w: 1.25, d: 0.70, h: 0.12, count: 4 }],
  }), [], w.entryKey);
  assert.equal(flat.ok, true, flat.explanation);
  assert.equal(flat.mode, 'package');
});

test('house: the door list follows the room graph', () => {
  const w = house();
  const r = planRoute(w, item('f', 'fridge', { w: 0.7, d: 0.73, h: 1.78 }, { x: 1.0, z: -5.0 }), [], w.entryKey);
  assert.equal(r.fit.doors.length, 2);
  assert.equal(r.fit.doors[0].key, 'hall/hd0');
  assert.equal(r.ok, true, r.explanation);
});

test('corner analysis finds the hallway inner corner', () => {
  const w = house();
  const c = analyseCorner(w, { x: 3.0, z: 0.5 }, 0.5);
  assert.ok(c);
  assert.deepEqual(c.corner, [3.1, 0.9]);
});

test('build order: farthest first, and never block a later route', () => {
  const w = lHall();
  const items = [
    item('near', 'near stool', { w: 0.4, d: 0.4, h: 0.45 }, { x: 4.55, z: 1.6 }),
    item('far', 'far cabinet', { w: 0.8, d: 0.4, h: 0.9 }, { x: 4.55, z: 3.7 }),
    item('mid', 'mid shelf', { w: 0.8, d: 0.3, h: 1.2 }, { x: 4.55, z: 2.7 }),
  ];
  const r = planBuildOrder(w, items, w.entryKey);
  assert.deepEqual(r.steps.map((s) => s.itemId), ['far', 'mid', 'near']);
  assert.match(r.steps[0].reason, /farthest/i);
  assert.equal(r.warnings.length, 0);
});
