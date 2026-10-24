import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { worldFromRoom, worldFromHouse, doorsOnRoomPath } from '../js/core/world.js';
import { evaluatePlacement } from '../js/logistics/placement.js';

const here = dirname(fileURLToPath(import.meta.url));
const sample = (f) => JSON.parse(readFileSync(join(here, '..', 'samples', f), 'utf8'));

test('room: inward normals and opening endpoints', () => {
  const w = worldFromRoom(sample('bedroom.json'));
  const w0 = w.walls.find((x) => x.id === 'w0');
  assert.ok(Math.abs(w0.normalIn[0]) < 1e-9 && Math.abs(w0.normalIn[1] - 1) < 1e-9);
  const d0 = w.openings.find((o) => o.id === 'd0');
  assert.ok(Math.abs(d0.a[0] - 1.19) < 1e-9 && Math.abs(d0.b[0] - 1.91) < 1e-9);
  assert.equal(w.entryKey, 'room/d0');
});

test('house: transforms line up the doorways and the room graph is connected', () => {
  const w = worldFromHouse(sample('house.json'), { living: sample('living_l.json'), bedroom: sample('bedroom.json') });
  const hd2 = w.openings.find((o) => o.key === 'hall/hd2');
  const bd0 = w.openings.find((o) => o.key === 'bedroom/d0');
  assert.ok(Math.abs(hd2.center[0] - bd0.center[0]) < 1e-9);
  assert.ok(Math.abs(bd0.center[1] - hd2.center[1] - 0.1) < 1e-9);
  assert.equal(w.entryKey, 'hall/hd0');
  assert.deepEqual(doorsOnRoomPath(w, w.entryKey, 'bedroom'), ['hall/hd0', 'hall/hd2']);
  assert.ok(w.entryCandidates.includes('living/d1'));
});

test('rotated room transform', () => {
  const room = sample('bedroom.json');
  const w = worldFromHouse({ rooms: [{ id: 'r', transform: { x: 10, z: 0, rot: 90 }, room }] });
  // yaw 90 deg: local +x -> (0, -1), so local (3.6, 0) -> (10, -3.6)
  const c = w.rooms[0].polygon[1];
  assert.ok(Math.abs(c[0] - 10) < 1e-9 && Math.abs(c[1] + 3.6) < 1e-9);
});

test('placement: overlaps, walls and door swing are flagged; flush is fine', () => {
  const w = worldFromRoom(sample('bedroom.json'));
  const items = [
    { id: 'a', name: 'desk', assembled: { w: 1.4, d: 0.7, h: 0.75 }, pose: { x: 1.0, z: 2.8, yaw: 0 } },
    { id: 'b', name: 'chair', assembled: { w: 0.5, d: 0.5, h: 0.9 }, pose: { x: 1.2, z: 2.6, yaw: 0 } },
    { id: 'c', name: 'bin', assembled: { w: 0.3, d: 0.3, h: 0.4 }, pose: { x: 1.5, z: 0.4, yaw: 0 } },
    { id: 'd', name: 'shelf', assembled: { w: 0.8, d: 0.3, h: 1.0 }, pose: { x: 3.5, z: 0.5, yaw: 0 } },
  ];
  const s = evaluatePlacement(w, items);
  assert.ok(s.a.problems.some((p) => p.kind === 'item' && p.ref === 'b'));
  assert.ok(s.c.problems.some((p) => p.kind === 'swing'));
  assert.ok(s.d.problems.some((p) => p.kind === 'wall'));
  const bed = { id: 'e', name: 'bed', assembled: { w: 1.6, d: 2.0, h: 1.0 }, pose: { x: 2.5, z: 3.2 - 1.005, yaw: 0 } };
  const flush = evaluatePlacement(w, [bed]);
  assert.equal(flush.e.ok, true, JSON.stringify(flush.e.problems));
});
