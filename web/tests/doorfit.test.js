import test from 'node:test';
import assert from 'node:assert/strict';
import { fitBox, rectInWindow, carverFits, assessItemDoors } from '../js/logistics/doorfit.js';

const door = (w, h, label = 'the door') => ({ key: 'd', label, width: w, height: h });

test('fits upright when the depth clears the door', () => {
  // Fridge 70 x 73 x 178 through an 82 x 204 door.
  const r = fitBox({ w: 0.70, d: 0.73, h: 1.78 }, door(0.82, 2.04));
  assert.equal(r.fits, true);
  assert.equal(r.best.vertical, 'h');
  assert.equal(r.best.tilted, false);
});

test("numeric tilt search agrees with Carver's closed form", () => {
  const W = 0.80, H = 2.02;
  for (const [p, q] of [[2.05, 0.05], [2.08, 0.10], [2.03, 0.20], [2.03, 0.25], [2.10, 0.30], [1.9, 0.5], [2.2, 0.1]]) {
    const r = rectInWindow(p, q, W, H, 45);
    const r2 = rectInWindow(q, p, W, H, 45);
    const numeric = Math.max(r.slack, r2.slack) >= -1e-6;
    assert.equal(numeric, carverFits(p, q, H, W), `p=${p} q=${q}`);
  }
});

test('a 210 cm square panel only fits tilted in the door plane', () => {
  // e.g. a sliding wardrobe door: 2.10 x 2.10 x 0.08. Upright it is 8 cm too tall
  // for a 204 cm door (2 cm clearance); rotated about 17 degrees it slips through.
  const r = fitBox({ w: 2.10, d: 0.08, h: 2.10 }, door(0.82, 2.04));
  assert.equal(r.fits, true);
  assert.equal(r.best.tilted, true);
  assert.ok(r.best.theta > 1 && r.best.theta < 45);
  assert.ok(r.candidates.every((c) => c.tilted || !c.fits));
  assert.equal(carverFits(2.10, 0.08, 2.02, 0.80), true);
});

test('sofa that only fits tilted back on its diagonal depth', () => {
  const sofa = { name: '3-seat sofa', assembled: { w: 2.20, d: 0.93, h: 0.85 }, diagonal_depth: 0.76, flatpack: false };
  const res = assessItemDoors(sofa, [door(0.82, 2.04, 'the front door')]);
  assert.equal(res.verdict, 'fits');
  assert.equal(res.tightest.result.best.special, 'diagonal');
  assert.match(res.headline, /tipped back/);
  // As a plain box (no diagonal depth) it does not fit at all.
  const box = assessItemDoors({ ...sofa, diagonal_depth: undefined }, [door(0.82, 2.04)]);
  assert.equal(box.verdict, 'no');
  assert.match(box.headline, /cm too wide/);
});

test('wardrobe fails assembled but passes as flat-pack', () => {
  const wardrobe = {
    name: 'Wardrobe', assembled: { w: 1.0, d: 0.65, h: 2.0 }, flatpack: true,
    packages: [{ w: 2.02, d: 0.60, h: 0.10, count: 2 }, { w: 1.0, d: 0.58, h: 0.08, count: 1 }],
  };
  const res = assessItemDoors(wardrobe, [door(0.62, 2.04, 'the cupboard door')]);
  assert.equal(res.assembledFits, false);
  assert.equal(res.verdict, 'flatpack');
  assert.match(res.headline, /^Does not fit: 5 cm too wide even/);
  assert.match(res.detail, /flat-pack, so carry the boxes in and build it in the room/);
});

test('reports the shortfall in plain English', () => {
  // 86 cm deep armchair through a 72 cm door with 2 cm clearance: 16 cm.
  const res = assessItemDoors({ name: 'Armchair', assembled: { w: 0.9, d: 0.86, h: 0.95 } }, [door(0.72, 2.04)]);
  assert.equal(res.verdict, 'no');
  assert.match(res.headline, /16 cm too wide/);
});

test('clearance is applied (2 cm default)', () => {
  assert.equal(fitBox({ w: 0.81, d: 0.81, h: 2.5 }, door(0.82, 2.04)).fits, false);
  assert.equal(fitBox({ w: 0.80, d: 0.80, h: 2.5 }, door(0.82, 2.04)).fits, true);
  assert.equal(fitBox({ w: 0.81, d: 0.81, h: 2.5 }, door(0.82, 2.04), 0.0).fits, true);
});

test('the tightest of several doors decides', () => {
  const res = assessItemDoors({ name: 'Desk', assembled: { w: 1.4, d: 0.75, h: 0.75 } },
    [door(0.86, 2.04, 'the front door'), door(0.72, 2.04, 'the bedroom door')]);
  assert.equal(res.verdict, 'no');
  assert.match(res.detail, /bedroom door/);
});
