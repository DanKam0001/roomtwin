// Pure 2D geometry on the floor plane. Coordinates are [x, z] in metres.
// Yaw convention matches three.js rotation.y: local +x maps to (cos y, -sin y),
// local +z maps to (sin y, cos y).

export const EPS = 1e-9;

export function axisX(yaw) { return [Math.cos(yaw), -Math.sin(yaw)]; }
export function axisZ(yaw) { return [Math.sin(yaw), Math.cos(yaw)]; }

export function sub(a, b) { return [a[0] - b[0], a[1] - b[1]]; }
export function add(a, b) { return [a[0] + b[0], a[1] + b[1]]; }
export function scale(a, s) { return [a[0] * s, a[1] * s]; }
export function dot(a, b) { return a[0] * b[0] + a[1] * b[1]; }
export function cross(a, b) { return a[0] * b[1] - a[1] * b[0]; }
export function len(a) { return Math.hypot(a[0], a[1]); }
export function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
export function norm(a) { const l = len(a) || 1; return [a[0] / l, a[1] / l]; }
export function lerp(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]; }

export function polygonArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    s += a[0] * b[1] - b[0] * a[1];
  }
  return s / 2; // signed
}

export function pointInPolygon(p, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], zi = poly[i][1], xj = poly[j][0], zj = poly[j][1];
    if ((zi > p[1]) !== (zj > p[1]) && p[0] < ((xj - xi) * (p[1] - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}

export function polygonCentroid(poly) {
  let a = 0, cx = 0, cz = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i], q = poly[(i + 1) % poly.length];
    const c = p[0] * q[1] - q[0] * p[1];
    a += c; cx += (p[0] + q[0]) * c; cz += (p[1] + q[1]) * c;
  }
  if (Math.abs(a) < EPS) {
    const n = poly.length;
    return [poly.reduce((s, p) => s + p[0], 0) / n, poly.reduce((s, p) => s + p[1], 0) / n];
  }
  return [cx / (3 * a), cz / (3 * a)];
}

export function closestPointOnSegment(p, a, b) {
  const ab = sub(b, a);
  const l2 = dot(ab, ab);
  let t = l2 < EPS ? 0 : dot(sub(p, a), ab) / l2;
  t = Math.max(0, Math.min(1, t));
  return [a[0] + ab[0] * t, a[1] + ab[1] * t, t];
}

export function pointSegmentDistance(p, a, b) {
  const c = closestPointOnSegment(p, a, b);
  return Math.hypot(p[0] - c[0], p[1] - c[1]);
}

// Oriented box: {cx, cz, hx, hz, yaw}; hx is the half-extent along local x.
export function obbCorners(o) {
  const ux = axisX(o.yaw), uz = axisZ(o.yaw);
  const c = [o.cx, o.cz];
  const corners = [];
  for (const [sx, sz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
    corners.push([c[0] + ux[0] * o.hx * sx + uz[0] * o.hz * sz, c[1] + ux[1] * o.hx * sx + uz[1] * o.hz * sz]);
  }
  return corners;
}

export function obbAabb(o) {
  const ux = axisX(o.yaw), uz = axisZ(o.yaw);
  const ex = Math.abs(ux[0]) * o.hx + Math.abs(uz[0]) * o.hz;
  const ez = Math.abs(ux[1]) * o.hx + Math.abs(uz[1]) * o.hz;
  return [o.cx - ex, o.cz - ez, o.cx + ex, o.cz + ez];
}

export function toLocal(o, p) {
  const ux = axisX(o.yaw), uz = axisZ(o.yaw);
  const d = [p[0] - o.cx, p[1] - o.cz];
  return [dot(d, ux), dot(d, uz)];
}

// Exact segment vs oriented box test (Liang-Barsky clip in box frame).
export function segmentIntersectsObb(a, b, o) {
  const p = toLocal(o, a), q = toLocal(o, b);
  let t0 = 0, t1 = 1;
  const dx = q[0] - p[0], dz = q[1] - p[1];
  const clip = (den, num) => {
    // den * t <= num
    if (Math.abs(den) < EPS) return num >= 0;
    const t = num / den;
    if (den > 0) { if (t < t1) t1 = t; } else { if (t > t0) t0 = t; }
    return t0 <= t1;
  };
  return clip(dx, o.hx - p[0]) && clip(-dx, o.hx + p[0]) && clip(dz, o.hz - p[1]) && clip(-dz, o.hz + p[1]);
}

// Penetration-free overlap of two oriented boxes (SAT). Touching does not count.
export function obbOverlap(a, b, margin = 0) {
  const axes = [axisX(a.yaw), axisZ(a.yaw), axisX(b.yaw), axisZ(b.yaw)];
  const d = [b.cx - a.cx, b.cz - a.cz];
  const aAx = [axisX(a.yaw), axisZ(a.yaw)], bAx = [axisX(b.yaw), axisZ(b.yaw)];
  for (const ax of axes) {
    const ra = a.hx * Math.abs(dot(aAx[0], ax)) + a.hz * Math.abs(dot(aAx[1], ax));
    const rb = b.hx * Math.abs(dot(bAx[0], ax)) + b.hz * Math.abs(dot(bAx[1], ax));
    if (Math.abs(dot(d, ax)) >= ra + rb - margin) return false;
  }
  return true;
}

// Convex polygon vs oriented box (SAT). poly is a list of [x,z].
export function convexPolyOverlapsObb(poly, o, margin = 0) {
  const corners = obbCorners(o);
  const axes = [axisX(o.yaw), axisZ(o.yaw)];
  for (let i = 0; i < poly.length; i++) {
    const e = sub(poly[(i + 1) % poly.length], poly[i]);
    axes.push(norm([-e[1], e[0]]));
  }
  for (const ax of axes) {
    let pmin = Infinity, pmax = -Infinity, bmin = Infinity, bmax = -Infinity;
    for (const p of poly) { const v = dot(p, ax); pmin = Math.min(pmin, v); pmax = Math.max(pmax, v); }
    for (const p of corners) { const v = dot(p, ax); bmin = Math.min(bmin, v); bmax = Math.max(bmax, v); }
    if (pmax <= bmin + margin || bmax <= pmin + margin) return false;
  }
  return true;
}

export function pointInObb(p, o) {
  const l = toLocal(o, p);
  return Math.abs(l[0]) <= o.hx && Math.abs(l[1]) <= o.hz;
}

// Distance from a point to an oriented box (0 inside).
export function pointObbDistance(p, o) {
  const l = toLocal(o, p);
  const dx = Math.max(Math.abs(l[0]) - o.hx, 0), dz = Math.max(Math.abs(l[1]) - o.hz, 0);
  return Math.hypot(dx, dz);
}

// Ray from p along dir (unit) to segment; returns distance or Infinity.
export function raySegment(p, dir, a, b) {
  const e = sub(b, a);
  const den = cross(dir, e);
  if (Math.abs(den) < EPS) return Infinity;
  const ap = sub(a, p);
  const t = cross(ap, e) / den;
  const u = cross(ap, dir) / den;
  if (t > 1e-6 && u >= -1e-9 && u <= 1 + 1e-9) return t;
  return Infinity;
}

export function wrapAngle(a) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a <= -Math.PI) a += 2 * Math.PI;
  return a;
}

export function fmtM(v, digits = 2) { return `${v.toFixed(digits)} m`; }
export function fmtCm(v) { return `${Math.round(v * 100)} cm`; }
