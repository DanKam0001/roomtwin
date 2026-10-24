// Configuration-space route search (SPEC "Route"). The load is a rectangle
// footprint {hx, hz} (half extents along its local x / z). State = (x, z, yaw)
// on a grid: `res` metres in x/z and `yawSteps` bins over [0, pi) (a rectangle is
// symmetric under a half turn). Weighted A* with an admissible 2D heuristic: the
// shortest path for the largest disc that fits inside the footprint, which is
// always shorter than the footprint's own path. Collision tests are exact
// (segment vs oriented box, box vs box), the grid only discretises the poses.
import {
  segmentIntersectsObb, obbOverlap, obbAabb, pointInPolygon, pointInObb, pointSegmentDistance,
  pointObbDistance, obbCorners,
} from '../core/geom.js';

class Heap {
  constructor(cap = 1 << 16) { this.f = new Float64Array(cap); this.v = new Int32Array(cap); this.n = 0; }
  push(f, v) {
    if (this.n >= this.f.length) {
      const f2 = new Float64Array(this.f.length * 2); f2.set(this.f); this.f = f2;
      const v2 = new Int32Array(this.v.length * 2); v2.set(this.v); this.v = v2;
    }
    let i = this.n++;
    const F = this.f, V = this.v;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (F[p] <= f) break;
      F[i] = F[p]; V[i] = V[p]; i = p;
    }
    F[i] = f; V[i] = v;
  }
  pop() {
    const F = this.f, V = this.v;
    const top = V[0];
    const n = --this.n;
    if (n > 0) {
      const f = F[n], v = V[n];
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && F[c + 1] < F[c]) c++;
        if (F[c] >= f) break;
        F[i] = F[c]; V[i] = V[c]; i = c;
      }
      F[i] = f; V[i] = v;
    }
    return top;
  }
}

// Collision checker over an env (see env.js) with a bucket grid for segments.
export function makeCollider(env, opts = {}) {
  const segs = env.segments;
  const B = opts.bucket ?? 0.5;
  const [x0, z0, x1, z1] = env.bounds;
  const bw = Math.max(1, Math.ceil((x1 - x0) / B)), bh = Math.max(1, Math.ceil((z1 - z0) / B));
  const buckets = Array.from({ length: bw * bh }, () => []);
  segs.forEach((s, idx) => {
    const ia = Math.max(0, Math.floor((Math.min(s[0], s[2]) - x0) / B)), ib = Math.min(bw - 1, Math.floor((Math.max(s[0], s[2]) - x0) / B));
    const ja = Math.max(0, Math.floor((Math.min(s[1], s[3]) - z0) / B)), jb = Math.min(bh - 1, Math.floor((Math.max(s[1], s[3]) - z0) / B));
    for (let j = ja; j <= jb; j++) for (let i = ia; i <= ib; i++) buckets[j * bw + i].push(idx);
  });
  const stamp = new Int32Array(segs.length);
  let tick = 0;
  const boxes = env.boxes || [];

  function forSegmentsNear(minx, minz, maxx, maxz, fn) {
    tick++;
    const ia = Math.max(0, Math.floor((minx - x0) / B)), ib = Math.min(bw - 1, Math.floor((maxx - x0) / B));
    const ja = Math.max(0, Math.floor((minz - z0) / B)), jb = Math.min(bh - 1, Math.floor((maxz - z0) / B));
    for (let j = ja; j <= jb; j++) for (let i = ia; i <= ib; i++) {
      const bk = buckets[j * bw + i];
      for (let n = 0; n < bk.length; n++) {
        const idx = bk[n];
        if (stamp[idx] === tick) continue;
        stamp[idx] = tick;
        if (fn(idx)) return true;
      }
    }
    return false;
  }

  // Returns null when free, else a description of the first thing hit.
  function hit(o) {
    const [ax, az, bx, bz] = obbAabb(o);
    let found = null;
    forSegmentsNear(ax, az, bx, bz, (idx) => {
      const s = segs[idx];
      if (segmentIntersectsObb([s[0], s[1]], [s[2], s[3]], o)) { found = { kind: 'wall', ref: env.segTags?.[idx], seg: idx }; return true; }
      return false;
    });
    if (found) return found;
    for (const b of boxes) if (obbOverlap(o, b, 1e-4)) return { kind: 'box', ref: b.ref, label: b.label };
    return null;
  }

  function clearance(o, maxR = 1) {
    const [ax, az, bx, bz] = obbAabb(o);
    let best = maxR, ref = null, at = null;
    const corners = obbCorners(o);
    forSegmentsNear(ax - maxR, az - maxR, bx + maxR, bz + maxR, (idx) => {
      const s = segs[idx];
      const a = [s[0], s[1]], b = [s[2], s[3]];
      if (segmentIntersectsObb(a, b, o)) { best = 0; ref = env.segTags?.[idx]; at = a; return true; }
      for (const p of [a, b]) { const d = pointObbDistance(p, o); if (d < best) { best = d; ref = env.segTags?.[idx]; at = p; } }
      for (const c of corners) { const d = pointSegmentDistance(c, a, b); if (d < best) { best = d; ref = env.segTags?.[idx]; at = c; } }
      return false;
    });
    for (const b of boxes) {
      for (const c of obbCorners(b)) { const d = pointObbDistance(c, o); if (d < best) { best = d; ref = b.ref; at = c; } }
      for (const c of corners) { const d = pointObbDistance(c, b); if (d < best) { best = d; ref = b.ref; at = c; } }
    }
    return { clearance: best, ref, at };
  }

  function discFree(p, r) {
    const hitSeg = forSegmentsNear(p[0] - r, p[1] - r, p[0] + r, p[1] + r, (idx) => {
      const s = segs[idx];
      return pointSegmentDistance(p, [s[0], s[1]], [s[2], s[3]]) < r;
    });
    if (hitSeg) return false;
    for (const b of boxes) if (pointObbDistance(p, b) < r) return false;
    return true;
  }

  function allowedAt(p) {
    for (const poly of env.allowedPolys) if (pointInPolygon(p, poly)) return true;
    for (const b of env.allowedBoxes) if (pointInObb(p, b)) return true;
    return false;
  }

  return { hit, clearance, discFree, allowedAt };
}

function yawDiffPi(a, b) {
  // smallest difference between two rectangle headings (period pi)
  let d = (b - a) % Math.PI;
  if (d > Math.PI / 2) d -= Math.PI;
  if (d < -Math.PI / 2) d += Math.PI;
  return d;
}

// fp: {hx, hz}; start/goal: {x, z, yaw}; goal.yawFree, goal.tol (m).
export function searchRoute(env, fp, start, goal, opts = {}) {
  const res = opts.res ?? 0.04;
  const K = opts.yawSteps ?? 24;
  const tol = opts.tol ?? 0.02;
  const W = opts.weight ?? 1.5;
  const maxExpand = opts.maxExpand ?? 2500000;
  const col = makeCollider(env);
  const [bx0, bz0, bx1, bz1] = env.bounds;
  const nx = Math.ceil((bx1 - bx0) / res) + 1, nz = Math.ceil((bz1 - bz0) / res) + 1;
  const N2 = nx * nz, N = N2 * K;
  const hx = Math.max(0.005, fp.hx - tol), hz = Math.max(0.005, fp.hz - tol);
  const R = Math.max(hx, hz);
  const dYaw = Math.PI / K;
  const rotCost = R * dYaw * 0.7 + 0.004;
  const X = (i) => bx0 + i * res, Z = (j) => bz0 + j * res;
  const yawOf = (k) => k * dYaw;
  const obbAt = (x, z, yaw) => ({ cx: x, cz: z, hx, hz, yaw });

  // 2D maps: allowed centres, disc-free cells, heuristic distances to goal.
  const discR = Math.max(0.005, Math.min(hx, hz) - res * 0.75);
  const free2d = new Uint8Array(N2);
  for (let j = 0; j < nz; j++) for (let i = 0; i < nx; i++) {
    const p = [X(i), Z(j)];
    if (col.allowedAt(p) && col.discFree(p, discR)) free2d[j * nx + i] = 1;
  }
  const goalTol = Math.max(goal.tol ?? res * 1.5, res * 1.5);
  const hdist = new Float32Array(N2).fill(Infinity);
  const h2 = new Heap();
  const gi0 = Math.max(0, Math.floor((goal.x - goalTol - bx0) / res)), gi1 = Math.min(nx - 1, Math.ceil((goal.x + goalTol - bx0) / res));
  const gj0 = Math.max(0, Math.floor((goal.z - goalTol - bz0) / res)), gj1 = Math.min(nz - 1, Math.ceil((goal.z + goalTol - bz0) / res));
  const isGoalCell = new Uint8Array(N2);
  for (let j = gj0; j <= gj1; j++) for (let i = gi0; i <= gi1; i++) {
    if (Math.hypot(X(i) - goal.x, Z(j) - goal.z) <= goalTol) {
      const c = j * nx + i;
      isGoalCell[c] = 1;
      if (free2d[c]) { hdist[c] = 0; h2.push(0, c); }
    }
  }
  const D8 = [[1, 0, 1], [-1, 0, 1], [0, 1, 1], [0, -1, 1], [1, 1, Math.SQRT2], [1, -1, Math.SQRT2], [-1, 1, Math.SQRT2], [-1, -1, Math.SQRT2]];
  while (h2.n) {
    const c = h2.pop();
    const ci = c % nx, cj = (c / nx) | 0;
    const dc = hdist[c];
    for (const [di, dj, w] of D8) {
      const ni = ci + di, nj = cj + dj;
      if (ni < 0 || nj < 0 || ni >= nx || nj >= nz) continue;
      const n = nj * nx + ni;
      if (!free2d[n]) continue;
      const nd = dc + w * res;
      if (nd < hdist[n]) { hdist[n] = nd; h2.push(nd, n); }
    }
  }

  const snapI = (x) => Math.min(nx - 1, Math.max(0, Math.round((x - bx0) / res)));
  const snapJ = (z) => Math.min(nz - 1, Math.max(0, Math.round((z - bz0) / res)));
  const snapK = (yaw) => ((Math.round((((yaw % Math.PI) + Math.PI) % Math.PI) / dYaw) % K) + K) % K;

  const result = { ok: false, path: [], length: 0, expanded: 0, stuck: null, reason: null, footprint: { hx: fp.hx, hz: fp.hz } };

  const goalObb = obbAt(goal.x, goal.z, goal.yaw || 0);
  if (!goal.yawFree && col.hit(goalObb)) { result.reason = 'goal-blocked'; result.stuck = { x: goal.x, z: goal.z, yaw: goal.yaw }; result.blocker = col.hit(goalObb); return result; }

  // Start: nearest collision-free grid state around the requested pose.
  let s0 = -1;
  const si = snapI(start.x), sj = snapJ(start.z), sk = snapK(start.yaw || 0);
  outer: for (let r = 0; r <= 6; r++) {
    for (let dj = -r; dj <= r; dj++) for (let di = -r; di <= r; di++) {
      if (Math.max(Math.abs(di), Math.abs(dj)) !== r) continue;
      const i = si + di, j = sj + dj;
      if (i < 0 || j < 0 || i >= nx || j >= nz) continue;
      for (let dk = 0; dk < K; dk++) {
        const k = (sk + (dk % 2 ? -1 : 1) * ((dk + 1) >> 1) + K) % K;
        if (!col.allowedAt([X(i), Z(j)])) continue;
        if (!col.hit(obbAt(X(i), Z(j), yawOf(k)))) { s0 = (k * nz + j) * nx + i; break outer; }
      }
    }
  }
  if (s0 < 0) { result.reason = 'start-blocked'; result.stuck = { x: start.x, z: start.z, yaw: start.yaw }; return result; }

  const s0c = s0 % N2;
  if (!isFinite(hdist[s0c])) {
    // Even the inscribed disc cannot reach the goal: report the closest reachable cell.
    const seen = new Uint8Array(N2); const q = [s0c]; seen[s0c] = 1; let best = s0c, bestD = Infinity;
    while (q.length) {
      const c = q.pop();
      const d = Math.hypot(X(c % nx) - goal.x, Z((c / nx) | 0) - goal.z);
      if (d < bestD) { bestD = d; best = c; }
      const ci = c % nx, cj = (c / nx) | 0;
      for (const [di, dj] of D8) {
        const ni = ci + di, nj = cj + dj;
        if (ni < 0 || nj < 0 || ni >= nx || nj >= nz) continue;
        const n = nj * nx + ni;
        if (!seen[n] && free2d[n]) { seen[n] = 1; q.push(n); }
      }
    }
    result.reason = 'too-narrow';
    result.stuck = { x: X(best % nx), z: Z((best / nx) | 0), yaw: start.yaw || 0 };
    result.discRadius = discR;
    return result;
  }

  const g = new Float32Array(N).fill(Infinity);
  const parent = new Int32Array(N).fill(-1);
  const closed = new Uint8Array(N);
  const cache = new Uint8Array(N); // 0 unknown, 1 free, 2 hit
  const open = new Heap(1 << 18);
  g[s0] = 0;
  open.push(W * hdist[s0c], s0);
  const kGoal = snapK(goal.yaw || 0);
  let bestS = s0, bestH = hdist[s0c], found = -1, expanded = 0;

  const isFree = (s, i, j, k) => {
    let c = cache[s];
    if (c === 0) { c = col.hit(obbAt(X(i), Z(j), yawOf(k))) ? 2 : 1; cache[s] = c; }
    return c === 1;
  };

  while (open.n) {
    const s = open.pop();
    if (closed[s]) continue;
    closed[s] = 1;
    expanded++;
    const c = s % N2, k = (s / N2) | 0, i = c % nx, j = (c / nx) | 0;
    const hc = hdist[c];
    if (hc < bestH) { bestH = hc; bestS = s; }
    if (isGoalCell[c] && (goal.yawFree || Math.abs(((k - kGoal + K + K / 2) % K) - K / 2) <= 1)) { found = s; break; }
    if (expanded > maxExpand) { result.reason = 'search-limit'; break; }
    const gs = g[s];
    for (let n = 0; n < 10; n++) {
      let ni = i, nj = j, nk = k, w;
      if (n < 8) { ni = i + D8[n][0]; nj = j + D8[n][1]; w = D8[n][2] * res; }
      else { nk = (k + (n === 8 ? 1 : K - 1)) % K; w = rotCost; }
      if (ni < 0 || nj < 0 || ni >= nx || nj >= nz) continue;
      const nc = nj * nx + ni;
      const hn = hdist[nc];
      if (!isFinite(hn)) continue;
      const ns = nk * N2 + nc;
      if (closed[ns]) continue;
      const ng = gs + w;
      if (ng >= g[ns]) continue;
      if (!isFree(ns, ni, nj, nk)) continue;
      g[ns] = ng; parent[ns] = s;
      open.push(ng + W * hn, ns);
    }
  }
  result.expanded = expanded;

  const poseOf = (s) => { const c = s % N2; return { x: X(c % nx), z: Z((c / nx) | 0), yaw: yawOf((s / N2) | 0) }; };
  if (found < 0) {
    result.reason = result.reason || 'no-path';
    result.stuck = poseOf(bestS);
    result.stuckHit = col.hit(obbAt(result.stuck.x, result.stuck.z, result.stuck.yaw));
    // Walk back the explored chain to show how far it got.
    const chain = []; let t = bestS;
    while (t >= 0) { chain.push(poseOf(t)); t = parent[t]; }
    result.partial = chain.reverse();
    return result;
  }
  let raw = []; let t = found;
  while (t >= 0) { raw.push(poseOf(t)); t = parent[t]; }
  raw.reverse();
  // Snap the first/last poses to the exact requested ones when they are free.
  if (!goal.yawFree) {
    const last = raw[raw.length - 1];
    const gp = { x: goal.x, z: goal.z, yaw: last.yaw + yawDiffPi(last.yaw, goal.yaw) };
    if (motionFree(col, last, gp, hx, hz, res)) raw.push(gp);
  }
  const path = smooth(col, raw, hx, hz, res);
  result.ok = true;
  result.path = path;
  result.length = path.reduce((s, p, idx) => (idx ? s + Math.hypot(p.x - path[idx - 1].x, p.z - path[idx - 1].z) : 0), 0);
  // Tightest point along the path, measured with the true (unshrunk) footprint.
  let tight = { clearance: Infinity };
  for (const p of densify(path, res)) {
    const c = col.clearance({ cx: p.x, cz: p.z, hx: fp.hx, hz: fp.hz, yaw: p.yaw }, 0.6);
    if (c.clearance < tight.clearance) tight = { ...c, pose: p };
  }
  result.tightest = tight;
  return result;
}

function motionFree(col, a, b, hx, hz, res) {
  const R = Math.max(hx, hz);
  const dy = yawDiffPi(a.yaw, b.yaw);
  const n = Math.max(1, Math.ceil(Math.max(Math.hypot(b.x - a.x, b.z - a.z), Math.abs(dy) * R) / (res * 0.5)));
  for (let i = 1; i <= n; i++) {
    const t = i / n;
    if (col.hit({ cx: a.x + (b.x - a.x) * t, cz: a.z + (b.z - a.z) * t, hx, hz, yaw: a.yaw + dy * t })) return false;
  }
  return true;
}

function smooth(col, raw, hx, hz, res) {
  if (raw.length < 3) return raw;
  const out = [raw[0]];
  let i = 0;
  while (i < raw.length - 1) {
    let j = Math.min(raw.length - 1, i + 80);
    while (j > i + 1 && !motionFree(col, raw[i], raw[j], hx, hz, res)) j--;
    // make yaw continuous for animation
    const prev = out[out.length - 1];
    const q = raw[j];
    out.push({ x: q.x, z: q.z, yaw: prev.yaw + yawDiffPi(prev.yaw, q.yaw) });
    i = j;
  }
  return out;
}

export function densify(path, step = 0.05) {
  const out = [];
  for (let i = 0; i < path.length - 1; i++) {
    const a = path[i], b = path[i + 1];
    const n = Math.max(1, Math.ceil(Math.max(Math.hypot(b.x - a.x, b.z - a.z), Math.abs(b.yaw - a.yaw) * 0.5) / step));
    for (let s = 0; s < n; s++) {
      const t = s / n;
      out.push({ x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t, yaw: a.yaw + (b.yaw - a.yaw) * t });
    }
  }
  if (path.length) out.push(path[path.length - 1]);
  return out;
}

// Tightest-corner test (SPEC): longest rectangle of width w that can be carried
// flat around a right-angle corner joining corridors of widths a and b.
// L(theta) = (a - w cos t)/sin t + (b - w sin t)/cos t, minimised over t.
export function cornerMaxLength(a, b, w) {
  if (w >= Math.min(a, b)) return 0;
  let best = Infinity;
  for (let i = 1; i < 2000; i++) {
    const t = (i / 2000) * Math.PI / 2;
    const L = (a - w * Math.cos(t)) / Math.sin(t) + (b - w * Math.sin(t)) / Math.cos(t);
    if (L < best) best = L;
  }
  return Math.max(0, best);
}

export { yawDiffPi };
