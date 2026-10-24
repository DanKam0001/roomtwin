// 2D floor plan on a canvas: measured walls with their uncertainty, openings,
// door swings, obstacles, furniture (drag / rotate / two-finger twist), routes.
import { obbCorners, axisZ, pointInObb, pointSegmentDistance, pointInPolygon, add, scale } from './core/geom.js';
import { isPassable, doorSwingPolygon } from './core/world.js';

const WALL_T = 0.09;
const C = {
  bg: '#0b0d10', floor: '#1b2027', grid: 'rgba(255,255,255,0.045)', grid5: 'rgba(255,255,255,0.08)',
  wall: '#d7dbe0', door: '#8d97a3', window: '#86aec4', text: '#e7e9ec', muted: '#8f98a3',
  accent: '#63d2be', bad: '#f07171', warn: '#e3b65f', obstacle: '#353b44', obstacleLine: '#4b535e',
};

export class PlanView {
  constructor(canvas, hooks) {
    this.cv = canvas;
    this.ctx = canvas.getContext('2d');
    this.hooks = hooks;
    this.world = null;
    this.view = { cx: 0, cz: 0, s: 80 };
    this.pointers = new Map();
    this.gesture = null;
    this.raf = 0;
    this.visible = true;
    this.pulse = 0;
    this.bind();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    this.resize();
  }

  setWorld(world) { this.world = world; this.fit(); }
  setVisible(v) { this.visible = v; if (v) { this.resize(); this.requestRender(); } }

  resize() {
    const r = this.cv.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    this.W = r.width; this.H = r.height; this.dpr = dpr;
    this.cv.width = Math.max(1, Math.round(r.width * dpr));
    this.cv.height = Math.max(1, Math.round(r.height * dpr));
    if (this.autoFit && this.world) this.fit();
    this.requestRender();
  }

  fit() {
    if (!this.world) return;
    this.autoFit = true;
    let [x0, z0, x1, z1] = this.world.bounds;
    // keep the entry arrow and its label in view
    const ek = this.hooks.getState().entryKey;
    const entry = this.world.openings.find((o) => o.key === ek);
    if (entry) {
      const q = add(entry.center, scale(entry.normalIn, -1.1));
      x0 = Math.min(x0, q[0]); x1 = Math.max(x1, q[0]); z0 = Math.min(z0, q[1]); z1 = Math.max(z1, q[1]);
    }
    const inset = this.hooks.insets ? this.hooks.insets() : { top: 0, bottom: 0 };
    const pad = this.W < 520 ? 30 : 70;
    const W = Math.max(100, this.W - pad * 2), H = Math.max(100, this.H - pad * 2 - inset.top - inset.bottom);
    const s = Math.min(W / Math.max(0.5, x1 - x0), H / Math.max(0.5, z1 - z0));
    this.view.s = Math.min(Math.max(s, 12), 400);
    this.view.cx = (x0 + x1) / 2;
    this.view.cz = (z0 + z1) / 2 + ((inset.bottom - inset.top) / 2) / this.view.s;
    this.requestRender();
  }

  toScreen(p) { return [(p[0] - this.view.cx) * this.view.s + this.W / 2, (p[1] - this.view.cz) * this.view.s + this.H / 2]; }
  toWorld(x, y) { return [(x - this.W / 2) / this.view.s + this.view.cx, (y - this.H / 2) / this.view.s + this.view.cz]; }

  requestRender() {
    if (this.raf || !this.visible) return;
    this.raf = requestAnimationFrame(() => { this.raf = 0; this.draw(); });
  }

  // ---------------- interaction ----------------
  bind() {
    const cv = this.cv;
    cv.addEventListener('pointerdown', (e) => this.onDown(e));
    cv.addEventListener('pointermove', (e) => this.onMove(e));
    cv.addEventListener('pointerup', (e) => this.onUp(e));
    cv.addEventListener('pointercancel', (e) => this.onUp(e));
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const k = Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015));
      this.zoomAt(e.offsetX, e.offsetY, k);
    }, { passive: false });
    cv.addEventListener('dblclick', (e) => { if (!this.hitItem(this.toWorld(e.offsetX, e.offsetY))) this.fit(); });
  }

  zoomAt(x, y, k) {
    this.autoFit = false;
    const before = this.toWorld(x, y);
    this.view.s = Math.min(Math.max(this.view.s * k, 10), 600);
    const after = this.toWorld(x, y);
    this.view.cx += before[0] - after[0];
    this.view.cz += before[1] - after[1];
    this.requestRender();
  }

  itemsTopFirst() {
    const st = this.hooks.getState();
    return [...st.items].filter((it) => !st.hidden?.has(it.id)).reverse();
  }

  hitItem(p) {
    for (const it of this.itemsTopFirst()) {
      const o = { cx: it.pose.x, cz: it.pose.z, hx: it.assembled.w / 2, hz: it.assembled.d / 2, yaw: it.pose.yaw };
      if (pointInObb(p, o)) return it;
    }
    return null;
  }

  handlePos(it) {
    const f = axisZ(it.pose.yaw);
    const r = it.assembled.d / 2 + 22 / this.view.s;
    return [it.pose.x + f[0] * r, it.pose.z + f[1] * r];
  }

  hitHandle(p) {
    const st = this.hooks.getState();
    const it = st.items.find((i) => i.id === st.selectedId);
    if (!it || st.locked) return null;
    const h = this.handlePos(it);
    return Math.hypot(p[0] - h[0], p[1] - h[1]) * this.view.s < 16 ? it : null;
  }

  hitDoor(p) {
    if (!this.world) return null;
    let best = null, bd = 0.25;
    for (const op of this.world.openings) {
      if (!isPassable(op) || op.partner) continue;
      const d = pointSegmentDistance(p, op.a, op.b);
      if (d < bd) { bd = d; best = op; }
    }
    return best;
  }

  onDown(e) {
    if (!this.world) return;
    this.cv.setPointerCapture(e.pointerId);
    const pt = { x: e.offsetX, y: e.offsetY, x0: e.offsetX, y0: e.offsetY, t0: performance.now() };
    this.pointers.set(e.pointerId, pt);
    const st = this.hooks.getState();
    if (this.pointers.size === 2) {
      const [a, b] = [...this.pointers.values()];
      const mid = this.toWorld((a.x + b.x) / 2, (a.y + b.y) / 2);
      const sel = st.items.find((i) => i.id === st.selectedId);
      if (sel && !st.locked && this.gesture && (this.gesture.type === 'drag' || this.gesture.type === 'rotate' || this.hitItem(mid) === sel)) {
        this.gesture = { type: 'twist', id: sel.id, yaw0: sel.pose.yaw, ang0: Math.atan2(b.y - a.y, b.x - a.x) };
      } else {
        this.gesture = { type: 'pinch', d0: Math.hypot(b.x - a.x, b.y - a.y), s0: this.view.s, mid0: [(a.x + b.x) / 2, (a.y + b.y) / 2], c0: [this.view.cx, this.view.cz] };
      }
      return;
    }
    if (this.pointers.size > 2) return;
    const p = this.toWorld(pt.x, pt.y);
    const h = this.hitHandle(p);
    if (h) { this.gesture = { type: 'rotate', id: h.id }; return; }
    const it = st.locked ? null : this.hitItem(p);
    if (it) {
      this.hooks.onSelect(it.id);
      this.gesture = { type: 'drag', id: it.id, off: [it.pose.x - p[0], it.pose.z - p[1]], moved: false };
      this.cv.style.cursor = 'grabbing';
      return;
    }
    this.gesture = { type: 'pan', c0: [this.view.cx, this.view.cz], moved: false };
  }

  onMove(e) {
    const pt = this.pointers.get(e.pointerId);
    if (!pt) { this.hover(e); return; }
    pt.x = e.offsetX; pt.y = e.offsetY;
    const g = this.gesture;
    if (!g) return;
    const st = this.hooks.getState();
    if ((g.type === 'pinch' || g.type === 'pan') && g.moved !== false) this.autoFit = false;
    if (g.type === 'pinch' && this.pointers.size >= 2) {
      const [a, b] = [...this.pointers.values()];
      const d = Math.hypot(b.x - a.x, b.y - a.y);
      this.view.s = Math.min(Math.max(g.s0 * d / Math.max(1, g.d0), 10), 600);
      const mid = [(a.x + b.x) / 2, (a.y + b.y) / 2];
      this.view.cx = g.c0[0] - (mid[0] - g.mid0[0]) / this.view.s - (g.mid0[0] - this.W / 2) * (1 / this.view.s - 1 / g.s0);
      this.view.cz = g.c0[1] - (mid[1] - g.mid0[1]) / this.view.s - (g.mid0[1] - this.H / 2) * (1 / this.view.s - 1 / g.s0);
      this.requestRender();
      return;
    }
    if (g.type === 'twist' && this.pointers.size >= 2) {
      const [a, b] = [...this.pointers.values()];
      const ang = Math.atan2(b.y - a.y, b.x - a.x);
      // screen y is world z: a clockwise screen turn is a negative yaw change
      let yaw = g.yaw0 - (ang - g.ang0);
      if (st.snap) yaw = Math.round(yaw / (Math.PI / 12)) * (Math.PI / 12);
      this.hooks.onRotate(g.id, yaw, 'move');
      return;
    }
    const dx = pt.x - pt.x0, dy = pt.y - pt.y0;
    if (Math.hypot(dx, dy) > 4) g.moved = true;
    if (g.type === 'pan') {
      this.view.cx = g.c0[0] - dx / this.view.s;
      this.view.cz = g.c0[1] - dy / this.view.s;
      this.requestRender();
    } else if (g.type === 'drag' && g.moved) {
      const p = this.toWorld(pt.x, pt.y);
      this.hooks.onMove(g.id, { x: p[0] + g.off[0], z: p[1] + g.off[1] }, 'move', e.shiftKey);
    } else if (g.type === 'rotate') {
      const it = st.items.find((i) => i.id === g.id);
      const p = this.toWorld(pt.x, pt.y);
      let yaw = Math.atan2(p[0] - it.pose.x, p[1] - it.pose.z);
      if (st.snap && !e.shiftKey) yaw = Math.round(yaw / (Math.PI / 12)) * (Math.PI / 12);
      this.hooks.onRotate(g.id, yaw, 'move');
    }
  }

  onUp(e) {
    const pt = this.pointers.get(e.pointerId);
    this.pointers.delete(e.pointerId);
    const g = this.gesture;
    if (!g) return;
    if (this.pointers.size > 0) {
      // finishing a two-finger gesture: keep the remaining finger idle
      if (g.type === 'twist') this.hooks.onRotate(g.id, null, 'end');
      this.gesture = { type: 'idle' };
      return;
    }
    this.cv.style.cursor = '';
    if (g.type === 'drag') this.hooks.onMove(g.id, null, 'end');
    if (g.type === 'rotate' || g.type === 'twist') this.hooks.onRotate(g.id, null, 'end');
    if (g.type === 'pan' && !g.moved && pt && performance.now() - pt.t0 < 500) {
      const p = this.toWorld(pt.x, pt.y);
      const door = this.hitDoor(p);
      if (door) this.hooks.onDoorTap(door.key);
      else this.hooks.onSelect(null);
    }
    this.gesture = null;
  }

  hover(e) {
    if (!this.world || e.pointerType !== 'mouse') return;
    const p = this.toWorld(e.offsetX, e.offsetY);
    let cur = '';
    if (this.hitHandle(p)) cur = 'alias';
    else if (this.hitItem(p) && !this.hooks.getState().locked) cur = 'grab';
    else if (this.hitDoor(p)) cur = 'pointer';
    this.cv.style.cursor = cur;
  }

  // ---------------- drawing ----------------
  draw() {
    const ctx = this.ctx, dpr = this.dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, this.W, this.H);
    if (!this.world) return;
    const st = this.hooks.getState();
    const w = this.world;
    const s = this.view.s;

    // floors + grid clipped to the floor
    ctx.save();
    ctx.beginPath();
    for (const r of w.rooms) this.polyPath(r.polygon);
    ctx.fillStyle = C.floor;
    ctx.fill('nonzero');
    ctx.clip('nonzero');
    this.drawGrid();
    ctx.restore();

    // coverage gaps (scan QA)
    for (const r of w.rooms) for (const g of r.coverage?.gaps || []) {
      const wall = w.walls.find((x) => x.roomId === r.id && x.id === g.wall);
      if (!wall) continue;
      const a = add(wall.a, scale(wall.dir, g.from)), b = add(wall.a, scale(wall.dir, g.to));
      const off = scale(wall.normalIn, 6 / s);
      ctx.save();
      ctx.strokeStyle = C.warn; ctx.lineWidth = 3; ctx.setLineDash([5, 4]);
      this.line(add(a, off), add(b, off));
      ctx.restore();
    }

    // obstacles
    for (const o of w.obstacles) {
      ctx.beginPath();
      this.polyPath(obbCorners(o));
      ctx.fillStyle = C.obstacle; ctx.fill();
      ctx.strokeStyle = C.obstacleLine; ctx.lineWidth = 1; ctx.stroke();
      if (s * Math.min(o.hx, o.hz) * 2 > 26 && st.showLabels) this.text(o.label, this.toScreen([o.cx, o.cz]), 11, C.muted, 'center');
    }

    // door swings
    const blocked = new Set();
    for (const it of st.items) for (const p of st.status[it.id]?.problems || []) if (p.kind === 'swing' || p.kind === 'doorway') blocked.add(p.ref);
    for (const op of w.openings) {
      if (op.type !== 'door' || op.swing === 'none') continue;
      const poly = doorSwingPolygon(op, 24);
      const bad = blocked.has(op.key);
      ctx.beginPath();
      this.polyPath(poly);
      ctx.fillStyle = bad ? 'rgba(240,113,113,0.16)' : 'rgba(141,151,163,0.07)';
      ctx.fill();
      ctx.strokeStyle = bad ? C.bad : C.door; ctx.lineWidth = bad ? 1.6 : 1;
      ctx.beginPath();
      const sc = poly.slice(1).map((p) => this.toScreen(p));
      sc.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])));
      ctx.stroke();
      ctx.lineWidth = 2;
      this.line(poly[0], poly[poly.length - 1]); // the open leaf
    }

    // build-order + route overlays under the items
    const ov = st.overlay;
    if (ov?.path?.length > 1) this.drawPath(ov.path, ov.ok ? C.accent : C.bad, !ov.ok);
    if (ov?.partial?.length > 1) this.drawPath(ov.partial, C.bad, true);

    if (st.showLabels) this.drawRoomNames();
    // items
    for (const it of st.items) {
      if (st.hidden?.has(it.id)) continue;
      this.drawItem(it, st, it.id === st.selectedId);
    }

    // walls on top so overlaps read as "through the wall"
    this.drawWalls(st);

    if (ov) this.drawOverlay(ov);
    const hl = st.highlightWall && this.drawWallHighlight(st.highlightWall);
    this.drawScaleBar();
    if (hl || (ov && !ov.ok && (ov.corner || ov.stuck))) { this.pulse = (this.pulse + 1) % 1000; this.requestRender(); }
  }

  // Scan coach: pulse the wall the coach wants rescanned. Returns true while drawn (keeps animating).
  drawWallHighlight(h) {
    const ctx = this.ctx, w = this.world, s = this.view.s;
    const wall = w.walls.find((x) => x.id === h.wall && (!h.room || x.roomId === h.room)) || w.walls.find((x) => x.key === h.wall);
    if (!wall) return false;
    const k = 0.5 + 0.5 * Math.sin(performance.now() / 260);
    const mid = scale(wall.normalIn, -WALL_T / 2);
    const a = add(wall.a, mid), b = add(wall.b, mid);
    ctx.save();
    ctx.lineCap = 'round';
    ctx.strokeStyle = `rgba(227,182,95,${0.18 + 0.3 * k})`; ctx.lineWidth = 14 + 12 * k;
    this.line(a, b);
    ctx.strokeStyle = C.warn; ctx.lineWidth = 3.5;
    this.line(a, b);
    ctx.restore();
    const c = add(add(wall.a, scale(wall.dir, wall.geomLength / 2)), scale(wall.normalIn, 30 / s));
    const p = this.toScreen(c);
    let ang = Math.atan2(wall.dir[1], wall.dir[0]);
    if (ang > Math.PI / 2) ang -= Math.PI; if (ang < -Math.PI / 2) ang += Math.PI;
    this.text(h.label || `Rescan wall ${wall.id}`, p, 12.5, '#ffe2a8', 'center', true, ang, C.warn);
    return true;
  }

  drawGrid() {
    const ctx = this.ctx, s = this.view.s;
    const [x0, z0] = this.toWorld(0, 0), [x1, z1] = this.toWorld(this.W, this.H);
    const step = s > 60 ? 0.5 : 1;
    for (let x = Math.floor(x0 / step) * step; x <= x1; x += step) {
      const major = Math.abs(x - Math.round(x)) < 1e-6;
      ctx.strokeStyle = major ? C.grid5 : C.grid; ctx.lineWidth = 1;
      const sx = Math.round(this.toScreen([x, 0])[0]) + 0.5;
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, this.H); ctx.stroke();
    }
    for (let z = Math.floor(z0 / step) * step; z <= z1; z += step) {
      const major = Math.abs(z - Math.round(z)) < 1e-6;
      ctx.strokeStyle = major ? C.grid5 : C.grid; ctx.lineWidth = 1;
      const sy = Math.round(this.toScreen([0, z])[1]) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(this.W, sy); ctx.stroke();
    }
  }

  drawWalls(st) {
    const ctx = this.ctx, w = this.world, s = this.view.s;
    for (const wall of w.walls) {
      const L = wall.geomLength;
      const ops = wall.openings.map((k) => w.openings.find((o) => o.key === k)).filter(Boolean).sort((a, b) => a.offset - b.offset);
      const out = scale(wall.normalIn, -WALL_T);
      let t = 0;
      const piece = (t0, t1, fill) => {
        if (t1 - t0 < 1e-4) return;
        const a = add(wall.a, scale(wall.dir, t0)), b = add(wall.a, scale(wall.dir, t1));
        ctx.beginPath(); this.polyPath([a, b, add(b, out), add(a, out)]);
        ctx.fillStyle = fill; ctx.fill();
      };
      for (const op of ops) {
        piece(t, op.offset, C.wall);
        const a = op.a, b = op.b;
        if (op.type === 'window') {
          piece(op.offset, op.offset + op.width, '#3a4652');
          ctx.strokeStyle = C.window; ctx.lineWidth = 1.5;
          this.line(add(a, scale(out, 0.33)), add(b, scale(out, 0.33)));
          this.line(add(a, scale(out, 0.66)), add(b, scale(out, 0.66)));
        } else if (op.type === 'opening' && !op.partner) {
          ctx.save(); ctx.strokeStyle = C.door; ctx.setLineDash([4, 4]); ctx.lineWidth = 1; this.line(a, b); ctx.restore();
        }
        // jamb ticks
        ctx.strokeStyle = C.wall; ctx.lineWidth = 2;
        this.line(a, add(a, out)); this.line(b, add(b, out));
        t = op.offset + op.width;
      }
      piece(t, L, C.wall);
    }
    // entry marker
    const entry = w.openings.find((o) => o.key === st.entryKey);
    if (entry) {
      const c = entry.center, n = entry.normalIn;
      const tip = add(c, scale(n, -0.18)), tail = add(c, scale(n, -0.75));
      ctx.strokeStyle = C.accent; ctx.fillStyle = C.accent; ctx.lineWidth = 2.5;
      this.line(tail, tip);
      const left = add(tip, add(scale(n, -0.16), scale(entry.dir, 0.12))), right = add(tip, add(scale(n, -0.16), scale(entry.dir, -0.12)));
      ctx.beginPath(); this.polyPath([tip, left, right]); ctx.fill();
      this.text('ENTRY', this.toScreen(add(c, scale(n, -0.98))), 10.5, C.accent, 'center', true);
    }
    // labels
    if (!st.showLabels) return;
    const inside = w.kind === 'house';
    for (const wall of w.walls) {
      if (wall.geomLength * s < 50) continue;
      // label the longest solid stretch so it never sits on a door
      const ops = wall.openings.map((k) => w.openings.find((o) => o.key === k)).filter(Boolean).sort((a, b) => a.offset - b.offset);
      let best = [0, wall.geomLength], t = 0, bestLen = -1;
      for (const op of [...ops, { offset: wall.geomLength, width: 0 }]) {
        if (op.offset - t > bestLen) { bestLen = op.offset - t; best = [t, op.offset]; }
        t = op.offset + op.width;
      }
      const midT = bestLen * s > 70 ? (best[0] + best[1]) / 2 : wall.geomLength / 2;
      const mid = add(wall.a, scale(wall.dir, midT));
      // outside the wall unless another room is there (shared walls in a house)
      const outPt = add(mid, scale(wall.normalIn, -0.3));
      const crowded = inside && w.rooms.some((r) => r.id !== wall.roomId && pointInPolygon(outPt, r.polygon));
      const off = crowded ? scale(wall.normalIn, 13 / s) : scale(wall.normalIn, -WALL_T - 13 / s);
      const p = this.toScreen(add(mid, off));
      const sig = wall.sigma != null ? ` ±${Math.max(1, Math.round(wall.sigma * 100))} cm` : '';
      let ang = Math.atan2(wall.dir[1], wall.dir[0]);
      if (ang > Math.PI / 2) ang -= Math.PI; if (ang < -Math.PI / 2) ang += Math.PI;
      this.text(`${wall.length.toFixed(2)} m${sig}`, p, 11.5, '#cfe9e3', 'center', false, ang, wall.sigma != null && wall.sigma > 0.025 ? C.warn : null);
    }
    const labelled = new Set();
    for (const op of w.openings) {
      if (op.width * s < 34 || labelled.has(op.key)) continue;
      if (op.partner) labelled.add(op.partner);
      const p = this.toScreen(add(op.center, scale(op.normalIn, 12 / s)));
      let ang = Math.atan2(op.dir[1], op.dir[0]);
      if (ang > Math.PI / 2) ang -= Math.PI; if (ang < -Math.PI / 2) ang += Math.PI;
      const label = op.type === 'window' ? `${Math.round(op.width * 100)} win` : `${Math.round(op.width * 100)} x ${Math.round(op.height * 100)}`;
      this.text(label, p, 10, C.muted, 'center', false, ang);
    }
  }

  drawRoomNames() {
    const w = this.world, s = this.view.s;
    for (const r of w.rooms) {
      const p = this.toScreen(r.centroid);
      const ceil = `ceiling ${r.ceiling.value.toFixed(2)} m${r.ceiling.sigma != null ? ` ±${Math.max(1, Math.round(r.ceiling.sigma * 100))} cm` : ''}`;
      if (s * 2 > 60) {
        this.text(r.name, [p[0], p[1] - 8], 13, 'rgba(231,233,236,0.55)', 'center', true);
        this.text(`${r.area.toFixed(1)} m²  ·  ${ceil}`, [p[0], p[1] + 9], 10.5, 'rgba(148,156,167,0.8)', 'center');
      }
    }
  }

  drawItem(it, st, selected) {
    const ctx = this.ctx, s = this.view.s;
    const o = { cx: it.pose.x, cz: it.pose.z, hx: it.assembled.w / 2, hz: it.assembled.d / 2, yaw: it.pose.yaw };
    const bad = st.status[it.id] && !st.status[it.id].ok;
    const corners = obbCorners(o);
    ctx.beginPath(); this.polyPath(corners);
    ctx.globalAlpha = 0.86; ctx.fillStyle = it.color; ctx.fill(); ctx.globalAlpha = 1;
    if (bad) { ctx.fillStyle = 'rgba(240,113,113,0.33)'; ctx.fill(); }
    ctx.lineWidth = selected || bad ? 2.2 : 1;
    ctx.strokeStyle = bad ? C.bad : selected ? C.accent : 'rgba(0,0,0,0.45)';
    ctx.stroke();
    // front edge
    ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 2.5;
    this.line(corners[2], corners[3]);
    // label
    const minSide = Math.min(it.assembled.w, it.assembled.d) * s;
    const c = this.toScreen([o.cx, o.cz]);
    let ang = -it.pose.yaw;
    if (it.assembled.d > it.assembled.w * 1.3) ang += Math.PI / 2;
    ang = Math.atan2(Math.sin(ang), Math.cos(ang));
    if (ang > Math.PI / 2) ang -= Math.PI; if (ang < -Math.PI / 2) ang += Math.PI;
    if (minSide > 22) {
      this.text(it.name, [c[0], c[1] - (minSide > 44 ? 6 : 0)], 12, '#0d1014', 'center', true, ang);
      if (minSide > 44) this.text(`${Math.round(it.assembled.w * 100)} x ${Math.round(it.assembled.d * 100)}`, [c[0], c[1] + 8], 10.5, 'rgba(13,16,20,0.75)', 'center', false, ang);
    }
    const n = st.orderIndex?.[it.id];
    if (n) {
      const p = this.toScreen(corners[0]);
      ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(p[0], p[1], 10, 0, Math.PI * 2); ctx.fill();
      this.text(String(n), [p[0], p[1] + 0.5], 11, '#06231e', 'center', true);
    }
    if (selected && !st.locked) {
      const h = this.toScreen(this.handlePos(it));
      const f = this.toScreen(add([o.cx, o.cz], scale(axisZ(o.yaw), o.hz)));
      ctx.strokeStyle = C.accent; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(f[0], f[1]); ctx.lineTo(h[0], h[1]); ctx.stroke();
      ctx.fillStyle = '#0e1013'; ctx.beginPath(); ctx.arc(h[0], h[1], 8, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(h[0], h[1], 3.5, 0, Math.PI * 2); ctx.fillStyle = C.accent; ctx.fill();
      const deg = ((Math.round(it.pose.yaw * 180 / Math.PI) % 360) + 360) % 360;
      this.text(`${deg}°`, [h[0], h[1] - 16], 10.5, C.accent, 'center', true);
    }
  }

  drawPath(path, color, dashed) {
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.globalAlpha = 0.9;
    if (dashed) ctx.setLineDash([6, 5]);
    ctx.beginPath();
    path.forEach((p, i) => { const q = this.toScreen([p.x, p.z]); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
    ctx.stroke();
    ctx.restore();
  }

  drawFootprint(pose, fp, color, fillAlpha) {
    const ctx = this.ctx;
    const corners = obbCorners({ cx: pose.x, cz: pose.z, hx: fp.hx, hz: fp.hz, yaw: pose.yaw });
    ctx.beginPath(); this.polyPath(corners);
    ctx.save(); ctx.globalAlpha = fillAlpha; ctx.fillStyle = color; ctx.fill(); ctx.restore();
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
  }

  drawOverlay(ov) {
    const ctx = this.ctx;
    if (ov.ghost && ov.fp) this.drawFootprint(ov.ghost, ov.fp, ov.ok ? C.accent : C.bad, 0.22);
    if (!ov.ok && ov.stuck && ov.fp && !ov.ghost) this.drawFootprint(ov.stuck, ov.fp, C.bad, 0.18);
    if (!ov.ok && ov.corner) {
      const p = this.toScreen(ov.corner.corner);
      const t = (performance.now() % 1400) / 1400;
      ctx.strokeStyle = C.bad; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p[0], p[1], 7 + t * 22, 0, Math.PI * 2); ctx.globalAlpha = 1 - t; ctx.stroke(); ctx.globalAlpha = 1;
      ctx.fillStyle = C.bad; ctx.beginPath(); ctx.arc(p[0], p[1], 5, 0, Math.PI * 2); ctx.fill();
      this.text(`max ${ov.corner.maxLength.toFixed(2)} m long here`, [p[0] + 12, p[1] - 14], 11.5, C.bad, 'left', true);
    } else if (!ov.ok && ov.stuck) {
      const p = this.toScreen([ov.stuck.x, ov.stuck.z]);
      const t = (performance.now() % 1400) / 1400;
      ctx.strokeStyle = C.bad; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p[0], p[1], 7 + t * 22, 0, Math.PI * 2); ctx.globalAlpha = 1 - t; ctx.stroke(); ctx.globalAlpha = 1;
    }
    if (ov.ok && ov.tightest?.pose && isFinite(ov.tightest.clearance) && !ov.ghost) {
      const at = ov.tightest.at || [ov.tightest.pose.x, ov.tightest.pose.z];
      const p = this.toScreen(at);
      ctx.fillStyle = C.accent; ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI * 2); ctx.fill();
      const cmv = Math.round(ov.tightest.clearance * 100);
      this.text(cmv < 2 ? 'tight: < 2 cm' : `tightest: ${cmv} cm`, [p[0] + 9, p[1] - 10], 11, C.accent, 'left', true);
    }
  }

  drawScaleBar() {
    const ctx = this.ctx, s = this.view.s;
    const len = s >= 50 ? 1 : 2;
    const px = len * s;
    const inset = this.hooks.insets ? this.hooks.insets() : { bottom: 0 };
    const x = this.W - px - 20, y = this.H - 22 - (inset.bottomTools ?? inset.bottom ?? 0);
    ctx.strokeStyle = C.muted; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, y - 4); ctx.lineTo(x, y); ctx.lineTo(x + px, y); ctx.lineTo(x + px, y - 4); ctx.stroke();
    this.text(`${len} m`, [x + px / 2, y - 9], 11, C.muted, 'center');
  }

  // ---------------- primitives ----------------
  polyPath(pts) {
    pts.forEach((p, i) => { const q = this.toScreen(p); i ? this.ctx.lineTo(q[0], q[1]) : this.ctx.moveTo(q[0], q[1]); });
    this.ctx.closePath();
  }

  line(a, b) {
    const p = this.toScreen(a), q = this.toScreen(b);
    this.ctx.beginPath(); this.ctx.moveTo(p[0], p[1]); this.ctx.lineTo(q[0], q[1]); this.ctx.stroke();
  }

  text(str, p, size, color, align = 'left', bold = false, angle = 0, badge = null) {
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(p[0], p[1]);
    if (angle) ctx.rotate(angle);
    ctx.font = `${bold ? 650 : 500} ${size}px ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
    ctx.textAlign = align; ctx.textBaseline = 'middle';
    if (badge) {
      const w = ctx.measureText(str).width + 10;
      const x = align === 'center' ? -w / 2 : align === 'right' ? -w : 0;
      ctx.fillStyle = 'rgba(227,182,95,0.16)';
      ctx.strokeStyle = badge; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, -size * 0.75, w, size * 1.5, 5) : ctx.rect(x, -size * 0.75, w, size * 1.5);
      ctx.fill(); ctx.stroke();
    }
    ctx.fillStyle = color;
    ctx.fillText(str, 0, 0);
    ctx.restore();
  }
}

