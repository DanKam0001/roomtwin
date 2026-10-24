// 2D floor plan on a canvas: measured walls with their uncertainty, openings,
// door swings, obstacles, furniture (drag / rotate / two-finger twist), routes.
import { obbCorners, axisZ, pointInObb, pointSegmentDistance, pointInPolygon, pointObbDistance, add, scale } from './core/geom.js';
import { isPassable, doorSwingPolygon } from './core/world.js';

const WALL_T = 0.09;
const C = {
  bg: '#0b0d10', floor: '#1b2027', grid: 'rgba(255,255,255,0.045)', grid5: 'rgba(255,255,255,0.08)',
  wall: '#d7dbe0', door: '#8d97a3', window: '#86aec4', text: '#e7e9ec', muted: '#8f98a3',
  accent: '#63d2be', bad: '#f07171', warn: '#e3b65f', obstacle: 'rgba(148,156,167,0.10)', obstacleLine: 'rgba(148,156,167,0.42)',
  hatch: 'rgba(148,156,167,0.13)',
};
const FONT = 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
// Scanned furniture is "existing"; the pipeline calls unclassified boxes "unknown".
export const obstacleName = (label) => (!label || /^(unknown|obstacle|object)$/i.test(label) ? 'existing' : label);
const uprightAngle = (a) => { a = Math.atan2(Math.sin(a), Math.cos(a)); if (a > Math.PI / 2) a -= Math.PI; if (a < -Math.PI / 2) a += Math.PI; return a; };

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

  setAutoFit(v) {
    if (this.autoFit === v) return;
    this.autoFit = v;
    this.hooks.onViewChange?.(v);
  }

  fit() {
    if (!this.world) return;
    this.setAutoFit(true);
    let [x0, z0, x1, z1] = this.world.bounds;
    // keep the entry arrow and its label in view
    const ek = this.hooks.getState().entryKey;
    const entry = this.world.openings.find((o) => o.key === ek);
    if (entry) {
      const q = add(entry.center, scale(entry.normalIn, -1.1));
      x0 = Math.min(x0, q[0]); x1 = Math.max(x1, q[0]); z0 = Math.min(z0, q[1]); z1 = Math.max(z1, q[1]);
    }
    const inset = this.hooks.insets ? this.hooks.insets() : { top: 0, bottom: 0 };
    // room for the dimension labels that sit outside the walls
    const pad = this.W < 520 ? 36 : 72;
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
    this.setAutoFit(false);
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
    return Math.hypot(p[0] - h[0], p[1] - h[1]) * this.view.s < 22 ? it : null;
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
    if ((g.type === 'pinch' || g.type === 'pan') && g.moved !== false) this.setAutoFit(false);
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

    // Every text label is collected first, then placed by priority so nothing overlaps.
    const placed = this.layoutLabels(this.collectLabels(st), this.reservedBoxes());
    const layer = (name) => { for (const l of placed) if (l.layer === name) this.drawLabel(l); };

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

    // existing furniture from the scan: hatched and dashed, so it reads as "already there"
    for (const o of w.obstacles) {
      ctx.save();
      ctx.beginPath();
      this.polyPath(obbCorners(o));
      ctx.fillStyle = C.obstacle; ctx.fill();
      ctx.clip();
      const [cx, cy] = this.toScreen([o.cx, o.cz]);
      const R = (Math.hypot(o.hx, o.hz) * s) + 2;
      ctx.strokeStyle = C.hatch; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let k = -R * 2; k <= R * 2; k += 7) { ctx.moveTo(cx + k - R, cy + R); ctx.lineTo(cx + k + R, cy - R); }
      ctx.stroke();
      ctx.restore();
      ctx.save();
      ctx.beginPath();
      this.polyPath(obbCorners(o));
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = C.obstacleLine; ctx.lineWidth = 1; ctx.stroke();
      ctx.restore();
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

    layer('under');
    // items
    for (const it of st.items) {
      if (st.hidden?.has(it.id)) continue;
      this.drawItem(it, st, it.id === st.selectedId);
    }
    layer('item');

    // walls on top so overlaps read as "through the wall"
    this.drawWalls(st);
    const hl = st.highlightWall && this.drawWallHighlight(st.highlightWall);
    layer('top');

    if (ov) this.drawOverlay(ov);
    this.drawScaleBar();
    if (hl || (ov && !ov.ok && (ov.corner || ov.stuck))) { this.pulse = (this.pulse + 1) % 1000; this.requestRender(); }
  }

  // ---------------- labels ----------------
  // A label: { layer, prio, cands: [[x, y], ...] screen anchors tried in order,
  //   variants: [[{ t, size, color, bold }, ...lines], ...] tried at each anchor (full text first, then shorter),
  //   angle, badge, maxW (optional: the text must fit in this width) }.
  collectLabels(st) {
    const w = this.world, s = this.view.s, out = [];
    const line = (t, size, color, bold = false) => ({ t, size, color, bold });
    const show = st.showLabels;
    const small = this.W < 520;

    // entry marker
    const entry = w.openings.find((o) => o.key === st.entryKey);
    if (entry) {
      const c = entry.center, n = entry.normalIn;
      // just beyond the tail of the arrow, whichever way the arrow points on screen
      const ext = (Math.abs(n[0]) * 20 + Math.abs(n[1]) * 8 + 6) / s;
      out.push({
        layer: 'top', prio: 100, angle: 0,
        cands: [0.75 + ext, 0.75 + ext + 0.15].map((k) => this.toScreen(add(c, scale(n, -k))))
          .concat([0.45, -0.45].map((k) => this.toScreen(add(add(c, scale(n, -0.45)), scale(entry.dir, k + Math.sign(k) * ext))))),
        variants: [[line('ENTRY', 10.5, C.accent, true)]],
      });
    }

    // the wall the scan coach wants rescanned
    const hw = st.highlightWall && this.highlightedWall(st.highlightWall);
    if (hw) {
      const ang = uprightAngle(Math.atan2(hw.dir[1], hw.dir[0]));
      const c0 = add(hw.a, scale(hw.dir, hw.geomLength / 2));
      out.push({
        layer: 'top', prio: 99, angle: ang, badge: C.warn,
        cands: [30, 52, -30].map((k) => this.toScreen(add(c0, scale(hw.normalIn, k / s)))),
        variants: [[line(st.highlightWall.label || `Rescan wall ${hw.id}`, 12.5, '#ffe2a8', true)], [line(`Wall ${hw.id}`, 12, '#ffe2a8', true)]],
      });
    }

    // route findings: where it gets stuck, and the tightest squeeze
    const ov = st.overlay;
    const callout = (txt, at, color, prio) => {
      const p = this.toScreen(at);
      const ln = line(txt, 11.5, color, true);
      const hw = this.measureLines([ln]).w / 2;
      // important enough to sit over a wall: drawn on a dark pill so it stays readable
      out.push({ layer: 'top', prio, angle: 0, variants: [[ln]], pill: true, overReserved: true,
        cands: [[p[0] + 12 + hw, p[1] - 14], [p[0] - 12 - hw, p[1] - 14], [p[0] + 12 + hw, p[1] + 16], [p[0] - 12 - hw, p[1] + 16], [p[0], p[1] - 28], [p[0], p[1] + 30]] });
    };
    if (ov && !ov.ok && ov.corner) callout(`max ${ov.corner.maxLength.toFixed(2)} m long here`, ov.corner.corner, C.bad, 98);
    if (ov && ov.ok && ov.tightest?.pose && isFinite(ov.tightest.clearance) && !ov.ghost) {
      const cmv = Math.round(ov.tightest.clearance * 100);
      callout(cmv < 2 ? 'tight: < 2 cm' : `tightest: ${cmv} cm`, ov.tightest.at || [ov.tightest.pose.x, ov.tightest.pose.z], C.accent, 97);
    }

    // furniture names, along the long side, only if they fit inside the piece
    for (const it of st.items) {
      if (st.hidden?.has(it.id)) continue;
      const W = it.assembled.w * s, D = it.assembled.d * s;
      if (Math.min(W, D) < 18) continue;
      const along = it.assembled.d > it.assembled.w * 1.3;
      const ang = uprightAngle(-it.pose.yaw + (along ? Math.PI / 2 : 0));
      const room = (along ? D : W) - 8;
      const c = this.toScreen([it.pose.x, it.pose.z]);
      const short = it.name.split(/\s+/)[0];
      const big = Math.min(W, D) > 44;
      const dimsTxt = `${Math.round(it.assembled.w * 100)} x ${Math.round(it.assembled.d * 100)}`;
      out.push({
        layer: 'item', prio: it.id === st.selectedId ? 92 : 80, angle: ang, maxW: room,
        cands: [c],
        variants: [
          ...(big ? [[line(it.name, 12, '#0d1014', true), line(dimsTxt, 10.5, 'rgba(13,16,20,0.75)')]] : []),
          [line(it.name, 12, '#0d1014', true)],
          [line(short, 11, '#0d1014', true)],
        ],
      });
    }

    if (!show) return out;

    // labels drawn under the furniture must not end up hidden beneath it
    const itemBoxes = st.items.filter((it) => !st.hidden?.has(it.id)).map((it) => {
      const pts = obbCorners({ cx: it.pose.x, cz: it.pose.z, hx: it.assembled.w / 2, hz: it.assembled.d / 2, yaw: it.pose.yaw }).map((q) => this.toScreen(q));
      const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
      return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
    });

    // room names: placed in the clearest open floor, away from scanned furniture and door swings
    const swings = w.openings.filter((op) => op.type === 'door' && op.swing !== 'none').map((op) => {
      const pts = doorSwingPolygon(op, 8).map((q) => this.toScreen(q));
      const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
      return { x0: Math.min(...xs), y0: Math.min(...ys), x1: Math.max(...xs), y1: Math.max(...ys) };
    });
    for (const r of w.rooms) {
      const spots = this.roomSpots(r).map((p) => this.toScreen(p));
      if (!spots.length) continue;
      const ceil = `ceiling ${r.ceiling.value.toFixed(2)} m${r.ceiling.sigma != null ? ` ±${Math.max(1, Math.round(r.ceiling.sigma * 100))} cm` : ''}`;
      const nameL = line(r.name, small ? 12.5 : 13, 'rgba(231,233,236,0.62)', true);
      const subL = (t) => line(t, 10.5, 'rgba(148,156,167,0.85)');
      out.push({
        layer: 'under', prio: 70, angle: 0, cands: spots, inside: r.polygon, avoid: [...swings, ...itemBoxes],
        variants: [[nameL, subL(`${r.area.toFixed(1)} m²  ·  ${ceil}`)], [nameL, subL(`${r.area.toFixed(1)} m²`)], [nameL], [line(r.name, 11, 'rgba(231,233,236,0.62)', true)]],
        // a furnished room may have no open floor left: then a small pill on top of everything
        fallback: { layer: 'top', prio: 70, angle: 0, cands: spots, inside: r.polygon, avoid: swings, pill: true, variants: [[line(r.name, 11.5, '#e7e9ec', true)]] },
      });
    }

    // wall lengths: on the longest solid stretch, outside the wall unless another room is there
    const inside = w.kind === 'house';
    for (const wall of w.walls) {
      if (wall.geomLength * s < 40) continue;
      const ops = wall.openings.map((k) => w.openings.find((o) => o.key === k)).filter(Boolean).sort((a, b) => a.offset - b.offset);
      let best = [0, wall.geomLength], t = 0, bestLen = -1;
      for (const op of [...ops, { offset: wall.geomLength, width: 0 }]) {
        if (op.offset - t > bestLen) { bestLen = op.offset - t; best = [t, op.offset]; }
        t = op.offset + op.width;
      }
      const useStretch = bestLen * s > 60;
      const [t0, t1] = useStretch ? best : [0, wall.geomLength];
      // outside the wall by default; a spot that lands inside another room (shared walls in a
      // house) is skipped so a length never reads as belonging to the neighbouring room
      const inOther = (q) => inside && w.rooms.some((r) => r.id !== wall.roomId && pointInPolygon(q, r.polygon));
      const outOff = -WALL_T - 13 / s, inOff = 13 / s;
      const fracs = [0.5, 0.32, 0.68, 0.18, 0.82];
      const cands = [];
      for (const off of [outOff, inOff, inOff + 14 / s]) {
        for (const f of fracs) {
          const q = add(add(wall.a, scale(wall.dir, t0 + (t1 - t0) * f)), scale(wall.normalIn, off));
          if (off === outOff && inOther(add(q, scale(wall.normalIn, -0.05)))) continue;
          cands.push(this.toScreen(q));
        }
      }
      const sigma = wall.sigma != null ? Math.max(1, Math.round(wall.sigma * 100)) : null;
      const col = '#cfe9e3';
      const warn = wall.sigma != null && wall.sigma > 0.025;
      out.push({
        layer: 'top', prio: 60 + Math.min(9, wall.geomLength), angle: uprightAngle(Math.atan2(wall.dir[1], wall.dir[0])), badge: warn ? C.warn : null,
        cands,
        variants: [
          ...(sigma != null && !(small && sigma <= 1) ? [[line(`${wall.length.toFixed(2)} m ±${sigma} cm`, 11.5, col)]] : []),
          [line(`${wall.length.toFixed(2)} m`, 11.5, col)],
          [line(wall.length.toFixed(2), 10, col)],
        ],
      });
    }

    // door / window sizes
    const labelled = new Set();
    for (const op of w.openings) {
      if (op.width * s < 30 || labelled.has(op.key)) continue;
      if (op.partner) labelled.add(op.partner);
      const ang = uprightAngle(Math.atan2(op.dir[1], op.dir[0]));
      const full = op.type === 'window' ? `${Math.round(op.width * 100)} win` : `${Math.round(op.width * 100)} x ${Math.round(op.height * 100)}`;
      out.push({
        layer: 'top', prio: 40, angle: ang,
        cands: [12, 24, -WALL_T * s - 12].map((k) => this.toScreen(add(op.center, scale(op.normalIn, k / s)))),
        variants: [[line(full, 10, C.muted)], [line(`${Math.round(op.width * 100)}`, 10, C.muted)]],
      });
    }

    // scanned furniture: a quiet "existing" tag, only when zoomed in enough to read it
    if (s >= 55) {
      for (const o of w.obstacles) {
        const bw = o.hx * 2 * s, bh = o.hz * 2 * s;
        if (Math.min(bw, bh) < 18) continue;
        const along = bh > bw * 1.3;
        out.push({
          layer: 'under', prio: 20, angle: along ? -Math.PI / 2 : 0, maxW: (along ? bh : bw) - 8, avoid: itemBoxes,
          cands: [this.toScreen([o.cx, o.cz])],
          variants: [[line(obstacleName(o.label), 10, 'rgba(148,156,167,0.7)')]],
        });
      }
    }
    return out;
  }

  // Boxes no label may cover (screen px).
  reservedBoxes() {
    const b = this.scaleBarGeom();
    const out = [{ x0: b.x - 4, y0: b.y - 18, x1: b.x + b.px + 4, y1: b.y + 4 }];
    // walls (the axis-aligned ones: a slanted wall's box would block far too much)
    for (const wall of this.world.walls) {
      if (Math.min(Math.abs(wall.dir[0]), Math.abs(wall.dir[1])) > 0.12) continue;
      const o = scale(wall.normalIn, -WALL_T);
      const pts = [wall.a, wall.b, add(wall.a, o), add(wall.b, o)].map((p) => this.toScreen(p));
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      out.push({ x0: Math.min(...xs) - 1, y0: Math.min(...ys) - 1, x1: Math.max(...xs) + 1, y1: Math.max(...ys) + 1 });
    }
    // the entry arrow
    const entry = this.world.openings.find((op) => op.key === this.hooks.getState().entryKey);
    if (entry) {
      const pts = [add(entry.center, scale(entry.normalIn, -0.12)), add(entry.center, scale(entry.normalIn, -0.78))].map((p) => this.toScreen(p));
      out.push({ x0: Math.min(pts[0][0], pts[1][0]) - 9, y0: Math.min(pts[0][1], pts[1][1]) - 9, x1: Math.max(pts[0][0], pts[1][0]) + 9, y1: Math.max(pts[0][1], pts[1][1]) + 9 });
    }
    return out;
  }

  scaleBarGeom() {
    const s = this.view.s;
    const len = s >= 50 ? 1 : 2;
    const px = len * s;
    const inset = this.hooks.insets ? this.hooks.insets() : { bottom: 0 };
    const y = this.H - 22 - (inset.bottomTools ?? inset.bottom ?? 0);
    const x = inset.scaleLeft ? 16 : this.W - px - 20;
    return { x, y, px, len };
  }

  measureLines(lines) {
    const ctx = this.ctx;
    let w = 0, h = 0;
    for (const l of lines) {
      ctx.font = `${l.bold ? 650 : 500} ${l.size}px ${FONT}`;
      w = Math.max(w, ctx.measureText(l.t).width);
      h += l.size * 1.28;
    }
    return { w, h };
  }

  layoutLabels(labels, reserved = []) {
    const placed = [];
    const boxes = [];
    const overlaps = (b, o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0;
    const tryPlace = (l) => {
      for (const lines of l.variants) {
        const m = this.measureLines(lines);
        const bw = m.w + (l.badge || l.pill ? 12 : 0);
        if (l.maxW != null && bw > l.maxW) continue;
        const c = Math.abs(Math.cos(l.angle || 0)), sn = Math.abs(Math.sin(l.angle || 0));
        const hw = (bw * c + m.h * sn) / 2 + 2, hh = (bw * sn + m.h * c) / 2 + 1;
        for (const p of l.cands) {
          const b = { x0: p[0] - hw, y0: p[1] - hh, x1: p[0] + hw, y1: p[1] + hh };
          if (boxes.some((o) => overlaps(b, o))) continue;
          if (!l.overReserved && reserved.some((o) => overlaps(b, o))) continue;
          if (l.inside && !this.boxInside(b, l.inside)) continue;
          if (l.avoid && l.avoid.some((o) => overlaps(b, o))) continue;
          boxes.push(b);
          placed.push({ ...l, lines, p, h: m.h });
          return true;
        }
      }
      return false;
    };
    const order = labels.map((l, i) => [l, i]).sort((a, b) => b[0].prio - a[0].prio || a[1] - b[1]);
    for (const [l] of order) {
      // highest priority first; a label that fits nowhere tries its fallback, else is dropped
      for (let x = l; x && !tryPlace(x); x = x.fallback);
    }
    return placed;
  }

  // Is a screen box entirely on the floor of a room polygon? (corners + edge midpoints)
  boxInside(b, poly) {
    const xs = [b.x0, (b.x0 + b.x1) / 2, b.x1], ys = [b.y0, (b.y0 + b.y1) / 2, b.y1];
    for (const x of xs) for (const y of ys) {
      if (x === xs[1] && y === ys[1]) continue;
      if (!pointInPolygon(this.toWorld(x, y), poly)) return false;
    }
    return true;
  }

  drawLabel(l) {
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(l.p[0], l.p[1]);
    if (l.angle) ctx.rotate(l.angle);
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    if (l.badge) {
      const m = this.measureLines(l.lines);
      const w = m.w + 10;
      ctx.fillStyle = 'rgba(227,182,95,0.16)';
      ctx.strokeStyle = l.badge; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(-w / 2, -m.h / 2, w, m.h, 5) : ctx.rect(-w / 2, -m.h / 2, w, m.h);
      ctx.fill(); ctx.stroke();
    }
    if (l.pill) {
      const m = this.measureLines(l.lines);
      const w = m.w + 12;
      ctx.fillStyle = 'rgba(14,16,19,0.8)';
      ctx.beginPath(); ctx.roundRect ? ctx.roundRect(-w / 2, -m.h / 2 - 1, w, m.h + 2, 6) : ctx.rect(-w / 2, -m.h / 2 - 1, w, m.h + 2);
      ctx.fill();
    }
    let y = -l.h / 2;
    for (const ln of l.lines) {
      const lh = ln.size * 1.28;
      ctx.font = `${ln.bold ? 650 : 500} ${ln.size}px ${FONT}`;
      ctx.fillStyle = ln.color;
      ctx.fillText(ln.t, 0, y + lh / 2);
      y += lh;
    }
    ctx.restore();
  }

  // Open-floor points for a room's name, best first: inside the polygon, far from walls,
  // and off the scanned furniture. Cached per world.
  roomSpots(r) {
    this.spotCache ||= new WeakMap();
    let byRoom = this.spotCache.get(this.world);
    if (!byRoom) { byRoom = new Map(); this.spotCache.set(this.world, byRoom); }
    if (byRoom.has(r.id)) return byRoom.get(r.id);
    const xs = r.polygon.map((p) => p[0]), zs = r.polygon.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), z0 = Math.min(...zs), z1 = Math.max(...zs);
    const step = Math.max(0.06, Math.min(x1 - x0, z1 - z0) / 28);
    const obs = this.world.obstacles.filter((o) => o.roomId === r.id || pointInPolygon([o.cx, o.cz], r.polygon));
    const cands = [];
    for (let x = x0 + step / 2; x < x1; x += step) {
      for (let z = z0 + step / 2; z < z1; z += step) {
        const p = [x, z];
        if (!pointInPolygon(p, r.polygon)) continue;
        let d = Infinity;
        for (let i = 0; i < r.polygon.length; i++) d = Math.min(d, pointSegmentDistance(p, r.polygon[i], r.polygon[(i + 1) % r.polygon.length]));
        let free = Infinity;
        for (const o of obs) free = Math.min(free, pointInObb(p, o) ? -1 : pointObbDistance(p, o));
        // a spot on top of furniture is a last resort; otherwise prefer wide open floor
        const score = free < 0 ? d * 0.25 : Math.min(d, free + 0.15);
        cands.push({ p, score });
      }
    }
    cands.sort((a, b) => b.score - a.score);
    // the ten best spots first, then a spread of every other open spot as fallbacks
    const spots = [];
    for (const c of cands) {
      if (spots.length >= 10) break;
      if (spots.every((q) => Math.hypot(q[0] - c.p[0], q[1] - c.p[1]) > 0.3)) spots.push(c.p);
    }
    for (const c of cands) {
      if (spots.length >= 70) break;
      if (spots.every((q) => Math.hypot(q[0] - c.p[0], q[1] - c.p[1]) > 0.18)) spots.push(c.p);
    }
    if (!spots.length) spots.push(r.centroid);
    byRoom.set(r.id, spots);
    return spots;
  }

  highlightedWall(h) {
    const w = this.world;
    return w.walls.find((x) => x.id === h.wall && (!h.room || x.roomId === h.room)) || w.walls.find((x) => x.key === h.wall) || null;
  }

  // Scan coach: pulse the wall the coach wants rescanned. Returns true while drawn (keeps animating).
  drawWallHighlight(h) {
    const ctx = this.ctx;
    const wall = this.highlightedWall(h);
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
    const ctx = this.ctx, w = this.world;
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
    // entry marker (its label is placed with the others)
    const entry = w.openings.find((o) => o.key === st.entryKey);
    if (entry) {
      const c = entry.center, n = entry.normalIn;
      const tip = add(c, scale(n, -0.18)), tail = add(c, scale(n, -0.75));
      ctx.strokeStyle = C.accent; ctx.fillStyle = C.accent; ctx.lineWidth = 2.5;
      this.line(tail, tip);
      const left = add(tip, add(scale(n, -0.16), scale(entry.dir, 0.12))), right = add(tip, add(scale(n, -0.16), scale(entry.dir, -0.12)));
      ctx.beginPath(); this.polyPath([tip, left, right]); ctx.fill();
    }
  }

  drawItem(it, st, selected) {
    const ctx = this.ctx;
    const o = { cx: it.pose.x, cz: it.pose.z, hx: it.assembled.w / 2, hz: it.assembled.d / 2, yaw: it.pose.yaw };
    const bad = st.status[it.id] && !st.status[it.id].ok;
    const corners = obbCorners(o);
    ctx.beginPath(); this.polyPath(corners);
    ctx.globalAlpha = 0.9; ctx.fillStyle = it.color; ctx.fill(); ctx.globalAlpha = 1;
    if (bad) { ctx.fillStyle = 'rgba(240,113,113,0.33)'; ctx.fill(); }
    ctx.lineWidth = selected || bad ? 2.2 : 1;
    ctx.strokeStyle = bad ? C.bad : selected ? C.accent : 'rgba(0,0,0,0.45)';
    ctx.stroke();
    // front edge
    ctx.strokeStyle = 'rgba(255,255,255,0.55)'; ctx.lineWidth = 2.5;
    this.line(corners[2], corners[3]);
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
      ctx.fillStyle = '#0e1013'; ctx.beginPath(); ctx.arc(h[0], h[1], 9, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.beginPath(); ctx.arc(h[0], h[1], 3.5, 0, Math.PI * 2); ctx.fillStyle = C.accent; ctx.fill();
      const deg = ((Math.round(it.pose.yaw * 180 / Math.PI) % 360) + 360) % 360;
      this.text(`${deg}°`, [h[0], h[1] - 17], 10.5, C.accent, 'center', true);
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
    }
  }

  drawScaleBar() {
    const ctx = this.ctx;
    const { x, y, px, len } = this.scaleBarGeom();
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

