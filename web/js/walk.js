// First-person walk mode: WASD / arrows + mouse look (pointer lock) on desktop,
// on-screen joystick + drag-to-look on touch. Eye height 1.6 m; the body is a
// 0.22 m circle that slides along walls and furniture.
import { closestPointOnSegment, toLocal, axisX, axisZ } from './core/geom.js';

const EYE = 1.6;
const RADIUS = 0.22;

export class WalkController {
  constructor(camera, dom, uiGetter) {
    this.camera = camera;
    this.dom = dom;
    this.uiGetter = uiGetter;
    this.enabled = false;
    this.yaw = 0; this.pitch = 0;
    this.pos = [0, 0];
    this.keys = new Set();
    this.stick = [0, 0];
    this.getColliders = () => ({ segments: [], boxes: [] });
    this.isTouch = matchMedia('(pointer: coarse)').matches;

    window.addEventListener('keydown', (e) => {
      if (!this.enabled || /INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName)) return;
      const k = e.key.toLowerCase();
      if (['w', 'a', 's', 'd', 'arrowup', 'arrowdown', 'arrowleft', 'arrowright', 'shift', 'q', 'e'].includes(k)) {
        this.keys.add(k);
        if (k.startsWith('arrow')) e.preventDefault();
      }
    });
    window.addEventListener('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    window.addEventListener('blur', () => this.keys.clear());

    dom.addEventListener('click', () => {
      if (!this.enabled || this.isTouch || document.pointerLockElement === dom) return;
      try { const p = dom.requestPointerLock(); if (p && p.catch) p.catch(() => {}); } catch { /* not allowed (iframe) */ }
    });
    document.addEventListener('pointerlockchange', () => this.updateHint());
    document.addEventListener('mousemove', (e) => {
      if (!this.enabled || document.pointerLockElement !== dom) return;
      this.look(e.movementX, e.movementY, 0.0022);
    });
    // drag-to-look (touch, or mouse without pointer lock)
    this.lookPtr = null;
    dom.addEventListener('pointerdown', (e) => {
      if (!this.enabled || document.pointerLockElement === dom) return;
      this.lookPtr = { id: e.pointerId, x: e.clientX, y: e.clientY };
    });
    dom.addEventListener('pointermove', (e) => {
      if (!this.enabled || !this.lookPtr || this.lookPtr.id !== e.pointerId) return;
      const k = e.pointerType === 'touch' ? 0.005 : 0.004;
      this.look(e.clientX - this.lookPtr.x, e.clientY - this.lookPtr.y, k);
      this.lookPtr.x = e.clientX; this.lookPtr.y = e.clientY;
    });
    const endLook = (e) => { if (this.lookPtr && this.lookPtr.id === e.pointerId) this.lookPtr = null; };
    dom.addEventListener('pointerup', endLook);
    dom.addEventListener('pointercancel', endLook);
  }

  ui() { return this.uiGetter ? this.uiGetter() : {}; }

  bindJoystick() {
    const { joystick, knob } = this.ui();
    if (!joystick || joystick.dataset.bound) return;
    joystick.dataset.bound = '1';
    let pid = null, cx = 0, cy = 0;
    const R = 46;
    const set = (dx, dy) => {
      const l = Math.hypot(dx, dy), k = l > R ? R / l : 1;
      dx *= k; dy *= k;
      knob.style.transform = `translate(${dx}px, ${dy}px)`;
      this.stick = [dx / R, dy / R];
    };
    joystick.addEventListener('pointerdown', (e) => {
      e.stopPropagation();
      pid = e.pointerId; joystick.setPointerCapture(pid);
      const r = joystick.getBoundingClientRect(); cx = r.left + r.width / 2; cy = r.top + r.height / 2;
      set(e.clientX - cx, e.clientY - cy);
    });
    joystick.addEventListener('pointermove', (e) => { if (e.pointerId === pid) set(e.clientX - cx, e.clientY - cy); });
    const end = (e) => { if (e.pointerId === pid) { pid = null; set(0, 0); } };
    joystick.addEventListener('pointerup', end);
    joystick.addEventListener('pointercancel', end);
  }

  look(dx, dy, k) {
    this.yaw -= dx * k;
    this.pitch = Math.max(-1.3, Math.min(1.3, this.pitch - dy * k));
  }

  enable(start) {
    this.enabled = true;
    this.isTouch = matchMedia('(pointer: coarse)').matches;
    if (start) {
      this.pos = [start.x, start.z];
      // start.dir: facing direction on the floor
      this.yaw = Math.atan2(-start.dir[0], -start.dir[1]);
      this.pitch = -0.08;
    }
    this.camera.fov = 70;
    this.camera.updateProjectionMatrix();
    const { joystick, crosshair } = this.ui();
    if (joystick) { joystick.hidden = !this.isTouch; this.bindJoystick(); }
    if (crosshair) crosshair.hidden = this.isTouch;
    this.updateHint();
  }

  disable() {
    if (!this.enabled) return;
    this.enabled = false;
    this.keys.clear();
    if (document.pointerLockElement === this.dom) document.exitPointerLock?.();
    const { joystick, hint, crosshair } = this.ui();
    if (joystick) joystick.hidden = true;
    if (hint) hint.hidden = true;
    if (crosshair) crosshair.hidden = true;
  }

  updateHint() {
    const { hint } = this.ui();
    if (!hint) return;
    hint.hidden = !this.enabled;
    if (this.isTouch) hint.textContent = 'Left stick to walk · drag to look around';
    else if (document.pointerLockElement === this.dom) hint.textContent = 'WASD or arrows to walk · Shift to hurry · Esc to release the mouse';
    else hint.textContent = 'Click to look around with the mouse · WASD or arrows to walk';
  }

  update(dt) {
    if (!this.enabled) return;
    const k = this.keys;
    let f = 0, s = 0;
    if (k.has('w') || k.has('arrowup')) f += 1;
    if (k.has('s') || k.has('arrowdown')) f -= 1;
    if (k.has('d')) s += 1;
    if (k.has('a')) s -= 1;
    if (k.has('arrowleft') || k.has('q')) this.yaw += 1.8 * dt;
    if (k.has('arrowright') || k.has('e')) this.yaw -= 1.8 * dt;
    f += -this.stick[1];
    s += this.stick[0];
    const mag = Math.hypot(f, s);
    if (mag > 1) { f /= mag; s /= mag; }
    const speed = k.has('shift') ? 2.8 : 1.4;
    const fwd = [-Math.sin(this.yaw), -Math.cos(this.yaw)];
    const right = [Math.cos(this.yaw), -Math.sin(this.yaw)];
    const step = speed * dt;
    const next = [this.pos[0] + (fwd[0] * f + right[0] * s) * step, this.pos[1] + (fwd[1] * f + right[1] * s) * step];
    this.pos = this.collide(next);
    this.camera.position.set(this.pos[0], EYE, this.pos[1]);
    this.camera.rotation.order = 'YXZ';
    this.camera.rotation.set(this.pitch, this.yaw, 0);
  }

  collide(p) {
    const { segments, boxes } = this.getColliders();
    for (let iter = 0; iter < 4; iter++) {
      let moved = false;
      for (const s of segments) {
        const c = closestPointOnSegment(p, [s[0], s[1]], [s[2], s[3]]);
        const dx = p[0] - c[0], dz = p[1] - c[1], d = Math.hypot(dx, dz);
        if (d < RADIUS && d > 1e-9) { const k = (RADIUS - d) / d; p = [p[0] + dx * k, p[1] + dz * k]; moved = true; }
      }
      for (const o of boxes) {
        const l = toLocal(o, p);
        const qx = Math.max(-o.hx, Math.min(o.hx, l[0])), qz = Math.max(-o.hz, Math.min(o.hz, l[1]));
        let dx = l[0] - qx, dz = l[1] - qz, d = Math.hypot(dx, dz);
        if (d < 1e-9) {
          // inside: push out through the nearest face
          const px = o.hx - Math.abs(l[0]), pz = o.hz - Math.abs(l[1]);
          if (px < pz) { dx = Math.sign(l[0]) || 1; dz = 0; d = -px; } else { dx = 0; dz = Math.sign(l[1]) || 1; d = -pz; }
          const push = RADIUS - d;
          const ax = axisX(o.yaw), az = axisZ(o.yaw);
          p = [p[0] + (ax[0] * dx + az[0] * dz) * push, p[1] + (ax[1] * dx + az[1] * dz) * push];
          moved = true;
        } else if (d < RADIUS) {
          const k = (RADIUS - d) / d;
          const ax = axisX(o.yaw), az = axisZ(o.yaw);
          p = [p[0] + (ax[0] * dx + az[0] * dz) * k, p[1] + (ax[1] * dx + az[1] * dz) * k];
          moved = true;
        }
      }
      if (!moved) break;
    }
    return p;
  }
}
