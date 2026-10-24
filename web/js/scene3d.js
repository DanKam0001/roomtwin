// three.js view: walls extruded from the measured wall segments with openings cut
// out, floor, optional ceiling, frames, door leaves, obstacles, furniture boxes
// with product images, measurement labels, route / build-order animation.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { WalkController } from './walk.js';
import { proxiedImage } from './api.js';

const WALL_T = 0.09;
const COL = { wall: 0xe6e3dd, frame: 0xf3f1ec, leaf: 0xb8a68a, floor: 0xffffff, ceiling: 0xd9d6d0, obstacle: 0x6b717a, accent: 0x63d2be, bad: 0xf07171 };

function yawFromDir(dir) { return Math.atan2(-dir[1], dir[0]); }

function floorTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  g.fillStyle = '#56514a'; g.fillRect(0, 0, 256, 256);
  // subtle planks
  for (let i = 0; i < 8; i++) {
    g.fillStyle = `rgba(255,255,255,${0.012 + (i * 37 % 7) * 0.004})`;
    g.fillRect(0, i * 32, 256, 32);
    g.fillStyle = 'rgba(0,0,0,0.12)'; g.fillRect(0, i * 32, 256, 1);
    g.fillRect(((i * 97) % 256), i * 32, 1, 32);
  }
  g.strokeStyle = 'rgba(255,255,255,0.10)'; g.lineWidth = 2; g.strokeRect(1, 1, 254, 254);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function label(html, cls = '') {
  const el = document.createElement('div');
  el.className = `lbl3d ${cls}`;
  el.innerHTML = html;
  return new CSS2DObject(el);
}

export class Scene3D {
  constructor(container, hooks) {
    this.container = container;
    this.hooks = hooks;
    this.visible = false;
    this.mode = 'orbit';
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    container.appendChild(r.domElement);
    this.labels = new CSS2DRenderer();
    Object.assign(this.labels.domElement.style, { position: 'absolute', inset: '0', pointerEvents: 'none' });
    container.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0d10);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.03, 300);
    this.camera.position.set(6, 7, 8);

    this.scene.add(new THREE.HemisphereLight(0xf4f1ea, 0x6d6862, 1.2));
    const sun = this.sun = new THREE.DirectionalLight(0xffffff, 1.5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.02;
    this.scene.add(sun, sun.target);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.18));

    this.roomGroup = new THREE.Group();
    this.meshGroup = new THREE.Group();
    this.itemGroup = new THREE.Group();
    this.overlayGroup = new THREE.Group();
    this.scene.add(this.roomGroup, this.meshGroup, this.itemGroup, this.overlayGroup);
    this.walls = [];
    this.wallLabels = [];
    this.itemObjs = new Map();
    this.texCache = new Map();
    this.floorTex = floorTexture();

    this.raycaster = new THREE.Raycaster();
    this.drag = null;
    const dom = r.domElement;
    dom.style.touchAction = 'none';
    // Registered before OrbitControls so an item grab can disable orbiting.
    dom.addEventListener('pointerdown', (e) => this.onDown(e));
    dom.addEventListener('pointermove', (e) => this.onMove(e));
    dom.addEventListener('pointerup', (e) => this.onUp(e));
    dom.addEventListener('pointercancel', (e) => this.onUp(e));
    this.controls = new OrbitControls(this.camera, dom);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 0.6;
    this.controls.maxDistance = 60;
    this.controls.screenSpacePanning = true;

    this.walk = new WalkController(this.camera, dom, hooks.walkUi);
    this.clock = new THREE.Clock();
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  resize() {
    const w = this.container.clientWidth || 1, h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setVisible(v) {
    this.visible = v;
    if (v) {
      this.resize();
      this.clock.getDelta();
      this.renderer.setAnimationLoop(() => this.tick());
    } else {
      this.renderer.setAnimationLoop(null);
      this.walk.disable();
    }
  }

  // ---------------- room ----------------
  setWorld(world, meshUrls = {}) {
    this.world = world;
    for (const g of [this.roomGroup, this.meshGroup, this.itemGroup, this.overlayGroup]) this.clearGroup(g);
    this.itemObjs.clear();
    this.walls = [];
    this.wallLabels = [];
    this.ceilings = [];
    this.lamps = [];
    const wallMatBase = new THREE.MeshStandardMaterial({ color: COL.wall, roughness: 0.92, metalness: 0 });
    const frameMat = new THREE.MeshStandardMaterial({ color: COL.frame, roughness: 0.6 });
    const leafMat = new THREE.MeshStandardMaterial({ color: COL.leaf, roughness: 0.7 });
    const glassMat = new THREE.MeshStandardMaterial({ color: 0x9cc7e0, transparent: true, opacity: 0.22, roughness: 0.1, depthWrite: false });

    for (const room of world.rooms) {
      const shape = new THREE.Shape(room.polygon.map((p) => new THREE.Vector2(p[0], -p[1])));
      const fg = new THREE.ShapeGeometry(shape);
      const floor = new THREE.Mesh(fg, new THREE.MeshStandardMaterial({ map: this.floorTex, roughness: 0.85 }));
      floor.rotation.x = -Math.PI / 2;
      floor.receiveShadow = true;
      this.roomGroup.add(floor);
      const ceil = new THREE.Mesh(fg.clone(), new THREE.MeshStandardMaterial({ color: COL.ceiling, roughness: 0.95, side: THREE.BackSide }));
      ceil.rotation.x = -Math.PI / 2;
      ceil.position.y = room.ceiling.value;
      ceil.visible = false;
      this.ceilings.push(ceil);
      this.roomGroup.add(ceil);
      const lamp = new THREE.PointLight(0xfff1de, 0, Math.max(6, Math.sqrt(room.area) * 3), 1.6);
      lamp.position.set(room.centroid[0], room.ceiling.value - 0.25, room.centroid[1]);
      this.lamps.push(lamp);
      this.roomGroup.add(lamp);
      const rl = label(`${room.name}`, 'room');
      rl.position.set(room.centroid[0], 0.05, room.centroid[1]);
      rl.userData.kind = 'room';
      this.roomGroup.add(rl);
      this.wallLabels.push(rl);
    }

    for (const wall of world.walls) {
      const g = new THREE.Group();
      const mat = wallMatBase.clone();
      const H = wall.height;
      const L = wall.geomLength;
      const yaw = yawFromDir(wall.dir);
      const box = (s0, s1, y0, y1, m = mat, depth = WALL_T, inward = 0) => {
        if (s1 - s0 < 1e-3 || y1 - y0 < 1e-3) return null;
        const geo = new THREE.BoxGeometry(s1 - s0, y1 - y0, depth);
        const mesh = new THREE.Mesh(geo, m);
        const mid = (s0 + s1) / 2;
        const off = -depth / 2 + inward;
        mesh.position.set(wall.a[0] + wall.dir[0] * mid + wall.normalIn[0] * off, (y0 + y1) / 2, wall.a[1] + wall.dir[1] * mid + wall.normalIn[1] * off);
        mesh.rotation.y = yaw;
        mesh.castShadow = true; mesh.receiveShadow = true;
        g.add(mesh);
        return mesh;
      };
      const ops = wall.openings.map((k) => world.openings.find((o) => o.key === k)).filter(Boolean).sort((a, b) => a.offset - b.offset);
      let t = 0;
      for (const op of ops) {
        box(t, op.offset, 0, H);
        const top = Math.min(H, op.sill + op.height);
        box(op.offset, op.offset + op.width, top, H);
        if (op.sill > 0.005) box(op.offset, op.offset + op.width, 0, op.sill);
        // frame: jambs + head (+ sill), slightly proud of the inner face
        const fd = WALL_T + 0.03, fw = 0.045;
        box(op.offset - fw, op.offset, op.sill, top, frameMat, fd, 0.015);
        box(op.offset + op.width, op.offset + op.width + fw, op.sill, top, frameMat, fd, 0.015);
        box(op.offset - fw, op.offset + op.width + fw, top, Math.min(H, top + fw), frameMat, fd, 0.015);
        if (op.type === 'window') {
          box(op.offset - fw, op.offset + op.width + fw, op.sill - 0.03, op.sill, frameMat, WALL_T + 0.08, 0.04);
          const glass = box(op.offset, op.offset + op.width, op.sill, top, glassMat, 0.01, -WALL_T / 2 + 0.005);
          if (glass) { glass.castShadow = false; glass.renderOrder = 2; }
        }
        if (op.type === 'door' && op.swing !== 'none') {
          const hingeAtA = op.hinge !== 'b';
          const hinge = hingeAtA ? op.a : op.b;
          const side = op.swing === 'out' ? -1 : 1;
          const leafH = Math.min(op.height, H) - 0.01;
          const geo = new THREE.BoxGeometry(0.04, leafH, op.width - 0.01);
          const leaf = new THREE.Mesh(geo, leafMat);
          // open 90 degrees: leaf runs along the inward normal from the hinge
          const n = [wall.normalIn[0] * side, wall.normalIn[1] * side];
          const along = hingeAtA ? 0.025 : -0.025;
          leaf.position.set(hinge[0] + n[0] * (op.width / 2) + wall.dir[0] * along, leafH / 2, hinge[1] + n[1] * (op.width / 2) + wall.dir[1] * along);
          leaf.rotation.y = yaw;
          leaf.castShadow = true;
          g.add(leaf);
        }
        t = op.offset + op.width;
      }
      box(t, L, 0, H);
      g.userData = { normalIn: wall.normalIn, mid: [(wall.a[0] + wall.b[0]) / 2, (wall.a[1] + wall.b[1]) / 2], mat };
      this.roomGroup.add(g);
      this.walls.push(g);
      const sig = wall.sigma != null ? ` <small>±${Math.max(1, Math.round(wall.sigma * 100))} cm</small>` : '';
      const wl = label(`${wall.length.toFixed(2)} m${sig}`, 'dim');
      const midp = g.userData.mid;
      wl.position.set(midp[0] + wall.normalIn[0] * 0.15, H + 0.12, midp[1] + wall.normalIn[1] * 0.15);
      wl.userData.kind = 'wall';
      this.roomGroup.add(wl);
      this.wallLabels.push(wl);
    }
    const ceilLbl = world.rooms.map((r) => {
      const l = label(`ceiling ${r.ceiling.value.toFixed(2)} m${r.ceiling.sigma != null ? ` <small>±${Math.max(1, Math.round(r.ceiling.sigma * 100))} cm</small>` : ''}`, 'dim');
      l.position.set(r.centroid[0], r.ceiling.value + 0.12, r.centroid[1]);
      l.userData.kind = 'ceiling';
      this.roomGroup.add(l);
      return l;
    });
    this.wallLabels.push(...ceilLbl);

    for (const o of world.obstacles) {
      const h = Math.max(0.02, o.ymax - o.ymin);
      const geo = new THREE.BoxGeometry(o.hx * 2, h, o.hz * 2);
      const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: COL.obstacle, roughness: 0.8 }));
      m.position.set(o.cx, o.ymin + h / 2, o.cz);
      m.rotation.y = o.yaw;
      m.castShadow = m.receiveShadow = true;
      const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x9aa1aa, transparent: true, opacity: 0.5 }));
      m.add(edges);
      this.roomGroup.add(m);
    }

    // light + shadow frustum around the world
    const [x0, z0, x1, z1] = world.bounds;
    const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2, R = Math.max(x1 - x0, z1 - z0) * 0.75 + 1;
    this.sun.position.set(cx + R * 0.6, 8, cz + R * 0.35);
    this.sun.target.position.set(cx, 0, cz);
    const sc = this.sun.shadow.camera;
    sc.left = -R; sc.right = R; sc.top = R; sc.bottom = -R; sc.near = 0.5; sc.far = 30;
    sc.updateProjectionMatrix();

    this.loadMeshes(meshUrls);
    this.fit();
  }

  loadMeshes(meshUrls) {
    const entries = Object.entries(meshUrls || {});
    this.hasMesh = false;
    if (!entries.length) return;
    const loader = new GLTFLoader();
    for (const [roomId, url] of entries) {
      loader.load(url, (gltf) => {
        const room = this.world.rooms.find((r) => r.id === roomId) || this.world.rooms[0];
        const obj = gltf.scene;
        obj.position.set(room.transform.x, 0, room.transform.z);
        obj.rotation.y = room.transform.yaw;
        obj.traverse((o) => { if (o.isMesh) { o.receiveShadow = true; } });
        this.meshGroup.add(obj);
        this.hasMesh = true;
        this.hooks.onMeshLoaded?.();
      }, undefined, () => this.hooks.onMeshError?.(url));
    }
  }

  clearGroup(g) {
    for (const c of [...g.children]) {
      g.remove(c);
      c.traverse?.((o) => {
        if (o.isCSS2DObject && o.element?.parentNode) o.element.parentNode.removeChild(o.element);
        o.geometry?.dispose?.();
      });
    }
  }

  fit() {
    if (!this.world) return;
    const [x0, z0, x1, z1] = this.world.bounds;
    const cx = (x0 + x1) / 2, cz = (z0 + z1) / 2;
    const size = Math.max(x1 - x0, z1 - z0, 2);
    this.controls.target.set(cx, 0.6, cz);
    this.camera.position.set(cx + size * 0.45, size * 0.95 + 1.5, cz + size * 0.85 + 0.5);
    this.controls.update();
  }

  setOptions({ ceiling, labels, mesh }) {
    this.showCeiling = ceiling; this.showLabels = labels;
    for (const c of this.ceilings || []) c.visible = !!ceiling;
    for (const l of this.lamps || []) l.intensity = ceiling ? 5 : 0;
    this.meshGroup.visible = mesh !== false;
    this.refreshLabelVisibility();
  }

  refreshLabelVisibility() {
    const walk = this.mode === 'walk';
    for (const l of this.wallLabels) l.visible = this.showLabels && !walk;
  }

  // ---------------- items ----------------
  setItems(items, status, selectedId, hidden = new Set()) {
    const keep = new Set(items.map((i) => i.id));
    for (const [id, obj] of this.itemObjs) if (!keep.has(id)) { this.itemGroup.remove(obj.group); obj.label.element.remove(); this.itemObjs.delete(id); }
    for (const it of items) {
      let obj = this.itemObjs.get(it.id);
      const sig = `${it.assembled.w}|${it.assembled.d}|${it.assembled.h}|${it.color}|${it.image}`;
      if (!obj || obj.sig !== sig) {
        if (obj) { this.itemGroup.remove(obj.group); obj.label.element.remove(); }
        obj = this.buildItem(it);
        obj.sig = sig;
        this.itemObjs.set(it.id, obj);
        this.itemGroup.add(obj.group);
      }
      obj.group.position.set(it.pose.x, 0, it.pose.z);
      obj.group.rotation.y = it.pose.yaw;
      obj.group.visible = !hidden.has(it.id);
      obj.label.visible = obj.group.visible;
      const bad = status[it.id] && !status[it.id].ok;
      const sel = it.id === selectedId;
      obj.edges.material.color.set(bad ? COL.bad : sel ? COL.accent : 0x101317);
      obj.edges.material.opacity = bad || sel ? 1 : 0.55;
      for (const m of obj.mats) m.emissive.set(bad ? 0x5a1414 : sel ? 0x0d2e29 : 0x000000);
      obj.label.element.className = `lbl3d item${bad ? ' bad' : ''}`;
      obj.label.element.textContent = it.name + (bad ? '  !' : '');
    }
  }

  buildItem(it) {
    const { w, d, h } = it.assembled;
    const geo = new THREE.BoxGeometry(w, h, d);
    const base = () => new THREE.MeshStandardMaterial({ color: it.color, roughness: 0.78 });
    const mats = [base(), base(), base(), base(), base(), base()];
    const mesh = new THREE.Mesh(geo, mats);
    mesh.position.y = h / 2;
    mesh.castShadow = mesh.receiveShadow = true;
    mesh.userData.itemId = it.id;
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color: 0x101317, transparent: true, opacity: 0.55 }));
    mesh.add(edges);
    const group = new THREE.Group();
    group.add(mesh);
    const lbl = label(it.name, 'item');
    lbl.position.set(0, h + 0.14, 0);
    group.add(lbl);
    if (it.image) this.applyImage(it.image, mats);
    return { group, mesh, edges, mats, label: lbl };
  }

  applyImage(url, mats) {
    const apply = (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      for (const i of [2, 4]) { mats[i].map = tex; mats[i].color.set(0xffffff); mats[i].needsUpdate = true; }
    };
    if (this.texCache.has(url)) { const t = this.texCache.get(url); if (t) apply(t); return; }
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin('anonymous');
    loader.load(url, (t) => { this.texCache.set(url, t); apply(t); }, undefined, () => {
      loader.load(proxiedImage(url), (t) => { this.texCache.set(url, t); apply(t); }, undefined, () => this.texCache.set(url, null));
    });
  }

  // ---------------- overlays ----------------
  setOverlay(ov) {
    this.clearGroup(this.overlayGroup);
    this.ghost = null;
    this.ring = null;
    if (!ov) return;
    const color = ov.ok ? COL.accent : COL.bad;
    const pts = (ov.path?.length > 1 ? ov.path : ov.partial || []).map((p) => new THREE.Vector3(p.x, 0.02, p.z));
    if (pts.length > 1) {
      // a flat ribbon on the floor (GL lines are 1 px wide)
      const pos = [];
      const wHalf = 0.035;
      for (let i = 0; i < pts.length - 1; i++) {
        const a = pts[i], b = pts[i + 1];
        const dx = b.x - a.x, dz = b.z - a.z, l = Math.hypot(dx, dz) || 1;
        const nx = (-dz / l) * wHalf, nz = (dx / l) * wHalf;
        pos.push(a.x + nx, 0.012, a.z + nz, b.x + nx, 0.012, b.z + nz, b.x - nx, 0.012, b.z - nz);
        pos.push(a.x + nx, 0.012, a.z + nz, b.x - nx, 0.012, b.z - nz, a.x - nx, 0.012, a.z - nz);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
      const ribbon = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: ov.ok ? 0.85 : 0.6, side: THREE.DoubleSide, depthWrite: false }));
      ribbon.renderOrder = 3;
      this.overlayGroup.add(ribbon);
    }
    if (ov.fp && ov.carry) {
      const dims = [ov.fp.hx * 2, Math.max(0.12, ov.carry.vertical || 0.5), ov.fp.hz * 2];
      const geo = new THREE.BoxGeometry(...dims);
      const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.55, depthWrite: false, emissive: color, emissiveIntensity: 0.45 }));
      m.add(new THREE.LineSegments(new THREE.EdgesGeometry(geo), new THREE.LineBasicMaterial({ color })));
      m.userData.h = dims[1];
      const pose = ov.ghost || ov.stuck || (ov.path && ov.path[0]);
      if (pose) { m.position.set(pose.x, dims[1] / 2 + 0.005, pose.z); m.rotation.y = pose.yaw; }
      this.ghost = m;
      this.overlayGroup.add(m);
    }
    if (!ov.ok && (ov.corner || ov.stuck)) {
      const c = ov.corner ? ov.corner.corner : [ov.stuck.x, ov.stuck.z];
      const ring = new THREE.Mesh(new THREE.RingGeometry(0.16, 0.22, 40), new THREE.MeshBasicMaterial({ color: COL.bad, transparent: true, side: THREE.DoubleSide, depthWrite: false }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(c[0], 0.03, c[1]);
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 2.2, 8), new THREE.MeshBasicMaterial({ color: COL.bad }));
      post.position.set(c[0], 1.1, c[1]);
      this.overlayGroup.add(ring, post);
      if (ov.corner) {
        const l = label(`max ${ov.corner.maxLength.toFixed(2)} m long here`, 'bad');
        l.position.set(c[0], 2.3, c[1]);
        this.overlayGroup.add(l);
      }
      this.ring = ring;
    }
  }

  setGhostPose(pose) {
    if (!this.ghost || !pose) return;
    this.ghost.position.set(pose.x, this.ghost.userData.h / 2 + 0.005, pose.z);
    this.ghost.rotation.y = pose.yaw;
  }

  // ---------------- modes ----------------
  setMode(mode, startPose) {
    this.mode = mode;
    this.scene.background.set(mode === 'walk' ? 0xb4c3cf : 0x0b0d10);
    if (mode === 'walk') {
      this.controls.enabled = false;
      this.walk.enable(startPose);
      for (const g of this.walls) { g.userData.mat.opacity = 1; g.userData.mat.transparent = false; g.userData.mat.depthWrite = true; }
    } else {
      this.walk.disable();
      this.controls.enabled = true;
      if (this.savedOrbit) {
        this.camera.position.copy(this.savedOrbit.pos);
        this.controls.target.copy(this.savedOrbit.target);
      }
      this.camera.fov = 50; this.camera.updateProjectionMatrix();
    }
    this.refreshLabelVisibility();
  }

  saveOrbit() { this.savedOrbit = { pos: this.camera.position.clone(), target: this.controls.target.clone() }; }

  setColliders(fn) { this.walk.getColliders = fn; }

  tick() {
    const dt = Math.min(0.05, this.clock.getDelta());
    if (this.mode === 'walk') this.walk.update(dt);
    else {
      this.controls.update();
      // dollhouse cutaway: fade walls between the camera and the room
      const cp = this.camera.position;
      for (const g of this.walls) {
        const { normalIn, mid, mat } = g.userData;
        const d = (cp.x - mid[0]) * normalIn[0] + (cp.z - mid[1]) * normalIn[1];
        const fade = d < -0.05 && cp.y > 0.3;
        const target = fade ? 0.14 : 1;
        if (Math.abs(mat.opacity - target) > 0.01 || mat.transparent !== fade) {
          mat.transparent = fade || target < 1;
          mat.opacity = target;
          mat.depthWrite = !fade;
          mat.needsUpdate = true;
        }
      }
    }
    if (this.ring) { const s = 1 + 0.35 * Math.sin(performance.now() / 220); this.ring.scale.set(s, s, s); }
    this.hooks.onTick?.(dt);
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }

  // ---------------- picking / dragging ----------------
  ndc(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  floorPoint(e) {
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    const p = new THREE.Vector3();
    return this.raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 1, 0), 0), p) ? p : null;
  }

  onDown(e) {
    if (this.mode !== 'orbit') return;
    const st = this.hooks.getState();
    this.downAt = { x: e.clientX, y: e.clientY, t: performance.now() };
    if (st.locked || (e.pointerType === 'touch' && this.touchCount > 0)) return;
    this.raycaster.setFromCamera(this.ndc(e), this.camera);
    const meshes = [...this.itemObjs.values()].filter((o) => o.group.visible).map((o) => o.mesh);
    const hit = this.raycaster.intersectObjects(meshes, false)[0];
    if (!hit) return;
    const id = hit.object.userData.itemId;
    const it = st.items.find((i) => i.id === id);
    const p = this.floorPoint(e);
    if (!it || !p) return;
    this.hooks.onSelect(id);
    this.controls.enabled = false;
    this.drag = { id, off: [it.pose.x - p.x, it.pose.z - p.z], pid: e.pointerId, moved: false };
    this.renderer.domElement.setPointerCapture(e.pointerId);
  }

  onMove(e) {
    if (!this.drag || e.pointerId !== this.drag.pid) return;
    const p = this.floorPoint(e);
    if (!p) return;
    this.drag.moved = true;
    this.hooks.onMove(this.drag.id, { x: p.x + this.drag.off[0], z: p.z + this.drag.off[1] }, 'move', e.shiftKey);
  }

  onUp(e) {
    if (this.drag && e.pointerId === this.drag.pid) {
      this.hooks.onMove(this.drag.id, null, 'end');
      this.drag = null;
      this.controls.enabled = this.mode === 'orbit';
      return;
    }
    if (this.mode === 'orbit' && this.downAt && Math.hypot(e.clientX - this.downAt.x, e.clientY - this.downAt.y) < 5 && performance.now() - this.downAt.t < 400) {
      this.hooks.onSelect(null);
    }
  }

  // ---------------- screenshot ----------------
  snapshot() {
    this.renderer.render(this.scene, this.camera);
    const src = this.renderer.domElement;
    const c = document.createElement('canvas');
    c.width = src.width; c.height = src.height;
    const g = c.getContext('2d');
    g.drawImage(src, 0, 0);
    const k = src.width / src.clientWidth;
    const base = this.container.getBoundingClientRect();
    g.font = `600 ${12 * k}px ui-sans-serif, system-ui, sans-serif`;
    g.textBaseline = 'middle';
    for (const el of this.labels.domElement.querySelectorAll('.lbl3d')) {
      if (el.style.display === 'none' || el.parentElement?.style.display === 'none') continue;
      const r = el.getBoundingClientRect();
      if (!r.width) continue;
      const x = (r.left - base.left) * k, y = (r.top - base.top) * k, w = r.width * k, h = r.height * k;
      if (!el.classList.contains('room')) {
        g.fillStyle = 'rgba(14,16,19,0.78)';
        g.beginPath(); g.roundRect ? g.roundRect(x, y, w, h, 6 * k) : g.rect(x, y, w, h); g.fill();
      }
      g.fillStyle = el.classList.contains('bad') ? '#ffd6d6' : el.classList.contains('dim') ? '#cfe9e3' : '#e7e9ec';
      g.fillText(el.textContent, x + 7 * k, y + h / 2);
    }
    return c;
  }
}
