// RoomTwin web app: state, panels and glue between the plan, 3D and logistics.
import { worldFromRoom, worldFromHouse, roomAt, openingByKey, openingLabel } from './core/world.js';
import { evaluatePlacement, snapToWall } from './logistics/placement.js';
import { checkDoors } from './logistics/plan.js';
import { buildEnv, itemObb, obstacleObb } from './logistics/env.js';
import { densify } from './logistics/route.js';
import { orientationPhrase } from './logistics/doorfit.js';
import { PlanView } from './plan2d.js';
import { loadFromUrl, loadFromFiles, sourceForSave } from './loader.js';
import { CATALOGUE, ITEM_COLOURS } from './catalogue.js';
import { extractFromUrl, uploadScan, coachScan, scanRoomUrl, SCAN_EXT } from './api.js';
import { scanCardHtml } from './scancheck.js';

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const cm = (v) => Math.round(v * 100);
const dims = (a) => `${cm(a.w)} x ${cm(a.d)} x ${cm(a.h)} cm`;
const sigmaTxt = (s) => (s != null ? ` ±${Math.max(1, Math.round(s * 100))} cm` : '');

const S = {
  src: null, world: null, items: [], selectedId: null, entryKey: null,
  view: 'plan', snap: true, labels: true, ceiling: false, mesh: true,
  status: {}, fits: {}, routes: {}, order: null, overlay: null, hidden: new Set(), locked: false,
  stamp: 0, orderIndex: null, nextId: 1,
  // uploaded scans: the open attempt, this session's attempts, a running coach call, the wall to pulse on the plan
  scan: null, attempts: [], coachRun: null, highlightWall: null, traceOpen: false,
};

// ---------------------------------------------------------------- views
const planCanvas = $('#plan');
const plan = new PlanView(planCanvas, {
  getState: () => S,
  insets: () => {
    if (!isSheetLayout()) return { top: 0, bottom: 56 };
    const p = $('#panel').getBoundingClientRect();
    return { top: 0, bottom: p.height + 54, top: 52, bottomTools: p.height + 56 };
  },
  onSelect: (id) => select(id),
  onMove: (id, pos, phase, noSnap) => moveItem(id, pos, phase, noSnap),
  onRotate: (id, yaw, phase) => rotateItem(id, yaw, phase),
  onDoorTap: (key) => {
    if (!S.world.entryCandidates.includes(key)) return;
    setEntry(key);
    toast(`Entry set to ${openingLabel(S.world, openingByKey(S.world, key))}.`, 'ok');
  },
});

let scene = null;
let sceneLoading = null;
async function ensureScene() {
  if (scene) return scene;
  if (!sceneLoading) {
    $('#loading3d').hidden = false;
    sceneLoading = import('./scene3d.js').then(({ Scene3D }) => {
      $('#loading3d').hidden = true;
      const el = $('#three');
      scene = new Scene3D(el, {
        getState: () => S,
        onSelect: (id) => select(id),
        onMove: (id, pos, phase, noSnap) => moveItem(id, pos, phase, noSnap),
        walkUi: () => ({ joystick: $('#joystick'), knob: $('#joystick .knob'), hint: $('#walkHint'), crosshair: $('#crosshair') }),
        onTick: () => {},
        onMeshLoaded: () => { $('#optMeshWrap').hidden = false; },
        onMeshError: (u) => toast(`Could not load the scan mesh (${u.split('/').pop()}).`, 'bad'),
      });
      scene.setColliders(walkColliders);
      if (S.world) scene.setWorld(S.world, S.src?.meshUrls);
      syncScene();
      return scene;
    }).catch((err) => {
      sceneLoading = null;
      $('#loading3d').hidden = true;
      toast('3D needs three.js from the CDN and could not load it. The plan still works.', 'bad', 6000);
      throw err;
    });
  }
  return sceneLoading;
}

function syncScene() {
  if (!scene) return;
  scene.setOptions({ ceiling: S.ceiling || S.view === 'walk', labels: S.labels, mesh: S.mesh });
  scene.setItems(S.items, S.status, S.selectedId, S.hidden);
}

async function setView(v) {
  if (!S.world && v !== 'plan') { toast('Load a room first.'); return; }
  const prev = S.view;
  S.view = v;
  $$('#viewTabs button').forEach((b) => b.classList.toggle('on', b.dataset.view === v));
  document.body.classList.toggle('view-plan', v === 'plan');
  document.body.classList.toggle('view-walk', v === 'walk');
  if (v === 'walk' && isSheetLayout()) { document.body.classList.remove('sheet-full'); document.body.classList.add('sheet-collapsed'); }
  if (v === 'plan') {
    $('#three').hidden = true;
    planCanvas.hidden = false;
    plan.setVisible(true);
    scene?.setVisible(false);
    $('#walkHint').hidden = true;
    return;
  }
  plan.setVisible(false);
  planCanvas.hidden = true;
  $('#three').hidden = false;
  try { await ensureScene(); } catch { setView('plan'); return; }
  if (S.view !== v) return;
  if (prev === 'orbit' && v === 'walk') scene.saveOrbit();
  scene.setVisible(true);
  if (v === 'walk') scene.setMode('walk', walkStart());
  else scene.setMode('orbit');
  syncScene();
  updateItemTools();
}

function walkStart() {
  const op = openingByKey(S.world, S.entryKey);
  if (op) {
    const p = [op.center[0] + op.normalIn[0] * 0.7, op.center[1] + op.normalIn[1] * 0.7];
    return { x: p[0], z: p[1], dir: op.normalIn };
  }
  const r = S.world.rooms[0];
  return { x: r.centroid[0], z: r.centroid[1], dir: [0, -1] };
}

let colliderCache = { stamp: -1, value: null };
function walkColliders() {
  if (colliderCache.stamp === S.stamp && colliderCache.world === S.world) return colliderCache.value;
  const env = buildEnv(S.world, { boxes: [] });
  const value = {
    segments: env.segments,
    boxes: [...S.world.obstacles.filter((o) => o.ymax > 0.3).map(obstacleObb), ...S.items.filter((i) => i.assembled.h > 0.3).map((i) => itemObb(i))],
  };
  colliderCache = { stamp: S.stamp, world: S.world, value };
  return value;
}

// ---------------------------------------------------------------- loading
async function loadSource(src) {
  let world;
  if (src.kind === 'house') world = worldFromHouse(src.raw, src.roomsById);
  else world = worldFromRoom(src.raw, { name: src.raw.name || (src.label || 'Room').replace(/\.json$/i, '') });
  S.src = src;
  S.world = world;
  S.items = [];
  S.selectedId = null;
  S.routes = {};
  S.order = null;
  S.orderIndex = null;
  S.overlay = null;
  S.hidden = new Set();
  S.entryKey = world.entryKey;
  S.stamp++;
  if (src.layout) {
    if (src.layout.entry && world.openings.some((o) => o.key === src.layout.entry)) S.entryKey = src.layout.entry;
    for (const it of src.layout.items || []) addItemRaw({ ...it, id: undefined });
  }
  S.scan = src.scan || null;
  S.highlightWall = null;
  setScanParam(S.scan?.id || null);
  $('#empty').hidden = true;
  document.body.classList.remove('no-room');
  $('#optMeshWrap').hidden = true;
  plan.setWorld(world);
  if (scene) scene.setWorld(world, src.meshUrls);
  else if (Object.keys(src.meshUrls || {}).length) ensureScene().catch(() => {});
  refreshAll();
  renderRoomCard();
  renderScanCard();
  if (S.scan?.coach) applyCoach(S.scan.coach, false);
  renderEntry();
  renderOrder();
  const t = world.kind === 'house' ? `${world.rooms.length} rooms` : `${world.rooms[0].area.toFixed(1)} m²`;
  $('#roomTitle').innerHTML = `<b>${esc(world.name)}</b> · ${t}`;
  document.title = `${world.name} · RoomTwin`;
}

async function openUrl(url) {
  try {
    const src = await loadFromUrl(url);
    await loadSource(src);
    return true;
  } catch (err) {
    toast(err.message || String(err), 'bad', 6000);
    return false;
  }
}

async function openFiles(files) {
  try { await loadSource(await loadFromFiles(files)); } catch (err) { toast(err.message || String(err), 'bad', 6000); }
}

// ---------------------------------------------------------------- scan upload + coach
// Upload a raw scan -> the server runs the OpenCV pipeline -> load its room.json.
// Attempts are kept for the session so a rescan loop reads as attempt 1, 2, ...
const ATTEMPTS_KEY = 'roomtwin.scanAttempts';
function loadAttempts() {
  try { const a = JSON.parse(sessionStorage.getItem(ATTEMPTS_KEY) || '[]'); return Array.isArray(a) ? a : []; } catch { return []; }
}
function saveAttempts() {
  try { sessionStorage.setItem(ATTEMPTS_KEY, JSON.stringify(S.attempts.slice(-8))); } catch { /* storage unavailable */ }
}
const nextAttemptN = () => S.attempts.reduce((m, a) => Math.max(m, a.n || 0), 0) + 1;

function setScanParam(id) {
  try {
    const u = new URL(location.href);
    if (id) { u.searchParams.set('scan', id); u.searchParams.delete('room'); u.searchParams.delete('layout'); } else u.searchParams.delete('scan');
    if (u.href !== location.href) history.replaceState(null, '', u);
  } catch { /* sandboxed */ }
}

async function openScan(attempt) {
  const src = await loadFromUrl(scanRoomUrl(attempt.id));
  attempt.coverage = src.raw.coverage?.score ?? attempt.coverage ?? null;
  if (!attempt.qa?.length) attempt.qa = src.raw.qa || [];
  if (!src.raw.name) src.raw = { ...src.raw, name: `Room scan ${attempt.n}` };
  src.scan = attempt;
  await loadSource(src);
  saveAttempts();
}

async function openScanById(id) {
  let a = S.attempts.find((x) => x.id === id);
  const isNew = !a;
  if (isNew) { a = { id, n: nextAttemptN(), coverage: null, processing_s: null, qa: [], coach: null }; S.attempts.push(a); }
  try {
    await openScan(a);
    return true;
  } catch (err) {
    if (isNew) S.attempts = S.attempts.filter((x) => x !== a);
    if (!S.scan) setScanParam(null);
    toast(`Could not open that scan: ${err.message || err}`, 'bad', 6000);
    return false;
  }
}

const mb = (b) => `${(b / 1048576).toFixed(b < 10485760 ? 1 : 0)} MB`;
const up = {
  box: $('#uploadBox'), card: $('#uploadBox .up-card'), title: $('#upTitle'), file: $('#upFile'), bar: $('#upBar'), fill: $('#upFill'),
  pct: $('#upPct'), size: $('#upSize'), note: $('#upNote'), cancel: $('#upCancel'), retry: $('#upRetry'), spin: $('#upSpin'),
  job: null, phase: '', timer: 0,
};

function upShow(file) {
  clearInterval(up.timer);
  up.phase = 'upload';
  up.box.hidden = false;
  up.card.classList.remove('bad');
  up.spin.hidden = false;
  up.title.textContent = 'Uploading your scan';
  up.file.textContent = `${file.name} · ${mb(file.size)}`;
  up.bar.className = 'up-bar';
  up.fill.style.width = '0%';
  up.pct.textContent = '0%';
  up.size.textContent = `0 of ${mb(file.size)}`;
  up.note.textContent = 'Keep this page open until it finishes.';
  up.cancel.textContent = 'Cancel';
  up.retry.hidden = true;
}

function upProgress(loaded, total) {
  if (up.phase !== 'upload') return;
  const p = total ? Math.min(1, loaded / total) : 0;
  up.fill.style.width = `${(p * 100).toFixed(1)}%`;
  up.pct.textContent = `${Math.floor(p * 100)}%`;
  up.size.textContent = `${mb(loaded)} of ${mb(total)}`;
}

function upMeasuring() {
  if (up.phase !== 'upload') return;
  up.phase = 'measure';
  up.title.textContent = 'Measuring your room with OpenCV…';
  up.bar.className = 'up-bar indet';
  up.size.textContent = 'uploaded';
  up.note.textContent = 'Finding the floor, walls, doors and ceiling, and checking what the scan missed. Takes a few seconds to about a minute.';
  const t0 = performance.now();
  up.pct.textContent = '0 s';
  up.timer = setInterval(() => { up.pct.textContent = `${Math.round((performance.now() - t0) / 1000)} s`; }, 500);
}

function upError(msg) {
  clearInterval(up.timer);
  up.phase = 'error';
  up.card.classList.add('bad');
  up.spin.hidden = true;
  up.title.textContent = 'That scan did not go through';
  up.bar.className = 'up-bar bad';
  up.pct.textContent = ''; up.size.textContent = '';
  up.note.textContent = msg;
  up.cancel.textContent = 'Close';
  up.retry.hidden = false;
}

function upHide() { clearInterval(up.timer); up.phase = ''; up.box.hidden = true; }

async function startScanUpload(file) {
  if (!file) return;
  if (!SCAN_EXT.test(file.name)) { toast('Upload a .ply, .obj or .zip from the scanner app.', 'bad', 5000); return; }
  if (file.size > 400 * 1048576) { toast('That file is over the 400 MB limit.', 'bad', 5000); return; }
  up.job?.abort();
  upShow(file);
  const job = up.job = uploadScan(file, {
    onProgress: (loaded, total) => { upProgress(loaded, total); if (total && loaded >= total) upMeasuring(); },
    onUploaded: upMeasuring,
  });
  try {
    const res = await job.promise;
    if (up.job !== job) return;
    clearInterval(up.timer);
    up.title.textContent = 'Loading the plan…';
    const attempt = { id: res.id, n: nextAttemptN(), coverage: null, processing_s: res.processing_s ?? null, qa: res.qa || [], coach: null, file: file.name };
    S.attempts.push(attempt);
    try { await openScan(attempt); } catch (err) {
      S.attempts = S.attempts.filter((x) => x !== attempt);
      throw new Error(`Measured, but the plan would not load: ${err.message}`);
    }
    upHide();
    if (isSheetLayout()) document.body.classList.remove('sheet-collapsed');
    $('#panel').scrollTop = 0;
    toast(`Measured in ${res.processing_s ?? '?'} s. Ask the coach whether the scan is good enough.`, 'ok', 5000);
  } catch (err) {
    if (up.job !== job) return;
    if (err.name === 'AbortError') upHide(); else upError(err.message || String(err));
  } finally {
    if (up.job === job) up.job = null;
  }
}

function pickScan() { const el = $('#scanInput'); el.value = ''; el.click(); }

function renderScanCard() {
  const card = $('#scanCard');
  const sc = S.scan;
  card.hidden = !sc;
  $('#qaList').hidden = !!sc; // the scan card lists the QA instead
  if (!sc) return;
  const run = S.coachRun?.id === sc.id ? S.coachRun : null;
  card.innerHTML = scanCardHtml({ scan: sc, attempts: S.attempts, coach: run || sc.coach, traceOpen: S.traceOpen });
}

function applyCoach(res, fresh) {
  S.highlightWall = res.decision === 'rescan' && res.wall_id ? { wall: res.wall_id, label: `Rescan wall ${res.wall_id}` } : null;
  if (fresh && S.highlightWall && S.view !== 'plan') setView('plan');
  plan.requestRender();
}

function showWall(id) {
  if (!S.world?.walls.some((w) => w.id === id)) { toast(`Wall ${id} is not in this plan.`); return; }
  S.highlightWall = { wall: id, label: S.scan?.coach?.decision === 'rescan' && S.scan.coach.wall_id === id ? `Rescan wall ${id}` : `Wall ${id}` };
  if (S.view !== 'plan') setView('plan');
  plan.requestRender();
}

async function askCoach() {
  const sc = S.scan;
  if (!sc || (S.coachRun?.pending && S.coachRun.id === sc.id)) return;
  const run = S.coachRun = { id: sc.id, pending: true, elapsed: 0 };
  const t0 = performance.now();
  const tick = setInterval(() => {
    run.elapsed = Math.round((performance.now() - t0) / 1000);
    const el = $('#scanCard [data-elapsed]');
    if (el && S.coachRun === run) el.textContent = ` · ${run.elapsed} s`;
  }, 1000);
  S.highlightWall = null;
  plan.requestRender();
  renderScanCard();
  try {
    const res = await coachScan(sc.id);
    sc.coach = res;
    if (res.coverage != null) sc.coverage = res.coverage;
    saveAttempts();
    if (S.coachRun === run) S.coachRun = null;
    if (S.scan === sc) applyCoach(res, true);
  } catch (err) {
    if (S.coachRun === run) S.coachRun = { id: sc.id, error: err.message || String(err), status: err.status };
  } finally {
    clearInterval(tick);
  }
  if (S.scan === sc) {
    renderScanCard();
    $('#scanCard .coach')?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

// ---------------------------------------------------------------- items
function addItemRaw(spec) {
  const id = spec.id || `f${S.nextId++}`;
  const it = {
    id,
    name: spec.name || 'Item',
    url: spec.url || null,
    image: spec.image || null,
    assembled: { w: spec.assembled.w, d: spec.assembled.d, h: spec.assembled.h },
    packages: (spec.packages || []).map((p) => ({ ...p })),
    flatpack: !!spec.flatpack,
    needs_assembly_space: spec.needs_assembly_space ?? (spec.flatpack ? 0.5 : 0),
    diagonal_depth: spec.diagonal_depth || undefined,
    color: spec.color || ITEM_COLOURS[(S.nextId + S.items.length) % ITEM_COLOURS.length],
    pose: spec.pose ? { ...spec.pose } : findFreePose(spec),
  };
  S.items.push(it);
  return it;
}

function activeRoom() {
  const sel = S.items.find((i) => i.id === S.selectedId);
  if (sel) { const r = roomAt(S.world, [sel.pose.x, sel.pose.z]); if (r) return r; }
  if (S.world.rooms.length === 1) return S.world.rooms[0];
  return [...S.world.rooms].sort((a, b) => b.area - a.area)[0];
}

function findFreePose(spec) {
  const room = activeRoom();
  const rooms = [room, ...S.world.rooms.filter((r) => r !== room)];
  const a = spec.assembled;
  const ok = (pose) => evaluatePlacement(S.world, [...S.items, { id: '__probe', name: 'probe', assembled: a, pose }]).__probe.ok;
  // Prefer a free spot with its back to a wall, as far from the entry as possible.
  const entry = openingByKey(S.world, S.entryKey);
  const ref = entry ? entry.center : room.centroid;
  for (const r of rooms) {
    let best = null;
    for (const w of S.world.walls.filter((x) => x.roomId === r.id)) {
      const yaw = Math.atan2(w.normalIn[0], w.normalIn[1]);
      for (let t = a.w / 2 + 0.02; t <= w.geomLength - a.w / 2 - 0.02; t += 0.1) {
        const pose = { x: w.a[0] + w.dir[0] * t + w.normalIn[0] * (a.d / 2 + 0.01), z: w.a[1] + w.dir[1] * t + w.normalIn[1] * (a.d / 2 + 0.01), yaw };
        const d = Math.hypot(pose.x - ref[0], pose.z - ref[1]);
        if ((!best || d > best.d) && ok(pose)) best = { d, pose };
      }
    }
    if (best) return best.pose;
  }
  for (const r of rooms) {
    const [cx, cz] = r.centroid;
    const xs = r.polygon.map((p) => p[0]), zs = r.polygon.map((p) => p[1]);
    const cands = [];
    for (let x = Math.min(...xs); x <= Math.max(...xs); x += 0.1) for (let z = Math.min(...zs); z <= Math.max(...zs); z += 0.1) cands.push([x, z]);
    cands.sort((p, q) => Math.hypot(p[0] - cx, p[1] - cz) - Math.hypot(q[0] - cx, q[1] - cz));
    for (const [x, z] of cands) for (const yaw of [0, Math.PI / 2]) {
      const probe = { id: '__probe', name: 'probe', assembled: a, pose: { x, z, yaw } };
      const st = evaluatePlacement(S.world, [...S.items, probe]).__probe;
      if (st.ok) return { x, z, yaw };
    }
  }
  return { x: room.centroid[0], z: room.centroid[1], yaw: 0 };
}

function addItem(spec) {
  if (!S.world) { toast('Load a room first.'); return null; }
  const it = addItemRaw(spec);
  S.stamp++;
  select(it.id);
  refreshAll();
  return it;
}

function removeItem(id) {
  S.items = S.items.filter((i) => i.id !== id);
  delete S.routes[id];
  if (S.selectedId === id) S.selectedId = null;
  if (S.overlay?.itemId === id) S.overlay = null;
  S.stamp++;
  invalidateOrder();
  refreshAll();
}

function select(id) {
  if (S.locked) return;
  if (S.selectedId === id) return;
  S.selectedId = id;
  const r = id && S.routes[id];
  setOverlay(r && !r.pending && r.stamp === S.stamp ? overlayFromRoute(r, id) : null);
  refreshAll({ scrollToSelected: true });
}

let dragFrame = 0;
function moveItem(id, pos, phase, noSnap) {
  const it = S.items.find((i) => i.id === id);
  if (!it || S.locked) return;
  if (phase === 'move' && pos) {
    let pose = { x: pos.x, z: pos.z, yaw: it.pose.yaw };
    if (S.snap && !noSnap) { const sn = snapToWall(S.world, it, pose); if (sn) pose = sn.pose; }
    it.pose = pose;
    S.stamp++;
    if (S.overlay) S.overlay = null, scene?.setOverlay(null);
    if (!dragFrame) dragFrame = requestAnimationFrame(() => { dragFrame = 0; refreshAll({ light: true }); });
  } else if (phase === 'end') {
    invalidateOrder();
    refreshAll();
  }
}

function rotateItem(id, yaw, phase) {
  const it = S.items.find((i) => i.id === id);
  if (!it || S.locked) return;
  if (phase === 'move' && yaw != null) {
    it.pose = { ...it.pose, yaw };
    S.stamp++;
    if (S.overlay) S.overlay = null, scene?.setOverlay(null);
    if (!dragFrame) dragFrame = requestAnimationFrame(() => { dragFrame = 0; refreshAll({ light: true }); });
  } else if (phase === 'end') {
    invalidateOrder();
    refreshAll();
  }
}

function rotateSelected(deltaDeg) {
  const it = S.items.find((i) => i.id === S.selectedId);
  if (!it) return;
  let yaw = it.pose.yaw + deltaDeg * Math.PI / 180;
  yaw = Math.round(yaw / (Math.PI / 36)) * (Math.PI / 36);
  rotateItem(it.id, yaw, 'move');
  rotateItem(it.id, null, 'end');
}

function invalidateOrder() {
  if (S.order) { S.order = { ...S.order, stale: true }; renderOrder(); }
}

// ---------------------------------------------------------------- evaluation
function evaluate() {
  S.status = S.world ? evaluatePlacement(S.world, S.items) : {};
  S.fits = {};
  for (const it of S.items) S.fits[it.id] = checkDoors(S.world, it, S.entryKey);
}

function refreshAll(opts = {}) {
  if (!S.world) return;
  evaluate();
  plan.requestRender();
  syncScene();
  renderItems(opts);
  updateItemTools();
}

function updateItemTools() {
  const show = !!S.selectedId && !S.locked && S.view !== 'walk';
  $('#itemTools').hidden = !show;
}

// ---------------------------------------------------------------- panels
function renderRoomCard() {
  const w = S.world;
  const area = w.rooms.reduce((s, r) => s + r.area, 0);
  const ceil = w.rooms.map((r) => r.ceiling);
  const cmin = Math.min(...ceil.map((c) => c.value)), cmax = Math.max(...ceil.map((c) => c.value));
  const csig = Math.max(...ceil.map((c) => c.sigma ?? 0));
  const covs = w.rooms.map((r) => r.coverage?.score).filter((x) => x != null);
  const cov = covs.length ? covs.reduce((a, b) => a + b, 0) / covs.length : null;
  const sigmas = w.walls.map((x) => x.sigma).filter((x) => x != null);
  const worst = sigmas.length ? Math.max(...sigmas) : null;
  const srcs = w.rooms.map((r) => r.source).filter(Boolean);
  const pts = srcs.reduce((s, x) => s + (x.points || 0), 0);
  $('#roomSource').textContent = srcs.length ? `${[...new Set(srcs.map((x) => x.type === 'rgbd' ? 'RGB-D' : 'point cloud'))].join(' + ')}${pts ? ` · ${pts.toLocaleString()} pts` : ''}` : '';
  $('#roomStats').innerHTML = `
    <div class="stat"><div class="k">Floor area</div><div class="v">${area.toFixed(1)} <small>m²</small></div></div>
    <div class="stat"><div class="k">Ceiling</div><div class="v">${cmin === cmax ? cmin.toFixed(2) : `${cmin.toFixed(2)}-${cmax.toFixed(2)}`} <small>m${csig ? sigmaTxt(csig) : ''}</small></div></div>
    <div class="stat"><div class="k">Walls measured</div><div class="v">${w.walls.length} <small>${worst != null ? `worst ±${Math.max(1, Math.round(worst * 100))} cm` : ''}</small></div></div>
    <div class="stat"><div class="k">Scan coverage</div><div class="v">${cov != null ? `${Math.round(cov * 100)}<small>%</small>` : '<small>n/a</small>'}</div>${cov != null ? `<div class="meter"><i style="width:${Math.round(cov * 100)}%"></i></div>` : ''}</div>`;
  const qa = w.rooms.flatMap((r) => (r.qa || []).map((q) => ({ ...q, room: r.name })));
  $('#qaList').innerHTML = qa.map((q) => `
    <div class="q ${q.severity === 'warn' || q.severity === 'error' ? 'warn' : 'info'}"><span class="dot"></span>
      <div>${q.action && q.action !== 'ok' ? `<div class="tag">${esc(q.action === 'rescan' ? 'Rescan suggested' : q.action === 'marker' ? 'Add a marker' : q.action)}</div>` : ''}
      ${w.kind === 'house' ? `<b>${esc(q.room)}:</b> ` : ''}${esc(q.message)}</div></div>`).join('');
  const rows = [];
  for (const r of w.rooms) {
    if (w.kind === 'house') rows.push(`<tr class="hl"><td colspan="2">${esc(r.name)}</td></tr>`);
    rows.push(`<tr><td>Ceiling</td><td>${r.ceiling.value.toFixed(2)} m${sigmaTxt(r.ceiling.sigma)}</td></tr>`);
    for (const wall of w.walls.filter((x) => x.roomId === r.id)) rows.push(`<tr><td>Wall ${esc(wall.id)}</td><td>${wall.length.toFixed(2)} m${sigmaTxt(wall.sigma)}</td></tr>`);
    for (const op of w.openings.filter((x) => x.roomId === r.id)) rows.push(`<tr><td>${esc(op.name ? op.name[0].toUpperCase() + op.name.slice(1) : `${op.type[0].toUpperCase() + op.type.slice(1)} ${op.id}`)}</td><td>${cm(op.width)} x ${cm(op.height)} cm${sigmaTxt(op.sigma)}${op.sill ? `, sill ${cm(op.sill)}` : ''}</td></tr>`);
  }
  $('#wallTable').innerHTML = rows.join('');
}

function renderCatalogue() {
  $('#catalogue').innerHTML = CATALOGUE.map((c) => `<button class="chip" data-key="${c.key}" type="button">${esc(c.name)}<small>${esc(c.short)}</small></button>`).join('');
}

function statusPill(it) {
  const st = S.status[it.id];
  if (st && !st.ok) return `<span class="st no">${esc(st.problems[0].kind === 'swing' ? 'Blocks door' : st.problems[0].kind === 'outside' ? 'Outside' : 'Overlap')}</span>`;
  const f = S.fits[it.id];
  if (!f) return '';
  if (f.verdict === 'fits') return '<span class="st ok">Fits door</span>';
  if (f.verdict === 'flatpack') return '<span class="st flat">Boxes fit</span>';
  if (f.verdict === 'no' && f.alternative) return '<span class="st flat">Other door</span>';
  if (f.verdict === 'no') return '<span class="st no">Won\'t fit door</span>';
  return '';
}

function doorBlock(it) {
  const f = S.fits[it.id];
  if (!f || !f.doors) return f ? `<div class="verdict"><div class="t">${esc(f.headline)}</div></div>` : '';
  const cls = f.verdict === 'fits' ? 'ok' : f.verdict === 'flatpack' ? 'flat' : 'no';
  const doorsLi = (f.perDoor || []).map((p) => {
    const b = p.result.best;
    const txt = b.fits ? `${b.slack < 0.005 ? '<1' : cm(b.slack)} cm spare` : `${Math.max(1, cm(b.excess))} cm short`;
    return `<li><span>${esc(p.door.label)} <small>${cm(p.door.width)} x ${cm(p.door.height)}</small></span><span>${txt}</span></li>`;
  }).join('');
  return `<div class="verdict ${cls}">
    <div class="t"><span class="lbl">Door</span>${esc(f.headline)}</div>
    <p>${esc(f.detail)}</p>
    ${(f.perDoor || []).length > 1 ? `<ul class="doors">${doorsLi}</ul>` : ''}
    ${f.alternative ? `<div class="row mt"><button class="btn small" data-act="altentry" data-key="${esc(f.alternative.entryKey)}" data-id="${it.id}" data-noroute="1" type="button">Use ${esc(f.alternative.label)}</button></div>` : ''}
  </div>`;
}

function routeBlock(it) {
  const r = S.routes[it.id];
  const btn = `<button class="btn small" data-act="route" data-id="${it.id}" type="button">${r && !r.pending ? 'Check again' : 'Check the route in'}</button>`;
  if (!r) return `<div class="row gap mt">${btn}<span class="subtle">Finds a way from the entry, turning it where needed.</span></div>`;
  if (r.pending) return `<div class="verdict"><div class="t"><span class="lbl">Route</span>Searching (x, z, turn) space...</div></div>`;
  const stale = r.stamp !== S.stamp;
  const cls = stale ? '' : r.ok ? 'ok' : 'no';
  const warn = (r.warnings || []).map((w) => `<p style="color:var(--warn)">${esc(w)}</p>`).join('');
  const alt = r.alternative ? `<button class="btn small" data-act="altentry" data-key="${esc(r.alternative.entryKey)}" data-id="${it.id}" type="button">Use ${esc(r.alternative.label)}</button>` : '';
  return `<div class="verdict ${cls}">
    <div class="t"><span class="lbl">Route</span>${stale ? 'Layout changed since this check' : r.ok ? 'Gets in' : 'Gets stuck'}</div>
    <p>${esc(r.explanation)}</p>${warn}
    <div class="row gap mt wrap">${!stale && (r.path?.length || r.partial?.length) ? `<button class="btn small" data-act="play" data-id="${it.id}" type="button"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>${r.ok ? 'Play route' : 'Show where'}</button>` : ''}${btn}${alt}</div>
  </div>`;
}

function renderItems(opts = {}) {
  const list = $('#itemList');
  $('#itemCount').textContent = S.items.length ? `${S.items.length} item${S.items.length > 1 ? 's' : ''}` : '';
  if (opts.light) {
    // during drags: only refresh the header pills
    for (const li of $$('li[data-id]', list)) {
      const it = S.items.find((i) => i.id === li.dataset.id);
      if (!it) continue;
      li.classList.toggle('bad', !!(S.status[it.id] && !S.status[it.id].ok));
      const pill = $('.pill', li);
      if (pill) pill.innerHTML = statusPill(it);
    }
    return;
  }
  if (!S.items.length) {
    list.innerHTML = '<li style="border-style:dashed;padding:12px;color:var(--muted);font-size:13px">No furniture yet. Add a link, type sizes, or pick a common item above.</li>';
    return;
  }
  list.innerHTML = S.items.map((it) => {
    const sel = it.id === S.selectedId;
    const st = S.status[it.id];
    const bad = st && !st.ok;
    const pk = it.flatpack && it.packages.length ? ` · flat-pack, ${it.packages.reduce((s, p) => s + (p.count || 1), 0)} box${it.packages.reduce((s, p) => s + (p.count || 1), 0) > 1 ? 'es' : ''}` : '';
    return `<li data-id="${it.id}" class="${sel ? 'sel' : ''} ${bad ? 'bad' : ''}">
      <div class="ih" data-act="select" data-id="${it.id}">
        <span class="sw" style="background:${it.color}"></span>
        <span class="nm"><b>${esc(it.name)}</b><small>${dims(it.assembled)}${pk}</small></span>
        <span class="pill">${statusPill(it)}</span>
      </div>
      ${sel ? `<div class="body">
        ${bad ? `<ul class="problems">${st.problems.map((p) => `<li>${esc(p.text)}</li>`).join('')}</ul>` : ''}
        ${doorBlock(it)}
        ${routeBlock(it)}
        <div class="row gap mt wrap">
          ${it.url ? `<a class="link small" href="${esc(it.url)}" target="_blank" rel="noopener">Product page</a>` : ''}
          <button class="link small" data-act="edit" data-id="${it.id}" type="button">Edit sizes</button>
          <button class="link small" data-act="remove" data-id="${it.id}" type="button" style="color:var(--bad)">Remove</button>
        </div>
      </div>` : ''}
    </li>`;
  }).join('');
  if (opts.scrollToSelected && S.selectedId) {
    const li = $(`li[data-id="${S.selectedId}"]`, list);
    li?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

function renderEntry() {
  const sel = $('#entrySel');
  const w = S.world;
  sel.innerHTML = w.entryCandidates.map((k) => `<option value="${esc(k)}">${esc(cap(openingLabel(w, openingByKey(w, k))))}</option>`).join('') || '<option value="">No exterior door</option>';
  sel.value = S.entryKey || '';
}
const cap = (s) => s.replace(/^the /, '').replace(/^./, (c) => c.toUpperCase());

function setEntry(key) {
  S.entryKey = key;
  S.stamp++;
  S.overlay = null;
  scene?.setOverlay(null);
  $('#entrySel').value = key;
  invalidateOrder();
  refreshAll();
}

function renderOrder() {
  const o = S.order;
  const list = $('#orderList');
  $('#btnPlayOrder').disabled = !o || !o.steps?.length || o.stale || !!o.pending;
  if (!o) {
    list.innerHTML = '';
    $('#orderStatus').textContent = S.items.length ? 'Works out what to carry in first so nothing blocks the way.' : 'Add furniture first.';
    return;
  }
  if (o.pending) { $('#orderStatus').textContent = o.progress ? `Routing ${o.progress.item} (${o.progress.done} searches)...` : 'Planning...'; return; }
  $('#orderStatus').innerHTML = o.stale ? '<span style="color:var(--warn)">The layout changed: plan again.</span>' : `${o.steps.length} step${o.steps.length !== 1 ? 's' : ''}, from the entry inwards.`;
  list.innerHTML = o.steps.map((s, i) => `<li data-i="${i}" class="${S.playStep === i ? 'now' : ''}"><span class="n">${s.n}</span><div><b>${esc(s.name)}</b><p>${esc(s.reason)}</p></div></li>`).join('') +
    (o.warnings || []).map((w) => `<li class="warn"><span class="n">!</span><div><p style="color:var(--bad)">${esc(w)}</p></div></li>`).join('');
}

// ---------------------------------------------------------------- jobs (worker)
let worker = null, workerBroken = false, jobId = 0;
const jobs = new Map();
function getWorker() {
  if (worker || workerBroken) return worker;
  try {
    worker = new Worker(new URL('./logistics/worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (e) => {
      const j = jobs.get(e.data.id);
      if (!j) return;
      if (e.data.progress) { j.onProgress?.(e.data.progress); return; }
      jobs.delete(e.data.id);
      if (e.data.ok) j.resolve(e.data.result); else j.reject(new Error(e.data.error));
    };
    worker.onerror = (e) => {
      e.preventDefault?.();
      workerBroken = true; worker = null;
      for (const [id, j] of jobs) { jobs.delete(id); runInline(j.msg, j.onProgress).then(j.resolve, j.reject); }
    };
  } catch { workerBroken = true; worker = null; }
  return worker;
}

async function runInline(msg, onProgress) {
  const { planRouteWithAlternatives, planBuildOrder } = await import('./logistics/plan.js');
  await new Promise((r) => setTimeout(r, 30));
  if (msg.type === 'route') {
    const item = msg.items.find((i) => i.id === msg.itemId);
    return planRouteWithAlternatives(msg.world, item, msg.items.filter((i) => i.id !== msg.itemId), msg.entryKey, msg.opts);
  }
  return planBuildOrder(msg.world, msg.items, msg.entryKey, msg.opts, onProgress);
}

function runJob(msg, onProgress) {
  const clean = JSON.parse(JSON.stringify(msg));
  const w = getWorker();
  if (!w) return runInline(clean, onProgress);
  return new Promise((resolve, reject) => {
    const id = ++jobId;
    jobs.set(id, { resolve, reject, onProgress, msg: clean });
    w.postMessage({ id, ...clean });
  });
}

const plainItems = () => S.items.map(({ color, ...rest }) => rest);

async function checkRoute(id, { play = true } = {}) {
  if (!S.entryKey) { toast('This room has no exterior door to come in through.', 'bad'); return; }
  const stamp = S.stamp;
  S.routes[id] = { pending: true };
  renderItems();
  try {
    const r = await runJob({ type: 'route', world: S.world, items: plainItems(), itemId: id, entryKey: S.entryKey });
    S.routes[id] = { ...r, stamp };
    if (S.selectedId === id && stamp === S.stamp) {
      setOverlay(overlayFromRoute(S.routes[id], id));
      if (play && r.ok) playRoute(id);
    }
  } catch (err) {
    delete S.routes[id];
    toast(`Route search failed: ${err.message}`, 'bad');
  }
  renderItems();
}

function overlayFromRoute(r, itemId) {
  if (!r || !r.carry) return null;
  return {
    itemId, ok: r.ok, path: r.path || null, partial: r.partial || null, stuck: r.stuck || null, corner: r.corner || null,
    fp: { hx: r.carry.across / 2, hz: r.carry.travel / 2 }, carry: r.carry, tightest: r.tightest || null, ghost: null,
  };
}

function setOverlay(ov) {
  S.overlay = ov;
  scene?.setOverlay(ov);
  plan.requestRender();
}

// ---------------------------------------------------------------- animation
let anim = null;
function animatePath(path, onDone, speed = 1.1) {
  const pts = densify(path, 0.03);
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.max(Math.hypot(pts[i].x - pts[i - 1].x, pts[i].z - pts[i - 1].z), Math.abs(pts[i].yaw - pts[i - 1].yaw) * 0.4));
  const total = cum[cum.length - 1] || 1;
  const t0 = performance.now();
  const my = anim = { stop: false };
  const step = () => {
    if (my.stop || anim !== my) return;
    const d = Math.min(total, ((performance.now() - t0) / 1000) * speed);
    let i = 1;
    while (i < cum.length - 1 && cum[i] < d) i++;
    const a = pts[i - 1], b = pts[i];
    const t = cum[i] > cum[i - 1] ? (d - cum[i - 1]) / (cum[i] - cum[i - 1]) : 1;
    const pose = { x: a.x + (b.x - a.x) * t, z: a.z + (b.z - a.z) * t, yaw: a.yaw + (b.yaw - a.yaw) * t };
    if (S.overlay) S.overlay.ghost = pose;
    scene?.setGhostPose(pose);
    plan.requestRender();
    if (d >= total) { anim = null; onDone?.(); return; }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
  return my;
}

function stopAnim() { if (anim) { anim.stop = true; anim = null; } }

function playRoute(id) {
  const r = S.routes[id];
  if (!r || r.stamp !== S.stamp) return;
  setOverlay(overlayFromRoute(r, id));
  if (!r.ok) { plan.requestRender(); return; }
  stopAnim();
  animatePath(r.path, () => { if (S.overlay) { S.overlay.ghost = null; scene?.setOverlay(S.overlay); plan.requestRender(); } });
}

async function planOrder() {
  if (!S.items.length) { toast('Add some furniture first.'); return; }
  if (!S.entryKey) { toast('This room has no exterior door to come in through.', 'bad'); return; }
  const stamp = S.stamp;
  S.order = { pending: true };
  renderOrder();
  $('#btnOrder').disabled = true;
  try {
    const r = await runJob({ type: 'order', world: S.world, items: plainItems(), entryKey: S.entryKey }, (p) => { if (S.order?.pending) { S.order.progress = p; renderOrder(); } });
    S.order = { ...r, stamp, stale: stamp !== S.stamp };
    S.orderIndex = Object.fromEntries(r.steps.map((s) => [s.itemId, s.n]));
  } catch (err) {
    S.order = null;
    toast(`Build order failed: ${err.message}`, 'bad');
  }
  $('#btnOrder').disabled = false;
  renderOrder();
  plan.requestRender();
}

function playOrder() {
  const o = S.order;
  if (!o || o.stale || !o.steps.length) return;
  stopAnim();
  S.locked = true;
  S.selectedId = null;
  S.hidden = new Set(o.steps.map((s) => s.itemId));
  $('#playbar').hidden = false;
  refreshAll();
  let i = 0;
  const next = () => {
    if (!S.locked) return;
    if (i >= o.steps.length) { finishPlay(); return; }
    const s = o.steps[i];
    S.playStep = i;
    renderOrder();
    $('#playText').textContent = `Step ${s.n} of ${o.steps.length}: ${s.name}${s.mode === 'package' ? ' (boxes)' : ''}`;
    const ov = { itemId: s.itemId, ok: true, path: s.path, fp: { hx: s.carry.across / 2, hz: s.carry.travel / 2 }, carry: s.carry, ghost: s.path[0] };
    setOverlay(ov);
    animatePath(s.path, () => {
      S.hidden.delete(s.itemId);
      syncScene();
      plan.requestRender();
      i++;
      setTimeout(next, 350);
    }, 1.6);
  };
  next();
}

function finishPlay() {
  stopAnim();
  S.locked = false;
  S.playStep = null;
  S.hidden = new Set();
  $('#playbar').hidden = true;
  setOverlay(null);
  renderOrder();
  refreshAll();
}

// ---------------------------------------------------------------- add-item dialog
const dlg = $('#itemDialog');
let editingId = null;
function pkgRow(p = {}) {
  const row = document.createElement('div');
  row.className = 'pkg';
  row.innerHTML = `<input type="number" placeholder="W" step="0.5" min="1" value="${p.w ? cm(p.w) : ''}" aria-label="Package width cm">
    <input type="number" placeholder="D" step="0.5" min="1" value="${p.d ? cm(p.d) : ''}" aria-label="Package depth cm">
    <input type="number" placeholder="H" step="0.5" min="1" value="${p.h ? cm(p.h) : ''}" aria-label="Package height cm">
    <input type="number" placeholder="Qty" step="1" min="1" value="${p.count || 1}" aria-label="Count">
    <button type="button" aria-label="Remove package">&times;</button>`;
  row.querySelector('button').onclick = () => row.remove();
  $('#pkgRows').appendChild(row);
}

function openDialog({ title = 'Add an item', note = '', item = null, url = '', name = '' } = {}) {
  editingId = item?.id || null;
  $('#dlgTitle').textContent = title;
  $('#dlgNote').textContent = note;
  $('#dlgNote').hidden = !note;
  $('#dlgOk').textContent = item?.id ? 'Save' : 'Add item';
  $('#fName').value = item?.name || name || '';
  $('#fW').value = item ? cm(item.assembled.w) : '';
  $('#fD').value = item ? cm(item.assembled.d) : '';
  $('#fH').value = item ? cm(item.assembled.h) : '';
  $('#fDiag').value = item?.diagonal_depth ? cm(item.diagonal_depth) : '';
  $('#fFlat').checked = !!item?.flatpack;
  $('#fAsm').value = item?.needs_assembly_space ? cm(item.needs_assembly_space) : 60;
  $('#fImg').value = item?.image || '';
  $('#fUrl').value = item?.url || url || '';
  $('#pkgRows').innerHTML = '';
  for (const p of item?.packages || []) pkgRow(p);
  $('#pkgBox').hidden = !$('#fFlat').checked;
  if ($('#fFlat').checked && !$('#pkgRows').children.length) pkgRow();
  dlg.showModal();
  setTimeout(() => (item ? $('#fW') : $('#fName')).focus(), 30);
}

$('#fFlat').addEventListener('change', () => {
  $('#pkgBox').hidden = !$('#fFlat').checked;
  if ($('#fFlat').checked && !$('#pkgRows').children.length) pkgRow();
});
$('#btnAddPkg').addEventListener('click', () => pkgRow());
dlg.addEventListener('close', () => {
  if (dlg.returnValue !== 'ok') return;
  const num = (sel) => parseFloat($(sel).value) / 100;
  const assembled = { w: num('#fW'), d: num('#fD'), h: num('#fH') };
  if (!(assembled.w > 0 && assembled.d > 0 && assembled.h > 0)) { toast('Width, depth and height are needed.', 'bad'); return; }
  const packages = $$('#pkgRows .pkg').map((r) => {
    const v = $$('input', r).map((i) => parseFloat(i.value));
    return v[0] > 0 && v[1] > 0 && v[2] > 0 ? { w: v[0] / 100, d: v[1] / 100, h: v[2] / 100, count: Math.max(1, Math.round(v[3] || 1)) } : null;
  }).filter(Boolean);
  const flat = $('#fFlat').checked;
  const spec = {
    name: $('#fName').value.trim() || 'Item', url: $('#fUrl').value || null, image: $('#fImg').value.trim() || null,
    assembled, packages: flat ? packages : [], flatpack: flat && packages.length > 0,
    needs_assembly_space: flat ? (parseFloat($('#fAsm').value) || 0) / 100 : 0,
    diagonal_depth: num('#fDiag') > 0 ? num('#fDiag') : undefined,
  };
  if (flat && !packages.length) toast('No package sizes given, so it is treated as delivered assembled.');
  if (editingId) {
    const it = S.items.find((i) => i.id === editingId);
    if (it) Object.assign(it, spec);
    delete S.routes[editingId];
    S.stamp++;
    invalidateOrder();
    refreshAll();
  } else addItem(spec);
});

function nameFromUrl(u) {
  try {
    const p = new URL(u).pathname.split('/').filter(Boolean);
    const slug = p.reverse().find((s) => /[a-z]{3,}/i.test(s) && !/^(p|dp|product|products|item|en|au|us|gb)$/i.test(s)) || '';
    return slug.replace(/\.(html?|php)$/i, '').replace(/[-_]+/g, ' ').split(' ').filter((t) => !/\d{4,}/.test(t)).join(' ').replace(/\s+/g, ' ').trim().replace(/^./, (c) => c.toUpperCase()).slice(0, 48);
  } catch { return ''; }
}

$('#urlForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const url = $('#urlInput').value.trim();
  if (!url) { openDialog(); return; }
  if (!S.world) { toast('Load a room first.'); return; }
  const btn = $('#btnFetch');
  btn.disabled = true; btn.textContent = 'Reading...';
  try {
    const item = await extractFromUrl(url);
    const it = addItem(item);
    $('#urlInput').value = '';
    toast(`Added ${it.name}: ${dims(it.assembled)}${item.flatpack ? ', flat-pack' : ''}. Check the sizes against the listing.`, 'ok', 5000);
  } catch (err) {
    openDialog({ title: 'Type the sizes', url, name: nameFromUrl(url), note: `Couldn't read that page automatically (${err.name === 'AbortError' ? 'timed out' : err.message}). Copy the sizes from the product page:` });
  } finally {
    btn.disabled = false; btn.textContent = 'Add';
  }
});

// ---------------------------------------------------------------- misc UI
function toast(msg, kind = '', ms = 3200) {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = msg;
  const box = $('#toasts');
  while (box.children.length >= 2) box.firstChild.remove();
  box.appendChild(el);
  setTimeout(() => el.remove(), ms);
}

function download(name, blob) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
}

const slug = (s) => String(s || 'room').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function exportLayout() {
  if (!S.world) { toast('Nothing to save yet.'); return; }
  const data = {
    kind: 'roomtwin-layout', version: 1, savedAt: new Date().toISOString(), name: S.world.name,
    entry: S.entryKey,
    items: S.items.map((it) => ({
      id: it.id, name: it.name, url: it.url, image: it.image, assembled: it.assembled, packages: it.packages,
      flatpack: it.flatpack, needs_assembly_space: it.needs_assembly_space, diagonal_depth: it.diagonal_depth,
      color: it.color, pose: { x: +it.pose.x.toFixed(4), z: +it.pose.z.toFixed(4), yaw: +it.pose.yaw.toFixed(5) },
      door_fit: S.fits[it.id] ? { verdict: S.fits[it.id].verdict, headline: S.fits[it.id].headline } : null,
    })),
    build_order: S.order && !S.order.stale ? S.order.steps.map((s) => ({ n: s.n, item: s.itemId, name: s.name, reason: s.reason })) : null,
    source: sourceForSave(S.src),
  };
  download(`${slug(S.world.name)}-layout.json`, new Blob([JSON.stringify(data, null, 1)], { type: 'application/json' }));
  toast('Layout saved. Open it again with Open or by dropping it here.', 'ok');
}

async function screenshot() {
  if (!S.world) return;
  let canvas;
  if (S.view === 'plan') { plan.draw(); canvas = planCanvas; } else { await ensureScene(); canvas = scene.snapshot(); }
  canvas.toBlob((b) => b && download(`${slug(S.world.name)}-${S.view}.png`, b), 'image/png');
}

function isSheetLayout() { return matchMedia('(max-width: 820px)').matches; }

function cycleSheet() {
  const b = document.body;
  if (b.classList.contains('sheet-collapsed')) { b.classList.remove('sheet-collapsed'); }
  else if (b.classList.contains('sheet-full')) { b.classList.remove('sheet-full'); b.classList.add('sheet-collapsed'); }
  else b.classList.add('sheet-full');
  setTimeout(() => plan.resize(), 220);
}

// ---------------------------------------------------------------- wiring
function wire() {
  renderCatalogue();
  document.body.classList.add('view-plan');
  $$('#viewTabs button').forEach((b) => b.addEventListener('click', () => setView(b.dataset.view)));
  const fileInput = $('#fileInput');
  const pick = () => fileInput.click();
  $('#btnOpen').addEventListener('click', pick);
  $('#btnOpen2').addEventListener('click', pick);
  $('#btnScan').addEventListener('click', pickScan);
  $('#btnScan2').addEventListener('click', pickScan);
  $('#scanInput').addEventListener('change', (e) => { const f = e.target.files[0]; e.target.value = ''; if (f) startScanUpload(f); });
  up.cancel.addEventListener('click', () => { const j = up.job; up.job = null; j?.abort(); upHide(); });
  up.retry.addEventListener('click', () => { upHide(); pickScan(); });
  const scanCard = $('#scanCard');
  scanCard.addEventListener('click', (e) => {
    const el = e.target.closest('[data-act]');
    if (!el) return;
    const act = el.dataset.act;
    if (act === 'coach') askCoach();
    else if (act === 'rescan') pickScan();
    else if (act === 'showwall') showWall(el.dataset.wall);
    else if (act === 'openscan' && el.dataset.scan !== S.scan?.id) openScanById(el.dataset.scan);
  });
  scanCard.addEventListener('toggle', (e) => { if (e.target.matches?.('details.trace')) S.traceOpen = e.target.open; }, true);
  fileInput.addEventListener('change', () => { if (fileInput.files.length) openFiles(fileInput.files); fileInput.value = ''; });
  $('#sampleSel').addEventListener('change', (e) => { if (e.target.value) openUrl(e.target.value); e.target.value = ''; });
  $$('.sample').forEach((b) => b.addEventListener('click', () => openUrl(b.dataset.src)));
  $('#btnExport').addEventListener('click', exportLayout);
  $('#btnShot').addEventListener('click', screenshot);
  $('#btnManual').addEventListener('click', () => (S.world ? openDialog() : toast('Load a room first.')));
  $('#catalogue').addEventListener('click', (e) => {
    const b = e.target.closest('.chip');
    if (!b) return;
    const c = CATALOGUE.find((x) => x.key === b.dataset.key);
    const it = addItem({ ...c, packages: (c.packages || []).map((p) => ({ ...p })) });
    if (it) toast(`${it.name} added. Drag it into place.`, 'ok');
  });
  $('#itemList').addEventListener('click', (e) => {
    const el = e.target.closest('[data-act]');
    if (!el) return;
    const id = el.dataset.id;
    const act = el.dataset.act;
    if (act === 'select') select(S.selectedId === id ? null : id);
    else if (act === 'remove') removeItem(id);
    else if (act === 'route') checkRoute(id);
    else if (act === 'play') playRoute(id);
    else if (act === 'edit') openDialog({ title: 'Edit sizes', item: S.items.find((i) => i.id === id) });
    else if (act === 'altentry') {
      setEntry(el.dataset.key);
      toast(`Entry set to ${openingLabel(S.world, openingByKey(S.world, el.dataset.key))}.`, 'ok');
      if (!el.dataset.noroute) checkRoute(id);
    }
  });
  $('#entrySel').addEventListener('change', (e) => setEntry(e.target.value));
  $('#btnOrder').addEventListener('click', planOrder);
  $('#btnPlayOrder').addEventListener('click', playOrder);
  $('#btnStopPlay').addEventListener('click', finishPlay);
  $('#optSnap').addEventListener('change', (e) => { S.snap = e.target.checked; });
  $('#optLabels').addEventListener('change', (e) => { S.labels = e.target.checked; S.showLabels = S.labels; plan.requestRender(); syncScene(); });
  $('#optCeiling').addEventListener('change', (e) => { S.ceiling = e.target.checked; syncScene(); });
  $('#optMesh').addEventListener('change', (e) => { S.mesh = e.target.checked; syncScene(); });
  S.showLabels = S.labels;
  $('#itemTools').addEventListener('click', (e) => {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    if (b.dataset.act === 'rotl') rotateSelected(-15);
    if (b.dataset.act === 'rotr') rotateSelected(15);
    if (b.dataset.act === 'rot90') rotateSelected(90);
    if (b.dataset.act === 'del' && S.selectedId) removeItem(S.selectedId);
  });
  $('#sheetHandle').addEventListener('click', cycleSheet);

  window.addEventListener('keydown', (e) => {
    if (/INPUT|SELECT|TEXTAREA/.test(document.activeElement?.tagName) || dlg.open) return;
    if (S.view === 'walk') { if (e.key === 'Escape') setView('orbit'); return; }
    const k = e.key.toLowerCase();
    if (k === 'escape') { if (S.locked) finishPlay(); else select(null); }
    if (!S.selectedId || S.locked) return;
    if (k === 'r') { rotateSelected(e.shiftKey ? 15 : 90); e.preventDefault(); }
    else if (k === 'delete' || k === 'backspace') { removeItem(S.selectedId); e.preventDefault(); }
    else if (k.startsWith('arrow')) {
      const it = S.items.find((i) => i.id === S.selectedId);
      const d = e.shiftKey ? 0.1 : 0.01;
      const v = { arrowleft: [-d, 0], arrowright: [d, 0], arrowup: [0, -d], arrowdown: [0, d] }[k];
      moveItem(it.id, { x: it.pose.x + v[0], z: it.pose.z + v[1] }, 'move', true);
      moveItem(it.id, null, 'end');
      e.preventDefault();
    }
  });

  // drag and drop anywhere
  let dragDepth = 0;
  window.addEventListener('dragenter', (e) => { e.preventDefault(); dragDepth++; $('#dropHint').hidden = false; });
  window.addEventListener('dragleave', () => { dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $('#dropHint').hidden = true; });
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault(); dragDepth = 0; $('#dropHint').hidden = true;
    const files = [...(e.dataTransfer?.files || [])];
    const scanFile = files.find((f) => SCAN_EXT.test(f.name));
    if (scanFile) startScanUpload(scanFile);
    else if (files.length) openFiles(files);
  });

  const qs = new URLSearchParams(location.search);
  const room = qs.get('room') || qs.get('layout');
  const view = qs.get('view');
  const scanId = qs.get('scan');
  S.attempts = loadAttempts();
  if (scanId) openScanById(scanId).then((ok) => { if (ok && view && ['plan', 'orbit', '3d', 'walk'].includes(view)) setView(view === '3d' ? 'orbit' : view); });
  else if (room) openUrl(room).then((ok) => { if (ok && view && ['plan', 'orbit', '3d', 'walk'].includes(view)) setView(view === '3d' ? 'orbit' : view); });
}

wire();

// Exposed for debugging and automated checks.
window.roomtwin = { S, setView, openUrl, addItem, checkRoute, planOrder, playOrder, select, setEntry, startScanUpload, openScanById, askCoach, CATALOGUE, plan, get scene() { return scene; } };
