// Loads room.json, house.json or a saved layout from a URL or dropped files.
// Returns {kind, raw, roomsById, meshUrls, layout} for app.js to turn into a world.

function kindOf(json) {
  if (json && json.kind === 'roomtwin-layout') return 'layout';
  if (json && Array.isArray(json.rooms)) return 'house';
  if (json && (json.walls || json.floor)) return 'room';
  return null;
}

function baseName(p) { return String(p || '').split(/[\\/]/).pop().split('?')[0]; }

async function fetchJson(url) {
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`Could not load ${baseName(url)} (${res.status})`);
  try { return await res.json(); } catch { throw new Error(`${baseName(url)} is not valid JSON`); }
}

function meshRefs(kind, raw, roomsById) {
  const refs = {};
  if (kind === 'room' && raw.mesh && /\.(glb|gltf)$/i.test(String(raw.mesh).split(' ')[0])) refs.room = String(raw.mesh).split(' ')[0];
  if (kind === 'house') for (const r of raw.rooms) {
    const room = r.room || roomsById[r.id];
    if (room?.mesh && /\.(glb|gltf)$/i.test(String(room.mesh).split(' ')[0])) refs[r.id] = String(room.mesh).split(' ')[0];
  }
  return refs;
}

async function resolveSource(json, base, fileMap) {
  const kind = kindOf(json);
  if (!kind) throw new Error('This file is not a room.json, house.json or RoomTwin layout');
  if (kind === 'layout') {
    const inner = await resolveSource(json.source, base, fileMap);
    return { ...inner, layout: json };
  }
  const roomsById = {};
  if (kind === 'house') {
    for (const r of json.rooms) {
      if (r.room) continue;
      if (!r.file) throw new Error(`house.json room "${r.id}" needs "room" or "file"`);
      if (fileMap) {
        const f = fileMap[baseName(r.file)];
        if (!f) throw new Error(`Also drop ${baseName(r.file)} (used by house.json)`);
        roomsById[r.id] = JSON.parse(await f.text());
      } else {
        roomsById[r.id] = await fetchJson(new URL(r.file, base).href);
      }
    }
  }
  const meshUrls = {};
  for (const [id, ref] of Object.entries(meshRefs(kind, json, roomsById))) {
    if (fileMap) { const f = fileMap[baseName(ref)]; if (f) meshUrls[id] = URL.createObjectURL(f); }
    else if (base) meshUrls[id] = new URL(ref, base).href;
  }
  return { kind, raw: json, roomsById, meshUrls, layout: null };
}

export async function loadFromUrl(url) {
  const abs = new URL(url, location.href).href;
  const json = await fetchJson(abs);
  const out = await resolveSource(json, abs, null);
  out.label = baseName(url);
  out.url = abs;
  return out;
}

export async function loadFromFiles(files) {
  const list = [...files];
  const fileMap = {};
  for (const f of list) fileMap[f.name] = f;
  const jsons = [];
  for (const f of list.filter((x) => /\.json$/i.test(x.name) || x.type === 'application/json')) {
    try { jsons.push({ f, json: JSON.parse(await f.text()) }); } catch { /* skip */ }
  }
  if (!jsons.length) throw new Error('Drop a room.json, house.json or saved layout');
  const rank = { layout: 0, house: 1, room: 2 };
  jsons.sort((a, b) => (rank[kindOf(a.json)] ?? 9) - (rank[kindOf(b.json)] ?? 9));
  const top = jsons[0];
  const out = await resolveSource(top.json, null, fileMap);
  out.label = top.f.name;
  return out;
}

// For saved layouts: embed everything so the file is self-contained.
export function sourceForSave(src) {
  if (src.kind === 'room') return src.raw;
  const house = JSON.parse(JSON.stringify(src.raw));
  for (const r of house.rooms) if (!r.room && src.roomsById[r.id]) { r.room = src.roomsById[r.id]; delete r.file; }
  return house;
}
