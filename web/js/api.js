// Server API client. The static app works without a server; these calls are
// optional enhancements and every caller has a fallback.
//
//   POST /api/extract   {url}  ->  furniture item JSON (docs/SPEC.md "Furniture item")
//   GET  /api/image?url=...    ->  image bytes (CORS-safe proxy for product photos)

const API_BASE = (new URLSearchParams(location.search).get('api') || '').replace(/\/$/, '');

function toMetres(v) {
  const n = Number(v);
  if (!isFinite(n) || n <= 0) return null;
  return n > 10 ? n / 100 : n; // tolerate centimetres
}

function normBox(b) {
  if (!b) return null;
  const w = toMetres(b.w ?? b.width), d = toMetres(b.d ?? b.depth), h = toMetres(b.h ?? b.height);
  if (!w || !d || !h) return null;
  return { w, d, h, ...(b.count ? { count: Number(b.count) || 1 } : {}) };
}

export function normaliseItem(raw, fallbackUrl) {
  const assembled = normBox(raw.assembled || raw.dimensions || raw);
  if (!assembled) throw new Error('no dimensions in response');
  const packages = (raw.packages || []).map(normBox).filter(Boolean);
  return {
    name: String(raw.name || raw.title || 'Item').slice(0, 80),
    url: raw.url || fallbackUrl || null,
    image: raw.image || raw.image_url || null,
    assembled,
    packages,
    flatpack: Boolean(raw.flatpack ?? packages.length > 0),
    needs_assembly_space: toMetres(raw.needs_assembly_space) ?? (packages.length ? 0.5 : 0),
    diagonal_depth: toMetres(raw.diagonal_depth ?? raw.assembled?.diagonal_depth) || undefined,
    source: raw.source || 'link',
    confidence: raw.confidence ?? null,
  };
}

export async function extractFromUrl(url, { timeoutMs = 25000 } = {}) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/api/extract`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ url }),
      signal: ctl.signal,
    });
    if (!res.ok) {
      let msg = `server said ${res.status}`;
      try { const j = await res.json(); if (j.detail || j.error) msg = String(j.detail || j.error); } catch { /* not json */ }
      throw new Error(msg);
    }
    const ct = res.headers.get('content-type') || '';
    if (!ct.includes('json')) throw new Error('no extraction server here');
    return normaliseItem(await res.json(), url);
  } finally {
    clearTimeout(t);
  }
}

export function proxiedImage(url) {
  return `${API_BASE}/api/image?url=${encodeURIComponent(url)}`;
}

// ---------------------------------------------------------------- scans
//   POST /api/scan   multipart `file` (.ply/.obj/.zip) -> {id, room_url, processing_s, qa, files}
//   GET  /api/scans/{id}/room.json
//   POST /api/coach  {id} -> {decision, summary?, caveats?, wall_id?, instruction?, reason?, question?, trace, seconds, coverage}

export const SCAN_EXT = /\.(ply|obj|zip)$/i;

// Server paths come back root-relative; keep them on the API host when ?api= is set.
export function apiUrl(path) {
  if (/^https?:/i.test(path)) return path;
  return `${API_BASE}${path.startsWith('/') ? '' : '/'}${path}`;
}

export const scanRoomUrl = (id) => apiUrl(`/api/scans/${encodeURIComponent(id)}/room.json`);

function detailText(j, status) {
  const d = j && (j.detail ?? j.error);
  if (Array.isArray(d)) return d.map((x) => x.msg || String(x)).join('; ');
  if (d) return String(d);
  if (status === 413) return 'That file is too large (400 MB max).';
  return `server said ${status || 'nothing'}`;
}

// XMLHttpRequest rather than fetch so the upload reports real progress.
export function uploadScan(file, { onProgress, onUploaded } = {}) {
  const xhr = new XMLHttpRequest();
  const promise = new Promise((resolve, reject) => {
    xhr.open('POST', `${API_BASE}/api/scan`);
    xhr.setRequestHeader('Accept', 'application/json');
    xhr.upload.onprogress = (e) => onProgress?.(e.loaded, e.lengthComputable ? e.total : file.size);
    xhr.upload.onload = () => onUploaded?.();
    xhr.onload = () => {
      let j = null;
      try { j = JSON.parse(xhr.responseText); } catch { /* not json */ }
      if (xhr.status >= 200 && xhr.status < 300 && j?.id) resolve(j);
      else reject(new Error(xhr.status >= 200 && xhr.status < 300 ? 'No scan server here.' : detailText(j, xhr.status)));
    };
    xhr.onerror = () => reject(new Error('Could not reach the RoomTwin server.'));
    xhr.onabort = () => { const e = new Error('Upload cancelled.'); e.name = 'AbortError'; reject(e); };
    const fd = new FormData();
    fd.append('file', file, file.name);
    xhr.send(fd);
  });
  return { promise, abort: () => xhr.abort() };
}

export async function coachScan(id, { timeoutMs = 120000 } = {}) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/api/coach`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ id }),
      signal: ctl.signal,
    });
    let j = null;
    try { j = await res.json(); } catch { /* not json */ }
    if (!res.ok) { const e = new Error(detailText(j, res.status)); e.status = res.status; throw e; }
    if (!j || !j.decision) throw new Error('The coach gave no decision.');
    return j;
  } catch (err) {
    if (err.name === 'AbortError') throw new Error('The coach took too long. Try again.');
    throw err;
  } finally {
    clearTimeout(t);
  }
}
