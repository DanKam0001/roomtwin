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
