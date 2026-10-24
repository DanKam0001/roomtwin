// "Scan check" card: the pipeline's QA findings, the scan coach's decision, the
// readable trace of how it decided, and this session's scan attempts.
// Pure HTML builders; app.js owns the state and the click handling.

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const pct = (v) => (v != null && isFinite(v) ? `${Math.round(v * 100)}%` : 'n/a');

// One line per tool step, written for a person reading the trace.
export function describeStep(step) {
  const inp = step.input || {};
  switch (step.tool) {
    case 'get_scan_summary':
      return { icon: 'list', title: 'Looked at the scan summary', detail: 'Wall lengths and uncertainty, openings, ceiling, coverage gaps and the pipeline\'s QA findings.' };
    case 'view_wall': {
      const missing = step.result === 'missing';
      return { icon: 'eye', title: `Viewed wall ${inp.wall_id || '?'}`, wall: inp.wall_id, detail: missing ? 'No image exists for that wall.' : 'Checked which parts of the wall the scanner actually saw (holes are gaps or openings).' };
    }
    case 'view_floorplan':
      return { icon: 'eye', title: 'Viewed the floor plan', detail: 'The top-down plan the pipeline drew, with its dimensions.' };
    case 'accept_scan':
      return { icon: 'ok', title: 'Accepted the scan', detail: inp.summary || '', final: 'accept' };
    case 'request_rescan':
      return { icon: 'warn', title: `Requested a rescan of wall ${inp.wall_id || '?'}`, wall: inp.wall_id, detail: inp.reason || inp.instruction || '', final: 'rescan' };
    case 'ask_user':
      return { icon: 'ask', title: 'Asked you a question', detail: inp.reason || inp.question || '', final: 'ask' };
    default:
      return { icon: 'list', title: `Used ${step.tool}`, detail: Object.keys(inp).length ? JSON.stringify(inp) : '' };
  }
}

const ICONS = {
  list: '<path d="M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01"/>',
  eye: '<path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
  ok: '<path d="m5 12 5 5 9-10"/>',
  warn: '<path d="M12 4v10M12 19h.01"/>',
  ask: '<path d="M9 9a3 3 0 1 1 4 2.8c-.7.3-1 .9-1 1.6V14M12 18h.01"/>',
};

const DECISION_LABEL = { accept: 'Accepted', rescan: 'Rescan requested', ask: 'Question asked' };

// ---------------------------------------------------------------- QA findings
// Long QA lists are noisy: group them by room, rank them, show the top two and
// fold the rest behind "Show all N". Wall ids get a friendly position.
const SEV = { error: 3, warn: 2, info: 1 };
const ACT = { rescan: 20, marker: 15 };
const CODE = { WALL_LOW_COVERAGE: 9, CORNER_UNSEEN: 8, WALL_GAP: 7, WALL_LOW_DENSITY: 4, WALL_OCCLUDED: 1 };

function qaScore(q) {
  const t = q.target || {};
  const extent = t.to != null && t.from != null ? (t.to - t.from) * Math.max(0.2, (t.y_to ?? 2) - (t.y_from ?? 0)) : 0;
  return (SEV[q.severity] || 1) * 100 + (ACT[q.action] || 0) + (CODE[q.code] || 2) + Math.min(5, extent * 2);
}

const lenTxt = (m) => (m < 1 ? `${Math.max(1, Math.round(m * 100))} cm` : `${m.toFixed(2)} m`);
function heightTxt(t) {
  if (t.y_from == null || t.y_to == null) return '';
  if (t.y_to - t.y_from > 1.6) return ', floor to ceiling';
  if (t.y_from < 0.15) return `, the bottom ${lenTxt(t.y_to)}`;
  return `, ${t.y_from.toFixed(1)}-${t.y_to.toFixed(1)} m up`;
}

// Plain-language version of a pipeline finding (the original stays in the tooltip).
export function friendlyQa(q) {
  const t = q.target || {};
  const span = t.from != null && t.to != null ? t.to - t.from : null;
  switch (q.code) {
    case 'WALL_GAP': return span != null ? `No scan data on a ${lenTxt(span)} stretch${heightTxt(t)}. Rescan it.` : q.message;
    case 'WALL_LOW_DENSITY': return span != null ? `Thin data on a ${lenTxt(span)} patch${heightTxt(t)}. Sweep it again slowly from 1-2 m away.` : q.message;
    case 'WALL_LOW_COVERAGE': return String(q.message || '').replace(/\bwall\s+\w+/i, 'this wall');
    case 'WALL_OCCLUDED': return `Partly hidden behind furniture${span != null ? ` (${lenTxt(span)})` : ''}. Fine to accept.`;
    case 'CORNER_UNSEEN': return 'This corner is barely seen. Point the phone into it and rescan.';
    default: return q.message || '';
  }
}

// Older saved attempts have "Room name: message"; newer ones carry q.room.
function splitRoom(q) {
  if (q.room) return q;
  const m = /^([^:]{1,40}):\s+(.*)$/.exec(q.message || '');
  return m ? { ...q, room: m[1], message: m[2] } : q;
}

function qaItem(q, describeWall) {
  const warn = q.severity === 'warn' || q.severity === 'error';
  const tag = q.action === 'rescan' ? 'Rescan' : q.action === 'marker' ? 'Add a marker' : warn ? 'Check' : 'Note';
  const t = q.target || {};
  let where = '';
  if (t.wall) {
    const d = describeWall?.(q.room, t.wall) || {};
    const txt = `${d.side ? `${d.side} wall` : `wall ${t.wall}`}`;
    where = ` · <button class="link" type="button" data-act="showwall" data-wall="${esc(t.wall)}"${d.roomId ? ` data-room="${esc(d.roomId)}"` : ''}>${esc(txt)}</button>${d.side ? ` <span class="wid">${esc(t.wall)}</span>` : ''}${d.feature ? ` <span class="wid">${esc(d.feature)}</span>` : ''}`;
  } else if (Array.isArray(t.corner)) {
    where = ` · corner <span class="wid">${esc(t.corner.join('/'))}</span>`;
  }
  return `<div class="q ${warn ? 'warn' : 'info'}" title="${esc(q.message)}"><span class="dot"></span><div>
    <div class="tag ${warn ? '' : 'info'}">${esc(tag)}${where}</div>
    ${esc(friendlyQa(q))}</div></div>`;
}

// qa: [{severity, code, message, action, target, room?}]; open: Set of room names whose list is expanded.
export function qaGroupsHtml(qa, { describeWall, open = new Set(), top = 2 } = {}) {
  if (!qa.length) return '<div class="subtle">The pipeline found nothing to flag.</div>';
  const groups = new Map();
  for (const raw of qa) {
    const q = splitRoom(raw);
    const k = q.room || '';
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(q);
  }
  const named = groups.size > 1 || [...groups.keys()][0] !== '';
  return `<div class="qa">${[...groups].map(([room, list]) => {
    const sorted = [...list].sort((a, b) => qaScore(b) - qaScore(a));
    const fix = list.filter((q) => q.action === 'rescan' || q.action === 'marker').length;
    const head = named ? `<div class="qa-room"><b>${esc(room || 'Scan')}</b><span>${fix ? `${fix} to rescan` : 'looks fine'}${list.length > fix ? ` · ${list.length - fix} note${list.length - fix > 1 ? 's' : ''}` : ''}</span></div>` : '';
    const first = sorted.slice(0, top).map((q) => qaItem(q, describeWall)).join('');
    const rest = sorted.slice(top);
    const more = rest.length ? `<details class="qa-more" data-room="${esc(room)}" ${open.has(room) ? 'open' : ''}>
        <summary><span class="when-closed">Show all ${sorted.length}</span><span class="when-open">Show fewer</span></summary>
        <div class="qa-rest">${rest.map((q) => qaItem(q, describeWall)).join('')}</div></details>` : '';
    return `<div class="qa-group">${head}${first}${more}</div>`;
  }).join('')}</div>`;
}

function decisionBlock(c) {
  if (c.decision === 'accept') {
    return `<div class="coach accept" role="status">
      <div class="ct"><span class="badge">Accepted</span>Good to plan with</div>
      ${c.summary ? `<p class="big">${esc(c.summary)}</p>` : ''}
      ${(c.caveats || []).length ? `<div class="sub">Caveats</div><ul class="caveats">${c.caveats.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}
    </div>`;
  }
  if (c.decision === 'rescan') {
    return `<div class="coach rescan" role="status">
      <div class="ct"><span class="badge">Rescan</span>One more pass needed${c.wall_id ? ` <button class="wallchip" type="button" data-act="showwall" data-wall="${esc(c.wall_id)}">wall ${esc(c.wall_id)}</button>` : ''}</div>
      <p class="instr">${esc(c.instruction || 'Rescan the highlighted wall.')}</p>
      ${c.reason ? `<p class="why"><b>Why:</b> ${esc(c.reason)}</p>` : ''}
      <button class="btn accent wide" type="button" data-act="rescan">
        <svg viewBox="0 0 24 24"><path d="M12 16V5m0 0-4 4m4-4 4 4M5 19h14"/></svg>Rescan and upload again</button>
    </div>`;
  }
  return `<div class="coach ask" role="status">
    <div class="ct"><span class="badge">Question</span>The coach needs your help</div>
    <p class="instr">${esc(c.question || 'Check the floor plan.')}</p>
    ${c.reason ? `<p class="why"><b>Why:</b> ${esc(c.reason)}</p>` : ''}
    <button class="btn wide mt" type="button" data-act="rescan">Rescan and upload again</button>
  </div>`;
}

function traceBlock(c, open) {
  const steps = (c.trace || []).map(describeStep);
  if (!steps.length) return '';
  return `<details class="trace" ${open ? 'open' : ''} data-act-toggle="trace">
    <summary>How the coach decided <span class="muted">· ${steps.length} step${steps.length !== 1 ? 's' : ''}${c.seconds != null ? `, ${c.seconds} s` : ''}</span></summary>
    <ol>${steps.map((s, i) => `<li class="${s.final ? `final ${s.final}` : ''}">
      <span class="n"><svg viewBox="0 0 24 24">${ICONS[s.icon]}</svg></span>
      <div><b>${i + 1}. ${s.wall ? esc(s.title).replace(esc(`wall ${s.wall}`), `<button class="link" type="button" data-act="showwall" data-wall="${esc(s.wall)}">wall ${esc(s.wall)}</button>`) : esc(s.title)}</b>
      ${s.detail ? `<p>${esc(s.detail)}</p>` : ''}</div></li>`).join('')}</ol>
  </details>`;
}

function historyBlock(attempts, currentId) {
  if (!attempts.length) return '';
  return `<div class="attempts">
    <div class="sub">Scan attempts this session</div>
    <ol>${attempts.map((a) => {
      const cur = a.id === currentId;
      const d = a.coach?.decision;
      return `<li class="${cur ? 'cur' : ''}">
        <button type="button" data-act="openscan" data-scan="${esc(a.id)}" ${cur ? 'disabled' : ''}>
          <span class="an">Attempt ${a.n}</span>
          <span class="cov">coverage ${pct(a.coverage)}</span>
          <span class="dec ${d || 'none'}">${d ? DECISION_LABEL[d] : 'Not checked yet'}</span>
        </button></li>`;
    }).join('')}</ol>
  </div>`;
}

export function scanCardHtml({ scan, attempts, coach, traceOpen, describeWall, qaOpen }) {
  const meta = [`attempt ${scan.n}`, scan.processing_s != null ? `measured in ${scan.processing_s} s` : null].filter(Boolean).join(' · ');
  let coachHtml;
  if (coach?.pending) {
    coachHtml = `<div class="coach pending" role="status"><div class="ct"><span class="spin"></span>The coach is checking your scan<span class="muted" data-elapsed>${coach.elapsed ? ` · ${coach.elapsed} s` : ''}</span></div>
      <p class="sub">It reads the measurements, looks at the walls that seem doubtful, then decides. Usually 10-40 s.</p></div>`;
  } else if (coach?.error) {
    coachHtml = `<div class="coach err" role="alert"><div class="ct">The coach could not run</div><p>${esc(coach.error)}</p>
      ${coach.status === 429 ? '' : '<button class="btn small mt" type="button" data-act="coach">Try again</button>'}</div>`;
  } else if (coach?.decision) {
    coachHtml = `${decisionBlock(coach)}${traceBlock(coach, traceOpen)}
      <div class="row gap mt"><button class="link small" type="button" data-act="coach">Ask the coach again</button></div>`;
  } else {
    coachHtml = `<button class="btn accent wide mt" type="button" data-act="coach">
      <svg viewBox="0 0 24 24"><path d="M12 3v3M12 18v3M4.2 7.5l2.6 1.5M17.2 15l2.6 1.5M4.2 16.5 6.8 15M17.2 9l2.6-1.5"/><circle cx="12" cy="12" r="3.2"/></svg>Ask the coach</button>
      <div class="subtle">The coach looks at the evidence and says whether the scan is good enough, or exactly what to rescan.</div>`;
  }
  return `<div class="card-h"><h2>Scan check</h2><span class="muted small">${esc(meta)}</span></div>
    ${qaGroupsHtml(scan.qa || [], { describeWall, open: qaOpen })}
    ${coachHtml}
    ${historyBlock(attempts, scan.id)}`;
}
