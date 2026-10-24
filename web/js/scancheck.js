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

function qaBlock(qa) {
  if (!qa.length) return '<div class="subtle">The pipeline found nothing to flag.</div>';
  return `<div class="qa">${qa.map((q) => {
    const warn = q.severity === 'warn' || q.severity === 'error';
    const tag = q.action === 'rescan' ? 'Rescan suggested' : q.action === 'marker' ? 'Add a marker' : warn ? 'Check' : 'Note';
    const wall = q.target?.wall;
    return `<div class="q ${warn ? 'warn' : 'info'}"><span class="dot"></span><div>
      <div class="tag ${warn ? '' : 'info'}">${esc(tag)}${wall ? ` · <button class="link" type="button" data-act="showwall" data-wall="${esc(wall)}">wall ${esc(wall)}</button>` : ''}</div>
      ${esc(q.message)}</div></div>`;
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

export function scanCardHtml({ scan, attempts, coach, traceOpen }) {
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
    ${qaBlock(scan.qa || [])}
    ${coachHtml}
    ${historyBlock(attempts, scan.id)}`;
}
