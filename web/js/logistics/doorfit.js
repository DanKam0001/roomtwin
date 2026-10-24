// Door-fit maths (SPEC "Fit through an opening"). Pure functions, no DOM.
//
// A box is carried through an opening along one of its axes (the travel axis).
// The other two axes form the cross-section the opening has to admit. We try all
// 6 axis permutations (travel, horizontal, vertical) and, for each, tilting the
// cross-section in the door plane by 0..45 degrees, which together cover every
// rotation of the rectangle within the plane. Fit = the rotated rectangle's
// bounding extents are within (width - clearance) x (height - clearance).
//
// For sofas with a published "diagonal depth" (the L-shaped side profile tipped
// back), an extra end-first candidate uses that diagonal as the width it needs.

export const DEFAULT_CLEARANCE = 0.02;
const AXES = ['w', 'd', 'h'];
const VERT_PREF = { h: 0, w: 1, d: 2 };
const PERMS = [
  ['w', 'd', 'h'], ['d', 'w', 'h'], // upright, travelling end-first / front-first
  ['h', 'd', 'w'], ['d', 'h', 'w'], // on its side
  ['h', 'w', 'd'], ['w', 'h', 'd'], // on its back
];

// Best placement of a p x q rectangle (p horizontal at theta=0) in a W x H window,
// rotating by theta in [0, maxTilt]. Returns the theta with the largest slack.
export function rectInWindow(p, q, W, H, maxTiltDeg = 45) {
  const evalAt = (deg) => {
    const t = deg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
    const u = p * c + q * s, v = p * s + q * c;
    return { theta: deg, u, v, slack: Math.min(W - u, H - v), widthSlack: W - u, heightSlack: H - v };
  };
  let best = evalAt(0);
  const step = 0.25;
  for (let deg = step; deg <= maxTiltDeg + 1e-9; deg += step) {
    const r = evalAt(deg);
    if (r.slack > best.slack + 1e-12) best = r;
  }
  // Local refinement (slack is piecewise smooth; golden-section on +-step).
  let lo = Math.max(0, best.theta - step), hi = Math.min(maxTiltDeg, best.theta + step);
  for (let i = 0; i < 40; i++) {
    const m1 = lo + (hi - lo) * 0.382, m2 = lo + (hi - lo) * 0.618;
    if (evalAt(m1).slack < evalAt(m2).slack) lo = m1; else hi = m2;
  }
  const refined = evalAt((lo + hi) / 2);
  if (refined.slack > best.slack) best = refined;
  if (best.theta < 0.05) best = { ...evalAt(0) };
  return best;
}

// Carver (1956) closed-form check used to cross-validate the numeric search:
// can a p x q rectangle fit inside an a x b rectangle at any rotation?
export function carverFits(p, q, a, b) {
  if (p < q) [p, q] = [q, p];
  if (a < b) [a, b] = [b, a];
  if (p <= a && q <= b) return true;
  if (p > a && q <= b) {
    const lhs = ((a + b) / (p + q)) ** 2 + ((a - b) / (p - q)) ** 2;
    return lhs >= 2;
  }
  return false;
}

function dimsOf(box) {
  return { w: box.w, d: box.d, h: box.h };
}

// All carry candidates for a box through one window (W x H already include clearance).
export function evaluateOrientations(box, W, H) {
  const dims = dimsOf(box);
  const out = [];
  for (const [travel, horiz, vert] of PERMS) {
    const r = rectInWindow(dims[horiz], dims[vert], W, H, 45);
    out.push({
      travel, horizontal: horiz, vertical: vert, special: null,
      theta: r.theta, tilted: r.theta >= 0.5, u: r.u, v: r.v,
      slack: r.slack, fits: r.slack >= -1e-9,
      binding: r.widthSlack <= r.heightSlack ? 'wide' : 'tall',
      excess: Math.max(0, -r.slack),
      footprint: { travel: dims[travel], across: dims[horiz] },
    });
  }
  if (box.diagonal_depth && box.diagonal_depth > 0) {
    // End-first with the side profile (d x h) tipped back: needs diagonal_depth of width.
    const u = box.diagonal_depth, v = Math.hypot(dims.d, dims.h);
    const slack = Math.min(W - u, H - v);
    out.push({
      travel: 'w', horizontal: 'd', vertical: 'h', special: 'diagonal',
      theta: Math.atan2(dims.h, dims.d) * 180 / Math.PI, tilted: true, u, v,
      slack, fits: slack >= -1e-9, binding: W - u <= H - v ? 'wide' : 'tall', excess: Math.max(0, -slack),
      footprint: { travel: dims.w, across: u },
    });
  }
  return out;
}

export function orientationKey(c) {
  return `${c.travel}${c.horizontal}${c.vertical}${c.special ? ':' + c.special : ''}`;
}

export function rankCandidates(cands) {
  return [...cands].sort((a, b) => {
    if (a.fits !== b.fits) return a.fits ? -1 : 1;
    if (!a.fits) return a.excess - b.excess;
    if (a.tilted !== b.tilted) return a.tilted ? 1 : -1;
    if (VERT_PREF[a.vertical] !== VERT_PREF[b.vertical]) return VERT_PREF[a.vertical] - VERT_PREF[b.vertical];
    return b.slack - a.slack;
  });
}

export function orientationPhrase(c) {
  if (c.special === 'diagonal') return 'upright, end-first, tipped back on its diagonal depth';
  let base = c.vertical === 'h' ? 'upright' : c.vertical === 'w' ? 'on its side' : 'on its back';
  if (c.vertical === 'h') base += c.travel === 'w' ? ', end-first' : ', front-first';
  if (c.tilted) base += `, tilted ${Math.round(c.theta)}°`;
  return base;
}

function cm(v) { return Math.max(1, Math.round(v * 100)); }

// One box through one opening {width, height}.
export function fitBox(box, opening, clearance = DEFAULT_CLEARANCE) {
  const W = opening.width - clearance, H = opening.height - clearance;
  const cands = evaluateOrientations(box, W, H);
  const ranked = rankCandidates(cands);
  const best = ranked[0];
  return { fits: best.fits, best, candidates: cands, W, H, clearance };
}

function boxLabel(b) {
  return `${Math.round(b.w * 100)} x ${Math.round(b.d * 100)} x ${Math.round(b.h * 100)} cm`;
}

// Whole item through a list of doors [{key, label, width, height}].
export function assessItemDoors(item, doors, clearance = DEFAULT_CLEARANCE) {
  const assembledBox = { ...item.assembled, diagonal_depth: item.diagonal_depth || item.assembled?.diagonal_depth };
  const perDoor = doors.map((door) => ({ door, result: fitBox(assembledBox, door, clearance) }));
  const failing = perDoor.filter((p) => !p.result.fits);
  const assembledFits = failing.length === 0;
  const tightest = assembledFits
    ? perDoor.reduce((m, p) => (!m || p.result.best.slack < m.result.best.slack ? p : m), null)
    : failing.reduce((m, p) => (!m || p.result.best.excess > m.result.best.excess ? p : m), null);

  let packages = null;
  const pk = (item.packages || []).filter((p) => p && p.w && p.d && p.h);
  if (item.flatpack && pk.length) {
    const res = [];
    for (const p of pk) for (const door of doors) res.push({ pkg: p, door, result: fitBox(p, door, clearance) });
    const bad = res.filter((r) => !r.result.fits);
    packages = {
      fits: bad.length === 0,
      worst: bad.length ? bad.reduce((m, r) => (r.result.best.excess > m.result.best.excess ? r : m)) : null,
      results: res,
    };
  }

  let verdict, headline, detail = '';
  if (!doors.length) {
    verdict = 'unknown'; headline = 'No entry door chosen.';
  } else if (assembledFits) {
    verdict = 'fits';
    const b = tightest.result.best;
    headline = `Fits ${orientationPhrase(b)}`;
    detail = `${b.slack < 0.005 ? 'Under 1' : cm(b.slack)} cm to spare through ${tightest.door.label}` +
      (doors.length > 1 ? ` (tightest of the ${doors.length} doors on the way).` : '.');
    if (b.special) detail += ' Tip it back level once it is through.';
    else if (b.vertical !== 'h') detail += ' Stand it back up once it is through.';
  } else {
    const b = tightest.result.best;
    const how = b.special === 'diagonal' ? 'tipped back on its diagonal depth'
      : b.vertical === 'h' && !b.tilted ? 'upright' : orientationPhrase(b).replace(/, (end|front)-first/, '');
    headline = `Does not fit: ${cm(b.excess)} cm too ${b.binding} even ${how}`;
    detail = `Blocked at ${tightest.door.label} (${Math.round(tightest.door.width * 100)} x ${Math.round(tightest.door.height * 100)} cm opening, ${Math.round(clearance * 100)} cm clearance).`;
    if (packages && packages.fits) {
      verdict = 'flatpack';
      detail += " It's flat-pack, so carry the boxes in and build it in the room.";
    } else if (packages && !packages.fits) {
      verdict = 'no';
      const w = packages.worst;
      detail += ` Even the flat-pack box (${boxLabel(w.pkg)}) is ${cm(w.result.best.excess)} cm too ${w.result.best.binding} for ${w.door.label}.`;
    } else {
      verdict = 'no';
      if (b.excess <= 0.04) detail += ' Taking the door off its hinges usually gains 3-4 cm.';
    }
  }
  return { verdict, headline, detail, assembledFits, perDoor, tightest, packages };
}

export { AXES, PERMS };
