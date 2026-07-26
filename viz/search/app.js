"use strict";
/* Episode replay view: one search, replayed pop by pop.
 *
 * Data shapes (see docs/superpowers/specs/2026-07-26-search-viz-design.md), schema_version 3:
 *   trace = {meta:{..., search:{combine,agg,prior,raw,dive_bonus,discount,gamma,tau,eps,w0_mode,
 *                               free_strike_q,gtable,hmax,sim_budget}},
 *            scene, boards:[{board_id,depth,parent_edge,parent_depth,pool:[{obj,edge,depth,q}],grid|null,
 *                            w0,free_strikes,geom|null,regions|null}],
 *            pops:[{t,board_id,obj,edge,depth,q,bp,w,se,opened}], result:{solved,sims,plan_len,end}}
 *   gt    = null | {root:{openers:[[e,d]],setups:[[e,d]]}, finish:{"<parent_edge>_<parent_depth>":{openers,setups}}}
 *
 * v3 adds PER-BOARD state. `scene` is still the episode's START geometry (and the only geometry a v2
 * trace has), but every board now also carries `geom` -- the movable/robot poses and the target's 60
 * contact points AT THAT BOARD'S OWN STATE -- and `regions` -- the wavefront region decomposition
 * there, run-length encoded (scripts/viz/trace_schema.py rle_encode documents the format). So zone A
 * redraws for the board being viewed instead of freezing at the start pose, and the two regions the
 * whole problem is about (the robot's and the goal's, which a successful push MERGES) are visible.
 * Both fields are read defensively: a v2 trace has neither, and then zone A renders exactly as before.
 *
 * ORDERING IS RECOMPUTED, NOT READ OFF THE POPS. Two values set a candidate's place in the queue:
 *   bp -- the base priority. Only recorded for already-popped candidates, so queued ones need the formula:
 *         priority(q, V, combine) + dive_bonus*depth, V = the board's own pool aggregate (agg).
 *   w  -- the board weight. `pops[].w` is the weight the pop SAW, i.e. BEFORE its own failure demoted it, and
 *         most child boards are popped exactly once -- so a board's post-failure weight is simply NOT in the
 *         file. Replaying pops[].w therefore leaves nearly every board at 1.0 and shows an order the search
 *         never used. Instead w is recomputed by re-applying the generator's demotion rule to that board's
 *         failures in order (seeded from w0, honouring free_strikes, root frozen at 1).
 * Both use meta.search -- the generator's actual flags -- never assumed defaults, because e.g. `--combine q`
 * changes the formula outright. verifyReconstruction() then checks the result against every recorded (bp, w);
 * any mismatch raises a visible banner instead of quietly presenting a wrong order.
 */

// Generator defaults (scripts/sandbox/eval_bestfirst.py argparse) -- used ONLY for pre-v2 traces, which also
// raise the verification banner, since assuming these is exactly the bug this page had.
const SEARCH_DEFAULTS = {
  combine: "blend", agg: "mean5", prior: "model", dive_bonus: 0.0,
  discount: "off", gamma: 0.65, tau: 1.0, eps: 1e-3, gtable: null,
};
const RECON_TOL = 1e-9;
// The schema_version this page knows how to reconstruct (scripts/viz/trace_schema.py SCHEMA_VERSION). Keyed
// on the version itself, not on the presence of any one field -- a later schema bump that happens to keep
// meta.search around must still be caught here, not slip through unflagged.
const SUPPORTED_SCHEMA_VERSION = 3;

const state = { trace: null, gt: null, t: 0, hover: null, manifestRow: null };
const boardVCache = new Map(); // board_id -> V (the board's pool aggregate, fixed at board creation)

function qs(name) {
  return new URLSearchParams(window.location.search).get(name);
}

function sp() {
  return state.trace.meta.search || SEARCH_DEFAULTS;
}

function boardById(id) {
  return state.trace.boards.find((b) => b.board_id === id);
}

// V(board) = eval_bestfirst.py candidates(): the state value the board was pushed with. uniform prior has no
// state value at all (V=0); otherwise mean of the top 5 pool q (agg=mean5) or the single best (agg=max).
function boardV(board) {
  if (boardVCache.has(board.board_id)) return boardVCache.get(board.board_id);
  const qsSorted = board.pool.map((c) => c.q).sort((a, b) => b - a);
  let v = 0;
  if (sp().prior !== "uniform" && qsSorted.length) {
    v = sp().agg === "max" ? qsSorted[0] : qsSorted.slice(0, 5).reduce((a, b) => a + b, 0) / Math.min(5, qsSorted.length);
  }
  boardVCache.set(board.board_id, v);
  return v;
}

// bp = eval_bestfirst.py priority(q, V, combine) + the cascade dive bonus children were pushed with.
function bpOf(board, q) {
  const p = sp();
  const V = boardV(board);
  const base = p.combine === "q" ? q : p.combine === "product" ? q * V : 0.5 * q + 0.5 * V;
  return base + (p.dive_bonus || 0) * board.depth;
}

// One failed sim on `board` (its kFailed-th), scoring q. Mirrors _update_w_on_fail: root boards never demote,
// `free_strikes` initial failures are forgiven, w is floored at eps and only ever decreases.
// Parity reference: python/tests/test_viz_demotion_parity.py pins this rule against the generator's own
// `_update_w_on_fail` (scripts/sandbox/eval_bestfirst.py) -- keep this function's behavior matching that test.
function demote(board, w, kFailed, qFailed) {
  const p = sp();
  if (board.depth < 1 || p.discount === "off") return w;
  const k = kFailed - board.free_strikes;
  if (k <= 0) return w;
  let out = w;
  if (p.discount === "gamma") out *= p.gamma;
  else if (p.discount === "conf") out *= Math.pow(1 - qFailed, p.tau);
  else if (p.discount === "fitted") {
    const kmax = Math.max(...Object.keys(p.gtable).map(Number));
    out = board.w0 * p.gtable[String(Math.min(k, kmax))];
  }
  return Math.max(out, p.eps);
}

// ---- Pure state-derived functions (Step 1) ---------------------------------------------------

function poppedKeySet(t) {
  const s = new Set();
  for (let i = 0; i < t; i++) {
    const p = state.trace.pops[i];
    s.add(`${p.board_id}:${p.edge}:${p.depth}`);
  }
  return s;
}

// Every board's weight after pops[0..t) -- recomputed, since the post-failure value is never on disk.
function boardWAt(t) {
  const w = new Map();
  const kFailed = new Map();
  for (const b of state.trace.boards) {
    w.set(b.board_id, b.w0);
    kFailed.set(b.board_id, 0);
  }
  for (let i = 0; i < t; i++) {
    const p = state.trace.pops[i];
    if (p.opened) continue; // mirrors _update_w_on_fail's early return on success -- opened pops never demote
    const b = boardById(p.board_id);
    const k = kFailed.get(p.board_id) + 1;
    kFailed.set(p.board_id, k);
    w.set(p.board_id, demote(b, w.get(p.board_id), k, p.q));
  }
  return w;
}

// Check the reconstruction against the trace: at every pop, the bp and w this page would have displayed must
// equal what the search recorded. Returns the mismatch count (0 = the replayed order is the real one).
function verifyReconstruction() {
  let bad = 0;
  for (let i = 0; i < state.trace.pops.length; i++) {
    const p = state.trace.pops[i];
    const b = boardById(p.board_id);
    // `<=` (not `>` negated implicitly) so a NaN recomputation -- e.g. a `fitted` gtable gap the Python
    // generator would raise on -- counts as a mismatch instead of silently passing (NaN fails every
    // comparison, so `NaN > TOL` is false and a naive `>` check would let it through uncaught).
    if (!(Math.abs(bpOf(b, p.q) - p.bp) <= RECON_TOL)) bad++;
    else if (!(Math.abs(boardWAt(i).get(p.board_id) - p.w) <= RECON_TOL)) bad++;
  }
  return bad;
}

// Which boards exist by time t. Root (board 0) always exists; every other board is spawned by
// exactly one FAILED pop with room left to expand -- detected generically (no hmax hardcoding) by
// matching (parent_edge, parent_depth) to a failed pop whose own board is one depth shallower.
function revealedBoardsAt(t) {
  const revealed = new Set([0]);
  for (let i = 0; i < t; i++) {
    const p = state.trace.pops[i];
    if (p.opened) continue; // search stops on open -- no child spawned
    const popBoard = boardById(p.board_id);
    const child = state.trace.boards.find(
      (b) => b.parent_edge === p.edge && b.parent_depth === p.depth && b.depth === popBoard.depth + 1
    );
    if (child) revealed.add(child.board_id);
  }
  return revealed;
}

// queueAt(t): replay pops[0..t) to know what's left unsimulated and each board's live w, then
// sort the remainder by bp*w descending. This is the priority queue the search is holding at t.
function queueAt(t) {
  const popped = poppedKeySet(t);
  const wAt = boardWAt(t);
  const revealed = revealedBoardsAt(t);
  const rows = [];
  for (const b of state.trace.boards) {
    if (!revealed.has(b.board_id)) continue;
    const w = wAt.get(b.board_id);
    b.pool.forEach((c, idx) => {
      const key = `${b.board_id}:${c.edge}:${c.depth}`;
      if (popped.has(key)) return;
      const bp = bpOf(b, c.q);
      rows.push({ board_id: b.board_id, idx, obj: c.obj, edge: c.edge, depth: c.depth, q: c.q, bp, w, se: bp * w });
    });
  }
  // ties break the way the generator's heap does: by insertion counter, i.e. board creation order then pool order
  rows.sort((a, b) => b.se - a.se || a.board_id - b.board_id || a.idx - b.idx);
  return rows;
}

// greenAt(boardId): the opener/setup (edge,depth) sets for that board, PLUS whether a setup counts as green
// there. At the root both count: an opener merges robot+goal now, a setup earns a finish push. At a finish
// board a "setup" would only set up a THIRD push -- it does not open the way, so only openers are green.
function greenAt(boardId) {
  const empty = { openers: new Set(), setups: new Set(), setupsGreen: false };
  if (!state.gt) return empty;
  const b = boardById(boardId);
  const src = b.depth === 0 ? state.gt.root : state.gt.finish[`${b.parent_edge}_${b.parent_depth}`];
  if (!src) return empty;
  return {
    openers: new Set(src.openers.map(([e, d]) => `${e}:${d}`)),
    setups: new Set(src.setups.map(([e, d]) => `${e}:${d}`)),
    setupsGreen: b.depth === 0,
  };
}

// "setup-late" = ground truth calls it a setup, but on this board that buys nothing -- labelled, never green.
function truthOf(green, edge, depth) {
  const key = `${edge}:${depth}`;
  if (green.openers.has(key)) return "opener";
  if (green.setups.has(key)) return green.setupsGreen ? "setup" : "setup-late";
  return state.gt ? "dead" : "unknown";
}

function isGreen(truth) {
  return truth === "opener" || truth === "setup";
}

// The board whose candidates are live "now", at t: since pops[t] (0-indexed) IS the candidate the
// search tries next -- literally whatever queueAt(t)'s top row is, once bp/w reconstruction is
// faithful -- this keeps zone A (scene) and zone D (grid) showing the SAME board zone B's queue is
// about to act on. Before this fix it returned pops[t - 1].board_id, i.e. the board the MOST
// RECENT pop came FROM: one step behind the queue by construction (pops[t-1] is always exactly
// what queueAt(t-1)'s top row was, since that's what a best-first search just popped), which is the
// concrete mechanism behind "the reachable edges ... seem 1 sim step delayed". At the end of the
// search (t === pops.length, nothing left to try next) fall back to the final pop's own board.
function currentBoardIdAt(t) {
  const pops = state.trace.pops;
  return t < pops.length ? pops[t].board_id : pops[pops.length - 1].board_id;
}

function bestGreenRank(rows, greenFor) {
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    if (isGreen(truthOf(greenFor(r.board_id), r.edge, r.depth))) return i + 1;
  }
  return null;
}

// ---- Small render helpers ---------------------------------------------------------------------

function boardHue(boardId) {
  return (boardId * 47) % 360;
}

function boardTag(board) {
  return board.depth === 0 ? "root" : `finish(e${board.parent_edge},d${board.parent_depth})`;
}

function lerp(a, b, f) {
  return a + (b - a) * f;
}

function qColor(q, qmin, qmax) {
  const f = qmax > qmin ? (q - qmin) / (qmax - qmin) : 0.5;
  // light blue -> dark blue, distinct from the green/red truth palette
  const r = Math.round(lerp(207, 8, f));
  const g = Math.round(lerp(232, 48, f));
  const bch = Math.round(lerp(255, 107, f));
  return `rgb(${r},${g},${bch})`;
}

function fmtPct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

// ---- Zone A: the scene ---------------------------------------------------------------------

// Decode board.regions.rle straight into DRAWABLE runs, skipping the intermediate grid: each run is
// one region id filling a contiguous span of one column ix, i.e. exactly one rectangle. Format is
// pinned by scripts/viz/trace_schema.py rle_encode -- flat [value, count, ...] over the row-major
// flatten of the (nx, ny) id grid, runs never crossing a row -- but the walk below splits at row
// boundaries anyway, so a run that did span rows would still decode to correct rectangles.
function regionRuns(regions) {
  const { nx, ny, res, origin, rle } = regions;
  const runs = [];
  let ix = 0;
  let iy = 0;
  for (let i = 0; i < rle.length && ix < nx; i += 2) {
    const v = rle[i];
    let n = rle[i + 1];
    while (n > 0 && ix < nx) {
      const take = Math.min(n, ny - iy);
      if (v !== 0) runs.push({ v, x: origin[0] + ix * res, y: origin[1] + iy * res, h: take * res });
      iy += take;
      n -= take;
      if (iy >= ny) {
        iy = 0;
        ix += 1;
      }
    }
  }
  return runs;
}

// The problem in one picture: "robot" = where the robot can currently get to, "goal" = the pocket it
// is trying to reach, and a push succeeds exactly when the two become one region ("robot_goal").
// Everything else is background free space -- drawn, but deliberately dull.
function regionClass(label) {
  if (label === "robot") return "region-robot";
  if (label === "goal") return "region-goal";
  if (label === "robot_goal") return "region-merged";
  return "region-other";
}

function regionLayer(regions) {
  if (!regions) return "";
  const res = regions.res;
  const cells = regionRuns(regions).map((r) => {
    const label = regions.labels[String(r.v)] || `region_${r.v}`;
    // 2% overhang on the width so neighbouring columns of the same region overlap instead of
    // leaving antialiased hairlines between them (0.1 mm of overdraw at 5 mm cells).
    return (
      `<rect class="region-cell ${regionClass(label)}" x="${r.x}" y="${r.y}" ` +
      `width="${res * 1.02}" height="${r.h}"><title>${label}</title></rect>`
    );
  });
  return `<g class="region-layer">${cells.join("")}</g>`;
}

function renderSceneA() {
  const svg = document.getElementById("scene-svg");
  const scene = state.trace.scene;
  const [xmin, xmax, ymin, ymax] = scene.bounds;
  const w = xmax - xmin, h = ymax - ymin;
  // No Y-flip: contacts arrive as raw world (x,y) already rotated by contact_offsets_world's
  // standard [cos -sin; sin cos] matrix, and the rect rotate() below uses the same matrix on the
  // same raw coordinates, so leaving the frame as-is keeps contact points and rectangle edges
  // aligned. (Screen "up" ends up meaning +y is drawn toward larger SVG y, i.e. lower on screen --
  // a mirrored-but-internally-consistent convention, harmless for a diagnostic tool with no photo
  // to match against.)
  svg.setAttribute("viewBox", `${xmin} ${ymin} ${w} ${h}`);

  const boardId = currentBoardIdAt(state.t);
  const board = boardById(boardId);
  const green = greenAt(boardId);
  const popped = poppedKeySet(state.t);

  // best q per edge (max over that edge's depths in this board's pool), with the winning depth
  const bestByEdge = new Map();
  for (const c of board.pool) {
    const cur = bestByEdge.get(c.edge);
    if (!cur || c.q > cur.q) bestByEdge.set(c.edge, { q: c.q, depth: c.depth });
  }
  const qs_ = [...bestByEdge.values()].map((v) => v.q);
  const qmin = qs_.length ? Math.min(...qs_) : 0;
  const qmax = qs_.length ? Math.max(...qs_) : 1;

  // v3: everything that MOVES comes from the board's own geometry; sizes and walls never move, so they
  // stay on the episode-level `scene`. A v2 trace has no board.geom -- fall back to the start state,
  // which is exactly what this page drew before.
  const geom = board.geom || null;
  const poseOf = (m) => (geom && geom.movable[m.name]) || [m.x, m.y, m.theta];
  const robotPose = (geom && geom.robot) || scene.robot;
  const contacts = (geom && geom.contacts) || scene.contacts;

  const parts = [];
  const stroke = 0.0025;

  parts.push(regionLayer(board.regions));   // first = beneath everything else

  for (const s of scene.static) {
    const theta = 2 * Math.atan2(s.qz, s.qw);
    const deg = (theta * 180) / Math.PI;
    parts.push(
      `<rect x="${-s.hw}" y="${-s.hd}" width="${2 * s.hw}" height="${2 * s.hd}" class="wall-rect" ` +
        `transform="translate(${s.x},${s.y}) rotate(${deg})"/>`
    );
  }

  const [gx, gy] = scene.goal;
  const gr = 0.02;
  parts.push(
    `<g class="goal-marker" transform="translate(${gx},${gy})">` +
      `<line x1="${-gr}" y1="${-gr}" x2="${gr}" y2="${gr}" stroke-width="${stroke * 2}"/>` +
      `<line x1="${-gr}" y1="${gr}" x2="${gr}" y2="${-gr}" stroke-width="${stroke * 2}"/>` +
      `<circle r="${gr * 1.3}" class="goal-ring"/></g>`
  );

  for (const m of scene.movable) {
    const [mx, my, mtheta] = poseOf(m);
    const deg = (mtheta * 180) / Math.PI;
    const isTarget = m.name === state.trace.meta.object_id;
    parts.push(
      `<rect x="${-m.hw}" y="${-m.hd}" width="${2 * m.hw}" height="${2 * m.hd}" ` +
        `class="${isTarget ? "movable-target" : "movable-other"}" ` +
        `transform="translate(${mx},${my}) rotate(${deg})"><title>${m.name}</title></rect>`
    );
  }

  const [rx, ry, rtheta] = robotPose;
  const rr = 0.025;
  parts.push(
    `<g class="robot-marker" transform="translate(${rx},${ry}) rotate(${(rtheta * 180) / Math.PI})">` +
      `<circle r="${rr}"/><line x1="0" y1="0" x2="${rr * 1.6}" y2="0" stroke-width="${stroke * 2}"/></g>`
  );

  const cr = 0.008;
  contacts.forEach((pt, edge) => {
    const [cx, cy] = pt;
    const best = bestByEdge.get(edge);
    if (!best) {
      parts.push(`<circle cx="${cx}" cy="${cy}" r="${cr * 0.6}" class="contact-unreachable" data-edge="${edge}"/>`);
      return;
    }
    const truth = truthOf(green, edge, best.depth);
    const key = `${board.board_id}:${edge}:${best.depth}`;
    const isPopped = popped.has(key);
    const popEntry = isPopped ? state.trace.pops.slice(0, state.t).find(
      (p) => p.board_id === board.board_id && p.edge === edge && p.depth === best.depth
    ) : null;
    const fill = qColor(best.q, qmin, qmax);
    const classes = ["contact-pt", `truth-${truth}`];
    if (isPopped) classes.push("is-popped");
    parts.push(
      `<g class="${classes.join(" ")}" data-edge="${edge}" data-depth="${best.depth}">` +
        `<circle cx="${cx}" cy="${cy}" r="${cr}" fill="${fill}"/>` +
        (popEntry
          ? `<text x="${cx}" y="${cy}" class="pop-order" font-size="0.013" text-anchor="middle" dominant-baseline="central">${popEntry.t}</text>`
          : "") +
        `<title>edge ${edge} depth ${best.depth}: q=${best.q.toFixed(3)}${isPopped ? ` (sim #${popEntry.t}, ${popEntry.opened ? "opened" : "failed"})` : ""}</title>` +
        `</g>`
    );
  });

  svg.innerHTML = parts.join("");
  wireHover(svg, "g.contact-pt");

  const note = document.getElementById("scene-note");
  note.innerHTML = geom
    ? `Drawn at <strong>board ${boardId}</strong>'s own state (${boardTag(board)}) &mdash; objects, robot ` +
      `and contact points move as you scrub. Tint = free space split into regions: ` +
      `<span class="legend-swatch region-robot"></span>&nbsp;robot's region, ` +
      `<span class="legend-swatch region-goal"></span>&nbsp;goal's region, ` +
      `<span class="legend-swatch region-other"></span>&nbsp;elsewhere. A push succeeds exactly when ` +
      `the first two merge.`
    : `This trace has no per-board geometry (pre-v3), so the scene is drawn once from the state the ` +
      `search started from; only the contact colors/pop markers change as you scrub.`;
}

// ---- Zone B: the priority queue ---------------------------------------------------------------

function renderQueueB() {
  const rows = queueAt(state.t);
  const rank = bestGreenRank(rows, greenAt);
  const marker = document.getElementById("best-green-marker");
  marker.textContent = state.gt ? (rank === null ? "no green left in the queue" : `#${rank}`) : "no ground truth";
  marker.classList.toggle("no-gt", !state.gt);

  const bps = rows.map((r) => r.bp);
  const bpMax = bps.length ? Math.max(...bps, 0.001) : 0.001;

  const list = document.getElementById("queue-list");
  const frag = [];
  rows.forEach((r, i) => {
    const board = boardById(r.board_id);
    const green = greenAt(r.board_id);
    const truth = truthOf(green, r.edge, r.depth);
    const hue = boardHue(r.board_id);
    const bpW = fmtPct(r.bp / bpMax);
    const seW = fmtPct(r.se / bpMax);
    const badge = state.gt
      ? `<span class="badge badge-${truth}"${truth === "setup-late" ? ' title="ground truth calls this a setup, but on a finish board a setup only sets up a third push -- not green here"' : ""}>${truth === "setup-late" ? "setup*" : truth}</span>`
      : `<span class="badge badge-unknown">?</span>`;
    frag.push(
      `<div class="queue-row" data-edge="${r.edge}" data-depth="${r.depth}" ` +
        `style="border-left-color:hsl(${hue},60%,45%)">` +
        `<span class="q-rank">${i + 1}</span>` +
        `<span class="q-board" title="board ${r.board_id}">${boardTag(board)}</span>` +
        `<span class="q-cand mono">e${r.edge}/d${r.depth}</span>` +
        `<span class="q-bar"><span class="bar-bp" style="width:${bpW}"></span><span class="bar-se" style="width:${seW}"></span></span>` +
        `<span class="q-w mono">×${r.w.toFixed(2)}</span>` +
        badge +
        `</div>`
    );
  });
  list.innerHTML = frag.join("") || `<div class="queue-empty">queue is empty after ${state.t} sims</div>`;
  wireHover(list, ".queue-row");
}

// ---- Zone C: the timeline ---------------------------------------------------------------------

function renderTimelineC() {
  const sims = state.trace.result.sims;
  const slider = document.getElementById("timeline-slider");
  slider.max = String(sims);
  slider.value = String(state.t);

  // Ticks live inside the slider's own track box, and are placed on the thumb's travel (which is inset by
  // half a thumb at each end) -- so a tick sits exactly under the thumb position it annotates.
  const ticks = document.getElementById("timeline-ticks");
  const frag = state.trace.pops.map((p) => {
    const f = sims > 0 ? p.t / sims : 0;
    const left = `calc(var(--thumb) / 2 + (100% - var(--thumb)) * ${f})`;
    return `<div class="tick ${p.opened ? "tick-pass" : "tick-fail"}" style="left:${left}" data-t="${p.t}" title="t=${p.t} ${p.opened ? "opened" : "failed"}"></div>`;
  });
  ticks.innerHTML = frag.join("");
  ticks.querySelectorAll(".tick").forEach((el) => {
    // Each tick's own data-t is a pop's `t` field, i.e. clicking it means "show the state right
    // after THIS pop ran" -- exactly the new t semantics, so no translation needed here.
    el.classList.toggle("tick-current", Number(el.dataset.t) === state.t);
    el.addEventListener("click", () => {
      state.t = Number(el.dataset.t);
      renderAll();
    });
  });

  // "after t sims": say so explicitly, and name the most-recently-executed push (board + edge/depth
  // + outcome) so the scrubber's meaning and the scene/queue's current board can't be conflated.
  const label = document.getElementById("timeline-label");
  if (state.t === 0) {
    label.textContent = `after 0 of ${sims} sims -- nothing simulated yet`;
  } else {
    const last = state.trace.pops[state.t - 1];
    label.textContent = `after ${state.t} of ${sims} sims -- most recent: board ${last.board_id} ` +
      `e${last.edge}/d${last.depth} (${last.opened ? "opened" : "failed"})`;
  }
}

// ---- Zone D: rank space ---------------------------------------------------------------------

function renderGridD() {
  const boardId = currentBoardIdAt(state.t);
  const board = boardById(boardId);
  const left = document.getElementById("grid-left");
  const right = document.getElementById("grid-right");
  const rangeLabel = document.getElementById("grid-range");
  const rightWrap = document.getElementById("grid-right-wrap");
  const panes = document.getElementById("grid-panes");
  const noData = document.getElementById("grid-no-data");

  document.getElementById("grid-board-label").textContent = `board ${boardId} (${boardTag(board)})`;

  if (!board.grid) {
    panes.style.display = "none";
    rangeLabel.textContent = "";
    noData.style.display = "";
    return;
  }
  panes.style.display = "";
  noData.style.display = "none";

  const flat = [];
  for (let e = 0; e < 60; e++) for (let d = 0; d < 5; d++) flat.push({ e, d, v: board.grid[e][d] });
  const vmin = Math.min(...flat.map((c) => c.v));
  const vmax = Math.max(...flat.map((c) => c.v));
  rangeLabel.textContent = `raw range: [${vmin.toFixed(4)}, ${vmax.toFixed(4)}] (own scale -- not comparable across models)`;

  const ranked = [...flat].sort((a, b) => b.v - a.v);
  const rankOf = new Map();
  ranked.forEach((c, i) => rankOf.set(`${c.e}:${c.d}`, i + 1));
  const nCells = flat.length;

  const green = greenAt(boardId);

  const leftCells = [];
  const rightCells = [];
  for (let e = 0; e < 60; e++) {
    for (let d = 0; d < 5; d++) {
      const key = `${e}:${d}`;
      const rank = rankOf.get(key);
      const f = 1 - (rank - 1) / (nCells - 1); // rank 1 -> f=1 -> darkest
      leftCells.push(
        `<div class="grid-cell" data-edge="${e}" data-depth="${d}" ` +
          `style="background:${qColor(f, 0, 1)}" title="edge ${e} depth ${d}: rank ${rank}/${nCells}"></div>`
      );
      if (state.gt) {
        const truth = truthOf(green, e, d);
        rightCells.push(
          `<div class="grid-cell truth-${truth}" data-edge="${e}" data-depth="${d}" ` +
            `title="edge ${e} depth ${d}: ${truth}"></div>`
        );
      }
    }
  }
  left.innerHTML = leftCells.join("");
  if (state.gt) {
    rightWrap.style.display = "";
    right.innerHTML = rightCells.join("");
    wireHover(right, ".grid-cell");
  } else {
    rightWrap.style.display = "none";
  }
  wireHover(left, ".grid-cell");
}

// ---- Cross-highlighting (Step 6) ---------------------------------------------------------------
//
// A single `state.hover = {edge, depth} | null` field drives highlighting in all three zones (A's
// contact circles, B's queue rows, D's grid cells). Deliberately NOT a full renderAll() per hover:
// rebuilding a zone's innerHTML while the pointer sits over the element being replaced can misfire
// mouseleave on the destroyed node (or occasionally a phantom mouseenter on its replacement),
// producing a highlight/unhighlight flicker loop. So hover only toggles an `is-hovered` class over
// the DOM that already exists; full rebuilds stay reserved for scrubbing (renderAll below also
// calls this once after every rebuild, so a render triggered while a hover is already active stays
// consistent).

function wireHover(root, selector) {
  root.querySelectorAll(selector).forEach((el) => {
    const edge = Number(el.dataset.edge);
    const depth = el.dataset.depth === undefined || el.dataset.depth === "" ? null : Number(el.dataset.depth);
    el.addEventListener("mouseenter", () => {
      state.hover = { edge, depth };
      applyHover();
    });
    el.addEventListener("mouseleave", () => {
      state.hover = null;
      applyHover();
    });
  });
}

function applyHover() {
  const h = state.hover;
  document.querySelectorAll("[data-edge]").forEach((el) => {
    const edge = Number(el.dataset.edge);
    const depth = el.dataset.depth === undefined || el.dataset.depth === "" ? null : Number(el.dataset.depth);
    const match = h && h.edge === edge && (h.depth === null || depth === null || h.depth === depth);
    el.classList.toggle("is-hovered", !!match);
  });
}

// ---- Wiring ---------------------------------------------------------------------------------

function renderAll() {
  renderSceneA();
  renderQueueB();
  renderTimelineC();
  renderGridD();
  applyHover();
}

function setT(t) {
  const sims = state.trace.result.sims;
  state.t = Math.max(0, Math.min(sims, t));
  renderAll();
}

async function init() {
  const arm = qs("arm");
  const key = qs("key");
  const header = document.getElementById("episode-header");
  if (!arm || !key) {
    header.textContent = "Missing ?arm=<model>|<strategy>&key=<episode key> in the URL.";
    return;
  }
  const [model, strategy] = arm.split("|");

  let manifest, row, trace, gt;
  try {
    manifest = await (await fetch("manifest.json")).json();
    row = manifest.index[arm].find((r) => r.key === key);
    trace = await (await fetch(`trace/${model}/${strategy}/${key}.json`)).json();
    gt = row.has_gt ? await (await fetch(`gt/${key}.json`)).json() : null;
  } catch (err) {
    header.textContent = `Failed to load episode data: ${err}`;
    return;
  }
  state.manifestRow = row;
  state.trace = trace;
  state.gt = gt;

  document.getElementById("no-gt-banner").style.display = state.gt ? "none" : "";

  // The displayed order is a reconstruction; say so out loud when it fails to reproduce the recorded search.
  const banner = document.getElementById("recon-banner");
  const nBad = verifyReconstruction();
  const missing = trace.schema_version !== SUPPORTED_SCHEMA_VERSION;
  banner.style.display = nBad || missing ? "" : "none";
  banner.textContent = missing
    ? `This trace's schema_version (${trace.schema_version}) is not ${SUPPORTED_SCHEMA_VERSION}, the version` +
      ` this page understands, so the queue below is ordered with the generator's DEFAULTS` +
      ` (${SEARCH_DEFAULTS.combine} priority, discount ${SEARCH_DEFAULTS.discount}) and may not be the order` +
      ` the search used. Regenerate the trace.`
    : `The displayed order could NOT be verified: ${nBad} of ${trace.pops.length} recorded pops disagree with` +
      ` the bp/w this page recomputed from meta.search. Treat the ranking below as unreliable.`;

  const sceneName = trace.meta.xml.split("/").pop();
  header.innerHTML =
    `<a href="index.html">&larr; index</a>` +
    `<span class="mono">${sceneName}</span> &middot; <span class="mono">${trace.meta.object_id}</span>` +
    ` &middot; tier ${row.tier}` +
    ` &middot; ${trace.meta.model}/${trace.meta.strategy}` +
    ` &middot; ${trace.result.solved ? "solved" : trace.result.end} in ${trace.result.sims} sims`;

  document.getElementById("timeline-slider").addEventListener("input", (ev) => setT(Number(ev.target.value)));
  document.getElementById("step-back").addEventListener("click", () => setT(state.t - 1));
  document.getElementById("step-fwd").addEventListener("click", () => setT(state.t + 1));
  document.addEventListener("keydown", (ev) => {
    // the range input steps itself natively once focused -- stepping again here would double every press
    if (ev.target === document.getElementById("timeline-slider")) return;
    if (ev.key === "ArrowLeft") setT(state.t - 1);
    if (ev.key === "ArrowRight") setT(state.t + 1);
  });

  setT(0);
}

init();
