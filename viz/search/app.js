"use strict";
/* Episode replay view: one search, replayed pop by pop.
 *
 * Data shapes (see docs/superpowers/specs/2026-07-26-search-viz-design.md):
 *   trace = {meta, scene, boards:[{board_id,depth,parent_edge,parent_depth,pool:[{obj,edge,depth,q}],grid|null,w0}],
 *            pops:[{t,board_id,obj,edge,depth,q,bp,w,se,opened}], result:{solved,sims,plan_len,end}}
 *   gt    = null | {root:{openers:[[e,d]],setups:[[e,d]]}, finish:{"<parent_edge>_<parent_depth>":{openers,setups}}}
 *
 * REAL-DATA NOTE (found by reading a trace, not assumed): board.pool entries only ever carry
 * {obj, edge, depth, q} -- `bp` is recorded ONLY on already-popped candidates (scripts/sandbox/eval_bestfirst.py
 * push()/make_pop()). So there is no `bp` sitting on disk for a still-queued candidate to sort by. Reading
 * eval_bestfirst.py's priority(q, V, combine) (default combine="blend" => 0.5*q + 0.5*V) and its board-level V
 * (mean of the board's own top-5 pool q, agg="mean5" default) reproduces every recorded `bp` in all 5 sample
 * traces EXACTLY (max abs error 0.0 across 60+ pops checked). So `bp` for a queued candidate is derived here as
 * 0.5*q + 0.5*V(board) -- not a guess, a verified closed-form reconstruction of the same formula the search used.
 * The board WEIGHT `w`, in contrast, really is only ever recorded at pop time (a floating per-board multiplier
 * demoted on failure, root frozen at 1) -- so per the brief, `w` is replayed off the pops, never recomputed.
 */

const state = { trace: null, gt: null, t: 0, hover: null, manifestRow: null };
const boardVCache = new Map(); // board_id -> V (mean top-5 pool q)

function qs(name) {
  return new URLSearchParams(window.location.search).get(name);
}

function boardById(id) {
  return state.trace.boards.find((b) => b.board_id === id);
}

function boardV(board) {
  if (boardVCache.has(board.board_id)) return boardVCache.get(board.board_id);
  const qsSorted = board.pool.map((c) => c.q).sort((a, b) => b - a);
  const top = qsSorted.slice(0, Math.min(5, qsSorted.length));
  const v = top.length ? top.reduce((a, b) => a + b, 0) / top.length : 0;
  boardVCache.set(board.board_id, v);
  return v;
}

function bpOf(board, q) {
  return 0.5 * q + 0.5 * boardV(board);
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

function boardWAt(t) {
  const w = new Map();
  for (const b of state.trace.boards) w.set(b.board_id, b.w0);
  for (let i = 0; i < t; i++) {
    const p = state.trace.pops[i];
    w.set(p.board_id, p.w);
  }
  return w;
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
    for (const c of b.pool) {
      const key = `${b.board_id}:${c.edge}:${c.depth}`;
      if (popped.has(key)) continue;
      const bp = bpOf(b, c.q);
      rows.push({ board_id: b.board_id, obj: c.obj, edge: c.edge, depth: c.depth, q: c.q, bp, w, se: bp * w });
    }
  }
  rows.sort((a, b) => b.se - a.se || a.board_id - b.board_id || a.edge - b.edge || a.depth - b.depth);
  return rows;
}

// greenAt(boardId): opener/setup (edge,depth) sets for that board. Empty when gt is null.
function greenAt(boardId) {
  const empty = { openers: new Set(), setups: new Set() };
  if (!state.gt) return empty;
  const b = boardById(boardId);
  const src = b.depth === 0 ? state.gt.root : state.gt.finish[`${b.parent_edge}_${b.parent_depth}`];
  if (!src) return empty;
  return {
    openers: new Set(src.openers.map(([e, d]) => `${e}:${d}`)),
    setups: new Set(src.setups.map(([e, d]) => `${e}:${d}`)),
  };
}

function truthOf(green, edge, depth) {
  const key = `${edge}:${depth}`;
  if (green.openers.has(key)) return "opener";
  if (green.setups.has(key)) return "setup";
  return state.gt ? "dead" : "unknown";
}

function currentBoardIdAt(t) {
  if (t === 0) return 0;
  return state.trace.pops[t - 1].board_id;
}

function bestGreenRank(rows, greenCache) {
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    const green = greenCache(r.board_id);
    if (green.openers.has(`${r.edge}:${r.depth}`) || green.setups.has(`${r.edge}:${r.depth}`)) return i + 1;
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

  const parts = [];
  const stroke = 0.0025;

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
    const deg = (m.theta * 180) / Math.PI;
    const isTarget = m.name === state.trace.meta.object_id;
    parts.push(
      `<rect x="${-m.hw}" y="${-m.hd}" width="${2 * m.hw}" height="${2 * m.hd}" ` +
        `class="${isTarget ? "movable-target" : "movable-other"}" ` +
        `transform="translate(${m.x},${m.y}) rotate(${deg})"><title>${m.name}</title></rect>`
    );
  }

  const [rx, ry, rtheta] = scene.robot;
  const rr = 0.025;
  parts.push(
    `<g class="robot-marker" transform="translate(${rx},${ry}) rotate(${(rtheta * 180) / Math.PI})">` +
      `<circle r="${rr}"/><line x1="0" y1="0" x2="${rr * 1.6}" y2="0" stroke-width="${stroke * 2}"/></g>`
  );

  const cr = 0.008;
  scene.contacts.forEach((pt, edge) => {
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
      ? `<span class="badge badge-${truth}">${truth}</span>`
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
  list.innerHTML = frag.join("") || `<div class="queue-empty">queue is empty at t=${state.t}</div>`;
  wireHover(list, ".queue-row");
}

// ---- Zone C: the timeline ---------------------------------------------------------------------

function renderTimelineC() {
  const sims = state.trace.result.sims;
  const slider = document.getElementById("timeline-slider");
  slider.max = String(sims);
  slider.value = String(state.t);

  const ticks = document.getElementById("timeline-ticks");
  const frag = state.trace.pops.map((p) => {
    const left = sims > 0 ? fmtPct((p.t - 0.5) / sims) : "0%";
    return `<div class="tick ${p.opened ? "tick-pass" : "tick-fail"}" style="left:${left}" data-t="${p.t}" title="t=${p.t} ${p.opened ? "opened" : "failed"}"></div>`;
  });
  ticks.innerHTML = frag.join("");
  ticks.querySelectorAll(".tick").forEach((el) => {
    el.addEventListener("click", () => {
      state.t = Number(el.dataset.t);
      renderAll();
    });
  });

  document.getElementById("timeline-label").textContent =
    `t = ${state.t} / ${sims} sims` + (state.t > 0 ? ` -- last: board ${state.trace.pops[state.t - 1].board_id}, ` +
      `${state.trace.pops[state.t - 1].opened ? "opened" : "failed"}` : " -- start");
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

  let manifest;
  try {
    manifest = await (await fetch("manifest.json")).json();
  } catch (err) {
    header.textContent = "Failed to load manifest.json: " + err;
    return;
  }
  const rows = manifest.index[arm] || [];
  const row = rows.find((r) => r.key === key);
  state.manifestRow = row || null;

  let trace;
  try {
    const resp = await fetch(`trace/${model}/${strategy}/${key}.json`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    trace = await resp.json();
  } catch (err) {
    header.textContent = `Failed to load trace for ${key}: ${err}`;
    return;
  }
  state.trace = trace;

  const hasGt = row ? row.has_gt : true; // manifest is the authority; fall back to trying if unknown
  if (hasGt) {
    try {
      const resp = await fetch(`gt/${key}.json`);
      state.gt = resp.ok ? await resp.json() : null;
    } catch (err) {
      state.gt = null;
    }
  } else {
    state.gt = null;
  }

  document.getElementById("no-gt-banner").style.display = state.gt ? "none" : "";

  const sceneName = trace.meta.xml.split("/").pop();
  header.innerHTML =
    `<a href="index.html">&larr; index</a>` +
    `<span class="mono">${sceneName}</span> &middot; <span class="mono">${trace.meta.object_id}</span>` +
    ` &middot; tier ${row ? row.tier : "?"}` +
    ` &middot; ${trace.meta.model}/${trace.meta.strategy}` +
    ` &middot; ${trace.result.solved ? "solved" : trace.result.end} in ${trace.result.sims} sims`;

  document.getElementById("timeline-slider").addEventListener("input", (ev) => setT(Number(ev.target.value)));
  document.getElementById("step-back").addEventListener("click", () => setT(state.t - 1));
  document.getElementById("step-fwd").addEventListener("click", () => setT(state.t + 1));
  document.addEventListener("keydown", (ev) => {
    if (ev.key === "ArrowLeft") setT(state.t - 1);
    if (ev.key === "ArrowRight") setT(state.t + 1);
  });

  setT(0);
}

init();
