"use strict";
// Scene gallery: arrow through one filtered category (horizon x tier) one episode at a time and
// star the ones worth turning into a figure. Deliberately a SCANNING tool, not a report -- the
// numbers shown are only what the canonical labels already assert about the root state.
// Data: scenes.json (index) + cards/<file>.json (one self-contained episode), built by
// scripts/viz/build_scene_cards.py.

const NOCACHE = { cache: "no-store" };
// The page code lives once under viz/app/; every dataset (viz/scenes/, viz/real_scenes/, ...) is a
// sibling folder of data only. ?data=<path relative to viz/app/> says which one this load is for --
// e.g. "../scenes/" (see viz/app/datasets.json, the manifest a new dataset needs one line in).
const DATA_BASE = new URLSearchParams(window.location.search).get("data") || "";
// Scoped by dataset: two galleries sharing one page still share one browser origin, so an
// unscoped key would mix one dataset's saved filters and starred shortlist into the other's.
const STORE_KEY = "namo-viz-gallery-state:" + DATA_BASE;
const STAR_KEY = "namo-viz-gallery-stars:" + DATA_BASE;
// The shortlist itself lives in a file beside the dataset, written by the POST route in
// scripts/viz/serve.py. localStorage alone is per browser and per port, so a shortlist built over
// an evening is gone on the next laptop, and no script on the box can read it. The browser copy
// stays as a cache for when the page is served by something that will not take the write (plain
// `python -m http.server`), and the page says so on screen when that happens.
const STARS_URL = DATA_BASE + "stars.json";
let starsWritable = true;
// Human name of the dataset, exported with every shortlist entry so a mixed or
// mis-pasted shortlist is detectable downstream (the hardware side refuses
// car-envs entries by this field rather than by guessing from id shapes).
const DATASET_NAME = ({
  "../real_scenes/": "shipped-600",
  "../real_scenes_all/": "full-exhaustive-pool",
  "../scenes/": "car-envs",
})[DATA_BASE] || DATA_BASE;
const TIER_ORDER = { hard: 0, medium: 1, easy: 2 };
// Cards carry their own copy of the tier, so the index normalisation below does not reach them.
const tierName = (t) => (t === "med" ? "medium" : t);

const horizonSelect = document.getElementById("horizon-select");
const tierSelect = document.getElementById("tier-select");
const sortSelect = document.getElementById("sort-select");
const textFilter = document.getElementById("text-filter");
const starredOnly = document.getElementById("starred-only");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const starBtn = document.getElementById("star-btn");
const copyBtn = document.getElementById("copy-btn");
const positionEl = document.getElementById("position");
const summary = document.getElementById("summary");
const starCount = document.getElementById("star-count");
const storeNote = document.getElementById("store-note");

const famControl = document.getElementById("fam-control");
const famRow = document.getElementById("fam-row");
let famBoxes = [];      // built in init() from the families this dataset actually contains
const evalSummaryEl = document.getElementById("eval-summary");
const evalPanelEl = document.getElementById("eval-panel");
const evalRunControl = document.getElementById("eval-run-control");
const evalRunSelect = document.getElementById("eval-run-select");
const evalFilterControl = document.getElementById("eval-filter-control");
const evalOutcomeRow = document.getElementById("eval-outcome-row");
const labelChainControl = document.getElementById("label-chain-control");
const labelChainSelect = document.getElementById("label-chain-select");
const modelChainControl = document.getElementById("model-chain-control");
const modelChainSelect = document.getElementById("model-chain-select");
const FILTERS = [horizonSelect, tierSelect, sortSelect, textFilter, starredOnly,
  labelChainSelect, modelChainSelect];
const EVAL_K = [1, 5, 30];   // solve@k shown in the summary strip
// One room can be "model solved" (every model arm solved), "model unsolved" (no model arm solved),
// or "model split" (some seeds did, some did not) -- never more than one of the three at once.
const MODEL_BUCKETS = [
  { key: "all", label: "all" },
  { key: "solved", label: "model solved" },
  { key: "split", label: "model split" },
  { key: "unsolved", label: "model unsolved" },
  { key: "solved-random-unsolved", label: "model solved, random unsolved" },
  { key: "neither", label: "neither solved" },
];
let modelBucket = "all";   // selected #eval-outcome-row button
const LABEL_CHAIN_OPTIONS = [
  { key: "all", label: "all" },
  { key: "same", label: "same object" },
  { key: "cross", label: "cross object" },
  { key: "single", label: "1push (single)" },
];
const MODEL_CHAIN_OPTIONS = [
  { key: "all", label: "all" },
  { key: "same", label: "same object" },
  { key: "cross", label: "cross object" },
  { key: "mixed", label: "seeds disagree" },
  { key: "unsolved", label: "no model arm solved" },
];

let index = null;
let timing = null;       // per-problem seconds from the timed campaign; absent = the page hides them
// eval.json: HY5U vs random on this dataset's rooms, built by scripts/viz/add_eval_overlay.py.
// Keyed by room (scenes.json's "scene" string), not by card file -- a room can carry several cards
// (one per object/horizon) and the eval ran once per room, so every card of a room shows the same
// numbers. Absent = the summary strip, the eval sort orders, and the eval filter all no-op.
let evalData = null;
let evalRun = null;     // which run drives the strip / sort / filter; the per-card panel shows all
let rows = [];          // the current filtered + sorted list
let i = 0;              // cursor into rows
let stars = {};         // file -> the index row, so a shortlist survives a rebuild of the cards
const cardCache = new Map();
const replayCache = new Map();
let replay = null;    // the current episode's solution, or null when none was built
let step = 0;         // 0 = start state, 1..n = after that push

function saveState() {
  localStorage.setItem(STORE_KEY, JSON.stringify({
    horizon: horizonSelect.value, tier: tierSelect.value, sort: sortSelect.value,
    text: textFilter.value, starredOnly: starredOnly.checked,
    families: famBoxes.filter((b) => b.checked).map((b) => b.dataset.family),
    file: rows[i] ? rows[i].file : null,
  }));
}

function saveStars() {
  const text = JSON.stringify(stars);
  localStorage.setItem(STAR_KEY, text);   // cache, so a dead server still shows the list
  starCount.textContent = Object.keys(stars).length;
  if (!starsWritable) return;
  fetch(STARS_URL, { method: "POST", body: text })
    .then((r) => { if (!r.ok) markReadOnly(); })
    .catch(markReadOnly);
}

// Say it once, on screen. A star that only reached localStorage looks identical to a saved one, and
// finding out at handoff time that an evening of starring never left the browser is the whole
// failure this file-backed list exists to prevent.
function markReadOnly() {
  if (!starsWritable) return;
  starsWritable = false;
  storeNote.textContent = "browser-only: this server will not write stars.json";
}

// timing.json is optional: a gallery built without build_scene_timing.py still works, it just has
// no seconds to show or sort by.
if (!DATA_BASE) {
  summary.textContent = "Missing ?data=<path to a dataset folder, relative to viz/app/> in the URL.";
} else {
  Promise.all([
    fetch(DATA_BASE + "scenes.json", NOCACHE).then((r) => r.json()),
    fetch(DATA_BASE + "timing.json", NOCACHE).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    fetch(DATA_BASE + "eval.json", NOCACHE).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    fetch(STARS_URL, NOCACHE).then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]).then(([m, t, ev, savedStars]) => {
    index = m;
    // The car_envs pools spell the middle tier "medium", the real-table pools spell it "med", and
    // everything here keys on the string: TIER_ORDER, the dropdown, the per-tier counter. Left
    // alone, a real-table gallery reports "medium 0" beside 183 medium episodes and sorts them
    // to a random spot on an undefined rank. Normalise once, on the way in.
    index.cards.forEach((r) => { if (r.tier === "med") r.tier = "medium"; });
    timing = t && t.cards ? t.cards : null;
    evalData = ev && ev.rooms ? ev : null;
    init(savedStars);
  }).catch((err) => { summary.textContent = "Failed to load scenes.json: " + err; });
}

// One shared page serves several datasets, so the room batches cannot be baked into the markup:
// the car_envs pools are feb_car/aug9_car, the real-table set is real_table. Read them off the
// index instead, and skip the control entirely when there is only one batch to choose.
function buildFamilyBoxes() {
  const fams = [...new Set(index.cards.map((r) => r.family))].sort();
  famRow.textContent = "";
  famBoxes = fams.map((f) => {
    const label = document.createElement("label");
    label.className = "checkbox";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.dataset.family = f;
    box.checked = true;
    box.addEventListener("input", () => applyFilters(null));
    label.append(box, " " + f);
    famRow.append(label);
    return box;
  });
  famControl.hidden = fams.length < 2;
}

// Same shape as buildFamilyBoxes: the run names come from eval.json, never hardcoded, so a later
// eval (a new campaign added with the same add_eval_overlay.py --run flag) needs no page edit.
function buildEvalRunControl() {
  if (!evalData) {
    evalRunControl.hidden = true; evalFilterControl.hidden = true;
    labelChainControl.hidden = true; modelChainControl.hidden = true;
    return;
  }
  evalRunSelect.innerHTML = evalData.runs.map((r) => `<option value="${r}">${r}</option>`).join("");
  evalRun = evalData.runs[0];
  evalRunControl.hidden = evalData.runs.length < 2;
  evalFilterControl.hidden = false;
  evalOutcomeRow.innerHTML = "";
  MODEL_BUCKETS.forEach((b) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "bucket-btn";
    btn.dataset.bucket = b.key;
    btn.textContent = b.label;
    btn.addEventListener("click", () => {
      modelBucket = b.key;
      applyFilters(rows[i] ? rows[i].file : null);
    });
    evalOutcomeRow.append(btn);
  });
  labelChainControl.hidden = !evalData.chain;
  modelChainControl.hidden = !evalData.chain;
  labelChainSelect.innerHTML = LABEL_CHAIN_OPTIONS.map((o) => `<option value="${o.key}">${o.label}</option>`).join("");
  modelChainSelect.innerHTML = MODEL_CHAIN_OPTIONS.map((o) => `<option value="${o.key}">${o.label}</option>`).join("");
}

// A room's model arms in one of three states for the SELECTED run: every arm solved, none did, or
// some did and some did not ("split"). null when the run has no model arms recorded for this room.
function roomModelBucket(rec) {
  if (!rec || !rec.model.length) return null;
  const solved = rec.model.filter((e) => e.solved).length;
  if (solved === rec.model.length) return "solved";
  if (solved === 0) return "unsolved";
  return "split";
}
function roomRandomAnySolved(rec) { return !!rec && rec.random.some((e) => e.solved); }
function roomRandomAllUnsolved(rec) { return !!rec && rec.random.length > 0 && rec.random.every((e) => !e.solved); }

function matchesModelBucket(rec, key) {
  if (key === "all") return true;
  const mb = roomModelBucket(rec);
  if (key === "solved") return mb === "solved";
  if (key === "split") return mb === "split";
  if (key === "unsolved") return mb === "unsolved";
  if (key === "solved-random-unsolved") return mb === "solved" && roomRandomAllUnsolved(rec);
  if (key === "neither") return mb === "unsolved" && roomRandomAllUnsolved(rec);
  return true;
}

// eval.json's per-card label chain (scripts/viz/add_eval_overlay.py:load_card_chains): "same",
// "cross", "single" (1push), or null (2push card with no replay).
function labelChainOf(row) {
  return evalData && evalData.chain ? (evalData.chain[row.file] || null) : null;
}

// The selected run's own model-arm chain for this ROOM: "same"/"cross" when every solved model arm
// agrees, "mixed" when seeds disagree, "unsolved" when no model arm solved (also covers 1push rooms,
// whose solution is always one push, so never carries a chain).
function modelChainBucket(rec) {
  if (!rec) return "unsolved";
  const chains = rec.model.filter((e) => e.solved && e.chain).map((e) => e.chain);
  if (!chains.length) return "unsolved";
  return chains.every((c) => c === chains[0]) ? chains[0] : "mixed";
}

function init(fileStars) {
  buildFamilyBoxes();
  buildEvalRunControl();
  // The file wins whenever it exists: it is the copy every other browser and every handoff script
  // reads. No file yet means either a fresh dataset or the first load since stars moved to disk, so
  // adopt whatever this browser is holding -- the saveStars() below writes it up.
  stars = fileStars || JSON.parse(localStorage.getItem(STAR_KEY) || "{}");
  // Stars persist the index ROW, so a star made before a field existed exports
  // without it forever. Refresh each saved row from the live index (matched by
  // file); keep the stale row only if the card vanished from the index.
  const byFile = Object.fromEntries(index.cards.map((r) => [r.file, r]));
  for (const f of Object.keys(stars)) if (byFile[f]) stars[f] = byFile[f];
  saveStars();
  starCount.textContent = Object.keys(stars).length;

  const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
  let wantFile = null;
  if (saved) {
    horizonSelect.value = saved.horizon || "1push";
    tierSelect.value = saved.tier || "all";
    sortSelect.value = saved.sort || "density-asc";
    if (!sortSelect.value) sortSelect.value = "density-asc";   // saved an option this build dropped
    textFilter.value = saved.text || "";
    starredOnly.checked = !!saved.starredOnly;
    // Never restore "no batch selected" -- that reads as an empty gallery, not as a filter.
    // Also skip the restore when none of the saved batches exist in this dataset any more --
    // unchecking every box reads as an empty gallery, not as a filter.
    if (saved.families && famBoxes.some((b) => saved.families.includes(b.dataset.family))) {
      famBoxes.forEach((b) => { b.checked = saved.families.includes(b.dataset.family); });
    }
    wantFile = saved.file;
  }

  FILTERS.forEach((el) => el.addEventListener("input", () => applyFilters(null)));
  evalRunSelect.addEventListener("input", () => {
    evalRun = evalRunSelect.value;
    renderEvalSummary();
    applyFilters(rows[i] ? rows[i].file : null);   // eval sort/filter may depend on the new run
  });
  prevBtn.addEventListener("click", () => stepEpisode(-1));
  nextBtn.addEventListener("click", () => stepEpisode(1));
  starBtn.addEventListener("click", toggleStar);
  copyBtn.addEventListener("click", copyShortlist);
  document.addEventListener("keydown", (ev) => {
    if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT") return;
    if (ev.key === "ArrowLeft") { stepEpisode(-1); ev.preventDefault(); }
    else if (ev.key === "ArrowRight") { stepEpisode(1); ev.preventDefault(); }
    else if (ev.key === "ArrowUp" || ev.key === "ArrowDown") {
      const n = replay ? replay.steps.length : 0;
      step = Math.min(n, Math.max(0, step + (ev.key === "ArrowDown" ? 1 : -1)));
      fetchCard(rows[i]).then((c) => { c.file_key = rows[i].file; render(c); });
      ev.preventDefault();
    }
    else if (ev.key === "s" || ev.key === "S") { toggleStar(); ev.preventDefault(); }
  });

  renderEvalSummary();
  applyFilters(wantFile);
}

// eval.json is keyed by room (scene), not by card file -- every card of a room reads the same entry.
function evalRoom(r) {
  return evalData && evalRun && evalData.rooms[r.scene] ? evalData.rooms[r.scene][evalRun] : null;
}

function applyFilters(wantFile) {
  const q = textFilter.value.trim().toLowerCase();
  const fams = new Set(famBoxes.filter((b) => b.checked).map((b) => b.dataset.family));
  // Everything except the eval-outcome bucket and the two chain filters: this is the denominator the
  // bucket/chain option counts are drawn from, so the numbers next to them agree with what a click
  // on "all" would show.
  const preRows = index.cards.filter((r) => {
    if (r.horizon !== horizonSelect.value) return false;
    if (tierSelect.value !== "all" && r.tier !== tierSelect.value) return false;
    if (starredOnly.checked && !stars[r.file]) return false;
    if (!fams.has(r.family)) return false;
    if (q && !(r.scene.toLowerCase().includes(q) || r.object_id.toLowerCase().includes(q))) return false;
    return true;
  });

  if (evalData) {
    evalOutcomeRow.querySelectorAll(".bucket-btn").forEach((btn) => {
      const key = btn.dataset.bucket;
      const n = preRows.filter((r) => matchesModelBucket(evalRoom(r), key)).length;
      btn.textContent = `${MODEL_BUCKETS.find((b) => b.key === key).label} (${n})`;
      btn.classList.toggle("bucket-active", key === modelBucket);
    });
    [[labelChainSelect, LABEL_CHAIN_OPTIONS, labelChainOf],
     [modelChainSelect, MODEL_CHAIN_OPTIONS, (r) => modelChainBucket(evalRoom(r))]].forEach(([sel, opts, of]) => {
      const cur = sel.value;
      Array.from(sel.options).forEach((opt) => {
        const key = opt.value;
        const n = key === "all" ? preRows.length : preRows.filter((r) => of(r) === key).length;
        opt.textContent = `${opts.find((o) => o.key === key).label} (${n})`;
      });
      sel.value = cur || "all";
    });
  }

  rows = preRows.filter((r) => {
    if (evalData && !matchesModelBucket(evalRoom(r), modelBucket)) return false;
    if (labelChainSelect.value !== "all" && labelChainOf(r) !== labelChainSelect.value) return false;
    if (modelChainSelect.value !== "all" && modelChainBucket(evalRoom(r)) !== modelChainSelect.value) return false;
    return true;
  });

  const mode = sortSelect.value;
  // A problem with no timing/eval row sorts to the END in BOTH directions -- "smallest speed-up"
  // must surface the ranker's real losses, not a pile of episodes we simply have no numbers for.
  const up = (r) => (timing && timing[r.file] && timing[r.file].up) || null;
  const evalUp = (r) => { const rec = evalRoom(r); return rec ? rec.speedup : null; };
  rows.sort((a, b) => {
    if (mode === "speedup-desc" || mode === "speedup-asc") {
      const ua = up(a), ub = up(b);
      if (ua === null || ub === null) return (ua === null) - (ub === null);
      return mode === "speedup-desc" ? ub - ua : ua - ub;
    }
    if (mode === "eval-speedup-desc" || mode === "eval-speedup-asc") {
      const ua = evalUp(a), ub = evalUp(b);
      if (ua === null || ub === null) return (ua === null) - (ub === null);
      return mode === "eval-speedup-desc" ? ub - ua : ua - ub;
    }
    if (mode === "scene") return a.scene.localeCompare(b.scene) || a.object_id.localeCompare(b.object_id);
    const t = TIER_ORDER[a.tier] - TIER_ORDER[b.tier];
    if (t) return t;
    return mode === "density-desc" ? b.density_pct - a.density_pct : a.density_pct - b.density_pct;
  });

  i = 0;
  if (wantFile) {
    const at = rows.findIndex((r) => r.file === wantFile);
    if (at >= 0) i = at;
  }

  const counts = TIER_ORDER;
  const byTier = Object.keys(counts)
    .map((t) => `${t} ${rows.filter((r) => r.tier === t).length}`)
    .join(" · ");
  // Median speed-up over exactly the rows on screen, so the filter bar and the number agree.
  let tail = "";
  if (timing) {
    const ups = rows.map((r) => timing[r.file] && timing[r.file].up).filter((v) => v).sort((a, b) => a - b);
    if (ups.length) tail = ` · median speed-up ${ups[ups.length >> 1].toFixed(1)}×`;
  }
  summary.textContent = `${rows.length} episodes in this filter · ${byTier} · ` +
    `${Object.keys(stars).length} starred overall` + tail;

  show();
}

function stepEpisode(d) {
  if (!rows.length) return;
  i = (i + d + rows.length) % rows.length;
  show();
}

// The ground-truth solution for this episode: the answer the test set already knows, simulated and
// snapshotted after each push (scripts/viz/build_scene_replay.py). Missing for a few episodes, and
// the page just shows the start state then.
function fetchReplay(row) {
  if (!row) return Promise.resolve(null);
  if (replayCache.has(row.file)) return Promise.resolve(replayCache.get(row.file));
  return fetch(DATA_BASE + "replay/" + row.file, NOCACHE)
    .then((r) => (r.ok ? r.json() : null))
    .catch(() => null)
    .then((v) => {
      if (replayCache.size > 40) replayCache.delete(replayCache.keys().next().value);
      replayCache.set(row.file, v);
      return v;
    });
}

function fetchCard(row) {
  if (!row) return Promise.resolve(null);
  if (cardCache.has(row.file)) return Promise.resolve(cardCache.get(row.file));
  return fetch(DATA_BASE + "cards/" + row.file, NOCACHE).then((r) => r.json()).then((c) => {
    if (cardCache.size > 60) cardCache.delete(cardCache.keys().next().value);
    cardCache.set(row.file, c);
    return c;
  });
}

function show() {
  positionEl.textContent = `${rows.length ? i + 1 : 0} / ${rows.length}`;
  const row = rows[i];
  updateStarBtn();
  if (!row) {
    document.getElementById("scene-svg").innerHTML = "";
    document.getElementById("card-title").textContent = "no episode matches this filter";
    document.getElementById("meta-table").innerHTML = "";
    document.getElementById("green-list").innerHTML = "";
    document.getElementById("xml-path").textContent = "";
    saveState();
    return;
  }
  step = 0;
  Promise.all([fetchCard(row), fetchReplay(row)]).then(([card, rep]) => {
    // A slow fetch must not paint over a newer selection.
    if (rows[i] !== row) return;
    card.file_key = row.file;
    replay = rep;
    render(card);
    saveState();
    fetchCard(rows[(i + 1) % rows.length]);   // prefetch so arrowing stays instant
    fetchReplay(rows[(i + 1) % rows.length]);
  });
}

// edge -> {green: [depths], tried: [depths]}
function edgeMap(card) {
  const m = new Map();
  const put = (e, d, k) => {
    if (!m.has(e)) m.set(e, { green: [], tried: [] });
    m.get(e)[k].push(d);
  };
  card.tried.forEach(([e, d]) => put(e, d, "tried"));
  card.green.forEach(([e, d]) => put(e, d, "green"));
  return m;
}

function render(card) {
  const svg = document.getElementById("scene-svg");
  const meta = card.meta;
  setSceneViewBox(svg, card.scene, true);
  const at = step > 0 && replay ? replay.steps[step - 1] : null;
  const parts = sceneLayers(card.scene, at ? at.geom : null,
                            at ? at.regions : card.regions, meta.object_id);

  // Truth dots belong to the START state: they say which of the pushes available THERE are right.
  // After a push the board is different, so showing them again would assert something untrue.
  const em = at ? new Map() : edgeMap(card);
  (at ? [] : card.contacts).forEach((pt, edge) => {
    const e = em.get(edge);
    const cls = !e ? "dot-untried" : (e.green.length ? "dot-green" : "dot-tried");
    const r = e && e.green.length ? 0.0055 : 0.0035;
    const depths = e ? `tried depths ${e.tried.sort((a, b) => a - b).join(",")}` +
      (e.green.length ? ` · WORKS at ${e.green.sort((a, b) => a - b).join(",")}` : "")
      : "not reachable from the robot's region";
    parts.push(
      `<circle class="contact-pt ${cls}" cx="${pt[0]}" cy="${pt[1]}" r="${r}">` +
        `<title>edge ${edge}: ${depths}</title></circle>`
    );
  });
  svg.innerHTML = parts.join("");

  document.getElementById("scene-legend").innerHTML =
    `Tint = free space split into regions: <span class="legend-swatch region-robot"></span>&nbsp;robot's region, ` +
    `<span class="legend-swatch region-goal"></span>&nbsp;goal's region, ` +
    `<span class="legend-swatch region-other"></span>&nbsp;elsewhere. The highlighted box is the ` +
    `blocking object this episode is about; a push on it has to merge the first two.`;

  document.getElementById("card-title").textContent =
    `${meta.horizon} · ${tierName(meta.tier)} · ${galleryId()} · ${meta.object_id}`;

  const green = meta.horizon === "1push" ? "openers" : "working setups";
  const kv = [
    ["gallery id", galleryId()],
    ["object", meta.object_id],
    ["goal region", meta.region],
    ["tier", `${tierName(meta.tier)} (${meta.density_pct.toFixed(2)}% of pushes work)`],
    [green, `${meta.n_green} of ${meta.n_tried} reachable pushes`],
    ["random draws to hit", meta.n_green ? (meta.n_tried / meta.n_green).toFixed(1) : "n/a"],
  ];
  // Guarded: this used to be an unconditional .toFixed on a field the two-movable cards did not
  // carry, and the TypeError aborted render() before renderSteps() ran, so those cards silently
  // lost their step pills. One missing meta field must not take the whole card down.
  if (meta.horizon === "2push" && typeof meta.solve_rate_1push === "number") {
    kv.push(["1push solve rate", meta.solve_rate_1push.toFixed(3)]);
  }

  // A doorway no single block opens. Measured over the 2220-card pool with best-first at budget
  // 900: the 240 rooms with no way around one of these exhaust the budget 13.8% of the time,
  // against 2.9% for the 136 that have an alternative route. Same shape of scene either way, so
  // the flag is the thing that predicts the failure and belongs on the card.
  if (meta.door_needs_both_blocks) {
    kv.push(["doorway", meta.has_route_around
      ? "needs BOTH blocks, but another route to the goal exists"
      : "needs BOTH blocks, and there is no way around it"]);
  }

  const t = timing && timing[card.file_key];
  if (t) {
    // Bold the seconds, plain the sims: the time is the number being compared, the call count is
    // context for it.
    const pm = (v) => `<b>${v[0].toFixed(2)} ± ${v[1].toFixed(2)} s</b>`;
    kv.push(["random search", pm(t.rand) + `  <span class="dim">(${t.rand_sims[0].toFixed(0)} sims)</span>`]);
    kv.push(["HY5U ranker", pm(t.model) + `  <span class="dim">(${t.model_sims[0].toFixed(0)} sims)</span>`]);
    kv.push(["speed-up", t.up >= 1
      ? `<span class="speedup" style="background:${greenFor(t.up)};color:${t.up >= 8 ? "#fff" : "inherit"}">` +
        `${t.up.toFixed(1)}× — ${t.saved_pct.toFixed(0)}% less time</span>`
      : `<span class="speedup slower" style="background:${redFor(t.up)};color:${t.up <= 0.125 ? "#fff" : "var(--dead)"}">` +
        `${t.up.toFixed(2)}× — ${(-t.saved_pct).toFixed(0)}% MORE time than random</span>`]);
    if (t.censored) {
      kv.push(["note", `budget exhausted on some seeds (ranker solved ${t.model_solved}/3, ` +
        `random ${t.rand_solved}/3) — these seconds are a lower bound`]);
    }
  }
  document.getElementById("meta-table").innerHTML = kv
    .map(([k, v]) => `<tr><th>${k}</th><td class="mono">${v}</td></tr>`).join("");

  const pairs = card.green.map(([e, d]) => `${e}/${d}`).join(" ");
  document.getElementById("green-list").innerHTML = card.green.length
    ? `<span class="green-label">${green} (edge/depth):</span> <span class="mono">${pairs}</span>`
    : `<span class="green-label">no ${green} recorded at the root</span>`;

  document.getElementById("xml-path").textContent = meta.xml;
  renderSteps(card);
  renderEvalPanel(rows[i]);
}

// One line per eval run this room appears in: HY5U vs random, sims to solve, speed-up. A room with
// a 0% solve rate on one side says so plainly instead of printing a speed-up that does not exist.
function renderEvalPanel(row) {
  if (!evalData) { evalPanelEl.hidden = true; return; }
  const byRun = row && evalData.rooms[row.scene];
  if (!byRun) {
    evalPanelEl.hidden = false;
    evalPanelEl.innerHTML = '<span class="eval-none">no eval run covers this room</span>';
    return;
  }
  evalPanelEl.hidden = false;
  evalPanelEl.innerHTML = evalData.runs.filter((r) => byRun[r])
    .map((r) => evalRunLine(r, byRun[r])).join("");
}

function evalArmText(list, medianSims) {
  if (!list.length) return "no arms run";
  const solvedN = list.filter((e) => e.solved).length;
  if (!solvedN) return `unsolved 0/${list.length}`;
  return `solved ${solvedN}/${list.length}, median ${medianSims} sims`;
}

function evalRunLine(run, rec) {
  const up = rec.speedup;
  const badge = up == null ? "" :
    ` &middot; <span class="speedup${up < 1 ? " slower" : ""}" ` +
    `style="background:${up >= 1 ? greenFor(up) : redFor(up)};` +
    `color:${(up >= 1 ? up >= 8 : up <= 0.125) ? "#fff" : "inherit"}">` +
    `${up.toFixed(up >= 10 ? 0 : 1)}&times;</span>`;
  return `<div class="eval-line"><span class="eval-run-name">${run}</span> ` +
    `HY5U ${evalArmText(rec.model, rec.model_median_sims)} <span class="dim">|</span> ` +
    `random ${evalArmText(rec.random, rec.random_median_sims)}${badge}</div>`;
}

// evalData.room_aggregate is already room-accurate and pre-split by scripts/viz/add_eval_overlay.py
// (summarize_rooms): one row per scope ("overall" / "open" / "door"), n = an actual room count, not
// a card-strata count. No client-side pooling needed any more.
function fmtSolveRate(group) {
  if (!group || !group.n) return "no rooms";
  return EVAL_K.map((k) => {
    const v = group.solve_at[String(k)];
    return `@${k} ${v == null ? "-" : (v * 100).toFixed(0) + "%"}`;
  }).join(" &middot; ");
}

// The top-of-page strip: this run's solve@k for model vs random, overall and split by whether the
// room's only doorway needs both blocks moved, then a room-count tally of model/random outcomes and
// (for 2push rooms the model solved) how often its own solution stayed on one object vs switched.
function renderEvalSummary() {
  if (!evalData || !evalRun || !evalData.room_aggregate || !evalData.room_aggregate[evalRun]) {
    evalSummaryEl.hidden = true;
    return;
  }
  const byScope = Object.fromEntries(evalData.room_aggregate[evalRun].map((s) => [s.scope, s]));
  evalSummaryEl.hidden = false;
  evalSummaryEl.innerHTML =
    `<b>${evalRun}</b> solve@k, model vs random &mdash; ` +
    `all rooms: ${fmtSolveRate(byScope.overall.model)} <span class="dim">vs</span> ` +
    `${fmtSolveRate(byScope.overall.random)} (n=${byScope.overall.n})<br>` +
    `has a route around the doorway: ${fmtSolveRate(byScope.open.model)} <span class="dim">vs</span> ` +
    `${fmtSolveRate(byScope.open.random)} (n=${byScope.open.n}) &nbsp;&nbsp; ` +
    `needs BOTH blocks moved: ${fmtSolveRate(byScope.door.model)} <span class="dim">vs</span> ` +
    `${fmtSolveRate(byScope.door.random)} (n=${byScope.door.n})<br>` +
    renderModelTally();
}

// Room-count tally for the selected run, over EVERY room it covers (not the current card filters --
// this is a fact about the run, like the solve@k lines above it).
function renderModelTally() {
  let solved = 0, unsolved = 0, split = 0, randomSolved = 0, chainSame = 0, chainCross = 0, chainMixed = 0;
  Object.values(evalData.rooms).forEach((byRun) => {
    const rec = byRun[evalRun];
    if (!rec) return;
    const mb = roomModelBucket(rec);
    if (mb === "solved") solved++; else if (mb === "unsolved") unsolved++; else if (mb === "split") split++;
    if (roomRandomAnySolved(rec)) randomSolved++;
    const cb = modelChainBucket(rec);
    if (cb === "same") chainSame++; else if (cb === "cross") chainCross++; else if (cb === "mixed") chainMixed++;
  });
  return `model solved ${solved} rooms, unsolved ${unsolved}, split ${split}; random solved ${randomSolved} ` +
    `<span class="dim">&middot;</span> model's own 2push solution (solved rooms): ` +
    `same-object ${chainSame}, cross-object ${chainCross}, seeds disagree ${chainMixed}`;
}

// Name the object a step pushed whenever it is not the card's own target. On the two-movable
// pools the finish lands on the OTHER block in 628 of 1296 two-push chains, and the note used to
// read "push on edge 15 at depth 2" for all of them, so you watched the neighbour move while the
// caption described an edge index on an object it never named.
function stepWho(st, card) {
  const who = st.object_id;
  if (!who || who === card.meta.object_id) return "push on ";
  return `push on <b>${who}</b>, the other block, `;
}

// "opened" is the labeller's bar: at least 20% of the poses sampled in the goal region became
// reachable. Merging the two regions is a stricter thing, and on this pool the two part company
// often -- 223 of 1805 solved replays (12% overall, 31% of hard 2push) clear the bar with the goal
// still standing as its own region. Both count as solved [USER, 2026-08-30]. The page said "the
// goal is now reachable" either way, so a card whose tint plainly showed two regions read as a
// broken render. Both lines below describe a solve; they differ in how the robot gets there.
function outcomeText(st) {
  const labs = (st.regions && st.regions.labels) || {};
  const split = Object.keys(labs).some((k) => labs[k] === "goal");
  return split
    ? "the goal is now reachable, through a gap -- the regions stay separate"
    : "the goal is now reachable, and the two regions have merged";
}

// start | after push 1 | after push 2 -- the solution the test set already knows, not the ranker's
// own path (that lives in the search traces).
function renderSteps(card) {
  const box = document.getElementById("steps");
  if (!replay || !replay.steps.length) {
    box.innerHTML = '<span class="steps-none">no solution replay built for this episode</span>';
    return;
  }
  const labels = ["start", ...replay.steps.map((s, k) => `after push ${k + 1}`)];
  const st = step > 0 ? replay.steps[step - 1] : null;
  box.innerHTML = '<span class="steps-cap">solution</span>' + labels.map((l, k) =>
    `<button type="button" class="step-pill${k === step ? " on" : ""}" data-step="${k}">${l}</button>`
  ).join("") + (st
    ? `<div class="step-note">${stepWho(st, card)}edge ${st.edge} at depth ${st.depth}, ` +
      (st.opened ? outcomeText(st) : "no opening yet, this is the setup") + `</div>`
    : `<div class="step-note">the state every number above describes</div>`);
  box.querySelectorAll(".step-pill").forEach((b) => b.addEventListener("click", () => {
    step = +b.dataset.step;
    fetchCard(rows[i]).then((c) => { c.file_key = rows[i].file; render(c); });
  }));
}

// Deeper green the bigger the win. Log-scaled: 1x is barely tinted, 50x and up is the full step,
// because speed-ups span three orders of magnitude and a linear ramp would paint everything under
// 10x the same near-white.
function greenFor(up) {
  const f = Math.min(1, Math.max(0, Math.log10(up) / Math.log10(50)));
  const a = (0.10 + 0.80 * f).toFixed(2);
  return `rgba(27, 94, 32, ${a})`;
}

// Mirror of greenFor for the losing side: 1x barely tinted, 1/50x and worse is the full step.
function redFor(up) {
  const f = Math.min(1, Math.max(0, Math.log10(1 / up) / Math.log10(50)));
  return `rgba(198, 40, 40, ${(0.10 + 0.80 * f).toFixed(2)})`;
}

// The name the shortlist exports as gallery_id and the hardware build sheets are numbered by
// (hard_006). It lives on the index row, never in the card. On the real-table datasets it is the
// only human-readable name there is: every one of those 600 scenes is a file called env.xml, so
// the old xml-basename title read "env" on all of them.
function galleryId() {
  return rows[i] ? rows[i].scene : "";
}

function updateStarBtn() {
  const row = rows[i];
  const on = row && stars[row.file];
  starBtn.classList.toggle("starred", !!on);
  starBtn.innerHTML = on ? "&#9733; starred" : "&#9734; star";
}

function toggleStar() {
  const row = rows[i];
  if (!row) return;
  if (stars[row.file]) delete stars[row.file];
  else stars[row.file] = row;
  saveStars();
  updateStarBtn();
  // Un-starring while filtered to starred-only must drop the row out of the list immediately.
  if (starredOnly.checked) applyFilters(rows[i] ? rows[i].file : null);
}

function copyShortlist() {
  const list = Object.values(stars).map((r) => ({
    ...r, gallery_id: r.scene, dataset: DATASET_NAME,
  })).sort((a, b) =>
    a.horizon.localeCompare(b.horizon) || TIER_ORDER[a.tier] - TIER_ORDER[b.tier] ||
    a.density_pct - b.density_pct);
  const text = JSON.stringify(list, null, 2);
  navigator.clipboard.writeText(text).then(
    () => { copyBtn.textContent = `copied ${list.length}`; setTimeout(() => {
      copyBtn.innerHTML = `copy shortlist (<span id="star-count">${list.length}</span>)`;
    }, 1200); },
    () => { window.prompt("copy the shortlist:", text); }
  );
}
