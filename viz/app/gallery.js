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
const clearStarsBtn = document.getElementById("clear-stars-btn");

const famControl = document.getElementById("fam-control");
const famRow = document.getElementById("fam-row");
const famSummary = document.getElementById("fam-summary");
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
// Simulator-call budgets the summary strip reports a solved share at. Spelled out as
// "simulator calls" on screen: "@k" needs a legend, and the strip is the first thing read.
const EVAL_K = [1, 5, 30];
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
// Both dropdowns read straight off scripts/viz/add_eval_overlay.py's own vocabulary now (eval.json's
// "chain" and "model_chain" values), so there is no key/label split left to keep in sync by hand.
const CARD_CHAIN_CATEGORIES = ["single push", "card object only", "other object only", "both objects"];
const LABEL_CHAIN_OPTIONS = ["all", ...CARD_CHAIN_CATEGORIES];   // label side never emits "other object only"
const MODEL_CHAIN_OPTIONS = ["all", ...CARD_CHAIN_CATEGORIES, "seeds disagree", "unsolved"];

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
let replay = null;    // the ACTIVE replay driving the step frames -- labelReplay, or one entry of
                       // modelArms, whichever replaySource currently names
let step = 0;         // 0 = start state, 1..n = after that push

// Replay source: the label (replay/) vs one arm's own plan (replay_eval/<run>/<arm>/), built by
// build_scene_replay.py --from-eval. The radios in the card's solution panel are the picker; see
// selectReplaySource() for how `replay` above gets pointed at one or the other.
const modelReplayCache = new Map();   // "<run>|<file>" -> Promise<[{arm, data}]>, data null = no file
let evalRunDirs = null;        // every basename under replay_eval/, or [] when there is no folder
let evalRunDirsPromise = null; // listed once per dataset load, never per card
let replayIndexPromise = null; // {dir, arms: Map(arm -> Set(card file))} for the selected run
let replayIndexRun = null;     // which run replayIndexPromise was built for
let labelReplay = null;        // this card's replay/ file, kept separate so switching back to
                               // "label" needs no refetch
let modelArms = [];            // [{arm, data}] for the CURRENT card; data null = arm has no file here
let replaysReady = false;      // the probe above has answered, so "no replay built" is now the truth
let replaySource = "label";    // "label" or one of this run's arm names

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
  clearStarsBtn.disabled = Object.keys(stars).length === 0;
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

// The <details> stays closed by default (a lot of checkboxes for something most sessions never
// touch); this is what the closed summary line says. Recomputed in applyFilters, same place every
// other live count (eval buckets, chain filters) gets refreshed, so it never drifts from the boxes.
function renderFamSummary() {
  if (!famBoxes.length) return;
  const checked = famBoxes.filter((b) => b.checked);
  famSummary.textContent = checked.length === famBoxes.length
    ? `batch: all ${famBoxes.length}`
    : `batch: ${checked.length} of ${famBoxes.length} (${checked.map((b) => b.dataset.family).join(", ") || "none"})`;
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
  evalRun = defaultEvalRun();
  evalRunSelect.value = evalRun;
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
  modelChainControl.hidden = !evalData.model_chain;
  labelChainSelect.innerHTML = LABEL_CHAIN_OPTIONS.map((o) => `<option value="${o}">${o}</option>`).join("");
  modelChainSelect.innerHTML = MODEL_CHAIN_OPTIONS.map((o) => `<option value="${o}">${o}</option>`).join("");
}

// Open on the newest run that covers the WHOLE gallery: the widest room count wins, and the later
// name breaks a tie (run names carry their date). eval.json's own run order is just the order the
// --run flags were typed, so it cannot be trusted to put the headline run first, and a narrow run
// (the 375-room doorway ablation) opening by default reads as a shrunken gallery.
function defaultEvalRun() {
  const n = (run) => {
    const agg = (evalData.room_aggregate || {})[run] || [];
    const overall = agg.find((sc) => sc.scope === "overall");
    return overall ? overall.n : 0;
  };
  return [...evalData.runs].sort((a, b) => n(b) - n(a) || b.localeCompare(a))[0];
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

// eval.json's per-card label chain (scripts/viz/add_eval_overlay.py:load_card_chains): one of
// CARD_CHAIN_CATEGORIES minus "other object only" (the label solution never skips the card's own
// object), or null (2push card with no replay).
function labelChainOf(row) {
  return evalData && evalData.chain ? (evalData.chain[row.file] || null) : null;
}

// The selected run's own chain for this CARD (scripts/viz/add_eval_overlay.py:card_model_bucket),
// already classified per card server-side against that card's own object_id -- a room with two
// cards can read differently on each. "seeds disagree" when solved model arms landed in different
// categories, "unsolved" when none did (also covers a room this run never touched at all).
function modelChainOf(row) {
  return evalData && evalData.model_chain && evalRun && evalData.model_chain[evalRun]
    ? (evalData.model_chain[evalRun][row.file] || "unsolved") : null;
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
  clearStarsBtn.addEventListener("click", clearShortlist);
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
  renderFamSummary();
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

  // The ground-truth solution of a 1push card is one push by definition, so this filter separates
  // nothing there. Grey it out and hold it at "all" rather than leave a live control that cannot
  // change the result. (The model side stays on: the model is free to spend two pushes where one
  // would do, and picking those out is the point.)
  const oneShot = horizonSelect.value === "1push";
  labelChainSelect.disabled = oneShot;
  labelChainControl.classList.toggle("control-off", oneShot);
  if (oneShot) labelChainSelect.value = "all";

  if (evalData) {
    // Every facet is counted over the rows the OTHER two filters leave, so the count beside the
    // option you have selected is exactly the episode count printed below the bar.
    const lc = labelChainSelect.value || "all";
    const mc = modelChainSelect.value || "all";
    const keepBucket = (r) => matchesModelBucket(evalRoom(r), modelBucket);
    const keepLabel = (r) => lc === "all" || labelChainOf(r) === lc;
    const keepModel = (r) => mc === "all" || modelChainOf(r) === mc;

    const forBuckets = preRows.filter((r) => keepLabel(r) && keepModel(r));
    evalOutcomeRow.querySelectorAll(".bucket-btn").forEach((btn) => {
      const key = btn.dataset.bucket;
      const n = forBuckets.filter((r) => matchesModelBucket(evalRoom(r), key)).length;
      btn.textContent = `${MODEL_BUCKETS.find((b) => b.key === key).label} (${n})`;
      btn.classList.toggle("bucket-active", key === modelBucket);
    });
    const fill = (sel, of, pool) => {
      const cur = sel.value;
      Array.from(sel.options).forEach((opt) => {
        const key = opt.value;
        const n = key === "all" ? pool.length : pool.filter((r) => of(r) === key).length;
        opt.textContent = `${key} (${n})`;
      });
      sel.value = cur || "all";
    };
    fill(labelChainSelect, labelChainOf, preRows.filter((r) => keepBucket(r) && keepModel(r)));
    fill(modelChainSelect, modelChainOf, preRows.filter((r) => keepBucket(r) && keepLabel(r)));
  }

  // Every eval-derived filter is guarded on evalData. Without one the two chain selects are never
  // built, so their .value is "" -- and an unguarded "!== all" test threw out every card, leaving a
  // dataset with no eval.json attached showing an empty gallery.
  rows = !evalData ? preRows : preRows.filter((r) => {
    if (!matchesModelBucket(evalRoom(r), modelBucket)) return false;
    if (labelChainSelect.value !== "all" && labelChainOf(r) !== labelChainSelect.value) return false;
    if (modelChainSelect.value !== "all" && modelChainOf(r) !== modelChainSelect.value) return false;
    return true;
  });
  if (evalData) renderChainTally(rows);

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
    if (ups.length) tail = ` · median speed-up ${ups[ups.length >> 1].toFixed(1)}× in seconds`;
  }
  summary.textContent = `${rows.length} episodes match these filters · ${byTier} · ` +
    `${Object.keys(stars).length} starred in this gallery` + tail;

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

// Anything under viz/, listed off the plain directory index scripts/viz/serve.py already serves for
// a folder with no index.html of its own. Returns the hrefs; a subfolder's href ends in "/".
function listDir(url) {
  return fetch(url, NOCACHE)
    .then((r) => (r.ok ? r.text() : ""))
    .then((html) => [...html.matchAll(/href="([^"?]+)"/g)].map((m) => m[1]))
    .catch(() => []);
}

// Every run folder under replay_eval/. One listing for the whole page load, not one per card. No
// replay_eval/ folder leaves every card with the label only.
function discoverEvalRunDirs() {
  if (evalRunDirsPromise) return evalRunDirsPromise;
  evalRunDirsPromise = listDir(DATA_BASE + "replay_eval/")
    .then((hrefs) => {
      evalRunDirs = hrefs.filter((h) => h.endsWith("/")).map((h) => h.slice(0, -1));
      return evalRunDirs;
    });
  return evalRunDirsPromise;
}

// eval.json calls a run by its nickname ("gallery_0902"); the replay folder is named after the eval
// output directory it was built from ("two_movable_1hop_gallery_20260902"). Fold a YYYYMMDD date to
// MMDD, drop the punctuation, and look for the nickname inside the folder name. No match means this
// run has no replays on disk and the card offers the label only, which beats stepping one run's
// frames under another run's numbers.
function normalizeRunName(name) {
  return name.toLowerCase().replace(/20\d{2}(\d{4})/g, "$1").replace(/[^a-z0-9]/g, "");
}

function replayDirForRun(dirs, run) {
  if (!dirs || !dirs.length || !run) return null;
  const want = normalizeRunName(run);
  return dirs.find((d) => normalizeRunName(d).includes(want)) || null;
}

// Which replays the SELECTED run actually has: its folder, then one listing per arm folder inside.
// Read once per run and reused for every card. This used to be a fetch per arm per card whose 404
// meant "no replay", which cost six failed requests on most of the 2200 cards and filled the
// console with them; the listing answers the same question for free and before the card is drawn.
function replayIndexFor(run) {
  if (replayIndexRun === run && replayIndexPromise) return replayIndexPromise;
  replayIndexRun = run;
  replayIndexPromise = discoverEvalRunDirs().then((dirs) => {
    const dir = replayDirForRun(dirs, run);
    if (!dir) return { dir: null, arms: new Map() };
    return listDir(`${DATA_BASE}replay_eval/${dir}/`)
      .then((hrefs) => Promise.all(hrefs.filter((h) => h.endsWith("/")).map((h) => {
        const arm = h.slice(0, -1);
        return listDir(`${DATA_BASE}replay_eval/${dir}/${arm}/`)
          .then((files) => [arm, new Set(files)]);
      })))
      .then((pairs) => ({ dir, arms: new Map(pairs) }));
  });
  return replayIndexPromise;
}

// The arms to show for THIS card: exactly what eval.json recorded for the room under the SELECTED
// run, model side first. Nothing hardcoded -- a run with other arm names (the doorway ablation's
// control_model / noreroute_uniform) needs no edit here.
function armRowsFor(row) {
  const rec = evalRoom(row);
  if (!rec) return [];
  return [...(rec.model || []).map((e) => ({ ...e, side: "model" })),
          ...(rec.random || []).map((e) => ({ ...e, side: "random" }))];
}

// Fetch only the replays the index above says are on disk, so nothing here 404s. Keyed by run as
// well as card: switching runs switches folders.
function fetchModelReplays(row) {
  if (!row) return Promise.resolve([]);
  const key = `${evalRun}|${row.file}`;
  if (modelReplayCache.has(key)) return modelReplayCache.get(key);
  const p = replayIndexFor(evalRun).then((ix) => {
    if (!ix.dir) return [];
    const have = armRowsFor(row).map((e) => e.arm)
      .filter((arm) => ix.arms.has(arm) && ix.arms.get(arm).has(row.file));
    return Promise.all(have.map((arm) =>
      fetch(`${DATA_BASE}replay_eval/${ix.dir}/${arm}/${row.file}`, NOCACHE)
        .then((r) => r.json())
        .catch(() => null)
        .then((data) => ({ arm, data }))
    ));
  });
  if (modelReplayCache.size > 40) modelReplayCache.delete(modelReplayCache.keys().next().value);
  modelReplayCache.set(key, p);
  return p;
}

function activeModelReplay() {
  const found = modelArms.find((a) => a.arm === replaySource);
  return found ? found.data : null;
}

// Clicking a radio in the solution panel: point the stepper at that solution and redraw from step 0.
function selectReplaySource(value) {
  if (!rows[i]) return;
  replaySource = value;
  replay = value === "label" ? labelReplay : activeModelReplay();
  step = 0;
  fetchCard(rows[i]).then((c) => { c.file_key = rows[i].file; render(c); });
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
  replaySource = "label";   // the ground truth is always the default, even for a card seen before
  modelArms = [];
  replaysReady = false;
  Promise.all([fetchCard(row), fetchReplay(row)]).then(([card, rep]) => {
    // A slow fetch must not paint over a newer selection.
    if (rows[i] !== row) return;
    card.file_key = row.file;
    labelReplay = rep;
    replay = labelReplay;
    render(card);
    saveState();
    fetchCard(rows[(i + 1) % rows.length]);   // prefetch so arrowing stays instant
    fetchReplay(rows[(i + 1) % rows.length]);
  });
  fetchModelReplays(row).then((arms) => {
    if (rows[i] !== row) return;   // a slow probe must not paint over a newer selection
    modelArms = arms;
    replaysReady = true;
    renderEvalPanel(row);          // fills in reproduced / no replay and enables the radios
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
    kv.push(["model search", pm(t.model) + `  <span class="dim">(${t.model_sims[0].toFixed(0)} sims)</span>`]);
    kv.push(["speed-up", t.up >= 1
      ? `<span class="speedup" style="background:${greenFor(t.up)};color:${t.up >= 8 ? "#fff" : "inherit"}">` +
        `${t.up.toFixed(1)}×, ${t.saved_pct.toFixed(0)}% less time</span>`
      : `<span class="speedup slower" style="background:${redFor(t.up)};color:${t.up <= 0.125 ? "#fff" : "var(--dead)"}">` +
        `${t.up.toFixed(2)}×, ${(-t.saved_pct).toFixed(0)}% MORE time than random</span>`]);
    if (t.censored) {
      kv.push(["note", `budget exhausted on some seeds (model solved ${t.model_solved}/3, ` +
        `random ${t.rand_solved}/3), so these seconds are a lower bound`]);
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

// The card's solution block, and the replay picker in one. First the ground truth, then the selected
// run's own arms, one row each: what it pushed, how many sims it took, whether the plan replayed.
// The radio on a row is what points the stepper below at that solution.
function renderEvalPanel(row) {
  // No eval attached to the dataset means there is nothing to choose between, and the stepper below
  // already names the label solution. Keep the simpler galleries as they were.
  if (!row || !evalData) { evalPanelEl.hidden = true; return; }
  const parts = ['<div class="sol-cap">Solutions on this room</div>', labelRowHtml()];
  if (evalData && evalRun) {
    const rec = evalRoom(row);
    if (rec) {
      parts.push(runHeadHtml(evalRun, rec));
      armRowsFor(row).forEach((e) => parts.push(armRowHtml(e)));
    } else {
      parts.push(`<div class="sol-run"><span class="sol-run-name">${evalRun}</span> ` +
        "did not cover this room</div>");
    }
    parts.push(otherRunsHtml(row));
  }
  evalPanelEl.hidden = false;
  evalPanelEl.innerHTML = parts.join("");
  evalPanelEl.querySelectorAll("input[name=replay-source]").forEach((b) =>
    b.addEventListener("change", () => selectReplaySource(b.value)));
}

// One row of the block. The push sequence goes on its own line under the radio, so a four-push chain
// wraps inside the column instead of running off the edge of it.
function solRowHtml(o) {
  const off = o.enabled ? "" : " sol-off";
  return `<label class="sol-row${off}" title="${o.hint || ""}">` +
    `<input type="radio" name="replay-source" value="${o.value}"` +
    `${replaySource === o.value ? " checked" : ""}${o.enabled ? "" : " disabled"}>` +
    `<span class="sol-side">${o.side}</span><span class="sol-who">${o.name}</span>` +
    `<span class="sol-facts">${o.facts}</span>` +
    (o.seq ? `<span class="sol-seq">${o.seq}</span>` : "") + "</label>";
}

// The label row is the default and says why: it is the answer the test set already holds, the one
// every arm is being judged against.
function labelRowHtml() {
  const n = labelReplay ? labelReplay.steps.length : 0;
  return solRowHtml({
    value: "label", side: "label", name: "ground truth",
    facts: labelReplay ? `${pushText(n)}, the solution the test set already holds`
                       : "no replay built for this episode",
    seq: labelReplay ? pushSeqText(labelReplay.steps.map(
      (st) => ({ object_id: st.object_id, edge_idx: st.edge, depth: st.depth }))) : "",
    enabled: !!labelReplay,
    hint: "The exhaustive sweep's own solution, replayed and snapshotted after each push.",
  });
}

function armRowHtml(e) {
  const found = modelArms.find((a) => a.arm === e.arm);
  const data = found ? found.data : null;
  const bits = [e.solved ? `solved in ${simText(e.sims)}` : "unsolved"];
  if (data) {
    bits.push(data.reproduced ? "reproduced"
      : `did not reproduce (${data.steps.length} of ${data.plan_len} pushes ran)`);
  } else if (!replaysReady) {
    bits.push("looking for a replay");
  } else {
    bits.push("no replay built");
  }
  return solRowHtml({
    value: e.arm, side: e.side, name: e.arm, facts: bits.join(" \u00b7 "),
    seq: e.pushes && e.pushes.length ? pushSeqText(e.pushes) : "",
    enabled: !!data,
    hint: data ? "Step this arm's own plan" : "This arm has no replay file for this room",
  });
}

function simText(n) { return n == null ? "an unknown number of sims" : `${n} sim${n === 1 ? "" : "s"}`; }
function pushText(n) { return `${n} push${n === 1 ? "" : "es"}`; }

// The run's headline for this room: how each side did, then how much cheaper the model was. "2.0x
// fewer sims" and "2.0x more sims" instead of a bare 0.5x, which reads as a win at a glance.
function runHeadHtml(run, rec) {
  const up = rec.speedup;
  const badge = up == null ? "" :
    ` <span class="speedup${up < 1 ? " slower" : ""}" ` +
    `style="background:${up >= 1 ? greenFor(up) : redFor(up)};` +
    `color:${(up >= 1 ? up >= 8 : up <= 0.125) ? "#fff" : "inherit"}">` +
    (up >= 1 ? `${up.toFixed(up >= 10 ? 0 : 1)}&times; fewer sims`
             : `${(1 / up).toFixed(1)}&times; more sims`) + "</span>";
  return `<div class="sol-run"><span class="sol-run-name">${run}</span> ` +
    `model ${armSideText(rec.model, rec.model_median_sims)}. ` +
    `random ${armSideText(rec.random, rec.random_median_sims)}.${badge}</div>`;
}

function armSideText(list, medianSims) {
  if (!list.length) return "had no arms here";
  const n = list.filter((e) => e.solved).length;
  const seeds = `${list.length} seed${list.length === 1 ? "" : "s"}`;
  if (!n) return `solved 0 of ${seeds}`;
  return `solved ${n} of ${seeds}, median ${simText(medianSims)}`;
}

// The other runs that also cover this room: one line each, no rows and no radios. Their replays live
// under a different replay_eval/ folder, and only the selected run's frames are on offer here.
function otherRunsHtml(row) {
  const byRun = evalData.rooms[row.scene] || {};
  return evalData.runs.filter((r) => r !== evalRun && byRun[r]).map((r) => {
    const rec = byRun[r];
    return `<div class="sol-other">also ran as <b>${r}</b>: ` +
      `model ${armSideText(rec.model, rec.model_median_sims)}, ` +
      `random ${armSideText(rec.random, rec.random_median_sims)}</div>`;
  }).join("");
}

function pushSeqText(pushes) {
  return pushes.map((p) => `${(p.object_id || "?").replace(/_movable$/, "")} e${p.edge_idx} d${p.depth}`)
    .join(" &rarr; ");
}

// evalData.room_aggregate is already room-accurate and pre-split by scripts/viz/add_eval_overlay.py
// (summarize_rooms): one row per scope ("overall" / "open" / "door"), n = an actual room count, not
// a card-strata count. No client-side pooling needed any more.
function pctList(group) {
  return EVAL_K.map((k) => {
    const v = group.solve_at[String(k)];
    return v == null ? "-" : (v * 100).toFixed(0) + "%";
  }).join(" / ");
}

// One sentence per line, and every number carries its unit: the strip is the first thing anybody
// reads, and "@30 83% (n=1432)" needs the source open to decode.
function scopeLine(lead, group) {
  if (!group || !group.n) return "";
  return `<div>${lead}: model ${pctList(group.model)}, ` +
    `random ${pctList(group.random)}</div>`;
}

// The top-of-page strip: how often the selected run solved a room within 1, 5 and 30 simulator
// calls, model against random, overall and split by whether the room's only doorway needs both
// blocks moved. Then two count lines: room outcomes across the run, and (from renderChainTally,
// which runs in applyFilters because it is card-filtered) what the model actually pushed.
function renderEvalSummary() {
  if (!evalData || !evalRun || !evalData.room_aggregate || !evalData.room_aggregate[evalRun]) {
    evalSummaryEl.hidden = true;
    return;
  }
  const byScope = Object.fromEntries(evalData.room_aggregate[evalRun].map((sc) => [sc.scope, sc]));
  const ks = EVAL_K.join(" / ");
  evalSummaryEl.hidden = false;
  evalSummaryEl.innerHTML =
    scopeLine(`Run <b>${evalRun}</b>, all ${byScope.overall.n} rooms, share solved within ` +
              `${ks} simulator calls`, byScope.overall) +
    scopeLine(`Of those, the ${byScope.open.n} rooms with a route around the doorway`, byScope.open) +
    scopeLine(`The ${byScope.door.n} rooms whose doorway needs both blocks moved`, byScope.door) +
    renderModelTally() + '<div id="eval-chain-tally"></div>';
}

// Room-count tally for the selected run, over EVERY room it covers (not the current card filters --
// this is a fact about the run, like the solve shares above it).
function renderModelTally() {
  let solved = 0, unsolved = 0, split = 0, randomSolved = 0;
  Object.values(evalData.rooms).forEach((byRun) => {
    const rec = byRun[evalRun];
    if (!rec) return;
    const mb = roomModelBucket(rec);
    if (mb === "solved") solved++; else if (mb === "unsolved") unsolved++; else if (mb === "split") split++;
    if (roomRandomAnySolved(rec)) randomSolved++;
  });
  return `<div>Across those ${solved + unsolved + split} rooms the model solved ${solved}, ` +
    `split across seeds on ${split} and left ${unsolved} unsolved; ` +
    `random solved ${randomSolved} of the same rooms.</div>`;
}

// Model-chain category tally, over the cards the current filters show (not dataset-wide, unlike
// renderModelTally above) -- called from applyFilters, which already has that row set.
function renderChainTally(shown) {
  const el = document.getElementById("eval-chain-tally");
  if (!el) return;
  if (!evalData || !evalData.model_chain || !evalRun) { el.textContent = ""; return; }
  const counts = Object.fromEntries(MODEL_CHAIN_OPTIONS.filter((o) => o !== "all").map((o) => [o, 0]));
  shown.forEach((r) => { const v = modelChainOf(r); if (v in counts) counts[v] += 1; });
  el.textContent = `What the model pushed on the ${shown.length} episodes now showing: ` +
    MODEL_CHAIN_OPTIONS.filter((o) => o !== "all").map((c) => `${c} ${counts[c]}`).join(", ") + ".";
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
  // Name the source in the caption. The frames look identical whoever produced them, and reading a
  // model arm's path as the ground truth is the one mistake this stepper can invite.
  const whose = replaySource === "label" ? "label solution" : `${replaySource} solution`;
  if (!replay || !replay.steps.length) {
    box.innerHTML = `<span class="steps-none">the ${whose} has no replay built for this episode</span>`;
    return;
  }
  const labels = ["start", ...replay.steps.map((s, k) => `after push ${k + 1}`)];
  const st = step > 0 ? replay.steps[step - 1] : null;
  box.innerHTML = `<span class="steps-cap">${whose}</span>` + labels.map((l, k) =>
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

function clearShortlist() {
  const n = Object.keys(stars).length;
  if (!n) return;
  if (!window.confirm(`Clear all ${n} starred cards?`)) return;
  stars = {};
  saveStars();     // same file-backed path toggleStar uses, so stars.json empties the same way
  updateStarBtn();
  // Re-run the filters even when starred-only is off: the summary line's starred count must catch
  // up now, not on the next filter change, and starred-only (if checked) must drop every row.
  applyFilters(rows[i] ? rows[i].file : null);
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
