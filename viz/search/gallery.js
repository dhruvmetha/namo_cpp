"use strict";
// Scene gallery: arrow through one filtered category (horizon x tier) one episode at a time and
// star the ones worth turning into a figure. Deliberately a SCANNING tool, not a report -- the
// numbers shown are only what the canonical labels already assert about the root state.
// Data: scenes.json (index) + cards/<file>.json (one self-contained episode), built by
// scripts/viz/build_scene_cards.py.

const NOCACHE = { cache: "no-store" };
const STORE_KEY = "namo-viz-gallery-state";
const STAR_KEY = "namo-viz-gallery-stars";
const TIER_ORDER = { hard: 0, medium: 1, easy: 2 };

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

const FILTERS = [horizonSelect, tierSelect, sortSelect, textFilter, starredOnly];

let index = null;
let timing = null;       // per-problem seconds from the timed campaign; absent = the page hides them
let rows = [];          // the current filtered + sorted list
let i = 0;              // cursor into rows
let stars = {};         // file -> the index row, so a shortlist survives a rebuild of the cards
const cardCache = new Map();

function saveState() {
  localStorage.setItem(STORE_KEY, JSON.stringify({
    horizon: horizonSelect.value, tier: tierSelect.value, sort: sortSelect.value,
    text: textFilter.value, starredOnly: starredOnly.checked,
    file: rows[i] ? rows[i].file : null,
  }));
}

function saveStars() {
  localStorage.setItem(STAR_KEY, JSON.stringify(stars));
  starCount.textContent = Object.keys(stars).length;
}

// timing.json is optional: a gallery built without build_scene_timing.py still works, it just has
// no seconds to show or sort by.
Promise.all([
  fetch("scenes.json", NOCACHE).then((r) => r.json()),
  fetch("timing.json", NOCACHE).then((r) => (r.ok ? r.json() : null)).catch(() => null),
]).then(([m, t]) => {
  index = m;
  timing = t && t.cards ? t.cards : null;
  init();
}).catch((err) => { summary.textContent = "Failed to load scenes.json: " + err; });

function init() {
  stars = JSON.parse(localStorage.getItem(STAR_KEY) || "{}");
  starCount.textContent = Object.keys(stars).length;

  const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
  let wantFile = null;
  if (saved) {
    horizonSelect.value = saved.horizon || "1push";
    tierSelect.value = saved.tier || "all";
    sortSelect.value = saved.sort || "density-asc";
    textFilter.value = saved.text || "";
    starredOnly.checked = !!saved.starredOnly;
    wantFile = saved.file;
  }

  FILTERS.forEach((el) => el.addEventListener("input", () => applyFilters(null)));
  prevBtn.addEventListener("click", () => step(-1));
  nextBtn.addEventListener("click", () => step(1));
  starBtn.addEventListener("click", toggleStar);
  copyBtn.addEventListener("click", copyShortlist);
  document.addEventListener("keydown", (ev) => {
    if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT") return;
    if (ev.key === "ArrowLeft") { step(-1); ev.preventDefault(); }
    else if (ev.key === "ArrowRight") { step(1); ev.preventDefault(); }
    else if (ev.key === "s" || ev.key === "S") { toggleStar(); ev.preventDefault(); }
  });

  applyFilters(wantFile);
}

function applyFilters(wantFile) {
  const q = textFilter.value.trim().toLowerCase();
  rows = index.cards.filter((r) => {
    if (r.horizon !== horizonSelect.value) return false;
    if (tierSelect.value !== "all" && r.tier !== tierSelect.value) return false;
    if (starredOnly.checked && !stars[r.file]) return false;
    if (q && !(r.scene.toLowerCase().includes(q) || r.object_id.toLowerCase().includes(q))) return false;
    return true;
  });

  const mode = sortSelect.value;
  const up = (r) => (timing && timing[r.file] ? timing[r.file].up : -Infinity);
  rows.sort((a, b) => {
    if (mode === "speedup-desc") return up(b) - up(a);
    if (mode === "speedup-asc") return up(a) - up(b);
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

function step(d) {
  if (!rows.length) return;
  i = (i + d + rows.length) % rows.length;
  show();
}

function fetchCard(row) {
  if (!row) return Promise.resolve(null);
  if (cardCache.has(row.file)) return Promise.resolve(cardCache.get(row.file));
  return fetch("cards/" + row.file, NOCACHE).then((r) => r.json()).then((c) => {
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
  fetchCard(row).then((card) => {
    // A slow fetch must not paint over a newer selection.
    if (rows[i] !== row) return;
    card.file_key = row.file;
    render(card);
    saveState();
    fetchCard(rows[(i + 1) % rows.length]);   // prefetch so arrowing stays instant
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
  setSceneViewBox(svg, card.scene);
  const parts = sceneLayers(card.scene, null, card.regions, meta.object_id);

  const em = edgeMap(card);
  card.contacts.forEach((pt, edge) => {
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
    `${meta.horizon} · ${meta.tier} · ${rowsSceneName(meta)} · ${meta.object_id}`;

  const green = meta.horizon === "1push" ? "openers" : "working setups";
  const kv = [
    ["scene", rowsSceneName(meta)],
    ["object", meta.object_id],
    ["goal region", meta.region],
    ["tier", `${meta.tier} (${meta.density_pct.toFixed(2)}% of pushes work)`],
    [green, `${meta.n_green} of ${meta.n_tried} reachable pushes`],
    ["random draws to hit", meta.n_green ? (meta.n_tried / meta.n_green).toFixed(1) : "n/a"],
  ];
  if (meta.horizon === "2push") kv.push(["1push solve rate", meta.solve_rate_1push.toFixed(3)]);

  const t = timing && timing[card.file_key];
  if (t) {
    const pm = (v) => `${v[0].toFixed(2)} ± ${v[1].toFixed(2)} s`;
    kv.push(["random search", pm(t.rand) + `  (${t.rand_sims[0].toFixed(0)} sims)`]);
    kv.push(["HY5U ranker", pm(t.model) + `  (${t.model_sims[0].toFixed(0)} sims)`]);
    kv.push(["speed-up", t.up >= 1
      ? `${t.up.toFixed(1)}× — ${t.saved_pct.toFixed(0)}% less time`
      : `${t.up.toFixed(2)}× — ${(-t.saved_pct).toFixed(0)}% MORE time than random`]);
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
}

function rowsSceneName(meta) {
  return meta.xml.split("/").pop().replace(".xml", "");
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
  const list = Object.values(stars).sort((a, b) =>
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
