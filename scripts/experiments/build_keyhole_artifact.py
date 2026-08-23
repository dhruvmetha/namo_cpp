#!/usr/bin/env python3
"""Build the self-contained interactive page for the paired per-keyhole comparison.

Reads paired_keyhole_compare.py's pairs_*.jsonl and inlines them into one HTML file: no CDN, no
fetch, no build step, so the page works as a published artifact or straight off disk.

    python scripts/experiments/build_keyhole_artifact.py \
        --data $NAMO_SCRATCH/analysis/keyhole --out $NAMO_SCRATCH/analysis/keyhole/keyhole.html
"""
import argparse
import json
import os

TEMPLATE = r"""<title>Ranker vs random — every region-opening problem</title>
<style>
:root {
  --bg: #f7f8fa; --surface: #ffffff; --sunk: #eef1f5;
  --ink: #10141a; --ink-2: #46505e; --ink-3: #78828f;
  --rule: #dde2e9;
  --ranker: #2a78d6; --random: #eb6834; --hard: #1baf7a; --censor: #98a0ac;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #12151a; --surface: #191d24; --sunk: #11141a;
    --ink: #eef2f7; --ink-2: #a8b3c1; --ink-3: #79838f;
    --rule: #2a3038;
    --ranker: #4d94ea; --random: #f47b4c; --hard: #2ac592; --censor: #6d7683;
  }
}
:root[data-theme="dark"] {
  --bg: #12151a; --surface: #191d24; --sunk: #11141a;
  --ink: #eef2f7; --ink-2: #a8b3c1; --ink-3: #79838f;
  --rule: #2a3038;
  --ranker: #4d94ea; --random: #f47b4c; --hard: #2ac592; --censor: #6d7683;
}
:root[data-theme="light"] {
  --bg: #f7f8fa; --surface: #ffffff; --sunk: #eef1f5;
  --ink: #10141a; --ink-2: #46505e; --ink-3: #78828f;
  --rule: #dde2e9;
  --ranker: #2a78d6; --random: #eb6834; --hard: #1baf7a; --censor: #98a0ac;
}

body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans);
       font-size: 15px; line-height: 1.55; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 40px 24px 72px; display: flex;
        flex-direction: column; gap: 28px; }

header h1 { font-size: clamp(24px, 3.4vw, 34px); line-height: 1.15; margin: 0 0 10px;
            text-wrap: balance; letter-spacing: -0.015em; }
header p { margin: 0; color: var(--ink-2); max-width: 68ch; }
.eyebrow { font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.13em;
           text-transform: uppercase; color: var(--ink-3); margin: 0 0 14px; }

.panel { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
         padding: 20px 22px; }
.panel > h2 { font-size: 15px; margin: 0 0 4px; letter-spacing: -0.005em; }
.panel > .sub { margin: 0 0 18px; color: var(--ink-2); font-size: 13.5px; }

.controls { display: flex; flex-wrap: wrap; gap: 10px 22px; align-items: flex-end; }
.control { display: flex; flex-direction: column; gap: 5px; }
.control > span { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.11em;
                  text-transform: uppercase; color: var(--ink-3); }
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 7px; overflow: hidden; }
.seg button { font: inherit; font-size: 13px; padding: 5px 13px; border: 0; cursor: pointer;
              background: var(--surface); color: var(--ink-2); }
.seg button + button { border-left: 1px solid var(--rule); }
.seg button[aria-pressed="true"] { background: var(--ink); color: var(--surface); }
.seg button:focus-visible { outline: 2px solid var(--ranker); outline-offset: -2px; }

.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr)); gap: 1px;
         background: var(--rule); border: 1px solid var(--rule); border-radius: 10px;
         overflow: hidden; }
.tile { background: var(--surface); padding: 16px 18px; }
.tile .k { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.11em;
           text-transform: uppercase; color: var(--ink-3); }
.tile .v { font-family: var(--mono); font-size: 27px; font-variant-numeric: tabular-nums;
           letter-spacing: -0.02em; margin-top: 4px; }
.tile .n { color: var(--ink-2); font-size: 12.5px; }
.tile.lose .v { color: var(--random); }
.tile.win .v { color: var(--ranker); }

.plotgrid { display: grid; grid-template-columns: 1fr 250px; gap: 20px; align-items: start; }
@media (max-width: 860px) { .plotgrid { grid-template-columns: 1fr; } }
svg { display: block; width: 100%; height: auto; overflow: visible; }
.axis line, .axis path { stroke: var(--rule); }
.axis text { fill: var(--ink-3); font-family: var(--mono); font-size: 10px; }
.gridline { stroke: var(--rule); stroke-dasharray: 2 3; }
.parity { stroke: var(--ink); stroke-width: 1.2; }
.tenx { stroke: var(--ink-3); stroke-width: 1; stroke-dasharray: 3 3; }
.pt { cursor: pointer; }
.pt.win { fill: var(--ranker); }
.pt.lose { fill: var(--random); }
.pt.cens { fill: none; stroke: var(--censor); stroke-width: 1.1; }
.pt.dim { opacity: 0.12; }
.axlabel { fill: var(--ink-2); font-size: 11.5px; font-family: var(--sans); }

.inspector { background: var(--sunk); border-radius: 8px; padding: 14px 16px; font-size: 13px; }
.inspector h3 { margin: 0 0 10px; font-size: 12px; font-family: var(--mono);
                letter-spacing: 0.11em; text-transform: uppercase; color: var(--ink-3); }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; }
.kv dt { color: var(--ink-3); }
.kv dd { margin: 0; font-family: var(--mono); font-variant-numeric: tabular-nums;
         word-break: break-all; }
.hint { color: var(--ink-3); font-size: 12.5px; margin: 10px 0 0; }

.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 12.5px; color: var(--ink-2);
          margin-top: 12px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
          margin-right: 6px; vertical-align: -1px; }
.swatch.sq { border-radius: 2px; background: none; border: 1.4px solid var(--censor); }

table { border-collapse: collapse; width: 100%; font-size: 13px; }
.tablewrap { overflow-x: auto; }
th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--rule);
         font-variant-numeric: tabular-nums; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { font-family: var(--mono); font-size: 10.5px; letter-spacing: 0.08em;
           text-transform: uppercase; color: var(--ink-3); font-weight: 400; }
tbody tr:hover { background: var(--sunk); }
td.mono { font-family: var(--mono); }

.tooltip { position: fixed; z-index: 20; pointer-events: none; background: var(--surface);
           border: 1px solid var(--rule); border-radius: 7px; padding: 9px 11px; font-size: 12.5px;
           box-shadow: 0 6px 22px rgba(0,0,0,0.16); max-width: 280px; opacity: 0;
           transition: opacity 90ms; }
.tooltip.on { opacity: 1; }
.tooltip .t { font-family: var(--mono); font-size: 11px; color: var(--ink-3); }
footer { color: var(--ink-3); font-size: 12.5px; border-top: 1px solid var(--rule); padding-top: 16px; }
footer code { font-family: var(--mono); font-size: 11.5px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>

<div class="wrap">
<header>
  <p class="eyebrow">HY5U ranker · uniform random · timed campaign, budget 4000</p>
  <h1>Every region-opening problem, priced twice</h1>
  <p>Each problem is one keyhole: a robot region, a goal region, and the single object between them.
     Both planners solved the same 2,320 of them on the same exclusive compute node, so the only
     difference is the order they try pushes in. One dot is one problem — never an average.</p>
</header>

<section class="panel">
  <div class="controls" id="controls"></div>
</section>

<div class="tiles" id="tiles"></div>

<section class="panel">
  <h2>Cost per problem: ranker against random</h2>
  <p class="sub">Below the black line the ranker was cheaper. Hover a dot for the problem it stands for;
     click to keep it in the inspector.</p>
  <div class="plotgrid">
    <div>
      <svg id="scatter" viewBox="0 0 640 470" role="img"
           aria-label="Scatter of ranker cost against random cost, one point per problem"></svg>
      <div class="legend">
        <span><i class="swatch" style="background:var(--ranker)"></i>ranker cheaper</span>
        <span><i class="swatch" style="background:var(--random)"></i>ranker slower</span>
        <span><i class="swatch sq"></i>one arm never solved it — cost is a lower bound</span>
      </div>
    </div>
    <div>
      <div class="inspector" id="inspector"></div>
      <p class="hint">Paste the scene name into the scene gallery to see the room this problem is.</p>
    </div>
  </div>
</section>

<section class="panel">
  <h2>How big the win is, problem by problem</h2>
  <p class="sub">Every problem's own speed-up, sorted worst to best. The shaded band is where random won.</p>
  <svg id="curve" viewBox="0 0 900 330" role="img"
       aria-label="Per-problem speed-up sorted into percentiles, one line per difficulty tier"></svg>
</section>

<section class="panel">
  <h2>The numbers</h2>
  <p class="sub">Per-problem speed-up: percentiles across problems, with the loss rate on the same measure.
     Censored problems are excluded from the ratios and counted separately.</p>
  <div class="tablewrap"><table id="table"></table></div>
</section>

<footer>
  <p>Both arms: 3 seeds, paired seed-to-seed, median within each problem, then percentiles across
     problems — the canonical statistic. Ratio-of-medians reads higher and is not what these numbers are.
     Cost in seconds is comparable because every run in this campaign took a whole exclusive
     cascadelake node, single-threaded. Source <code>scripts/experiments/paired_keyhole_compare.py</code>.</p>
</footer>
</div>

<div class="tooltip" id="tip"></div>

<script>
const DATA = __DATA__;
const TIERS = ["easy", "medium", "hard"];
const TIER_STROKE = { easy: "var(--ranker)", medium: "var(--random)", hard: "var(--hard)" };
const state = { leg: "2push", measure: "time", tiers: new Set(TIERS), pinned: null };

const el = (id) => document.getElementById(id);
const fmt = (x, d = 2) => Number(x).toFixed(d);
const cost = (r, which) => state.measure === "time" ? r[which + "_t"] : r[which + "_s"];
const speed = (r) => state.measure === "time" ? r.up_t : r.up_s;
const unit = () => state.measure === "time" ? "s" : " calls";

function rows() {
  return DATA.filter((r) => r.leg === state.leg && state.tiers.has(r.tier));
}
function clean(rs) { return rs.filter((r) => r.cl); }

function median(v) {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function pctl(v, p) {
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.ceil(p / 100 * s.length) - 1))];
}

/* ---- controls ------------------------------------------------------------------ */
function segment(label, opts, get, set) {
  const wrap = document.createElement("label");
  wrap.className = "control";
  const cap = document.createElement("span");
  cap.textContent = label;
  const seg = document.createElement("span");
  seg.className = "seg";
  opts.forEach(([val, text]) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = text;
    b.setAttribute("aria-pressed", String(get(val)));
    b.addEventListener("click", () => { set(val); renderAll(); });
    seg.appendChild(b);
  });
  wrap.append(cap, seg);
  return wrap;
}

function renderControls() {
  const c = el("controls");
  c.innerHTML = "";
  c.append(
    segment("Horizon", [["1push", "one-push"], ["2push", "two-push"]],
      (v) => state.leg === v, (v) => { state.leg = v; state.pinned = null; }),
    segment("Cost measured in", [["time", "seconds"], ["sims", "simulator calls"]],
      (v) => state.measure === v, (v) => { state.measure = v; }),
    segment("Difficulty", TIERS.map((t) => [t, t]),
      (v) => state.tiers.has(v),
      (v) => {
        if (state.tiers.has(v) && state.tiers.size > 1) state.tiers.delete(v);
        else state.tiers.add(v);
      })
  );
}

/* ---- summary tiles -------------------------------------------------------------- */
function renderTiles() {
  const rs = rows(), cl = clean(rs);
  const sp = cl.map(speed);
  const lose = sp.filter((x) => x < 1).length;
  const t = el("tiles");
  const tile = (k, v, n, cls) =>
    `<div class="tile ${cls || ""}"><div class="k">${k}</div><div class="v">${v}</div>` +
    `<div class="n">${n}</div></div>`;
  t.innerHTML =
    tile("median problem", fmt(median(sp), 1) + "×", "speed-up over random", "win") +
    tile("ranker loses", fmt(100 * lose / sp.length, 0) + "%", `${lose} of ${sp.length} problems`, "lose") +
    tile("typical ranker cost", fmt(median(cl.map((r) => cost(r, "model"))), 2) + unit(), "median problem") +
    tile("typical random cost", fmt(median(cl.map((r) => cost(r, "rand"))), 2) + unit(), "median problem") +
    tile("top decile", fmt(pctl(sp, 90), 0) + "×", "the best tenth of problems") +
    tile("problems", String(rs.length), `${rs.length - cl.length} censored`);
}

/* ---- scatter -------------------------------------------------------------------- */
const S = { w: 640, h: 470, l: 58, r: 14, t: 14, b: 46 };

function niceTicks(lo, hi) {
  const out = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) out.push(Math.pow(10, e));
  return out.filter((v) => v >= lo * 0.95 && v <= hi * 1.05);
}

function renderScatter() {
  const svg = el("scatter");
  const rs = rows();
  if (!rs.length) { svg.innerHTML = ""; return; }
  const vals = rs.flatMap((r) => [cost(r, "model"), cost(r, "rand")]).filter((v) => v > 0);
  const lo = Math.min(...vals) * 0.75, hi = Math.max(...vals) * 1.3;
  const lx = Math.log10(lo), hx = Math.log10(hi);
  const px = (v) => S.l + (Math.log10(v) - lx) / (hx - lx) * (S.w - S.l - S.r);
  const py = (v) => S.h - S.b - (Math.log10(v) - lx) / (hx - lx) * (S.h - S.t - S.b);

  const p = [];
  niceTicks(lo, hi).forEach((v) => {
    p.push(`<line class="gridline" x1="${px(v)}" y1="${S.t}" x2="${px(v)}" y2="${S.h - S.b}"/>`);
    p.push(`<line class="gridline" x1="${S.l}" y1="${py(v)}" x2="${S.w - S.r}" y2="${py(v)}"/>`);
    const lab = v >= 1 ? String(v) : String(v);
    p.push(`<text class="axis" x="${px(v)}" y="${S.h - S.b + 15}" text-anchor="middle">${lab}</text>`);
    p.push(`<text class="axis" x="${S.l - 8}" y="${py(v) + 3}" text-anchor="end">${lab}</text>`);
  });
  p.push(`<line class="parity" x1="${px(lo)}" y1="${py(lo)}" x2="${px(hi)}" y2="${py(hi)}"/>`);
  p.push(`<line class="tenx" x1="${px(lo)}" y1="${py(lo / 10)}" x2="${px(hi)}" y2="${py(hi / 10)}"/>`);
  p.push(`<text class="axlabel" x="${(S.l + S.w) / 2}" y="${S.h - 6}" text-anchor="middle">` +
         `random: ${state.measure === "time" ? "seconds" : "simulator calls"} to solve</text>`);
  p.push(`<text class="axlabel" transform="translate(13 ${(S.t + S.h - S.b) / 2}) rotate(-90)" ` +
         `text-anchor="middle">ranker: ${state.measure === "time" ? "seconds" : "calls"} to solve</text>`);

  rs.forEach((r, i) => {
    const x = px(cost(r, "rand")), y = py(cost(r, "model"));
    const cls = !r.cl ? "cens" : (speed(r) >= 1 ? "win" : "lose");
    const dim = state.pinned !== null && state.pinned !== r.i ? " dim" : "";
    p.push(!r.cl
      ? `<rect class="pt cens${dim}" data-i="${r.i}" x="${x - 3.4}" y="${y - 3.4}" width="6.8" height="6.8"/>`
      : `<circle class="pt ${cls}${dim}" data-i="${r.i}" cx="${x}" cy="${y}" r="3.1" opacity="0.62"/>`);
  });
  svg.innerHTML = p.join("");
  svg.querySelectorAll(".pt").forEach((n) => {
    n.addEventListener("pointerenter", (e) => showTip(e, byIndex(+n.dataset.i)));
    n.addEventListener("pointerleave", hideTip);
    n.addEventListener("click", () => {
      state.pinned = state.pinned === +n.dataset.i ? null : +n.dataset.i;
      renderScatter(); renderInspector();
    });
  });
}

function byIndex(i) { return DATA.find((r) => r.i === i); }

/* ---- inspector + tooltip -------------------------------------------------------- */
function describe(r) {
  const s = speed(r);
  return [
    ["scene", r.scene], ["object", r.obj], ["difficulty", r.tier],
    ["ranker", fmt(cost(r, "model"), 2) + unit()],
    ["random", fmt(cost(r, "rand"), 2) + unit()],
    ["speed-up", (r.cl ? fmt(s, 1) + "×" : "≥ " + fmt(s, 1) + "× (censored)")],
    ["solved by", r.cl ? "both arms, every seed" : (r.ms ? "ranker only" : (r.rs ? "random only" : "neither"))],
  ];
}

function renderInspector() {
  const box = el("inspector");
  const r = state.pinned !== null ? byIndex(state.pinned) : null;
  if (!r) {
    box.innerHTML = "<h3>Problem inspector</h3><p style='margin:0;color:var(--ink-2)'>" +
      "Click any dot to pin one problem here.</p>";
    return;
  }
  box.innerHTML = "<h3>Problem inspector</h3><dl class='kv'>" +
    describe(r).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") + "</dl>";
}

function showTip(e, r) {
  const tip = el("tip");
  tip.innerHTML = `<div class="t">${r.scene} · ${r.obj}</div>` +
    `<div><b>${r.cl ? fmt(speed(r), 1) + "×" : "≥" + fmt(speed(r), 1) + "×"}</b> ` +
    `— ranker ${fmt(cost(r, "model"), 2)}${unit()} vs random ${fmt(cost(r, "rand"), 2)}${unit()}</div>`;
  tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 300) + "px";
  tip.style.top = (e.clientY + 16) + "px";
  tip.classList.add("on");
}
function hideTip() { el("tip").classList.remove("on"); }

/* ---- percentile curve ------------------------------------------------------------ */
const C = { w: 900, h: 330, l: 62, r: 96, t: 16, b: 44 };

function renderCurve() {
  const svg = el("curve");
  const series = TIERS.filter((t) => state.tiers.has(t)).map((t) => ({
    tier: t,
    v: clean(DATA.filter((r) => r.leg === state.leg && r.tier === t)).map(speed).sort((a, b) => a - b),
  })).filter((s) => s.v.length);
  if (!series.length) { svg.innerHTML = ""; return; }
  const all = series.flatMap((s) => s.v);
  const lo = Math.max(Math.min(...all) * 0.8, 1e-3), hi = Math.max(...all) * 1.2;
  const ly = Math.log10(lo), hy = Math.log10(hi);
  const px = (p) => C.l + p / 100 * (C.w - C.l - C.r);
  const py = (v) => C.h - C.b - (Math.log10(v) - ly) / (hy - ly) * (C.h - C.t - C.b);

  const p = [];
  p.push(`<rect x="${C.l}" y="${py(1)}" width="${C.w - C.l - C.r}" height="${C.h - C.b - py(1)}" ` +
         `fill="var(--sunk)"/>`);
  niceTicks(lo, hi).forEach((v) => {
    p.push(`<line class="gridline" x1="${C.l}" y1="${py(v)}" x2="${C.w - C.r}" y2="${py(v)}"/>`);
    p.push(`<text class="axis" x="${C.l - 8}" y="${py(v) + 3}" text-anchor="end">${v}×</text>`);
  });
  [0, 25, 50, 75, 100].forEach((q) => {
    p.push(`<text class="axis" x="${px(q)}" y="${C.h - C.b + 16}" text-anchor="middle">p${q}</text>`);
  });
  p.push(`<line class="parity" x1="${C.l}" y1="${py(1)}" x2="${C.w - C.r}" y2="${py(1)}" ` +
         `stroke-dasharray="4 3"/>`);
  p.push(`<text class="axis" x="${C.w - C.r - 6}" y="${py(1) + 15}" text-anchor="end">` +
         `shaded: random was faster</text>`);
  series.forEach((s) => {
    const pts = s.v.map((v, i) => `${px(100 * (i + 0.5) / s.v.length)},${py(v)}`).join(" ");
    p.push(`<polyline points="${pts}" fill="none" stroke="${TIER_STROKE[s.tier]}" stroke-width="2"/>`);
    const mid = s.v[s.v.length >> 1];
    p.push(`<circle cx="${px(50)}" cy="${py(mid)}" r="4.2" fill="${TIER_STROKE[s.tier]}" ` +
           `stroke="var(--surface)" stroke-width="1.6"/>`);
    p.push(`<text class="axis" x="${C.w - C.r + 8}" y="${py(s.v[s.v.length - 1]) + 3}" ` +
           `fill="${TIER_STROKE[s.tier]}">${s.tier} (n=${s.v.length})</text>`);
  });
  p.push(`<text class="axlabel" x="${(C.l + C.w - C.r) / 2}" y="${C.h - 6}" text-anchor="middle">` +
         `problems, sorted by their own speed-up</text>`);
  p.push(`<text class="axlabel" transform="translate(14 ${(C.t + C.h - C.b) / 2}) rotate(-90)" ` +
         `text-anchor="middle">speed-up (random ÷ ranker)</text>`);
  svg.innerHTML = p.join("");
}

/* ---- table ---------------------------------------------------------------------- */
function renderTable() {
  const cols = ["", "problems", "ranker solved", "random solved", "ranker cost", "random cost",
                "p25", "median", "p75", "p90", "ranker loses"];
  const body = [];
  [["1push", "one-push"], ["2push", "two-push"]].forEach(([leg, name]) => {
    TIERS.forEach((tier) => {
      const rs = DATA.filter((r) => r.leg === leg && r.tier === tier);
      const cl = clean(rs), sp = cl.map(speed);
      const lose = sp.filter((x) => x < 1).length;
      const cens = rs.length - cl.length;
      body.push("<tr>" + [
        `${name} · ${tier}`,
        rs.length + (cens ? ` <span style="color:var(--ink-3)">(${cens} censored)</span>` : ""),
        fmt(100 * rs.filter((r) => r.ms).length / rs.length, 1) + "%",
        fmt(100 * rs.filter((r) => r.rs).length / rs.length, 1) + "%",
        fmt(median(cl.map((r) => cost(r, "model"))), 2) + unit(),
        fmt(median(cl.map((r) => cost(r, "rand"))), 2) + unit(),
        fmt(pctl(sp, 25), 1) + "×", "<b>" + fmt(median(sp), 1) + "×</b>",
        fmt(pctl(sp, 75), 1) + "×", fmt(pctl(sp, 90), 1) + "×",
        fmt(100 * lose / sp.length, 1) + "%",
      ].map((c, i) => `<td${i ? ' class="mono"' : ""}>${c}</td>`).join("") + "</tr>");
    });
  });
  el("table").innerHTML = "<thead><tr>" + cols.map((c) => `<th>${c}</th>`).join("") +
    "</tr></thead><tbody>" + body.join("") + "</tbody>";
}

function renderAll() {
  renderControls(); renderTiles(); renderScatter(); renderInspector(); renderCurve(); renderTable();
}
renderAll();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir written by paired_keyhole_compare.py")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for leg in ("1push", "2push"):
        for line in open(os.path.join(a.data, f"pairs_{leg}.jsonl")):
            r = json.loads(line)
            rows.append({
                "i": len(rows), "leg": leg, "tier": r["tier"],
                "scene": os.path.basename(r["xml"]).replace(".xml", ""),
                "obj": r["object_id"].replace("_movable", ""),
                "model_t": round(r["model_t"], 3), "rand_t": round(r["rand_t"], 3),
                "model_s": r["model_sims"], "rand_s": r["rand_sims"],
                "up_t": round(r["speedup_time"], 3), "up_s": round(r["speedup_sims"], 3),
                "cl": bool(r["clean"]), "ms": bool(r["model_solved"]), "rs": bool(r["rand_solved"]),
            })
    html = TEMPLATE.replace("__DATA__", json.dumps(rows, separators=(",", ":")))
    with open(a.out, "w") as f:
        f.write(html)
    print(f"wrote {a.out}  ({len(rows)} problems, {os.path.getsize(a.out)/1e3:.0f} KB)")


if __name__ == "__main__":
    main()
