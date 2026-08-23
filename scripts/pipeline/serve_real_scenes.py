#!/usr/bin/env python3
"""Browse the real-table scene pool in a browser and pick which ones to build.

Same shape as python/visualize_environment.py: a localhost HTTPServer, a `/api/render`
endpoint that returns a PNG, and a browser tab opened for you. It reuses that file's
`visualize_environment` renderer rather than drawing its own picture, so what you see here is the
same overhead view as everywhere else in the repo.

What it adds over the generic viewer is the label side. Every scene is joined to its row in
`handoff/real_scene_build_sheets/<axis>/<tier>.csv`, so you can filter by difficulty, by whether one
push suffices or the scene needs a two-chain, by which block, and by brick count. Ticking scenes
copies out their build-sheet rows in the same CSV columns, so selecting IS the handoff and nobody
has to turn ids back into centimetres.

  python scripts/pipeline/serve_real_scenes.py                 # hmax=2 axis, port 8000
  python scripts/pipeline/serve_real_scenes.py --axis 1push --port 8010
"""
import argparse
import csv
import io
import json
import os
import sys
import threading
import webbrowser
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "python"))
from environment_selection import visualize_environment  # noqa: E402

SHEETS = os.path.join(REPO, "handoff", "real_scene_build_sheets")


def load_scenes(axis, pools):
    """Join each shipped CSV row to the env.xml it was generated from.

    The CSV carries no path (it is a build sheet for a human, not a file index), so scenes are
    matched on blocker centre plus robot start, which is unique per scene. That is the same join
    used everywhere else in this pipeline and it has never produced an unmatched row.
    """
    index = {}
    for pool in pools:
        for root, _dirs, files in os.walk(pool):
            if "build_sheets.json" not in files:
                continue
            for s in json.load(open(os.path.join(root, "build_sheets.json"))):
                xml = os.path.join(root, s["scene_id"], "env.xml")
                if os.path.exists(xml):
                    index[(round(s["blocker"]["center_cm"][0], 1),
                           round(s["blocker"]["center_cm"][1], 1),
                           round(s["robot_start_cm"][0], 1),
                           round(s["robot_start_cm"][1], 1))] = xml

    scenes, missing = [], 0
    for tier in ("easy", "med", "hard"):
        path = os.path.join(SHEETS, axis, f"{tier}.csv")
        per = defaultdict(lambda: {"bricks": [], "block": None})
        for r in csv.DictReader(open(path)):
            if r["item"] == "brick":
                per[r["build_id"]]["bricks"].append(r)
            else:
                per[r["build_id"]]["block"] = r
        for bid, d in per.items():
            b = d["block"]
            key = (round(float(b["centre_x_cm"]), 1), round(float(b["centre_y_cm"]), 1),
                   round(float(b["robot_start_x_cm"]), 1), round(float(b["robot_start_y_cm"]), 1))
            xml = index.get(key)
            if xml is None:
                missing += 1
                continue
            scenes.append({
                "id": bid, "tier": tier, "axis": axis, "xml": xml,
                "kind": b["push_kind"], "obj": b["marker_hint"], "nb": int(b["n_bricks"]),
                "rate": float(b["solve_rate"]), "tried": int(b["tried"]),
                "v1": int(b["valid_1push"]), "vf": int(b["valid_first_push"]),
                "rc": int(b["n_contacts_reachable"]), "cc": int(b["n_contacts_cutoff"]),
                "xc": int(b["n_contacts_collision"]),
                "rows": [dict(r) for r in d["bricks"]] + [dict(b)],
            })
    if missing:
        print(f"WARNING: {missing} rows had no env.xml and were dropped", file=sys.stderr)
    return scenes


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Real-table scenes</title><style>
*{box-sizing:border-box}
body{margin:0;background:#eef1f5;color:#16202b;font:14px/1.5 system-ui,-apple-system,sans-serif}
header{position:sticky;top:0;z-index:9;background:#fff;border-bottom:1px solid #d3d9e2;padding:12px 18px}
h1{margin:0 0 2px;font-size:17px}
.sub{margin:0 0 10px;color:#6b7a8b;font-size:12.5px}
.rail{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end}
.g{display:flex;flex-direction:column;gap:4px}
.lab{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:#6b7a8b;font-weight:600}
.ch{display:flex;border:1px solid #d3d9e2;border-radius:5px;overflow:hidden}
.ch button{font:inherit;font-size:12px;padding:4px 10px;border:0;border-right:1px solid #d3d9e2;
  background:#fff;color:#3d4b5a;cursor:pointer}
.ch button:last-child{border-right:0}
.ch button.on{background:#e6eefa;color:#1d4e89;font-weight:600}
main{padding:16px 18px 110px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{background:#fff;border:1px solid #d3d9e2;border-radius:6px;overflow:hidden;cursor:pointer}
.card.on{border-color:#1d4e89;box-shadow:inset 0 0 0 1px #1d4e89}
.card img{display:block;width:100%;height:auto;background:#f7f9fb}
.top{display:flex;align-items:center;gap:7px;padding:7px 9px 5px;font-size:12.5px}
.id{font-family:ui-monospace,Menlo,monospace;font-weight:600}
.tick{width:14px;height:14px;border:1.5px solid #d3d9e2;border-radius:3px;display:grid;
  place-items:center;font-size:10px;color:transparent}
.card.on .tick{background:#1d4e89;border-color:#1d4e89;color:#fff}
.bars{margin-left:auto;display:flex;gap:2px}
.bars i{width:4px;height:10px;background:#d3d9e2;border-radius:1px}
.bars i.f{background:#3d4b5a}
.st{padding:5px 9px 8px;border-top:1px solid #e4e8ee;font-size:11.5px;color:#6b7a8b;
  display:flex;gap:9px;flex-wrap:wrap}
.st b{color:#16202b}
.k{font-size:9.5px;text-transform:uppercase;letter-spacing:.05em;font-weight:600;
  padding:1px 5px;border-radius:3px;background:#e4e8ee;color:#3d4b5a}
.k.c{background:#e6eefa;color:#1d4e89}
.tray{position:fixed;left:0;right:0;bottom:0;background:#fff;border-top:1px solid #d3d9e2;
  padding:10px 18px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
.tray .ids{flex:1;min-width:180px;font-family:ui-monospace,Menlo,monospace;font-size:11px;
  color:#6b7a8b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
button.a{font:inherit;font-size:12.5px;font-weight:600;padding:6px 12px;border:1px solid #d3d9e2;
  border-radius:5px;background:#fff;cursor:pointer}
button.a.p{background:#1d4e89;border-color:#1d4e89;color:#fff}
button.a:disabled{opacity:.4;cursor:default}
</style></head><body>
<header><h1>Real-table scenes</h1>
<p class="sub">__N__ scenes on the __AXIS__ axis that fit the 49.0 x 77.5 cm table. Tick what to
build, then copy the rows.</p>
<div class="rail">
  <div class="g"><span class="lab">Difficulty</span><div class="ch" id="f-tier"></div></div>
  <div class="g"><span class="lab">Pushes needed</span><div class="ch" id="f-kind"></div></div>
  <div class="g"><span class="lab">Object</span><div class="ch" id="f-obj"></div></div>
  <div class="g"><span class="lab">Bricks</span><div class="ch" id="f-nb"></div></div>
  <div class="g"><span class="lab">Order</span><div class="ch" id="f-sort"></div></div>
  <div class="g" style="margin-left:auto"><span class="lab">Showing</span>
    <span id="count" style="font-size:13px;font-weight:600"></span></div>
</div></header>
<main><div class="grid" id="grid"></div></main>
<div class="tray"><strong id="n">0 selected</strong><span class="ids" id="ids">Tick a scene.</span>
  <button class="a" id="clr" disabled>Clear</button>
  <button class="a" id="cid" disabled>Copy ids</button>
  <button class="a p" id="ccsv" disabled>Copy build sheet</button></div>
<script>
const S = __DATA__, BARS = {easy:1, med:2, hard:3};
const st = {tier:"all", kind:"all", obj:"all", nb:"all", sort:"id", sel:new Set()};
function chips(el, opts, k){ el.innerHTML="";
  opts.forEach(([v,t])=>{ const b=document.createElement("button"); b.textContent=t;
    if(st[k]===v) b.className="on"; b.onclick=()=>{st[k]=v; draw();}; el.appendChild(b); }); }
function vis(){ return S.filter(s=>
    (st.tier==="all"||s.tier===st.tier) && (st.kind==="all"||s.kind===st.kind) &&
    (st.obj==="all"||s.obj===st.obj) && (st.nb==="all"||String(s.nb)===st.nb))
  .sort((a,b)=> st.sort==="rate-"? b.rate-a.rate : st.sort==="rate+"? a.rate-b.rate
              : a.id.localeCompare(b.id)); }
function draw(){
  chips(document.getElementById("f-tier"),
    [["all","All"],["easy","Easy"],["med","Medium"],["hard","Hard"]],"tier");
  chips(document.getElementById("f-kind"),
    [["all","Any"],["one_push","One push"],["needs_2_chain","Needs two"]],"kind");
  chips(document.getElementById("f-obj"),[["all","Any"],["obj_1","obj_1"],["obj_4","obj_4"]],"obj");
  chips(document.getElementById("f-nb"),[["all","Any"],["1","1"],["2","2"],["3","3"]],"nb");
  chips(document.getElementById("f-sort"),
    [["id","Scene id"],["rate-","Most solving"],["rate+","Fewest solving"]],"sort");
  const list = vis(), g = document.getElementById("grid"); g.innerHTML="";
  document.getElementById("count").textContent = list.length + " of " + S.length;
  list.forEach(s=>{
    const d=document.createElement("div"); d.className="card"+(st.sel.has(s.id)?" on":"");
    d.innerHTML='<div class="top"><span class="tick">&check;</span>'
      +'<span class="id">'+s.id+'</span><span class="bars">'
      +[1,2,3].map(i=>'<i class="'+(i<=BARS[s.tier]?"f":"")+'"></i>').join("")+'</span></div>'
      +'<img loading="lazy" src="/api/render?id='+encodeURIComponent(s.id)+'" alt="'+s.id+'">'
      +'<div class="st"><span><b>'+Math.round(s.rate*100)+'%</b> of '+s.tried+'</span>'
      +'<span class="k'+(s.kind==="needs_2_chain"?" c":"")+'">'
      +(s.kind==="needs_2_chain"?"2-chain":"1 push")+'</span>'
      +'<span>'+s.nb+' brick'+(s.nb>1?"s":"")+'</span><span>'+s.obj+'</span></div>';
    d.onclick=()=>{ st.sel.has(s.id)?st.sel.delete(s.id):st.sel.add(s.id); draw(); };
    g.appendChild(d);
  });
  const n=st.sel.size;
  document.getElementById("n").textContent=n+" selected";
  document.getElementById("ids").textContent=n?[...st.sel].join("  "):"Tick a scene.";
  ["clr","cid","ccsv"].forEach(i=>document.getElementById(i).disabled=!n);
}
function csv(){
  const picked=S.filter(s=>st.sel.has(s.id)); if(!picked.length) return "";
  const cols=Object.keys(picked[0].rows[0]);
  return [cols.join(",")].concat(picked.flatMap(s=>s.rows.map(r=>
    cols.map(c=>{const v=String(r[c]); return v.includes(",")?'"'+v+'"':v;}).join(",")))).join("\\n");
}
function cp(t,b,msg){ navigator.clipboard.writeText(t).catch(()=>{
    const a=document.createElement("textarea"); a.value=t; document.body.appendChild(a);
    a.select(); document.execCommand("copy"); a.remove(); });
  const w=b.textContent; b.textContent=msg; setTimeout(()=>b.textContent=w,1300); }
document.getElementById("clr").onclick=()=>{st.sel.clear();draw();};
document.getElementById("cid").onclick=e=>cp([...st.sel].join("\\n"),e.target,"Copied");
document.getElementById("ccsv").onclick=e=>cp(csv(),e.target,"Copied CSV");
draw();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/":
            scenes = self.server.scenes
            slim = [{k: s[k] for k in
                     ("id", "tier", "kind", "obj", "nb", "rate", "tried", "v1", "vf",
                      "rc", "cc", "xc", "rows")} for s in scenes]
            body = (PAGE.replace("__DATA__", json.dumps(slim))
                        .replace("__N__", str(len(scenes)))
                        .replace("__AXIS__", self.server.axis))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())
        elif p.path == "/api/render":
            self.render(parse_qs(p.query).get("id", [""])[0])
        else:
            self.send_error(404)

    def render(self, sid):
        s = self.server.by_id.get(sid)
        if s is None:
            self.send_error(404, "unknown scene")
            return
        png = self.server.cache.get(sid)
        if png is None:
            try:
                img = visualize_environment(s["xml"], resolution=self.server.resolution,
                                            wall_color="grey")
            except Exception as exc:                       # a bad scene must not kill the browse
                print(f"render failed for {sid}: {exc}", file=sys.stderr)
                self.send_error(500, "render failed")
                return
            if img is None:
                self.send_error(500, "render returned nothing")
                return
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png = buf.getvalue()
            self.server.cache[sid] = png
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.end_headers()
        self.wfile.write(png)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--axis", choices=("hmax2", "1push"), default="hmax2",
                    help="hmax2 is the search we deploy; 1push is the same scenes tiered on "
                         "single-push solve rate alone")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resolution", type=int, default=420)
    ap.add_argument("--pools", nargs="+",
                    default=[os.path.join(os.environ.get("NAMO_SCRATCH", "/tmp"),
                                          "real_buildable")],
                    help="dirs holding the generated pools with build_sheets.json")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    scenes = load_scenes(args.axis, args.pools)
    if not scenes:
        sys.exit(f"no scenes found. checked pools: {args.pools}")

    srv = HTTPServer(("localhost", args.port), Handler)
    srv.scenes = scenes
    srv.by_id = {s["id"]: s for s in scenes}
    srv.cache = {}
    srv.axis = args.axis
    srv.resolution = args.resolution
    url = f"http://localhost:{args.port}"
    counts = {t: sum(1 for s in scenes if s["tier"] == t) for t in ("easy", "med", "hard")}
    print(f"{len(scenes)} scenes on the {args.axis} axis  "
          f"(easy {counts['easy']}, med {counts['med']}, hard {counts['hard']})")
    print(f"serving {url}   ctrl-c to stop")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
