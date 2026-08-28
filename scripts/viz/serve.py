#!/usr/bin/env python3
"""Static server for the viz pages that never lets a browser cache the assets.

`python -m http.server` sends no cache headers, so browsers apply heuristic freshness: they
revalidate the HTML but happily reuse a css/js file for minutes to hours. Editing the page then
shows new markup in the old layout with the old behaviour, which reads as a broken page rather than
a stale one. Everything here is served no-store.

    python scripts/viz/serve.py --root $NAMO_SCRATCH/viz --port 8899

Serving the parent viz/ directory puts both tools under one port:
    /scenes/scenes.html   the scene gallery
    /search/index.html    the search trace replay

One write is allowed, POST <dataset>/stars.json, which is where the gallery keeps its starred
shortlist. Nothing else here is writable.
"""
import argparse
import functools
import http.server
import json
import os
import socketserver


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass          # one line per card fetch is thousands of lines of noise

    # A shortlist kept only in localStorage is per browser AND per port: tunnel in from a second
    # laptop, or move the server off 8899, and the stars are gone with nothing on screen saying they
    # ever existed. The gallery POSTs its whole star map here after every toggle. Only the name
    # stars.json is writable, and only inside a dataset folder that already exists, so this stays a
    # static server that happens to keep one file.
    def do_POST(self):
        path = self.translate_path(self.path.split("?", 1)[0])
        if os.path.basename(path) != "stars.json" or not os.path.isdir(os.path.dirname(path)):
            self.send_error(403, "only <existing dataset>/stars.json is writable")
            return
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            json.loads(body)
        except ValueError:
            self.send_error(400, "body is not JSON")  # never store what the page cannot read back
            return
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, path)   # atomic: a concurrent read gets the old list or the new one
        self.send_response(204)
        self.end_headers()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--bind", default="127.0.0.1", help="loopback by default: tunnel in, do not "
                                                        "expose the page on a shared box")
    a = ap.parse_args()
    handler = functools.partial(NoCacheHandler, directory=a.root)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((a.bind, a.port), handler) as httpd:
        print(f"serving {a.root} on http://{a.bind}:{a.port} (no-store)", flush=True)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
