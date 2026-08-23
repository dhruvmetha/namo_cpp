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
"""
import argparse
import functools
import http.server
import socketserver


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass          # one line per card fetch is thousands of lines of noise


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
