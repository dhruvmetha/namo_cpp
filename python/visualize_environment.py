#!/usr/bin/env python3
"""
Fast Environment Viewer

A lightweight web-based viewer for browsing MuJoCo maze environments.
Uses the same image rendering as environment_selection.py but with
a simple next/previous interface.

Usage:
    python3 visualize_environment.py                    # View generated templates
    python3 visualize_environment.py --port 8080        # Custom port
    python3 visualize_environment.py --dir /path/to/xml  # Custom directory
"""

import argparse
import glob
import json
import os
import sys
import threading
import time
import webbrowser
import xml.etree.ElementTree as ET
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Import the visualization function from environment_selection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment_selection import (
    get_primitive_bounds,
    parse_geom,
    parse_site,
    visualize_environment,
)


def _colors_for_wall_color(wall_color: str):
    wall_rgb = (160, 160, 160)
    if wall_color == "white":
        wall_rgb = (255, 255, 255)
    elif wall_color == "red":
        wall_rgb = (255, 0, 0)
    elif wall_color == "grey":
        wall_rgb = (160, 160, 160)

    return {
        "robot": (0, 0, 255),
        "goal": (255, 0, 0),
        "wall": wall_rgb,
        "obstacle": (255, 255, 0),
        "target": (0, 255, 255),
        "floor": (255, 255, 255),
        "default": (200, 200, 200),
        "outline": (0, 0, 0),
        "robot_outline": (255, 255, 255),
        "halo": (255, 255, 255),
    }


def _compute_view_transform(xml_file_path: str, resolution: int, wall_color: str):
    """
    Recompute the same world->pixel transform used by environment_selection.visualize_environment
    so we can place text labels in the correct location.
    """
    colors = _colors_for_wall_color(wall_color)

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    geoms = []
    sites = []
    worldbody = root.find("worldbody")
    if worldbody is not None:
        for geom in worldbody.iter("geom"):
            geoms.append(geom)
        for site in worldbody.iter("site"):
            sites.append(site)

    primitives = []
    for geom in geoms:
        geom_data = parse_geom(geom, colors)
        if geom_data:
            primitives.append(geom_data)
    for site in sites:
        site_data = parse_site(site, colors)
        if site_data:
            primitives.append(site_data)

    if not primitives:
        return None

    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for prim in primitives:
        bounds = get_primitive_bounds(prim)
        min_x = min(min_x, bounds[0])
        max_x = max(max_x, bounds[1])
        min_y = min(min_y, bounds[2])
        max_y = max(max_y, bounds[3])

    padding = 0.05
    width = max_x - min_x
    height = max_y - min_y
    min_x -= width * padding
    max_x += width * padding
    min_y -= height * padding
    max_y += height * padding

    world_width = max_x - min_x
    world_height = max_y - min_y
    scale = (resolution * 0.9) / max(world_width, world_height)

    margin = resolution * 0.05

    def world_to_pixel(x, y):
        px = (x - min_x) * scale + margin
        py = (max_y - y) * scale + margin
        return px, py

    return world_to_pixel


def _iter_movable_obstacle_labels(xml_file_path: str):
    """
    Yield (name, x, y) for obstacles that should be labeled.

    Per user request: "any obstacle that doesn't have wall in the name".
    We treat geoms with 'obstacle' in the name (and without 'wall') as label candidates.
    """
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        return

    def traverse(node, inherited_label: str | None = None):
        if node.tag == "body":
            body_name = node.get("name", "") or ""
            body_lower = body_name.lower()
            if ("obstacle" in body_lower) and ("wall" not in body_lower):
                inherited_label = body_name

        for child in list(node):
            if child.tag == "body":
                yield from traverse(child, inherited_label=inherited_label)
                continue
            if child.tag != "geom":
                continue

            geom_name = child.get("name", "") or ""
            label_name = geom_name or (inherited_label or "")
            lowered = label_name.lower()
            if "obstacle" not in lowered:
                continue
            if "wall" in lowered:
                continue

            pos_str = child.get("pos", "0 0 0")
            try:
                pos = [float(v) for v in pos_str.split()]
            except Exception:
                continue
            if len(pos) < 2:
                continue
            yield label_name, pos[0], pos[1]

    yield from traverse(worldbody)


def add_movable_obstacle_labels(img, xml_file_path: str, wall_color: str = "grey"):
    """Annotate the rendered overhead view with movable-obstacle labels."""
    from PIL import ImageDraw, ImageFont

    resolution = img.size[0]
    world_to_pixel = _compute_view_transform(xml_file_path, resolution, wall_color)
    if world_to_pixel is None:
        return img

    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    def text_bbox(text: str):
        if hasattr(draw, "textbbox"):
            x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
            return x1 - x0, y1 - y0
        return draw.textsize(text, font=font)

    for name, x, y in _iter_movable_obstacle_labels(xml_file_path):
        px, py = world_to_pixel(x, y)
        label = name

        tw, th = text_bbox(label)
        x0 = px - tw / 2
        y0 = py - th / 2

        pad = 2
        draw.rectangle(
            (x0 - pad, y0 - pad, x0 + tw + pad, y0 + th + pad),
            fill=(0, 0, 0),
            outline=(255, 255, 255),
            width=1,
        )
        draw.text((x0, y0), label, fill=(255, 255, 255), font=font)

    return img


class EnvironmentViewerHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        """Suppress default HTTP server logging."""
        pass

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query = parse_qs(parsed_path.query)
        
        if path == '/':
            self.serve_main_page()
        elif path == '/api/environments':
            self.serve_environment_list()
        elif path == '/api/render':
            self.serve_environment_image(query)
        else:
            self.send_error(404)

    def serve_main_page(self):
        """Serve the main HTML page."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Environment Viewer</title>
    <style>
        body {
            font-family: system-ui, -apple-system, sans-serif;
            margin: 0;
            padding: 20px;
            background: #0b0d10;
            color: #e6edf3;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }
        
        .container {
            max-width: 900px;
            width: 100%;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .counter {
            color: #8b949e;
            margin-bottom: 10px;
        }
        
        .filename {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            word-break: break-all;
        }
        
        .image-container {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin-bottom: 30px;
        }
        
        .environment-image {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            background: #0d1117;
        }
        
        .loading {
            color: #8b949e;
            padding: 60px;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            align-items: center;
        }
        
        button {
            background: #238636;
            border: none;
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
        }
        
        button:hover {
            background: #2ea043;
        }
        
        button:disabled {
            background: #30363d;
            cursor: not-allowed;
        }
        
        .jump-controls {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .jump-input {
            background: #0d1117;
            border: 1px solid #30363d;
            color: #e6edf3;
            padding: 8px 12px;
            border-radius: 6px;
            width: 60px;
            text-align: center;
        }
        
        .error {
            color: #f85149;
            text-align: center;
            padding: 40px;
        }
        
        .controls-grid {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            align-items: center;
            gap: 20px;
            width: 100%;
        }
        
        .nav-left { justify-self: start; }
        .nav-center { justify-self: center; }
        .nav-right { justify-self: end; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="counter" id="counter">Loading...</div>
            <div class="filename" id="filename"></div>
        </div>
        
        <div class="image-container">
            <div class="loading" id="loading">Loading environments...</div>
            <img class="environment-image" id="image" style="display: none;" />
            <div class="error" id="error" style="display: none;"></div>
        </div>
        
        <div class="controls-grid">
            <div class="nav-left">
                <button id="prevBtn" onclick="navigate(-1)">← Previous</button>
            </div>
            
            <div class="nav-center">
                <div class="jump-controls">
                    <span>Go to:</span>
                    <input type="number" class="jump-input" id="jumpInput" min="1" />
                    <button onclick="jumpToEnvironment()">Go</button>
                </div>
            </div>
            
            <div class="nav-right">
                <button id="nextBtn" onclick="navigate(1)">Next →</button>
            </div>
        </div>
    </div>

    <script>
        let environments = [];
        let currentIndex = 0;
        
        // Load environment list
        async function loadEnvironments() {
            try {
                const response = await fetch('/api/environments');
                environments = await response.json();
                
                if (environments.length === 0) {
                    showError('No XML environments found!');
                    return;
                }
                
                document.getElementById('jumpInput').max = environments.length;
                loadCurrentEnvironment();
            } catch (error) {
                showError('Failed to load environments: ' + error.message);
            }
        }
        
        // Load and display current environment
        async function loadCurrentEnvironment() {
            if (environments.length === 0) return;
            
            updateUI();
            showLoading();
            
            try {
                const filename = environments[currentIndex];
                const response = await fetch(`/api/render?file=${encodeURIComponent(filename)}`);
                
                if (response.ok) {
                    const blob = await response.blob();
                    const imageUrl = URL.createObjectURL(blob);
                    showImage(imageUrl);
                } else {
                    showError('Failed to render environment');
                }
            } catch (error) {
                showError('Error loading environment: ' + error.message);
            }
        }
        
        // Navigation
        function navigate(delta) {
            if (environments.length === 0) return;
            currentIndex = (currentIndex + delta + environments.length) % environments.length;
            loadCurrentEnvironment();
        }
        
        function jumpToEnvironment() {
            const input = document.getElementById('jumpInput');
            const index = parseInt(input.value) - 1;
            if (index >= 0 && index < environments.length) {
                currentIndex = index;
                loadCurrentEnvironment();
            }
        }
        
        // UI updates
        function updateUI() {
            const counter = document.getElementById('counter');
            const filename = document.getElementById('filename');
            const prevBtn = document.getElementById('prevBtn');
            const nextBtn = document.getElementById('nextBtn');
            const jumpInput = document.getElementById('jumpInput');
            
            counter.textContent = `${currentIndex + 1} of ${environments.length}`;
            filename.textContent = environments[currentIndex] || '';
            jumpInput.value = currentIndex + 1;
            
            prevBtn.disabled = environments.length <= 1;
            nextBtn.disabled = environments.length <= 1;
        }
        
        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('image').style.display = 'none';
            document.getElementById('error').style.display = 'none';
        }
        
        function showImage(imageUrl) {
            const img = document.getElementById('image');
            img.src = imageUrl;
            img.style.display = 'block';
            document.getElementById('loading').style.display = 'none';
            document.getElementById('error').style.display = 'none';
        }
        
        function showError(message) {
            document.getElementById('error').textContent = message;
            document.getElementById('error').style.display = 'block';
            document.getElementById('loading').style.display = 'none';
            document.getElementById('image').style.display = 'none';
        }
        
        // Keyboard navigation
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
                navigate(-1);
                e.preventDefault();
            } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
                navigate(1);
                e.preventDefault();
            } else if (e.key === 'Home') {
                currentIndex = 0;
                loadCurrentEnvironment();
                e.preventDefault();
            } else if (e.key === 'End') {
                currentIndex = environments.length - 1;
                loadCurrentEnvironment();
                e.preventDefault();
            }
        });
        
        // Initialize
        loadEnvironments();
    </script>
</body>
</html>"""
        
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())

    def serve_environment_list(self):
        """Serve the list of available XML files."""
        xml_files = []
        for pattern in self.server.xml_patterns:
            xml_files.extend(glob.glob(pattern))
        
        # Sort and deduplicate by full path first
        xml_files = sorted(list(set(xml_files)))
        
        # Create filename->path mapping
        self.server.xml_file_map = {}
        filenames = []
        
        # Handle duplicate basenames by keeping only the first occurrence
        seen_basenames = set()
        for full_path in xml_files:
            basename = os.path.basename(full_path)
            if basename not in seen_basenames:
                seen_basenames.add(basename)
                filenames.append(basename)
                self.server.xml_file_map[basename] = full_path
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(filenames).encode())

    def serve_environment_image(self, query):
        """Render and serve an environment image."""
        filename = query.get('file', [''])[0]
        if not filename:
            self.send_error(400, 'Missing file parameter')
            return
            
        full_path = getattr(self.server, 'xml_file_map', {}).get(filename)
        if not full_path or not os.path.exists(full_path):
            self.send_error(404, 'File not found')
            return
            
        try:
            # Render the environment image
            resolution = int(getattr(self.server, "resolution", 800))
            wall_color = getattr(self.server, "wall_color", "grey")
            img = visualize_environment(full_path, resolution=resolution, wall_color=wall_color)
            if img is None:
                self.send_error(500, 'Failed to render environment')
                return

            # Label movable obstacles (anything with 'obstacle' in the name and without 'wall')
            img = add_movable_obstacle_labels(img, full_path, wall_color=wall_color)

            # Convert to PNG bytes
            import io

            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', str(len(img_bytes.getvalue())))
            self.end_headers()
            self.wfile.write(img_bytes.getvalue())

        except Exception as e:
            print(f"Error rendering {full_path}: {e}")
            self.send_error(500, f'Rendering error: {str(e)}')


def find_xml_files(directory):
    """Find XML files in directory."""
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, "*.xml")]


def start_server(port, xml_patterns, *, wall_color: str = "grey", resolution: int = 800):
    """Start the HTTP server."""
    class CustomHTTPServer(HTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.xml_patterns = xml_patterns
            self.xml_file_map = {}
            self.wall_color = wall_color
            self.resolution = resolution
    
    server = CustomHTTPServer(('localhost', port), EnvironmentViewerHandler)
    server.xml_patterns = xml_patterns
    
    print(f"Starting Environment Viewer on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    
    # Open browser after a short delay
    threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description='Fast Environment Viewer - Browse MuJoCo maze environments')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--dir', type=str, help='Directory containing XML files (default: ../generated_templates)')
    parser.add_argument(
        "--wall-color",
        type=str,
        default="grey",
        choices=["grey", "white", "red"],
        help="Wall color for rendering (default: grey). Note: white walls blend into the white floor.",
    )
    parser.add_argument("--resolution", type=int, default=800, help="Render resolution in pixels (default: 800)")
    
    args = parser.parse_args()
    
    # Default to generated_templates directory
    if args.dir:
        xml_patterns = [os.path.join(args.dir, "*.xml")]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_dir = os.path.join(os.path.dirname(script_dir), "generated_templates")
        xml_patterns = [
            os.path.join(default_dir, "*.xml"),
            os.path.join(script_dir, "../generated_templates", "*.xml"),
            os.path.join(script_dir, "generated_templates", "*.xml"),
        ]
    
    # Check if any XML files exist
    total_files = 0
    for pattern in xml_patterns:
        total_files += len(glob.glob(pattern))
    
    if total_files == 0:
        print("No XML files found!")
        print("Searched in patterns:", xml_patterns)
        print("\nGenerate some environments first with:")
        print("  python3 template_generation.py --num_mazes 5")
        return 1
    
    print(f"Found {total_files} XML files")
    start_server(args.port, xml_patterns, wall_color=args.wall_color, resolution=args.resolution)
    return 0


if __name__ == "__main__":
    sys.exit(main())
