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
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Import the visualization function from environment_selection
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment_selection import visualize_environment


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
            img = visualize_environment(full_path, resolution=800, wall_color="white")
            if img is None:
                self.send_error(500, 'Failed to render environment')
                return
                
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


def start_server(port, xml_patterns):
    """Start the HTTP server."""
    class CustomHTTPServer(HTTPServer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.xml_patterns = xml_patterns
            self.xml_file_map = {}
    
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
    start_server(args.port, xml_patterns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
