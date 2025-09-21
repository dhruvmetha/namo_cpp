import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os
import math
import shutil
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import json
import threading
import webbrowser
import time
import signal
import sys
import pickle
import argparse

def visualize_environment(xml_file_path, resolution=800, wall_color="white"):
    """
    Create overhead view image and save to file, return the image path.
    """
    # Color mapping
    colors = {
        'robot': (0, 102, 255),      # Blue
        'goal': (0, 255, 0),         # Green  
        'wall': (255, 255, 255) if wall_color == "white" else (255, 0, 0),  # White or Red
        'obstacle': (255, 212, 0),   # Yellow
        'floor': (240, 240, 240),    # Light gray background
        'default': (200, 200, 200)   # Default gray
    }
    
    try:
        # Parse XML
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Extract geoms and sites
        geoms = []
        sites = []
        
        # Find all geoms in worldbody
        worldbody = root.find('worldbody')
        if worldbody is not None:
            for geom in worldbody.iter('geom'):
                geoms.append(geom)
            for site in worldbody.iter('site'):
                sites.append(site)
        
        # Convert to drawable primitives
        primitives = []
        
        # Process geoms
        for geom in geoms:
            geom_data = parse_geom(geom, colors)
            if geom_data:
                primitives.append(geom_data)
        
        # Process sites (goal)
        for site in sites:
            site_data = parse_site(site, colors)
            if site_data:
                primitives.append(site_data)
        
        if not primitives:
            print(f"Warning: No drawable elements found in {xml_file_path}")
            return None
        
        # Calculate bounds
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        
        for prim in primitives:
            bounds = get_primitive_bounds(prim)
            min_x = min(min_x, bounds[0])
            max_x = max(max_x, bounds[1])
            min_y = min(min_y, bounds[2])
            max_y = max(max_y, bounds[3])
        
        # Add padding (5%)
        padding = 0.05
        width = max_x - min_x
        height = max_y - min_y
        pad_x = width * padding
        pad_y = height * padding
        
        min_x -= pad_x
        max_x += pad_x
        min_y -= pad_y
        max_y += pad_y
        
        # Calculate scale
        world_width = max_x - min_x
        world_height = max_y - min_y
        scale = (resolution * 0.9) / max(world_width, world_height)
        
        # Create image
        img = Image.new('RGB', (resolution, resolution), colors['floor'])
        draw = ImageDraw.Draw(img)
        
        # Sort primitives by z-order
        z_order = {'floor': 0, 'wall': 1, 'obstacle': 2, 'robot': 3, 'goal': 4, 'default': 2}
        primitives.sort(key=lambda p: z_order.get(p['type'], 2))
        
        # Draw primitives
        for prim in primitives:
            draw_primitive(draw, prim, min_x, max_y, scale, resolution)
        
        return img
        
    except Exception as e:
        print(f"Error processing {xml_file_path}: {e}")
        return None

# Keep all your existing helper functions (parse_geom, parse_site, etc.)
def parse_geom(geom, colors):
    """Parse a geom element into drawable primitive data."""
    name = geom.get('name', '')
    geom_type = geom.get('type', 'sphere')
    pos_str = geom.get('pos', '0 0 0')
    size_str = geom.get('size', '0.1 0.1 0.1')
    euler_str = geom.get('euler', '0 0 0')
    
    if geom_type == 'plane':
        return None
    
    pos = [float(x) for x in pos_str.split()]
    x, y = pos[0], pos[1]
    size = [float(x) for x in size_str.split()]
    euler = [float(x) for x in euler_str.split()]
    yaw = math.radians(euler[2]) if len(euler) > 2 else 0
    
    color_type = 'default'
    if name == 'robot' or 'robot' in name.lower():
        color_type = 'robot'
    elif name.startswith('wall') or 'wall' in name.lower():
        color_type = 'wall'
    elif 'obstacle' in name.lower():
        color_type = 'obstacle'
    
    if geom_type == 'sphere':
        return {
            'shape': 'circle',
            'type': color_type,
            'x': x, 'y': y,
            'radius': size[0],
            'color': colors[color_type]
        }
    elif geom_type in ['box', 'capsule', 'cylinder']:
        width = 2 * size[0] if len(size) > 0 else 0.2
        height = 2 * size[1] if len(size) > 1 else 0.2
        return {
            'shape': 'rectangle',
            'type': color_type,
            'x': x, 'y': y,
            'width': width, 'height': height,
            'yaw': yaw,
            'color': colors[color_type]
        }
    return None

def parse_site(site, colors):
    """Parse a site element (mainly for goal)."""
    name = site.get('name', '')
    if name != 'goal':
        return None
    
    pos_str = site.get('pos', '0 0 0')
    size_str = site.get('size', '0.25 0.25 0.25')
    
    pos = [float(x) for x in pos_str.split()]
    size = [float(x) for x in size_str.split()]
    
    return {
        'shape': 'circle',
        'type': 'goal',
        'x': pos[0], 'y': pos[1],
        'radius': size[0],
        'color': colors['goal']
    }

def get_primitive_bounds(prim):
    """Get bounding box of a primitive."""
    if prim['shape'] == 'circle':
        r = prim['radius']
        return (prim['x'] - r, prim['x'] + r, prim['y'] - r, prim['y'] + r)
    elif prim['shape'] == 'rectangle':
        w, h = prim['width'] / 2, prim['height'] / 2
        cx, cy = prim['x'], prim['y']
        yaw = prim['yaw']
        
        corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        x_coords, y_coords = [], []
        
        for dx, dy in corners:
            rx = dx * cos_yaw - dy * sin_yaw
            ry = dx * sin_yaw + dy * cos_yaw
            x_coords.append(cx + rx)
            y_coords.append(cy + ry)
        
        return (min(x_coords), max(x_coords), min(y_coords), max(y_coords))
    
    return (0, 0, 0, 0)

def draw_primitive(draw, prim, min_x, max_y, scale, resolution):
    """Draw a primitive on the PIL ImageDraw canvas."""
    margin = resolution * 0.05
    
    def world_to_pixel(x, y):
        px = (x - min_x) * scale + margin
        py = (max_y - y) * scale + margin
        return px, py
    
    if prim['shape'] == 'circle':
        cx, cy = world_to_pixel(prim['x'], prim['y'])
        r_pixels = prim['radius'] * scale
        bbox = (cx - r_pixels, cy - r_pixels, cx + r_pixels, cy + r_pixels)
        draw.ellipse(bbox, fill=prim['color'])
        
    elif prim['shape'] == 'rectangle':
        w, h = prim['width'] / 2, prim['height'] / 2
        cx, cy = prim['x'], prim['y']
        yaw = prim['yaw']
        
        corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        
        pixel_corners = []
        for dx, dy in corners:
            rx = dx * cos_yaw - dy * sin_yaw
            ry = dx * sin_yaw + dy * cos_yaw
            world_x = cx + rx
            world_y = cy + ry
            px, py = world_to_pixel(world_x, world_y)
            pixel_corners.append((px, py))
        
        draw.polygon(pixel_corners, fill=prim['color'])

def get_envs_from_pickle(pickle_file_path: str) -> list[str]:
    """Read XML file paths from a pickle file."""
    import pickle
    
    if not os.path.exists(pickle_file_path):
        print(f"Warning: Pickle file {pickle_file_path} does not exist")
        return []
    
    try:
        with open(pickle_file_path, 'rb') as f:
            xml_files = pickle.load(f)
        print(xml_files)
        return xml_files
    except Exception as e:
        print(f"Error reading pickle file {pickle_file_path}: {e}")
        return []

# Web interface
class EnvironmentHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, input_pickle=None, accept_pickle=None, **kwargs):
        self.input_pickle = input_pickle
        self.accept_pickle = accept_pickle
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Environment Selection</title>
                <style>
                    body { font-family: Arial, sans-serif; text-align: center; margin: 20px; }
                    .container { max-width: 900px; margin: 0 auto; }
                    .environment { margin: 20px 0; padding: 20px; border: 1px solid #ccc; }
                    .environment img { max-width: 600px; height: auto; }
                    .buttons { margin: 20px 0; }
                    .accept { background: green; color: white; padding: 15px 30px; font-size: 18px; margin: 10px; }
                    .reject { background: red; color: white; padding: 15px 30px; font-size: 18px; margin: 10px; }
                    .status { margin: 10px 0; font-weight: bold; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Environment Selection Tool</h1>
                    <div id="content">Loading...</div>
                </div>
                
                <script>
                    let environments = [];
                    let currentIndex = 0;
                    
                    async function loadEnvironments() {
                        const response = await fetch('/environments');
                        environments = await response.json();
                        showCurrentEnvironment();
                    }

                    function showCurrentEnvironment() {
                        if (currentIndex >= environments.length) {
                            document.getElementById('content').innerHTML = '<h2>All environments processed! Shutting down...</h2>';
                            // Send shutdown signal to server
                            fetch('/shutdown', {method: 'POST'});
                            setTimeout(() => window.close(), 2000);
                            return;
                        }

                        const env = environments[currentIndex];
                        document.getElementById('content').innerHTML = `
                            <div class="environment">
                                <h2>${env.name}</h2>
                                <div class="status">Environment ${currentIndex + 1} of ${environments.length}</div>
                                <img src="/image/${encodeURIComponent(env.filename)}" alt="${env.name}">
                                <div class="buttons">
                                    <button class="accept" onclick="selectEnvironment('accept')">ACCEPT</button>
                                    <button class="reject" onclick="selectEnvironment('reject')">REJECT</button>
                                </div>
                            </div>
                        `;
                    }
                    
                    async function selectEnvironment(choice) {
                        const env = environments[currentIndex];
                        await fetch('/select', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({filename: env.filename, choice: choice})
                        });
                        
                        currentIndex++;
                        showCurrentEnvironment();
                    }
                    
                    loadEnvironments();
                </script>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
            
        elif self.path == '/environments':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # Get list of environments
            xml_files = get_envs_from_pickle(self.input_pickle)
            envs = []
            
            for xml_file in xml_files:
                img = visualize_environment(xml_file)
                if img:
                    envs.append({
                        'filename': xml_file,
                        'name': os.path.splitext(os.path.basename(xml_file))[0]
                    })
            
            self.wfile.write(json.dumps(envs).encode())
            
        elif self.path.startswith('/image/'):
            # Extract filename from URL
            filename = urllib.parse.unquote(self.path[7:])  # Remove '/image/' prefix
            
            try:
                img = visualize_environment(filename)
                if img:
                    # Convert PIL image to bytes
                    from io import BytesIO
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/png')
                    self.end_headers()
                    self.wfile.write(img_bytes.getvalue())
                else:
                    self.send_error(404)
            except Exception as e:
                print(f"Error generating image: {e}")
                self.send_error(500)
            
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/select':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            filename = data['filename']
            choice = data['choice']
            
            basename = os.path.basename(filename)
            
            # Only handle accept - ignore reject completely
            if choice == 'accept':
                print(f"✓ Accepted: {basename}")
                
                # Load existing accepted environments or create new list
                try:
                    with open(self.accept_pickle, 'rb') as f:
                        file_list = pickle.load(f)
                except (FileNotFoundError, EOFError):
                    file_list = []

                # Add current file and save back
                file_list.append(filename)
                with open(self.accept_pickle, 'wb') as f:
                    pickle.dump(file_list, f)
            else:
                # For reject, just print and do nothing else
                print(f"✗ Rejected: {basename}")
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
        elif self.path == '/shutdown':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "shutting down"}')
            
            print("\nAll environments processed. Shutting down...")
            # Shutdown the server after a brief delay
            threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()

def start_web_interface(input_pickle: str, accept_pickle: str):
    """Start the web interface"""
    def signal_handler(sig, frame):
        print('\nShutting down...')
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    
    print("Starting web interface...")
    print("Open your browser and go to: http://localhost:8000")
    print("The server will automatically shut down when all environments are processed.")
    
    try:
        handler = lambda *args, **kwargs: EnvironmentHandler(*args, 
                                                     input_pickle=input_pickle,
                                                     accept_pickle=accept_pickle,
                                                     **kwargs)
        server = HTTPServer(('localhost', 8000), handler)
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down...')
        sys.exit(0)

if __name__ == "__main__":

    # input command from namo_cpp folder should look like: "python python/environment_selection.py train_envs_new_path/envs_names_hard.pkl"
    parser = argparse.ArgumentParser(description='Environment selection web interface')
    parser.add_argument('input_pickle', help='Path to input pickle file (e.g., train_envs_new_path/envs_names_very_hard.pkl)')
    
    args = parser.parse_args()

    # Extract filename from input path and create output path
    # Will save to the coresponding level of difficulty dir in /common/users/tdn39/Robotics/Mujoco/namo_cpp/train_envs_accepted
    input_filename = os.path.basename(args.input_pickle)
    accept_pickle = os.path.join("train_envs_accepted", input_filename)

    # Create output directory if it doesn't exist
    os.makedirs("train_envs_accepted", exist_ok=True)
    
    start_web_interface(args.input_pickle, accept_pickle)