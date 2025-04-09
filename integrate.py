#!/usr/bin/env python3
"""
Integration script to connect the React frontend with the Flask backend.
This script builds the React frontend and configures it to work with the backend.
"""

import os
import sys
import subprocess
import shutil
import logging
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    logger.info(f"Original PATH: {os.environ['PATH']}")

    # Check Python dependencies
    python_deps = ['flask', 'flask-cors', 'numpy', 'pandas', 'matplotlib', 'scipy', 'scikit-learn']
    missing_py_deps = []

    for dep in python_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_py_deps.append(dep)

    if missing_py_deps:
        logger.warning(f"Missing Python dependencies: {', '.join(missing_py_deps)}")
        logger.info("Installing missing Python dependencies...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_py_deps, check=True)
            logger.info("Python dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing Python dependencies: {e}")
            logger.info("Please install them manually with: pip install " + " ".join(missing_py_deps))

    # Forcefully prepend known node paths if npm is not found
    if not shutil.which("npm") or not shutil.which("node"):
        known_node_paths = [
            r"C:\Program Files\nodejs",
            r"C:\Users\gusta\AppData\Roaming\npm"
        ]
        os.environ["PATH"] = os.pathsep.join(known_node_paths + [os.environ["PATH"]])
        logger.info(f"Updated PATH: {os.environ['PATH']}")

    # Check Node.js and npm again
    node_installed = shutil.which("node") is not None
    npm_installed = shutil.which("npm") is not None

    if not node_installed:
        logger.warning("Node.js is not installed or not in PATH")
    if not npm_installed:
        logger.warning("npm is not installed or not in PATH")

    if not node_installed or not npm_installed:
        logger.error("Node.js and npm are required to build the frontend")
        logger.info("Please install Node.js and npm from https://nodejs.org/")
        logger.info("After installation, make sure they are added to your PATH")
        logger.info("You can verify installation with 'node --version' and 'npm --version'")
        return False

    logger.info(f"Detected node at: {shutil.which('node')}")
    logger.info(f"Detected npm at: {shutil.which('npm')}")
    return True


def build_frontend():
    """Build the React frontend."""
    logger.info("Building React frontend...")

    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'frontend')

    if not os.path.exists(frontend_dir):
        logger.error(f"Frontend directory not found: {frontend_dir}")
        return False

    npm_path = shutil.which("npm")  # full path to npm.cmd

    try:
        logger.info(f"Using npm path: {npm_path}")

        logger.info("Installing frontend dependencies...")
        subprocess.run([npm_path, 'install'], cwd=frontend_dir, check=True)

        logger.info("Building frontend production build...")
        subprocess.run([npm_path, 'run', 'build'], cwd=frontend_dir, check=True)

        build_dir = os.path.join(frontend_dir, 'build')
        if not os.path.exists(build_dir):
            logger.error(f"Build directory not found: {build_dir}")
            return False

        logger.info("Frontend build completed successfully")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Error building frontend: {e}")
        return False
    except FileNotFoundError as e:
        logger.error(f"Command not found: {e}")
        logger.error("Make sure Node.js and npm are installed and in your PATH")
        return False

def setup_backend():
    """Setup the Flask backend to serve the React frontend."""
    logger.info("Setting up Flask backend...")
    
    frontend_build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'frontend', 'build')
    backend_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'frontend', 'build')
    
    # Check if frontend build directory exists
    if not os.path.exists(frontend_build_dir):
        logger.error(f"Frontend build directory not found: {frontend_build_dir}")
        return False
    
    try:
        # Ensure backend static directory exists
        os.makedirs(os.path.dirname(backend_static_dir), exist_ok=True)
        
        logger.info("Backend setup completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"Error setting up backend: {e}")
        return False

def install_python_dependencies():
    """Install required Python dependencies."""
    logger.info("Installing Python dependencies...")
    
    try:
        # Install required packages
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'flask', 'flask-cors', 'numpy', 'pandas', 
            'matplotlib', 'scipy', 'scikit-learn'
        ], check=True)
        
        logger.info("Python dependencies installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error installing Python dependencies: {e}")
        return False

def create_dummy_frontend():
    """Create a minimal dummy frontend if Node.js/npm is not available."""
    logger.info("Creating minimal frontend without Node.js/npm...")
    
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard', 'frontend')
    build_dir = os.path.join(frontend_dir, 'build')
    
    # Create build directory if it doesn't exist
    os.makedirs(build_dir, exist_ok=True)
    
    # Create a minimal index.html file - using ASCII-compatible characters only
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EVE & EVS Simulation Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #121212;
            color: white;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }
        h1 {
            color: #3f51b5;
        }
        p {
            line-height: 1.6;
        }
        .card {
            background-color: #2d2d2d;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .button {
            background-color: #3f51b5;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-top: 10px;
        }
        .button:hover {
            background-color: #303f9f;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>EVE & EVS Simulation Dashboard</h1>
        <div class="card">
            <h2>Backend API Only Mode</h2>
            <p>You are viewing a minimal version of the dashboard because Node.js and npm are not available to build the full React frontend.</p>
            <p>The backend API is still fully functional and can be accessed at the following endpoints:</p>
            <ul>
                <li><strong>/api/models</strong> - Get list of available models</li>
                <li><strong>/api/simulation/run</strong> - Run simulation</li>
                <li><strong>/api/simulation/results</strong> - Get simulation results</li>
                <li><strong>/api/metrics</strong> - Get calculated metrics</li>
                <li><strong>/api/charts/*</strong> - Get various charts</li>
            </ul>
        </div>
        <div class="card">
            <h2>To enable the full dashboard:</h2>
            <p>1. Install Node.js and npm from <a href="https://nodejs.org/" style="color: #90caf9;">https://nodejs.org/</a></p>
            <p>2. Make sure Node.js and npm are in your PATH</p>
            <p>3. Run the integration script again: <code>python integrate.py</code></p>
        </div>
        <div class="card">
            <h2>API Status</h2>
            <p>The backend API is running and available at:</p>
            <p><strong>http://localhost:5000/api</strong></p>
            <button class="button" onclick="checkApi()">Check API Status</button>
            <div id="apiStatus" style="margin-top: 10px;"></div>
        </div>
    </div>
    <script>
        function checkApi() {
            document.getElementById('apiStatus').innerHTML = 'Checking API status...';
            fetch('/api/models')
                .then(response => {
                    if (response.ok) {
                        return response.json();
                    }
                    throw new Error('API request failed');
                })
                .then(data => {
                    document.getElementById('apiStatus').innerHTML = 
                        '<span style="color: #4caf50;">API is working! Found ' + 
                        (data.eve_models.length + data.evs_models.length) + 
                        ' models.</span>';
                })
                .catch(error => {
                    document.getElementById('apiStatus').innerHTML = 
                        '<span style="color: #f44336;">API is not responding. Make sure the backend is running.</span>';
                });
        }
    </script>
</body>
</html>
"""
    
    # Write with explicit UTF-8 encoding to avoid character encoding issues
    try:
        with open(os.path.join(build_dir, 'index.html'), 'w', encoding='utf-8') as f:
            f.write(index_html)
    except UnicodeEncodeError:
        # Fallback for systems with limited encoding support
        logger.warning("UTF-8 encoding not supported, using ASCII encoding")
        with open(os.path.join(build_dir, 'index.html'), 'w', encoding='ascii', errors='replace') as f:
            f.write(index_html)
    
    logger.info(f"Created minimal frontend in {build_dir}")
    return True

def main():
    """Main function to integrate frontend and backend."""
    logger.info("Starting integration process...")
    logger.info(f"Operating system: {platform.system()} {platform.release()}")
    
    # Install Python dependencies
    if not install_python_dependencies():
        logger.error("Failed to install Python dependencies")
        return 1
    
    # Check if Node.js and npm are available
    dependencies_available = check_dependencies()
    
    if dependencies_available:
        # Build frontend with Node.js/npm
        if not build_frontend():
            logger.error("Failed to build frontend with Node.js/npm")
            logger.info("Falling back to minimal frontend...")
            if not create_dummy_frontend():
                logger.error("Failed to create minimal frontend")
                return 1
    else:
        # Create minimal frontend without Node.js/npm
        logger.info("Node.js/npm not available, creating minimal frontend...")
        if not create_dummy_frontend():
            logger.error("Failed to create minimal frontend")
            return 1
    
    # Setup backend
    if not setup_backend():
        logger.error("Failed to setup backend")
        return 1
    
    logger.info("Integration completed successfully")
    logger.info("To start the application, run: python main.py")
    logger.info("Then access the dashboard in your web browser at: http://localhost:5000")
    return 0

if __name__ == "__main__":
    sys.exit(main())
