#!/usr/bin/env python3
"""Setup script for NAMO Python package."""

from setuptools import setup, find_packages
import os
import sys

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "NAMO (Navigation Among Movable Obstacles) Python package"

# Determine the C++ module path based on hostname
def get_cpp_module_path():
    """Get the path to the compiled namo_rl module."""
    import subprocess
    try:
        hostname = subprocess.check_output(['hostname', '-s']).decode().strip()
        base_dir = os.path.dirname(os.path.dirname(__file__))  # Go up from python/ to namo/
        cpp_module_dir = os.path.join(base_dir, f'build_python_mjxrl_{hostname}')

        if os.path.exists(cpp_module_dir):
            return cpp_module_dir
        else:
            # Fallback to generic build directory
            generic_dir = os.path.join(base_dir, 'build_python_mjxrl')
            if os.path.exists(generic_dir):
                return generic_dir
    except:
        pass

    return None

setup(
    name="namo",
    version="0.1.0",
    description="Navigation Among Movable Obstacles (NAMO) Planning Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="NAMO Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pyyaml",
        "dataclasses; python_version<'3.7'",
    ],
    extras_require={
        "visualization": ["matplotlib", "opencv-python"],
        "ml": ["torch", "torchvision"],
        "data": ["pandas", "scipy"],
    },
    entry_points={
        "console_scripts": [
            "namo-collect=namo.data_collection.modular_parallel_collection:main",
            "namo-visualize=namo.visualization.visual_test_single:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

# Post-install hook to add C++ module to path
class PostDevelopInstall:
    """Add the C++ module path to the Python path after installation."""

    def run(self):
        cpp_path = get_cpp_module_path()
        if cpp_path:
            print(f"\n🔧 C++ module found at: {cpp_path}")
            print("💡 You may need to add this to your PYTHONPATH:")
            print(f"   export PYTHONPATH=\"{cpp_path}:$PYTHONPATH\"")
            print("\n   Or add it to your shell configuration file.")
        else:
            print("\n⚠️  C++ module (namo_rl) not found.")
            print("   You may need to build it first with:")
            print("   cmake -B build_python_mjxrl -DBUILD_PYTHON_BINDINGS=ON")
            print("   cmake --build build_python_mjxrl")

# Run post-install notification
if 'develop' in sys.argv or 'install' in sys.argv:
    post_install = PostDevelopInstall()
    post_install.run()