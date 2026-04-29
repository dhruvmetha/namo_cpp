#!/usr/bin/env python3
"""Setup script for NAMO Python package."""

from pathlib import Path
from setuptools import setup, find_packages
import sys
from typing import Optional


def read_readme() -> str:
    readme_path = Path(__file__).resolve().parent / "README.md"
    if readme_path.exists():
        return readme_path.read_text(encoding="utf-8")
    return "NAMO (Navigation Among Movable Obstacles) Python package"


def get_cpp_module_path() -> Optional[Path]:
    """Return canonical build directory path if namo_rl shared object exists."""
    base_dir = Path(__file__).resolve().parent.parent
    build_dir = base_dir / "build_python"
    if build_dir.is_dir() and any(build_dir.glob("namo_rl*.so")):
        return build_dir
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


class PostDevelopInstall:
    """Show canonical namo_rl build guidance after install/develop."""

    def run(self) -> None:
        cpp_path = get_cpp_module_path()
        if cpp_path is not None:
            print(f"\nC++ module found at: {cpp_path}")
            print("Add this to your PYTHONPATH:")
            print(f"  export PYTHONPATH=\"{cpp_path}:$PYTHONPATH\"")
            return

        print("\nC++ module (namo_rl) not found in canonical build directory.")
        print("Build it with:")
        print("  cmake -S . -B build_python -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON")
        print("  cmake --build build_python --target namo_rl -j$(nproc)")


if "develop" in sys.argv or "install" in sys.argv:
    PostDevelopInstall().run()
