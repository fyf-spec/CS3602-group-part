"""
Setup script for accelerated_inference package.

Install in development mode with:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if it exists
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="accelerated_inference",
    version="0.1.0",
    description="Efficient KV cache strategies for LLM inference (H2O, StreamingLLM, Lazy H2O, SepLLM)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/accelerated_inference",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "evaluate", "outputs", "dataset"]),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=[
        "torch>=2.0.0",
        "transformers==4.33.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "eval": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "datasets>=2.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    
    # Entry points (optional CLI tools)
    entry_points={
        "console_scripts": [
            # Add CLI commands here if needed
            # "accel-benchmark=accelerated_inference.cli:benchmark_main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Include package data
    include_package_data=True,
    package_data={
        "accelerated_inference": ["*.json", "*.yaml"],
    },
    
    # Zip safety
    zip_safe=False,
)
