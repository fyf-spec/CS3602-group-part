"""
Setup script for accelerated_inference package.

Installation:
    # Basic install (Python only, no INT8)
    pip install -e .

    # Full install with INT8 CUDA extension
    pip install -e . --no-build-isolation

Requirements for INT8 extension:
    - GPU: NVIDIA Ampere or newer (A100, RTX 30xx/40xx), sm_80+
    - CUDA: 12.x (matching PyTorch version)
    - Python: 3.10+
    - PyTorch: 2.0+ (recommend 2.9.1 for CUDA 12.8)
    - Compiler: Visual Studio Build Tools (Windows) or GCC (Linux)
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read README if it exists
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Try to import CUDA extension build tools
ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    
    # Check if CUDA source files exist
    cuda_sources = [
        'accelerated_inference/quantization/int8_weight_only/binding.cpp',
        'accelerated_inference/quantization/int8_weight_only/w8a16_gemm.cu',
    ]
    
    if all(os.path.exists(src) for src in cuda_sources):
        ext_modules.append(
            CUDAExtension(
                name='accelerated_inference.quantization.int8_weight_only.w8a16_gemm',
                sources=cuda_sources,
                extra_compile_args={
                    'cxx': ['-O3'],
                    'nvcc': [
                        '-O3',
                        '-gencode=arch=compute_80,code=sm_80',  # Ampere (A100, RTX 30xx)
                        '-gencode=arch=compute_86,code=sm_86',  # Ampere (RTX 30xx laptop)
                        '-gencode=arch=compute_89,code=sm_89',  # Ada Lovelace (RTX 40xx)
                        '--threads=4'
                    ]
                }
            )
        )
        cmdclass['build_ext'] = BuildExtension
        print("INT8 CUDA extension will be built.")
    else:
        print("Warning: CUDA source files not found. Skipping INT8 extension.")
        
except ImportError:
    print("Warning: torch.utils.cpp_extension not available. Skipping INT8 extension.")

setup(
    name="accelerated_inference",
    version="0.2.0",
    description="Efficient KV cache strategies + INT8 quantization for LLM inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/accelerated_inference",
    
    # Package discovery
    packages=find_packages(exclude=["tests", "evaluate", "outputs", "dataset"]),
    
    # CUDA extension modules
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    
    # Python version requirement
    python_requires=">=3.10",
    
    # Dependencies
    install_requires=[
        "torch>=2.0.0",
        "transformers==4.33.0",
        "numpy>=1.20.0",
        "tqdm>=4.60.0",
        "datasets>=2.0.0",
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
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    
    # Include package data (CUDA source files)
    include_package_data=True,
    package_data={
        "accelerated_inference": ["*.json", "*.yaml"],
        "accelerated_inference.quantization.int8_weight_only": ["*.cpp", "*.cu"],
    },
    
    # Zip safety
    zip_safe=False,
)
