#!/usr/bin/env python3
"""
Setup script for CryptoGuard-LLM

A Multi-Modal Deep Learning Framework for Real-Time Cryptocurrency Fraud Detection
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().strip().split('\n')
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith('#')]

setup(
    name="cryptoguard-llm",
    version="1.0.0",
    author="Naga Sujitha Vummaneni, Usha Ratnam Jammula, Ramesh Chandra Aditya Komperla",
    author_email="nv262@cornell.edu",
    description="Multi-Modal Deep Learning Framework for Cryptocurrency Fraud Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nv262-lab/CryptoGuard-LLM",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cryptoguard-train=scripts.train:main",
            "cryptoguard-eval=scripts.evaluate:main",
            "cryptoguard-infer=scripts.inference:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
