from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

requirements = parse_requirements('requirements.txt')

setup(
    name="IANN",
    version="0.1.2",
    description="Interatomic Neural Network Package for materials science",
    author="Changzhi Ai",
    author_email="changzhi@stanford.edu",
    url="https://github.com/changzhiai/IANN",
    # include= keeps the top-level `test` package (it has an __init__.py) out of
    # the wheel; a bare find_packages() installs it as an importable `test`.
    packages=find_packages(include=["iann", "iann.*"]),
    # package_data, not MANIFEST.in: MANIFEST.in governs the sdist only, so the
    # wheel would still ship without these and they are read from the *installed*
    # package directory at runtime.
    #   iann/data/Jd.pt          -- Wigner-D tables; EquiformerV2/V3 and UMA all
    #                               load it via iann.__path__, and equiformerV3
    #                               does so at module import time.
    #   iann/foundations/*.pt    -- bundled foundation-model checkpoints reached
    #                               by foundation_model().
    #   iann/plugins/*           -- LAMMPS pair-style sources and example input.
    package_data={
        "iann.data": ["*.pt"],
        "iann.foundations": ["*.pt"],
        "iann.plugins": ["*.cpp", "*.h", "*.in", "README.md"],
    },
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pylint>=2.6.0",
            "jupyter>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
    ],
    keywords="machine learning, materials science, neural networks, molecular dynamics",
    python_requires=">=3.8",
) 