import re

from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

requirements = parse_requirements('requirements.txt')

# The PyPI project page renders this; without it the page is blank. Read rather
# than duplicated, so it cannot drift from the README.
#
# Relative image paths have to become absolute here. The README keeps them
# relative because that is what works on GitHub and when the file is read
# offline in a checkout, but PyPI renders the description with no repository
# context, so a relative src resolves to nothing and the figures silently do not
# appear. Neither GitHub nor PyPI allows a fallback chain -- <picture>/<source>
# selects on type and media rather than on load failure, and both sanitizers
# strip onerror handlers and <object> fallbacks -- so the rewrite happens at
# build time instead, leaving one source of truth.
_RAW = "https://raw.githubusercontent.com/changzhiai/IANN/master"

with open('README.md', encoding='utf-8') as fh:
    long_description = fh.read()

long_description = re.sub(
    r'(src=")(docs/|examples/)',          # HTML <img src="docs/...">
    lambda m: f'{m.group(1)}{_RAW}/{m.group(2)}', long_description)
long_description = re.sub(
    r'(\]\()(docs/|examples/)',            # markdown ![alt](docs/...)
    lambda m: f'{m.group(1)}{_RAW}/{m.group(2)}', long_description)

setup(
    name="pyiann",
    version="0.1.3",
    description="Interatomic Neural Network Package for materials science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Changzhi Ai",
    author_email="changzhi@stanford.edu",
    url="https://github.com/changzhiai/IANN",
    # include= keeps the top-level `test` package (it has an __init__.py) out of
    # the wheel; a bare find_packages() installs it as an importable `test`.
    # exclude= drops the agent-layer tests: they sit inside the package so the
    # whole agent surface is one reviewable directory, but they should not be
    # installed.
    packages=find_packages(include=["iann", "iann.*"],
                           exclude=["iann.agent.tests", "iann.agent.tests.*"]),
    # package_data, not MANIFEST.in: MANIFEST.in governs the sdist only, so the
    # wheel would still ship without these and they are read from the *installed*
    # package directory at runtime.
    #   iann/data/Jd.pt          -- Wigner-D tables; EquiformerV2/V3 and UMA all
    #                               load it via iann.__path__, and equiformerV3
    #                               does so at module import time.
    #   iann/foundations/*.pt    -- bundled foundation-model checkpoints reached
    #                               by foundation_model().
    #   iann/plugins/*           -- LAMMPS pair-style sources and example input.
    #   iann/agent/AGENTS.md     -- tool-neutral repository notes, and
    #   iann/agent/skills/*      -- Claude Code skill sources, both copied out by
    #                               `iann agent install`; they must ship or that
    #                               command has nothing to install from.
    package_data={
        "iann.data": ["*.pt"],
        "iann.foundations": ["*.pt"],
        "iann.plugins": ["*.cpp", "*.h", "*.in", "README.md"],
        "iann.agent": ["AGENTS.md", "skills/*/*.md"],
    },
    entry_points={
        "console_scripts": [
            "iann = iann.agent.cli:main",
        ],
    },
    install_requires=requirements,
    extras_require={
        # `pip install -e ".[agent]"` adds the MCP server's only dependency;
        # the CLI itself needs nothing beyond the base requirements.
        "agent": ["mcp>=1.2.0"],
        # Only `iann.foundations` needs this, and only to *download* a released
        # checkpoint -- resolving a bundled one or an explicit path does not
        # import it. Keeping it out of install_requires means a user who trains
        # on their own data never pulls it in; the download path raises with the
        # install command when it is missing.
        "foundations": ["huggingface_hub>=0.23.0"],
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