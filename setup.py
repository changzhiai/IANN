from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith('#')]

requirements = parse_requirements('requirements.txt')

setup(
    name="IANN",
    version="0.1.0",
    description="Interatomic Neural Network Package for materials science",
    author="Changzhi Ai",
    author_email="changzhi@stanford.edu",
    url="https://github.com/changzhiai/IANN",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pylint>=2.6.0",
            "jupyter>=1.0.0",
        ],
    },
    classifiers=[
        "Development Status :: Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7++",
        "Programming Language :: C++",
    ],
    keywords="machine learning, materials science, neural networks, molecular dynamics",
    python_requires=">=3.7",
) 