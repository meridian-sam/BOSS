from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="boss",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "astropy>=4.2",
        "pandas>=1.3.0",
        "pyyaml>=5.4.1",
        "sgp4>=2.20",
        "skyfield>=1.39",
        "spiceypy>=4.0.0",
        "quaternion>=2022.4.0",
        "matplotlib>=3.4.0",
        "plotly>=5.1.0",
        "dash>=2.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.12",
            "black>=21.5b2",
            "mypy>=0.910",
            "pylint>=2.8.2",
            "flake8>=3.9.2",
        ],
        "docs": [
            "sphinx>=4.0.2",
            "sphinx-rtd-theme>=0.5.2",
        ],
    },
    author="Meridian Space Command Ltd, UK",
    author_email="contact@meridianspacecommand.com",
    description="BOSS: Basic Open-source Spacecraft Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meridian-sam/BOSS",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
)