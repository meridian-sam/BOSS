# BOSS: Basic Open-source Satellite Simulator

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

BOSS is a comprehensive, open-source satellite simulation framework designed for spacecraft mission analysis, orbital dynamics, and space systems engineering. It provides a flexible, modular architecture for simulating satellite behavior in various orbital scenarios.

## Features

- **Orbital Dynamics**: High-fidelity propagation using SGP4/SDP4 models
- **Attitude Dynamics**: Quaternion-based attitude propagation and control
- **Environmental Models**: 
  - Solar radiation pressure
  - Atmospheric drag
  - Gravitational perturbations
- **Interactive Visualization**: Real-time 3D visualization of satellite orbits and attitudes
- **Extensible Architecture**: Easily add custom models and simulation components

## Installation

```bash
# Clone the repository
git clone https://github.com/meridian-sam/BOSS.git
cd BOSS

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# For development installation with additional tools
pip install -e ".[dev]"
```

## Quick Start

```python
from boss import SatelliteSimulator
from pathlib import Path

# Initialize simulator with configuration
sim = SatelliteSimulator(config_path=Path("config/default.yaml"))

# Run simulation
sim.run()
```

## Documentation

Detailed documentation is available at [docs/](docs/). To build the documentation locally:

```bash
pip install -e ".[docs]"
cd docs
make html
```

## Project Structure

```
BOSS/
├── src/
│   ├── utils/          # Utility functions and helpers
│   ├── models/         # Physical models and dynamics
│   ├── visualization/  # Visualization and dashboard components
│   └── main.py        # Main simulation interface
├── tests/             # Test suite
├── docs/              # Documentation
├── config/            # Configuration files
└── examples/          # Example scripts and notebooks
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and contribute to the project.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 src tests
black src tests
mypy src tests
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use BOSS in your research, please cite:

```bibtex
@software{boss2024,
  title = {BOSS: Basic Open-source Satellite Simulator},
  author = {{Meridian Space Command Ltd}},
  year = {2024},
  url = {https://github.com/meridian-sam/BOSS}
}
```

## Contact

For questions and support:
- Email: contact@meridianspacecommand.com
- Issues: [GitHub Issues](https://github.com/meridian-sam/BOSS/issues)

## Acknowledgments

BOSS builds upon several open-source projects and scientific papers. See [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for a complete list of references and acknowledgments.