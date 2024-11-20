"""
Physical models for satellite simulation.
"""

from .attitude import AttitudeDynamics, AttitudeState
from .orbit import OrbitPropagator, OrbitState
from .environment import Environment, CelestialBody
from .eclipse import EclipseCalculator, EclipseState
from .magnetic_field import MagneticFieldModel

__all__ = [
    'AttitudeDynamics',
    'AttitudeState',
    'OrbitPropagator', 
    'OrbitState',
    'Environment',
    'CelestialBody',
    'EclipseCalculator',
    'EclipseState',
    'MagneticFieldModel'
]