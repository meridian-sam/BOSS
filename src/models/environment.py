from skyfield.api import load, wgs84
from skyfield.positionlib import Geocentric
from datetime import datetime, timedelta
import numpy as np
from typing import Tuple, NamedTuple, Optional
from models.eclipse import EclipseCalculator, EclipseState
from dataclasses import dataclass

class CelestialBody(NamedTuple):
    position: np.ndarray  # km, ECI coordinates
    velocity: np.ndarray  # km/s, ECI coordinates
    sun_direction: np.ndarray  # unit vector

@dataclass
class EnvironmentState:
    """Environmental conditions at a point in time."""
    magnetic_field: np.ndarray  # [T]
    atmospheric_density: float  # [kg/m³]
    solar_flux: float  # [W/m²]
    eclipse_factor: float  # [0-1]
    timestamp: datetime

class Environment:
    class Environment:
    def __init__(self, config):
        """
        Initialize environment model.
        
        Args:
            config: Environment configuration
        """
        self.config = config
        self.state: Optional[EnvironmentState] = None

        # Load astronomical ephemerides
        self.ts = load.timescale()
        self.ephemeris = load('de421.bsp')  # NASA JPL ephemerides
        
        # Get the celestial bodies
        self.earth = self.ephemeris['earth']
        self.moon = self.ephemeris['moon']
        self.sun = self.ephemeris['sun']
        
        # Cache for performance
        self._last_update_time = None
        self._cached_bodies = None
        
        # Earth parameters
        self.EARTH_RADIUS = 6371.0  # km
        self.EARTH_MU = 398600.4418  # km³/s²
        self.J2 = 1.08262668e-3     # Earth's J2 perturbation

        # Eclipse calculations
        self.eclipse_calculator = EclipseCalculator()
        self._cached_eclipse_state = None
        
    def update(self, current_time: datetime) -> CelestialBody:
        """
        Update the positions of celestial bodies for the given time.
        Returns positions in Earth-Centered Inertial (ECI) frame.
        """
        # Check if we need to update (cache for performance)
        if (self._last_update_time is not None and 
            abs((current_time - self._last_update_time).total_seconds()) < 1.0):
            return self._cached_bodies
            
        # Convert to skyfield time
        t = self.ts.from_datetime(current_time)
        
        # Get positions relative to Earth center
        moon_geocentric = self.earth.at(t).observe(self.moon)
        sun_geocentric = self.earth.at(t).observe(self.sun)
        
        # Convert to kilometers in ECI frame
        moon_pos = moon_geocentric.position.km
        moon_vel = moon_geocentric.velocity.km_per_s
        sun_pos = sun_geocentric.position.km
        sun_vel = sun_geocentric.velocity.km_per_s
        
        # Calculate sun direction unit vector
        sun_direction = sun_pos / np.linalg.norm(sun_pos)
        
        # Cache the results
        self._cached_bodies = CelestialBody(
            position=np.array([moon_pos, sun_pos]),
            velocity=np.array([moon_vel, sun_vel]),
            sun_direction=sun_direction
        )
        self._last_update_time = current_time
        
        return self._cached_bodies
        
    def calculate_eclipse(self, satellite_position: np.ndarray) -> float:
        """
        Calculate eclipse factor (0 = full eclipse, 1 = full sun)
        """
        # Get vector from Earth center to satellite
        sat_r = np.linalg.norm(satellite_position)
        sat_direction = satellite_position / sat_r
        
        # Get angle between satellite and sun direction
        sun_angle = np.arccos(np.dot(sat_direction, self._cached_bodies.sun_direction))
        
        # Calculate eclipse geometry
        # Simple cylindrical shadow model
        apparent_earth_radius = np.arcsin(self.EARTH_RADIUS / sat_r)
        
        if sun_angle <= apparent_earth_radius:
            return 0.0  # Full eclipse
        elif sun_angle <= apparent_earth_radius + 0.25:  # Add penumbra region
            # Simple linear interpolation in penumbra
            return (sun_angle - apparent_earth_radius) / 0.25
        else:
            return 1.0  # Full sun
            
    def calculate_magnetic_field(self, position: np.ndarray, time: datetime) -> np.ndarray:
        """
        Calculate Earth's magnetic field vector at given position
        Using simple dipole model for now
        """
        # TODO: Implement more accurate IGRF model
        r = np.linalg.norm(position)
        m = 7.94e22  # Earth's magnetic dipole moment (A⋅m²)
        
        # Simplified dipole field
        B0 = (1e-9 * m) / (r**3)  # Convert to Tesla
        
        # Align with magnetic poles (simplified)
        magnetic_axis = np.array([0, 0, 1])  # Assuming aligned with Earth's axis for now
        
        B = B0 * (3 * np.dot(magnetic_axis, position/r) * (position/r) - magnetic_axis)
        
        return B
        
    def calculate_atmospheric_density(self, position: np.ndarray) -> float:
        """
        Calculate atmospheric density using exponential model.
        
        Args:
            position: Position vector in ECI frame [km]
            
        Returns:
            Atmospheric density [kg/m³]
        """
        altitude = np.linalg.norm(position) - self.EARTH_RADIUS
        
        # Base density at reference altitude
        rho_0 = 1.225  # kg/m³ at sea level
        H = 7.249  # Scale height in km
        
        # Exponential decay model
        density = rho_0 * np.exp(-altitude / H)
        
        return density

    def calculate_eclipse(self, satellite_position: np.ndarray) -> EclipseState:
        """Calculate eclipse state for given satellite position"""
        if self._cached_bodies is None:
            raise ValueError("Environment must be updated before calculating eclipse")
            
        eclipse_state = self.eclipse_calculator.calculate_eclipse(
            satellite_position,
            self._cached_bodies.position[1],  # Sun position
            self._cached_bodies.position[0]   # Moon position
        )
        
        self._cached_eclipse_state = eclipse_state
        return eclipse_state