from dataclasses import dataclass
import numpy as np
from typing import NamedTuple

class EclipseState(NamedTuple):
    """Represents the eclipse state of the spacecraft"""
    illumination: float  # 0.0 (full eclipse) to 1.0 (full sun)
    eclipse_type: str    # 'none', 'umbra', or 'penumbra'
    sun_angle: float    # Angle between spacecraft and sun direction (radians)

@dataclass
class CelestialBodyConstants:
    """Constants for eclipse calculations"""
    SUN_RADIUS = 696340.0  # km
    EARTH_RADIUS = 6371.0  # km
    MOON_RADIUS = 1737.1   # km
    
class EclipseCalculator:
    def __init__(self):
        self.constants = CelestialBodyConstants()
        
    def calculate_eclipse(self, sat_pos: np.ndarray, sun_pos: np.ndarray, 
                         moon_pos: np.ndarray) -> EclipseState:
        """
        Calculate spacecraft eclipse state considering both Earth and Moon shadows
        
        Args:
            sat_pos: Satellite position vector in ECI (km)
            sun_pos: Sun position vector in ECI (km)
            moon_pos: Moon position vector in ECI (km)
        """
        # Calculate Earth eclipse
        earth_eclipse = self._calculate_body_eclipse(
            sat_pos, sun_pos, np.zeros(3),
            self.constants.EARTH_RADIUS, self.constants.SUN_RADIUS
        )
        
        # Calculate Moon eclipse
        moon_eclipse = self._calculate_body_eclipse(
            sat_pos, sun_pos, moon_pos,
            self.constants.MOON_RADIUS, self.constants.SUN_RADIUS
        )
        
        # Take the minimum illumination (worst case)
        illumination = min(earth_eclipse[0], moon_eclipse[0])
        
        # Determine eclipse type
        if illumination == 0.0:
            eclipse_type = 'umbra'
        elif illumination < 1.0:
            eclipse_type = 'penumbra'
        else:
            eclipse_type = 'none'
            
        return EclipseState(
            illumination=illumination,
            eclipse_type=eclipse_type,
            sun_angle=earth_eclipse[2]  # Use Earth-Sun angle as reference
        )
    
    def _calculate_body_eclipse(self, sat_pos: np.ndarray, sun_pos: np.ndarray, 
                              body_pos: np.ndarray, body_radius: float, 
                              sun_radius: float) -> tuple:
        """
        Calculate eclipse state for a single occulting body
        Returns: (illumination, eclipse_type, sun_angle)
        """
        # Vectors from body to satellite and sun
        sat_to_sun = sun_pos - sat_pos
        sat_to_body = body_pos - sat_pos
        
        # Distances
        sun_dist = np.linalg.norm(sat_to_sun)
        body_dist = np.linalg.norm(sat_to_body)
        
        # Angular calculations
        sun_angle = np.arccos(np.dot(-sat_to_sun, sat_to_body) / 
                            (sun_dist * body_dist))
        
        # Apparent angular radii
        body_angular_radius = np.arcsin(body_radius / body_dist)
        sun_angular_radius = np.arcsin(sun_radius / sun_dist)
        
        # Calculate umbra and penumbra conditions
        umbra_angle = body_angular_radius - sun_angular_radius
        penumbra_angle = body_angular_radius + sun_angular_radius
        
        # Determine eclipse state
        if sun_angle <= umbra_angle:
            return 0.0, 'umbra', sun_angle
        elif sun_angle <= penumbra_angle:
            # Linear interpolation in penumbra
            illumination = (sun_angle - umbra_angle) / (penumbra_angle - umbra_angle)
            return illumination, 'penumbra', sun_angle
        else:
            return 1.0, 'none', sun_angle