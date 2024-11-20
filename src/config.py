from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class OrbitConfig:
    # Keplerian elements (defaults to 500km SSO)
    semi_major_axis_km: float = 6871.0  # Earth radius + 500km
    eccentricity: float = 0.0
    inclination_deg: float = 97.4
    raan_deg: float = 0.0  # Right Ascension of Ascending Node
    arg_perigee_deg: float = 0.0
    true_anomaly_deg: float = 0.0
    
    # Spacecraft properties
    mass_kg: float = 4.0  # 3U CubeSat typical mass
    drag_coefficient: float = 2.2
    drag_area_m2: float = 0.03  # Typical 3U cross-section

    # Solar radiation pressure properties
    reflectivity: float = 0.3  # Typical satellite reflectivity coefficient
    srp_area_m2: float = 0.03  # Area exposed to sun (default: 3U face)

@dataclass
class CommsConfig:
    uplink_rate_bps: int = 20_000  # 20 kbps
    downlink_rate_bps: int = 1_000_000  # 1 Mbps
    telemetry_rate_hz: float = 10.0
    yamcs_host: str = "localhost"
    yamcs_uplink_port: int = 10015
    yamcs_downlink_port: int = 10016

@dataclass
class SimConfig:
    orbit: OrbitConfig = OrbitConfig()
    comms: CommsConfig = CommsConfig()
    dt: float = 0.1  # 100ms simulation timestep 

@dataclass
class EnvironmentConfig:
    ephemeris_file: str = 'de421.bsp'
    magnetic_model: str = 'dipole'  # or 'igrf'
    atmosphere_model: str = 'exponential'  # or 'nrlmsise'

@dataclass
class SimConfig:
    orbit: OrbitConfig = OrbitConfig()
    comms: CommsConfig = CommsConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    dt: float = 0.1  # 100ms simulation timestep


@dataclass
class CameraConfig:
    # Camera specifications
    focal_length_mm: float = 70.0
    pixel_size_um: float = 5.5
    sensor_width_pixels: int = 2048
    sensor_height_pixels: int = 1536
    bit_depth: int = 12
    
    # Operational parameters
    max_frame_rate_hz: float = 1.0  # 1 frame per second
    power_consumption_w: float = 2.5  # Active power consumption
    standby_power_w: float = 0.1    # Standby power
    
    # Storage parameters
    max_images: int = 100  # Maximum images in storage
    compression_ratio: float = 0.7  # JPEG compression ratio
    
    # Mounting parameters (relative to spacecraft body)
    mounting_quaternion: tuple = (1.0, 0.0, 0.0, 0.0)  # Aligned with nadir face