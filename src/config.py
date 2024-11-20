from dataclasses import dataclass
from typing import Dict
import numpy as np
from datetime import datetime

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
    # Data rates
    uplink_rate_bps: int = 20_000  # 20 kbps
    downlink_rate_bps: int = 1_000_000  # 1 Mbps
    telemetry_rate_hz: float = 10.0
    
    # YAMCS UDP Connection
    yamcs_host: str = "localhost"
    tm_port: int = 10015      # Spacecraft telemetry send port
    tc_port: int = 10025      # Spacecraft telecommand receive port
    yamcs_tm_port: int = 10115  # YAMCS telemetry receive port
    yamcs_tc_port: int = 10125  # YAMCS telecommand send port
    
    # RF Parameters
    downlink_frequency_mhz: float = 437.0
    uplink_frequency_mhz: float = 145.9
    tx_power_dbm: float = 30.0
    rx_sensitivity_dbm: float = -110.0
    
    # CCSDS Configuration
    max_packet_size: int = 1024
    packet_overhead: int = 16  # CCSDS header size + CRC
    max_retry_count: int = 3
    ack_timeout_s: float = 1.0
    
    # Default APIDs
    housekeeping_apid: int = 100
    event_apid: int = 101
    payload_apid: int = 102
    adcs_apid: int = 103
    power_apid: int = 104
    thermal_apid: int = 105

@dataclass
class SimulationConfig:
    # Time Configuration
    start_time: datetime = datetime(2024, 1, 1, 0, 0, 0)  # Simulation start time
    time_factor: float = 1.0  # Real-time = 1.0, 2x speed = 2.0, etc.
    dt: float = 0.1  # 100ms simulation timestep
    
    # Simulation Control
    max_duration_s: float = 86400.0  # 24 hours default
    random_seed: int = 42
    enable_faults: bool = True
    logging_level: str = "INFO"
    
    # Telemetry Generation
    hk_generation_period_s: float = 1.0  # Housekeeping telemetry period
    event_check_period_s: float = 0.1    # Event checking period
    attitude_update_period_s: float = 0.1 # ADCS update period

@dataclass
class EnvironmentConfig:
    ephemeris_file: str = 'de421.bsp'
    magnetic_model: str = 'dipole'  # or 'igrf'
    atmosphere_model: str = 'exponential'  # or 'nrlmsise'
    
    # Space environment parameters
    solar_flux_f10_7: float = 150.0
    magnetic_index_kp: float = 3.0
    
    # Thermal environment
    solar_constant_w_m2: float = 1361.0
    earth_ir_w_m2: float = 237.0
    albedo_coefficient: float = 0.3

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

@dataclass
class Config:
    """Main configuration class."""
    orbit: OrbitConfig = OrbitConfig()
    comms: CommsConfig = CommsConfig()
    simulation: SimulationConfig = SimulationConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    camera: CameraConfig = CameraConfig()

# Create default configuration instance
DEFAULT_CONFIG = Config()

def get_config() -> Config:
    """Get configuration instance."""
    return DEFAULT_CONFIG

def set_simulation_start_time(start_time: datetime):
    """Set simulation start time."""
    DEFAULT_CONFIG.simulation.start_time = start_time

def set_time_factor(factor: float):
    """Set simulation time factor."""
    if factor <= 0:
        raise ValueError("Time factor must be positive")
    DEFAULT_CONFIG.simulation.time_factor = factor

def set_yamcs_host(host: str):
    """Set YAMCS host address."""
    DEFAULT_CONFIG.comms.yamcs_host = host

def set_yamcs_ports(tm_port: int, tc_port: int):
    """Set YAMCS communication ports."""
    DEFAULT_CONFIG.comms.yamcs_tm_port = tm_port
    DEFAULT_CONFIG.comms.yamcs_tc_port = tc_port

def validate_config(config: Config) -> List[str]:
    """
    Validate configuration values.
    Returns list of validation errors, empty if valid.
    """
    errors = []
    
    # Validate orbit configuration
    if config.orbit.semi_major_axis_km is None:
        errors.append("Semi-major axis cannot be None")
    elif config.orbit.semi_major_axis_km < 6371:
        errors.append("Semi-major axis must be above Earth's radius")
    
    if not 0 <= config.orbit.eccentricity < 1:
        errors.append("Eccentricity must be between 0 and 1")
        
    # Validate comms configuration
    if config.comms.uplink_rate_bps <= 0:
        errors.append("Uplink rate must be positive")
    
    if config.comms.downlink_rate_bps <= 0:
        errors.append("Downlink rate must be positive")
        
    if config.comms.tx_power_dbm > 33:  # Example limit
        errors.append("Transmit power exceeds maximum allowed")
        
    # Validate simulation configuration
    if config.simulation.dt <= 0:
        errors.append("Simulation timestep must be positive")
        
    if config.simulation.time_factor <= 0:
        errors.append("Time factor must be positive")
        
    return errors

def load_config(config_file: str) -> Config:
    """Load and validate configuration from file."""
    # Load config from file
    config = Config()  # Load from file implementation
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + 
                        "\n".join(f"- {error}" for error in errors))
        
    return config