from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from datetime import datetime
import logging
from models.attitude import AttitudeDynamics
from models.magnetic_field import MagneticFieldModel
from subsystems.thermal import ThermalZone
from common.events import EventType

class ADCSMode(Enum):
    """ADCS control modes."""
    IDLE = auto()
    DETUMBLE = auto()
    SUNSAFE = auto()
    NADIR = auto()
    TARGET = auto()
    CUSTOM = auto()

@dataclass
class ADCSState:
    """ADCS system state."""
    mode: ADCSMode
    quaternion: np.ndarray  # [w, x, y, z]
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s
    magnetic_field: np.ndarray  # [Bx, By, Bz] Tesla
    sun_vector: Optional[np.ndarray]  # Unit vector to Sun
    nadir_vector: np.ndarray  # Unit vector to Earth center
    wheel_speeds: np.ndarray  # [w1, w2, w3, w4] rad/s
    magnetorquer_dipoles: np.ndarray  # [m1, m2, m3] A⋅m²
    power_consumption: float  # Watts
    timestamp: datetime
    temperature_c: float = 20.0  # Add temperature tracking

@dataclass
class ADCSTelemetry:
    quaternion: np.ndarray
    angular_velocity: np.ndarray
    magnetometer: np.ndarray
    sun_sensors: List[float]
    gyro_temps: List[float]
    reaction_wheel_speeds: List[float]
    reaction_wheel_currents: List[float]
    magnetorquer_commands: List[float]
    control_mode: str
    control_error: float
    control_effort: float
    power_status: bool
    fault_flags: int
    cpu_temp: float

class ADCS:
    """Attitude Determination and Control System."""

    def __init__(self, config, attitude_dynamics: AttitudeDynamics):
        """
        Initialize ADCS.

        Args:
            config: ADCS configuration
            attitude_dynamics: Attitude dynamics model
        """
        self.config = config
        self.dynamics = attitude_dynamics
        self.magnetic_model = MagneticFieldModel()
        self.logger = logging.getLogger(__name__)

        # Initialize state
        self.state = ADCSState(
            mode=ADCSMode.IDLE,
            quaternion=np.array([1., 0., 0., 0.]),
            angular_velocity=np.zeros(3),
            magnetic_field=np.zeros(3),
            sun_vector=None,
            nadir_vector=np.array([0., 0., 1.]),
            wheel_speeds=np.zeros(4),
            magnetorquer_dipoles=np.zeros(3),
            power_consumption=0.0,
            timestamp=datetime.now(datetime.UTC)
        )

        # Hardware configuration
        self.num_wheels = 4
        self.wheel_max_torque = 0.01  # N⋅m
        self.wheel_max_momentum = 0.1  # N⋅m⋅s
        self.magnetorquer_max_dipole = 1.0  # A⋅m²
        
        # Control parameters
        self.detumble_gain = 5e-5
        self.pointing_gain_p = 0.1
        self.pointing_gain_d = 0.5
        
        # Power consumption
        self.wheel_power = 0.5  # Watts per wheel
        self.magnetorquer_power = 0.2  # Watts per magnetorquer
        
    def update(self, 
              position: np.ndarray, 
              velocity: np.ndarray,
              sun_vector: Optional[np.ndarray],
              dt: float) -> Dict:
        """
        Update ADCS state and control.

        Args:
            position: Position vector in ECI [km]
            velocity: Velocity vector in ECI [km/s]
            sun_vector: Unit vector to Sun in ECI (None if eclipsed)
            dt: Time step [s]

        Returns:
            Dictionary containing ADCS telemetry
        """
        # Update state vectors
        self.state.sun_vector = sun_vector
        self.state.nadir_vector = -position / np.linalg.norm(position)
        self.state.magnetic_field = self.magnetic_model.calculate_magnetic_field(position)
        
        # Calculate control torques based on mode
        control_torque = np.zeros(3)
        
        if self.state.mode == ADCSMode.DETUMBLE:
            control_torque = self._detumble_control()
        elif self.state.mode == ADCSMode.SUNSAFE:
            control_torque = self._sunsafe_control()
        elif self.state.mode == ADCSMode.NADIR:
            control_torque = self._nadir_control()
        elif self.state.mode == ADCSMode.TARGET:
            control_torque = self._target_control()
            
        # Allocate torques to actuators
        wheel_torques, magnetorquer_dipoles = self._allocate_torques(control_torque)
        
        # Update dynamics
        self.dynamics.update(control_torque, dt)
        
        # Update state
        self.state.quaternion = self.dynamics.get_state().quaternion
        self.state.angular_velocity = self.dynamics.get_state().angular_velocity
        self.state.wheel_speeds += wheel_torques * dt / self.wheel_max_momentum
        self.state.magnetorquer_dipoles = magnetorquer_dipoles
        self.state.timestamp = datetime.now(datetime.UTC)
        
        # Calculate power consumption
        self.state.power_consumption = (
            np.sum(np.abs(self.state.wheel_speeds)) * self.wheel_power +
            np.sum(np.abs(self.state.magnetorquer_dipoles)) * self.magnetorquer_power
        )

        if hasattr(self, 'thermal_subsystem'):
            self.state.temperature_c = self.thermal_subsystem.zones[ThermalZone.ADCS].average_temp
        
        return self.get_telemetry()
        
    def set_mode(self, mode: ADCSMode):
        """Set ADCS control mode."""
        self.logger.info(f"ADCS mode changed from {self.state.mode} to {mode}")
        self.state.mode = mode
        
    def get_telemetry(self) -> Dict:
        """Get ADCS telemetry."""
        return {
            'mode': self.state.mode.name,
            'quaternion': self.state.quaternion.tolist(),
            'angular_velocity_rads': self.state.angular_velocity.tolist(),
            'magnetic_field_t': self.state.magnetic_field.tolist(),
            'wheel_speeds_rads': self.state.wheel_speeds.tolist(),
            'magnetorquer_dipoles_am2': self.state.magnetorquer_dipoles.tolist(),
            'power_consumption_w': self.state.power_consumption,
            'timestamp': self.state.timestamp.isoformat()
        }
        
    def _detumble_control(self) -> np.ndarray:
        """B-dot detumbling controller."""
        # B-dot algorithm for detumbling
        b_dot = -np.cross(self.state.angular_velocity, self.state.magnetic_field)
        return -self.detumble_gain * b_dot
        
    def _sunsafe_control(self) -> np.ndarray:
        """Sun-safe pointing controller."""
        if self.state.sun_vector is None:
            return np.zeros(3)
            
        # Quaternion error between current and desired sun-pointing attitude
        q_error = self._calculate_sun_pointing_error()
        
        # PD control law
        return (self.pointing_gain_p * q_error[1:] - 
                self.pointing_gain_d * self.state.angular_velocity)
        
    def _nadir_control(self) -> np.ndarray:
        """Nadir pointing controller."""
        # Quaternion error between current and desired nadir-pointing attitude
        q_error = self._calculate_nadir_pointing_error()
        
        # PD control law
        return (self.pointing_gain_p * q_error[1:] - 
                self.pointing_gain_d * self.state.angular_velocity)
        
    def _target_control(self) -> np.ndarray:
        """Target pointing controller."""
        # Implementation depends on target definition
        return np.zeros(3)
        
    def _allocate_torques(self, 
                         desired_torque: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Allocate desired torque between reaction wheels and magnetorquers.
        
        Returns:
            Tuple of (wheel_torques, magnetorquer_dipoles)
        """
        # Simplified allocation - use wheels for primary control
        wheel_torques = np.clip(
            desired_torque,
            -self.wheel_max_torque,
            self.wheel_max_torque
        )
        
        # Use magnetorquers for momentum management
        wheel_momentum = self.state.wheel_speeds * self.wheel_max_momentum
        if np.any(np.abs(wheel_momentum) > 0.8 * self.wheel_max_momentum):
            # Cross product control law for momentum dumping
            magnetorquer_dipoles = np.cross(
                wheel_momentum,
                self.state.magnetic_field
            )
            magnetorquer_dipoles = np.clip(
                magnetorquer_dipoles,
                -self.magnetorquer_max_dipole,
                self.magnetorquer_max_dipole
            )
        else:
            magnetorquer_dipoles = np.zeros(3)
            
        return wheel_torques, magnetorquer_dipoles
        
    def _calculate_sun_pointing_error(self) -> np.ndarray:
        """Calculate quaternion error for sun pointing."""
        if self.state.sun_vector is None:
            return np.array([1., 0., 0., 0.])
            
        # Calculate desired attitude quaternion for sun pointing
        # Implementation depends on desired sun-pointing strategy
        return np.array([1., 0., 0., 0.])  # Placeholder
        
    def _calculate_nadir_pointing_error(self) -> np.ndarray:
        """Calculate quaternion error for nadir pointing."""
        # Calculate desired attitude quaternion for nadir pointing
        # Implementation depends on desired nadir-pointing strategy
        return np.array([1., 0., 0., 0.])  # Placeholder

    def get_telemetry(self) -> ADCSTelemetry:
        """Generate ADCS telemetry packet."""
        return ADCSTelemetry(
            # Attitude state
            quaternion=self.state.quaternion,
            angular_velocity=self.state.angular_velocity,
            
            # Sensor readings
            magnetometer=self.magnetometer.get_reading(),
            sun_sensors=self.sun_sensors.get_readings(),
            gyro_temps=self.gyros.get_temperatures(),
            
            # Actuator states
            reaction_wheel_speeds=self.reaction_wheels.get_speeds(),
            reaction_wheel_currents=self.reaction_wheels.get_currents(),
            magnetorquer_commands=self.magnetorquers.get_commands(),
            
            # Control state
            control_mode=self.control_mode.name,
            control_error=self.get_control_error(),
            control_effort=self.get_control_effort(),
            
            # System status
            power_status=self.power_enabled,
            fault_flags=self.get_fault_flags(),
            cpu_temp=self.get_cpu_temperature()
        )
    
    def publish_telemetry(self):
        """Publish ADCS telemetry packet."""
        telemetry = self.get_telemetry()
        packet = telemetry.to_ccsds()
        self.event_bus.publish(
            EventType.TELEMETRY,
            "ADCS",
            {"packet": packet.pack()}
        )