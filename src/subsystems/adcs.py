import numpy as np

class ADCS:
    OFF = 0
    SUNSAFE = 1
    NADIR = 2
    
    def __init__(self, config, attitude_dynamics):
        self.config = config
        self.attitude_dynamics = attitude_dynamics
        self.control_mode = self.OFF
        
        self.reaction_wheel_power = 0.5  # Watts per wheel
        self.magnetorquer_power = 0.2  # Watts per torquer
        self.num_reaction_wheels = 3
        self.num_magnetorquers = 3
        
    def set_control_mode(self, mode: int):
        """Set the ADCS control mode"""
        if mode in [self.OFF, self.SUNSAFE, self.NADIR]:
            self.control_mode = mode
    
    def update(self, dt: float, magnetic_field: np.ndarray, sun_direction: np.ndarray, nadir_direction: np.ndarray) -> dict:
        """
        Update ADCS state
        Returns ADCS telemetry dictionary
        """
        torque = np.zeros(3)
        
        if self.control_mode == self.SUNSAFE:
            # Align solar panels with the sun
            torque = self._calculate_sun_safe_torque(sun_direction)
        elif self.control_mode == self.NADIR:
            # Align with Earth's nadir
            torque = self._calculate_nadir_torque(nadir_direction)
        
        # Update attitude dynamics
        self.attitude_dynamics.update(torque, dt)
        
        # Calculate power usage
        reaction_wheel_power_usage = self.reaction_wheel_power * self.num_reaction_wheels
        magnetorquer_power_usage = self.magnetorquer_power * self.num_magnetorquers
        
        return {
            'angular_velocity_rad_s': self.attitude_dynamics.get_angular_velocity().tolist(),
            'attitude_quaternion': self.attitude_dynamics.get_attitude().tolist(),
            'reaction_wheel_power_w': reaction_wheel_power_usage,
            'magnetorquer_power_w': magnetorquer_power_usage,
            'control_mode': self.control_mode
        }
    
    def _calculate_sun_safe_torque(self, sun_direction: np.ndarray) -> np.ndarray:
        """Calculate torque to align with the sun"""
        # Placeholder for control logic
        # TODO: Implement control algorithm to align with sun
        return np.random.uniform(-0.01, 0.01, 3)
    
    def _calculate_nadir_torque(self, nadir_direction: np.ndarray) -> np.ndarray:
        """Calculate torque to align with nadir"""
        # Placeholder for control logic
        # TODO: Implement control algorithm to align with nadir
        return np.random.uniform(-0.01, 0.01, 3)