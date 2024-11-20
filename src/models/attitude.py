from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional

@dataclass
class AttitudeState:
    """Spacecraft attitude state."""
    quaternion: np.ndarray  # [w, x, y, z]
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s
    timestamp: Optional[float] = None  # Unix timestamp

class AttitudeDynamics:
    def __init__(self, inertia_matrix: np.ndarray):
        """
        Initialize attitude dynamics model.
        
        Args:
            inertia_matrix: 3x3 inertia tensor [kg⋅m²]
        """
        self.inertia_matrix = inertia_matrix
        self.inertia_inv = np.linalg.inv(inertia_matrix)
        
        # Initial state
        self.state = AttitudeState(
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            angular_velocity=np.zeros(3)
        )
        
    def update(self, torque: np.ndarray, dt: float) -> AttitudeState:
        """
        Update attitude dynamics.
        
        Args:
            torque: Applied torque vector [N⋅m]
            dt: Time step [s]
            
        Returns:
            Updated attitude state
        """
        # Update angular velocity using Euler's equation
        angular_acceleration = self.inertia_inv @ (torque - np.cross(self.angular_velocity, self.inertia_matrix @ self.angular_velocity))
        self.angular_velocity += angular_acceleration * dt
        
        # Update quaternion using angular velocity
        omega_quat = np.concatenate(([0], self.angular_velocity))
        quat_dot = 0.5 * self._quaternion_multiply(self.quaternion, omega_quat)
        self.quaternion += quat_dot * dt
        self.quaternion /= np.linalg.norm(self.quaternion)  # Normalize
        
    def get_attitude(self) -> np.ndarray:
        """
        Get the current attitude as a quaternion
        """
        return self.quaternion
    
    def get_angular_velocity(self) -> np.ndarray:
        """
        Get the current angular velocity
        """
        return self.angular_velocity
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Multiply two quaternions
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])