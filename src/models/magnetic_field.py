from dataclasses import dataclass
import numpy as np
from datetime import datetime
from typing import Optional

@dataclass
class MagneticFieldState:
    """Magnetic field state."""
    field_vector: np.ndarray  # [T]
    field_strength: float  # [T]
    timestamp: datetime

class MagneticFieldModel:
    def __init__(self, model_type: str = 'dipole'):
        """
        Initialize magnetic field model.
        
        Args:
            model_type: 'dipole' or 'igrf'
        """
        self.model_type = model_type
        self.state: Optional[MagneticFieldState] = None
        
        # Earth's magnetic dipole moment (A⋅m²)
        self.magnetic_moment = 7.94e22  # Tesla⋅m³
        
    def calculate_magnetic_field(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate Earth's magnetic field vector at given position
        Using a simple dipole model
        """
        r = np.linalg.norm(position)
        B0 = (1e-9 * self.magnetic_moment) / (r**3)  # Convert to Tesla
        
        # Align with magnetic poles (simplified)
        magnetic_axis = np.array([0, 0, 1])  # Assuming aligned with Earth's axis for now
        
        B = B0 * (3 * np.dot(magnetic_axis, position/r) * (position/r) - magnetic_axis)
        
        return B