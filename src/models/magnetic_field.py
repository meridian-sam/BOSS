import numpy as np

class MagneticFieldModel:
    def __init__(self):
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