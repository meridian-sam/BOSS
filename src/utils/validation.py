from typing import Any, Dict, List, Type, Union
import numpy as np

def validate_type(value: Any, expected_type: Union[Type, tuple]) -> bool:
    """Validate value is of expected type."""
    return isinstance(value, expected_type)

def validate_range(value: float, min_val: float = None, max_val: float = None) -> bool:
    """Validate numeric value is within range."""
    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True

def validate_quaternion(quat: List[float]) -> bool:
    """Validate quaternion format and normalization."""
    if not isinstance(quat, (list, np.ndarray)) or len(quat) != 4:
        return False
    # Check normalization within tolerance
    norm = np.sqrt(sum(x*x for x in quat))
    return abs(norm - 1.0) < 1e-6

def validate_dict_keys(data: Dict, required_keys: List[str]) -> bool:
    """Validate dictionary contains all required keys."""
    return all(key in data for key in required_keys)