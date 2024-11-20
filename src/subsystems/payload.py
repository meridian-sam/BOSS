from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging

@dataclass
class Image:
    timestamp: datetime
    image_id: str
    size_bytes: int
    metadata: dict
    compressed: bool = True

class CameraState(Enum):
    OFF = "OFF"
    STANDBY = "STANDBY"
    IMAGING = "IMAGING"

class CameraPayload:
    def __init__(self, config):
        self.config = config
        self.state = CameraState.OFF
        self.images = []  # List to store captured images
        self.last_capture_time = None
        self.power_consumption = 0.0
        
        # Calculate camera parameters
        self.fov_rad = 2 * np.arctan((self.config.sensor_width_pixels * 
                                     self.config.pixel_size_um * 1e-6) / 
                                    (2 * self.config.focal_length_mm * 1e-3))
        
        # Ground sampling distance at 500km (meters per pixel)
        self.gsd_at_500km = (500000 * self.config.pixel_size_um * 1e-6 / 
                            (self.config.focal_length_mm * 1e-3))
        
        logging.info(f"Camera initialized. FOV: {np.degrees(self.fov_rad):.1f} degrees, "
                    f"GSD @ 500km: {self.gsd_at_500km:.2f} m/pixel")
    
    def capture_image(self, position: np.ndarray, attitude_quaternion: np.ndarray, 
                     sun_in_fov: bool) -> bool:
        """
        Attempt to capture an image if conditions are suitable
        Returns True if image was captured successfully
        """
        current_time = datetime.utcnow()
        
        # Check if camera is ready
        if self.state == CameraState.OFF:
            logging.warning("Cannot capture image: Camera is OFF")
            return False
        
        # Check if enough time has passed since last capture
        if (self.last_capture_time and 
            (current_time - self.last_capture_time).total_seconds() < 1/self.config.max_frame_rate_hz):
            logging.warning("Cannot capture image: Frame rate limit")
            return False
        
        # Check storage capacity
        if len(self.images) >= self.config.max_images:
            logging.warning("Cannot capture image: Storage full")
            return False
        
        # Check if sun is in FOV
        if sun_in_fov:
            logging.warning("Cannot capture image: Sun in FOV")
            return False
        
        # Calculate image parameters
        altitude = np.linalg.norm(position) - 6371.0  # km
        gsd = self.gsd_at_500km * (altitude / 500.0)  # Scale GSD with altitude
        
        # Calculate ground footprint
        swath_width = gsd * self.config.sensor_width_pixels / 1000.0  # km
        swath_height = gsd * self.config.sensor_height_pixels / 1000.0  # km
        
        # Create image metadata
        metadata = {
            'position_km': position.tolist(),
            'attitude_quaternion': attitude_quaternion.tolist(),
            'altitude_km': altitude,
            'gsd_m': gsd,
            'swath_width_km': swath_width,
            'swath_height_km': swath_height
        }
        
        # Calculate image size (with compression)
        raw_size = (self.config.sensor_width_pixels * 
                   self.config.sensor_height_pixels * 
                   self.config.bit_depth) / 8  # bytes
        compressed_size = int(raw_size * self.config.compression_ratio)
        
        # Create and store image
        image = Image(
            timestamp=current_time,
            image_id=f"IMG_{current_time.strftime('%Y%m%d_%H%M%S')}",
            size_bytes=compressed_size,
            metadata=metadata
        )
        self.images.append(image)
        self.last_capture_time = current_time
        
        logging.info(f"Image captured: {image.image_id}, Size: {compressed_size/1e6:.1f}MB")
        return True
    
    def set_state(self, new_state: CameraState):
        """Change camera state and update power consumption"""
        self.state = new_state
        if new_state == CameraState.OFF:
            self.power_consumption = 0.0
        elif new_state == CameraState.STANDBY:
            self.power_consumption = self.config.standby_power_w
        elif new_state == CameraState.IMAGING:
            self.power_consumption = self.config.power_consumption_w
    
    def delete_image(self, image_id: str) -> bool:
        """Delete an image from storage"""
        initial_length = len(self.images)
        self.images = [img for img in self.images if img.image_id != image_id]
        return len(self.images) < initial_length
    
    def get_telemetry(self) -> dict:
        """Return current payload telemetry"""
        return {
            'state': self.state.value,
            'power_consumption_w': self.power_consumption,
            'storage_used': len(self.images),
            'storage_available': self.config.max_images - len(self.images),
            'last_capture_time': self.last_capture_time.isoformat() if self.last_capture_time else None,
            'storage_percentage': (len(self.images) / self.config.max_images) * 100
        }