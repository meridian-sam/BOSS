from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import numpy as np
import logging
from .fms import FileManagementSystem, FileType, StorageArea

class PayloadState(Enum):
    """Payload operational states."""
    OFF = auto()
    STANDBY = auto()
    ACTIVE = auto()
    CALIBRATING = auto()
    ERROR = auto()

class ImageQuality(Enum):
    """Image quality settings."""
    LOW = auto()      # 8-bit, high compression
    MEDIUM = auto()   # 12-bit, medium compression
    HIGH = auto()     # 16-bit, low compression
    RAW = auto()      # No compression

@dataclass
class CameraConfig:
    """Camera configuration."""
    exposure_time_ms: float
    gain: float
    binning: int
    quality: ImageQuality
    compression_ratio: float

@dataclass
class ImageMetadata:
    """Image metadata."""
    image_id: str
    timestamp: datetime
    exposure_time_ms: float
    gain: float
    binning: int
    quality: ImageQuality
    size_bytes: int
    compression_ratio: float
    position: np.ndarray  # Spacecraft position when image was taken
    attitude: np.ndarray  # Spacecraft attitude when image was taken
    target: Optional[str] = None  # Target name if applicable

@dataclass
class PayloadStatus:
    """Payload system status."""
    state: PayloadState
    temperature_c: float
    power_consumption_w: float
    storage_used_bytes: int
    images_captured: int
    last_image_time: Optional[datetime]
    calibration_age_hours: float
    error_count: int
    timestamp: datetime
    temperature_c: float = 20.0  # Add temperature tracking

class PayloadManager:
    """Spacecraft payload (camera) manager."""
    
    def __init__(self, config, fms: FileManagementSystem):
        """
        Initialize payload manager.
        
        Args:
            config: Payload configuration
            fms: File management system
        """
        self.config = config
        self.fms = fms
        self.logger = logging.getLogger(__name__)
        
        # Initialize camera configuration
        self.camera_config = CameraConfig(
            exposure_time_ms=10.0,
            gain=1.0,
            binning=1,
            quality=ImageQuality.HIGH,
            compression_ratio=0.7
        )
        
        # Initialize state
        self.status = PayloadStatus(
            state=PayloadState.OFF,
            temperature_c=20.0,
            power_consumption_w=0.0,
            storage_used_bytes=0,
            images_captured=0,
            last_image_time=None,
            calibration_age_hours=0.0,
            error_count=0,
            timestamp=datetime.utcnow()
        )
        
        # Image capture parameters
        self.sensor_width = 2048
        self.sensor_height = 1536
        self.bit_depth = 12
        self.max_frame_rate = 1.0  # Hz
        self.last_capture_time = datetime.utcnow()
        
        # Temperature limits
        self.temp_limits = {
            'min_operational': -20.0,
            'max_operational': 60.0,
            'warning_low': -10.0,
            'warning_high': 50.0
        }
        
        self.logger.info("Payload manager initialized")
        
    def update(self, dt: float) -> Dict:
        """
        Update payload status.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Payload telemetry dictionary
        """
        try:
            # Update timestamp
            self.status.timestamp = datetime.utcnow()
            
            # Update calibration age
            if self.status.state != PayloadState.OFF:
                self.status.calibration_age_hours += dt / 3600.0
            
            # Update temperature (simulated)
            self._update_temperature(dt)
            
            # Check temperature limits
            self._check_temperature_limits()
            
            # Update power consumption
            self._update_power_consumption()

            if hasattr(self, 'thermal_subsystem'):
                self.status.temperature_c = self.thermal_subsystem.zones[ThermalZone.PAYLOAD].average_temp
            if hasattr(self, 'thermal_subsystem'):
                zone_state = self.thermal_subsystem.zones[ThermalZone.PAYLOAD]
                if zone_state.average_temp < self.temp_limits['min_operational'] or \
                zone_state.average_temp > self.temp_limits['max_operational']:
                    self.status.state = PayloadState.OFF

            return self.get_telemetry()
            
        except Exception as e:
            self.logger.error(f"Error in payload update: {str(e)}")
            self.status.error_count += 1
            return self.get_telemetry()
            
    def capture_image(self, 
                     exposure_time_ms: Optional[float] = None,
                     target: Optional[str] = None) -> Optional[str]:
        """
        Capture an image with the current configuration.
        
        Args:
            exposure_time_ms: Optional override for exposure time
            target: Optional target name
            
        Returns:
            Image ID if successful, None otherwise
        """
        try:
            # Check if camera is ready
            if not self._can_capture():
                return None
                
            # Update exposure time if provided
            if exposure_time_ms is not None:
                self.camera_config.exposure_time_ms = exposure_time_ms
                
            # Simulate image capture
            image_data = self._simulate_image_capture()
            
            # Create metadata
            metadata = ImageMetadata(
                image_id=f"IMG_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.utcnow(),
                exposure_time_ms=self.camera_config.exposure_time_ms,
                gain=self.camera_config.gain,
                binning=self.camera_config.binning,
                quality=self.camera_config.quality,
                size_bytes=len(image_data),
                compression_ratio=self.camera_config.compression_ratio,
                position=np.zeros(3),  # Should come from ADCS
                attitude=np.zeros(4),  # Should come from ADCS
                target=target
            )
            
            # Store image
            image_id = self.fms.store_file(
                image_data,
                FileType.IMAGE,
                StorageArea.PAYLOAD,
                compress=True,
                attributes=metadata.__dict__
            )
            
            if image_id:
                self.status.images_captured += 1
                self.status.last_image_time = datetime.utcnow()
                self.last_capture_time = datetime.utcnow()
                
            return image_id
            
        except Exception as e:
            self.logger.error(f"Error capturing image: {str(e)}")
            self.status.error_count += 1
            return None
            
    def set_state(self, state: PayloadState):
        """Set payload state."""
        if state == self.status.state:
            return
            
        self.logger.info(f"Payload state changing from {self.status.state} to {state}")
        
        if state == PayloadState.ACTIVE:
            if not self._check_operational_limits():
                self.logger.error("Cannot activate payload - outside operational limits")
                return
                
        self.status.state = state
        
    def calibrate(self) -> bool:
        """Perform payload calibration."""
        try:
            if self.status.state not in [PayloadState.STANDBY, PayloadState.ACTIVE]:
                return False
                
            self.status.state = PayloadState.CALIBRATING
            
            # Perform calibration sequence
            # Implementation depends on specific payload requirements
            
            self.status.calibration_age_hours = 0.0
            self.status.state = PayloadState.ACTIVE
            return True
            
        except Exception as e:
            self.logger.error(f"Calibration error: {str(e)}")
            self.status.error_count += 1
            self.status.state = PayloadState.ERROR
            return False
            
    def get_telemetry(self) -> Dict:
        """Get payload telemetry."""
        return {
            'state': self.status.state.name,
            'temperature_c': self.status.temperature_c,
            'power_consumption_w': self.status.power_consumption_w,
            'storage_used_bytes': self.status.storage_used_bytes,
            'images_captured': self.status.images_captured,
            'last_image_time': self.status.last_image_time.isoformat() 
                if self.status.last_image_time else None,
            'calibration_age_hours': self.status.calibration_age_hours,
            'error_count': self.status.error_count,
            'timestamp': self.status.timestamp.isoformat()
        }
        
    def _can_capture(self) -> bool:
        """Check if camera can capture an image."""
        if self.status.state != PayloadState.ACTIVE:
            return False
            
        if not self._check_operational_limits():
            return False
            
        # Check frame rate limit
        time_since_last = (datetime.utcnow() - self.last_capture_time).total_seconds()
        if time_since_last < (1.0 / self.max_frame_rate):
            return False
            
        return True
        
    def _check_operational_limits(self) -> bool:
        """Check if payload is within operational limits."""
        if (self.status.temperature_c < self.temp_limits['min_operational'] or
            self.status.temperature_c > self.temp_limits['max_operational']):
            return False
        return True
        
    def _update_temperature(self, dt: float):
        """Update payload temperature (simulated)."""
        if self.status.state == PayloadState.OFF:
            # Cool down
            self.status.temperature_c = max(
                20.0,
                self.status.temperature_c - 0.1 * dt
            )
        else:
            # Heat up based on activity
            heat_rate = {
                PayloadState.STANDBY: 0.05,
                PayloadState.ACTIVE: 0.1,
                PayloadState.CALIBRATING: 0.15,
                PayloadState.ERROR: 0.0
            }[self.status.state]
            
            self.status.temperature_c += heat_rate * dt
            
    def _check_temperature_limits(self):
        """Check temperature limits and update state if necessary."""
        if self.status.temperature_c < self.temp_limits['min_operational']:
            self.logger.warning("Payload temperature below operational limit")
            self.set_state(PayloadState.OFF)
        elif self.status.temperature_c > self.temp_limits['max_operational']:
            self.logger.warning("Payload temperature above operational limit")
            self.set_state(PayloadState.OFF)
            
    def _update_power_consumption(self):
        """Update power consumption based on state."""
        power_draw = {
            PayloadState.OFF: 0.0,
            PayloadState.STANDBY: 0.5,
            PayloadState.ACTIVE: 2.5,
            PayloadState.CALIBRATING: 3.0,
            PayloadState.ERROR: 0.1
        }[self.status.state]
        
        self.status.power_consumption_w = power_draw
        
    def _simulate_image_capture(self) -> bytes:
        """Simulate image capture (placeholder)."""
        # Create dummy image data based on configuration
        image_size = (
            self.sensor_width // self.camera_config.binning,
            self.sensor_height // self.camera_config.binning
        )
        
        # Create random image data
        image_data = np.random.randint(
            0,
            2 ** self.bit_depth,
            size=image_size,
            dtype=np.uint16
        )
        
        return image_data.tobytes()