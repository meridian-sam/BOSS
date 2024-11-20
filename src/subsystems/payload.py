from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import numpy as np
import logging
from .fms import FileManagementSystem, FileType, StorageArea
import requests
import io
from PIL import Image, ImageEnhance
import ephem
from pathlib import Path
import os
from .thermal import ThermalZone
from protocols.events import EventType

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

@dataclass
class PayloadTelemetry:
    """Payload telemetry data."""
    timestamp: datetime
    payload_enabled: bool
    operating_mode: str
    exposure_time_ms: float
    gain: float
    binning: int
    region_of_interest: Tuple[int, int, int, int]
    color_mode: str
    sensor_temp: float
    sensor_voltage: float
    sensor_current: float
    ccd_temp: float
    images_captured: int
    images_stored: int
    failed_captures: int
    last_image_quality: float
    last_image_brightness: float
    last_image_contrast: float
    storage_used_bytes: int
    storage_total_bytes: int
    images_queued: int
    compression_ratio: float
    processing_queue_length: int
    last_processing_time_ms: float
    processing_errors: int
    voltage_3v3: float
    voltage_5v: float
    current_3v3: float
    current_5v: float
    power_consumption: float
    fault_flags: int
    board_temp: float
    uptime_seconds: int

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
        
        # Camera parameters
        self.fov = 2.0  # degrees
        self.resolution = (2048, 1536)
        self.pixel_size = 5.5e-6  # 5.5 microns
        self.focal_length = 0.5  # 500mm

        # API keys and configuration
        self.google_maps_api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        self.nasa_api_key = os.getenv('NASA_API_KEY')
        
        # Image cache directory
        self.cache_dir = Path("cache/images")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
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
                     lat: float, 
                     lon: float, 
                     alt: float,
                     quaternion: np.ndarray,
                     timestamp: datetime) -> Optional[str]:
        """
        Capture an image based on spacecraft position and attitude.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters
            quaternion: Spacecraft attitude quaternion
            timestamp: Image capture time
            
        Returns:
            Path to captured image file or None if capture failed
        """
        try:
            # Calculate pointing vector from quaternion
            pointing = self._quaternion_to_vector(quaternion)
            
            # Check if pointing at Earth
            if self._is_nadir_pointing(pointing):
                # Calculate ground footprint
                footprint = self._calculate_footprint(lat, lon, alt)
                
                # Get illumination conditions
                sun_elevation = self._calculate_sun_elevation(lat, lon, timestamp)
                
                # Get base map from Google Static Maps API
                base_image = self._get_earth_image(
                    footprint['center_lat'],
                    footprint['center_lon'],
                    footprint['width_m'],
                    footprint['height_m']
                )
                
                if base_image:
                    # Get cloud coverage from NASA GIBS
                    cloud_image = self._get_cloud_coverage(
                        footprint['center_lat'],
                        footprint['center_lon'],
                        timestamp
                    )
                    
                    # Combine images and adjust for lighting
                    final_image = self._process_image(
                        base_image, 
                        cloud_image,
                        sun_elevation
                    )
                    
                    # Save image
                    image_path = self.cache_dir / f"capture_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
                    final_image.save(image_path, "JPEG", quality=95)
                    
                    return str(image_path)
                    
            else:
                # Pointing at space, generate star field
                return self._capture_star_field(pointing, timestamp)
                
            return None
            
        except Exception as e:
            self.logger.error(f"Image capture error: {str(e)}")
            return None

    def capture_and_store_image(self, 
                          lat: float, 
                          lon: float, 
                          alt: float,
                          quaternion: np.ndarray) -> Optional[Dict]:
        """Capture image and store in FMS."""
        try:
            # Capture image
            timestamp = datetime.utcnow()
            image_path = self.capture_image(lat, lon, alt, quaternion, timestamp)
            
            if not image_path:
                return None
                
            # Prepare metadata
            metadata = {
                'timestamp': timestamp.isoformat(),
                'position': {
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt
                },
                'attitude': quaternion.tolist(),
                'camera_config': {
                    'exposure_time_ms': self.camera_config.exposure_time_ms,
                    'gain': self.camera_config.gain,
                    'binning': self.camera_config.binning,
                    'quality': self.camera_config.quality.name
                }
            }
            
            # Store in FMS
            file_id = self.fms.store_image(image_path, metadata)
            
            if file_id:
                # Create thumbnail
                thumb_id = self.fms.create_thumbnail(file_id)
                
                return {
                    'file_id': file_id,
                    'thumbnail_id': thumb_id,
                    'metadata': metadata
                }
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error in capture and store: {str(e)}")
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
            
    def get_telemetry(self) -> PayloadTelemetry:
        """Generate payload subsystem telemetry packet."""
        return PayloadTelemetry(
            timestamp=datetime.utcnow(),
            payload_enabled=self.is_enabled(),
            operating_mode=self.mode.name,
            
            # Camera Settings
            exposure_time_ms=self.camera.get_exposure_time(),
            gain=self.camera.get_gain(),
            binning=self.camera.get_binning(),
            region_of_interest=self.camera.get_roi(),
            color_mode=self.camera.get_color_mode(),
            
            # Sensor Status
            sensor_temp=self.camera.get_sensor_temperature(),
            sensor_voltage=self.camera.get_sensor_voltage(),
            sensor_current=self.camera.get_sensor_current(),
            ccd_temp=self.camera.get_ccd_temperature(),
            
            # Image Stats
            images_captured=self.image_stats['captured'],
            images_stored=self.image_stats['stored'],
            failed_captures=self.image_stats['failed'],
            last_image_quality=self.image_stats['last_quality'],
            last_image_brightness=self.image_stats['last_brightness'],
            last_image_contrast=self.image_stats['last_contrast'],
            
            # Storage Status
            storage_used_bytes=self.storage.get_used_space(),
            storage_total_bytes=self.storage.get_total_space(),
            images_queued=len(self.processing_queue),
            compression_ratio=self.get_compression_ratio(),
            
            # Processing Status
            processing_queue_length=len(self.processing_queue),
            last_processing_time_ms=self.processing_stats['last_time_ms'],
            processing_errors=self.processing_stats['errors'],
            
            # Power Status
            voltage_3v3=self.power_monitor.get_3v3_voltage(),
            voltage_5v=self.power_monitor.get_5v_voltage(),
            current_3v3=self.power_monitor.get_3v3_current(),
            current_5v=self.power_monitor.get_5v_current(),
            power_consumption=self.get_power_consumption(),
            
            # System Status
            fault_flags=self.get_fault_flags(),
            board_temp=self.get_board_temperature(),
            uptime_seconds=int((datetime.utcnow() - self.start_time).total_seconds())
        )

    def publish_telemetry(self):
        """Publish payload telemetry packet."""
        telemetry = self.get_telemetry()
        packet = telemetry.to_ccsds()
        self.event_bus.publish(
            EventType.TELEMETRY,
            "PAYLOAD",
            {"packet": packet.pack()}
        )

    def get_fault_flags(self) -> int:
        """Get payload subsystem fault flags."""
        flags = 0
        
        # Camera faults
        if self.camera.has_fault():
            flags |= 0x01
        if self.camera.temperature_exceeded():
            flags |= 0x02
        if self.camera.power_fault():
            flags |= 0x04
            
        # Storage faults
        if self.storage.is_full():
            flags |= 0x10
        if self.storage.has_error():
            flags |= 0x20
            
        # Processing faults
        if len(self.processing_queue) >= self.max_queue_size:
            flags |= 0x100
        if self.processing_stats['errors'] > self.max_processing_errors:
            flags |= 0x200
            
        # Power faults
        if not self.power_monitor.voltages_in_range():
            flags |= 0x1000
        if self.power_consumption > self.max_power:
            flags |= 0x2000
            
        return flags
        
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
    
    def _get_earth_image(self, 
                        lat: float, 
                        lon: float, 
                        width_m: float, 
                        height_m: float) -> Optional[Image.Image]:
        """Get Earth image from Google Static Maps API."""
        try:
            # Calculate zoom level based on footprint
            zoom = self._calculate_zoom_level(width_m)
            
            # Construct API URL
            url = (
                f"https://maps.googleapis.com/maps/api/staticmap?"
                f"center={lat},{lon}&zoom={zoom}&size={self.resolution[0]}x{self.resolution[1]}"
                f"&maptype=satellite&key={self.google_maps_api_key}"
            )
            
            response = requests.get(url)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            return None
            
        except Exception as e:
            self.logger.error(f"Earth image retrieval error: {str(e)}")
            return None

    def _get_cloud_coverage(self, 
                          lat: float, 
                          lon: float, 
                          timestamp: datetime) -> Optional[Image.Image]:
        """Get cloud coverage from NASA GIBS."""
        try:
            # NASA GIBS API endpoint for MODIS cloud coverage
            url = (
                f"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/"
                f"MODIS_Terra_CorrectedReflectance_TrueColor/default/"
                f"{timestamp.strftime('%Y-%m-%d')}/250m/"
                f"{lat}/{lon}/0/0.png"
            )
            
            response = requests.get(url)
            if response.status_code == 200:
                return Image.open(io.BytesIO(response.content))
            return None
            
        except Exception as e:
            self.logger.error(f"Cloud coverage retrieval error: {str(e)}")
            return None

    def _process_image(self, 
                      base_image: Image.Image,
                      cloud_image: Optional[Image.Image],
                      sun_elevation: float) -> Image.Image:
        """Process and combine images with lighting conditions."""
        try:
            # Resize cloud image if available
            if cloud_image:
                cloud_image = cloud_image.resize(base_image.size)
                # Blend cloud layer
                base_image = Image.blend(base_image, cloud_image, 0.3)
            
            # Adjust brightness based on sun elevation
            brightness_factor = np.clip(np.sin(np.radians(sun_elevation)), 0.0, 1.0)
            enhancer = ImageEnhance.Brightness(base_image)
            base_image = enhancer.enhance(brightness_factor)
            
            # Add noise based on sensor characteristics
            noise = np.random.normal(0, 5, base_image.size)
            noise_image = Image.fromarray(noise.astype('uint8'))
            base_image = Image.blend(base_image, noise_image, 0.1)
            
            return base_image
            
        except Exception as e:
            self.logger.error(f"Image processing error: {str(e)}")
            return base_image

    def _calculate_sun_elevation(self, 
                               lat: float, 
                               lon: float, 
                               timestamp: datetime) -> float:
        """Calculate sun elevation angle for given position and time."""
        observer = ephem.Observer()
        observer.lat = str(lat)
        observer.lon = str(lon)
        observer.date = timestamp
        
        sun = ephem.Sun()
        sun.compute(observer)
        
        return float(sun.alt) * 180.0 / np.pi

    def _calculate_footprint(self, 
                           lat: float, 
                           lon: float, 
                           alt: float) -> Dict:
        """Calculate image footprint on ground."""
        # Calculate ground footprint based on FOV and altitude
        footprint_angle = self.fov / 2.0
        footprint_radius = alt * np.tan(np.radians(footprint_angle))
        
        return {
            'center_lat': lat,
            'center_lon': lon,
            'width_m': footprint_radius * 2,
            'height_m': footprint_radius * 2
        }

    def _capture_star_field(self, 
                          pointing: np.ndarray, 
                          timestamp: datetime) -> Optional[str]:
        """Generate star field image when pointing at space."""
        try:
            # Convert pointing to RA/Dec
            ra, dec = self._vector_to_radec(pointing)
            
            # Use Astronomy.net API to get star field
            # Implementation depends on chosen star catalog/API
            return None
            
        except Exception as e:
            self.logger.error(f"Star field generation error: {str(e)}")
            return None