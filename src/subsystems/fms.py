from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional, List, Union
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from filelock import FileLock
from PIL import Image
import io
from protocols.events import EventType

class FileType(Enum):
    """Types of files managed by FMS."""
    IMAGE_RAW = auto()      # Raw camera data
    IMAGE_PROCESSED = auto() # Processed JPG/PNG
    IMAGE_THUMBNAIL = auto() # Small preview
    TELEMETRY = auto()
    CONFIG = auto()
    SEQUENCE = auto()
    LOG = auto()
    FIRMWARE = auto()

class StorageArea(Enum):
    """Storage partitions."""
    PAYLOAD = auto()     # For image data
    SYSTEM = auto()      # For ATS/RTS and config files
    TELEMETRY = auto()   # For stored telemetry
    LOG = auto()         # For system logs

@dataclass
class FileSystemStats:
    """Storage area statistics."""
    total_bytes: int
    used_bytes: int
    free_bytes: int
    file_count: int
    fragmentation: float
    last_write_time: datetime
    write_errors: int
    read_errors: int

@dataclass
class FileMetadata:
    """File metadata."""
    file_id: str
    file_type: FileType
    storage_area: StorageArea
    size_bytes: int
    creation_time: datetime
    last_access_time: datetime
    checksum: str
    compression_ratio: Optional[float] = None
    encryption_type: Optional[str] = None
    attributes: Dict = None

@dataclass
class StoragePartition:
    """Storage partition information."""
    total_bytes: int
    used_bytes: int
    max_files: int
    files: Dict[str, FileMetadata]
    read_only: bool = False
    encrypted: bool = False

@dataclass
class FMSTelemetry:
    """File Management System telemetry data."""
    timestamp: datetime
    system_enabled: bool
    operating_mode: str
    payload_storage: FileSystemStats
    telemetry_storage: FileSystemStats
    software_storage: FileSystemStats
    configuration_storage: FileSystemStats
    files_written: int
    files_read: int
    files_deleted: int
    bytes_written: int
    bytes_read: int
    failed_operations: int
    active_transfers: int
    queued_transfers: int
    failed_transfers: int
    last_transfer_rate_bps: float
    transfer_queue_bytes: int
    image_files: int
    telemetry_files: int
    log_files: int
    config_files: int
    software_files: int
    garbage_collection_running: bool
    last_gc_duration_s: float
    disk_health: float
    fault_flags: int
    board_temp: float
    uptime_seconds: int

class StorageError(Exception):
    """Base exception for storage-related errors."""
    pass

class FileManagementSystem:
    """Spacecraft file management system."""
    
    def __init__(self, config):
        """
        Initialize file management system.
        
        Args:
            config: FMS configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage partitions
        self.storage = {
            StorageArea.PAYLOAD: StoragePartition(
                total_bytes=1024*1024*1024,  # 1 GB for payload
                used_bytes=0,
                max_files=1000,
                files={},
                encrypted=True
            ),
            StorageArea.SYSTEM: StoragePartition(
                total_bytes=64*1024*1024,    # 64 MB for system
                used_bytes=0,
                max_files=100,
                files={},
                read_only=True
            ),
            StorageArea.TELEMETRY: StoragePartition(
                total_bytes=256*1024*1024,   # 256 MB for telemetry
                used_bytes=0,
                max_files=10000,
                files={}
            ),
            StorageArea.LOG: StoragePartition(
                total_bytes=32*1024*1024,    # 32 MB for logs
                used_bytes=0,
                max_files=1000,
                files={}
            )
        }
        
        # In-memory data store for quick access
        self.data_store: Dict[str, bytes] = {}
        
        # File operation locks
        self.file_locks: Dict[str, bool] = {}
        
        self.logger.info("File Management System initialized")

    def store_image(self, 
                   image_path: str, 
                   metadata: Dict = None) -> Optional[str]:
        """
        Store captured image with metadata.
        
        Args:
            image_path: Path to image file
            metadata: Image metadata (timestamp, position, etc.)
            
        Returns:
            File ID if successful, None otherwise
        """
        try:
            # Read image file
            with open(image_path, 'rb') as f:
                image_data = f.read()
                
            # Generate unique file ID
            timestamp = datetime.utcnow()
            file_id = f"IMG_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Create metadata
            file_metadata = FileMetadata(
                file_id=file_id,
                file_type=FileType.IMAGE_PROCESSED,
                storage_area=StorageArea.PAYLOAD,
                size_bytes=len(image_data),
                creation_time=timestamp,
                last_access_time=timestamp,
                checksum=hashlib.sha256(image_data).hexdigest(),
                compression_ratio=None,  # JPG already compressed
                encryption_type=None,
                attributes=metadata or {}
            )
            
            # Store in payload partition
            partition = self.storage[StorageArea.PAYLOAD]
            
            # Check storage limits
            if partition.used_bytes + len(image_data) > partition.total_bytes:
                self.logger.error("Insufficient storage space for image")
                return None
                
            # Store data and metadata
            partition.files[file_id] = file_metadata
            partition.used_bytes += len(image_data)
            self.data_store[file_id] = image_data
            
            self.logger.info(f"Stored image {file_id} ({len(image_data)} bytes)")
            return file_id
            
        except Exception as e:
            self.logger.error(f"Error storing image: {str(e)}")
            return None
            
    def create_thumbnail(self, file_id: str) -> Optional[str]:
        """Create thumbnail for image preview."""
        try:
            if file_id not in self.data_store:
                return None
                
            # Load image
            image_data = self.data_store[file_id]
            image = Image.open(io.BytesIO(image_data))
            
            # Create thumbnail
            thumbnail_size = (256, 256)
            image.thumbnail(thumbnail_size)
            
            # Save thumbnail
            thumb_buffer = io.BytesIO()
            image.save(thumb_buffer, format='JPEG', quality=70)
            thumb_data = thumb_buffer.getvalue()
            
            # Store thumbnail
            thumb_id = f"{file_id}_thumb"
            self.store_file(
                thumb_data,
                FileType.IMAGE_THUMBNAIL,
                StorageArea.PAYLOAD,
                file_id=thumb_id
            )
            
            return thumb_id
            
        except Exception as e:
            self.logger.error(f"Error creating thumbnail: {str(e)}")
            return None
        
    def store_file(self, 
                  data: Union[bytes, Dict], 
                  file_type: FileType,
                  storage_area: StorageArea,
                  compress: bool = True,
                  encrypt: bool = False,
                  attributes: Dict = None) -> Optional[str]:
        """
        Store a new file in the specified storage area.
        
        Args:
            data: File data (bytes or dictionary)
            file_type: Type of file
            storage_area: Storage area to use
            compress: Whether to compress the data
            encrypt: Whether to encrypt the data
            attributes: Additional file attributes
            
        Returns:
            File ID if successful, None otherwise
        """
        file_path = self._get_file_path(file_id, storage_area)
        lock_path = f"{file_path}.lock"
        with FileLock(lock_path):
            with open(file_path, 'wb') as f:
                f.write(processed_data)
                
        try:
            # Convert dict to bytes if necessary
            if isinstance(data, dict):
                data = json.dumps(data).encode()
                
            # Compress if requested
            original_size = len(data)
            if compress:
                data = self._compress_data(data)
                compression_ratio = len(data) / original_size
            else:
                compression_ratio = 1.0
                
            # Encrypt if requested or if partition requires it
            if encrypt or self.storage[storage_area].encrypted:
                data = self._encrypt_data(data)
                
            processed_data = data  # Store processed data
                
            file_path = self._get_file_path(file_id, storage_area)
            lock_path = f"{file_path}.lock"
            with FileLock(lock_path):
                with open(file_path, 'wb') as f:
                    f.write(processed_data)
                
            # Check storage capacity
            partition = self.storage[storage_area]
            if (partition.used_bytes + len(data) > partition.total_bytes or
                len(partition.files) >= partition.max_files):
                raise StorageError(f"Storage area {storage_area.name} full")
                
            # Generate file ID and checksum
            timestamp = datetime.utcnow()
            file_id = f"{file_type.name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            checksum = hashlib.sha256(data).hexdigest()
            
            # Create metadata
            metadata = FileMetadata(
                file_id=file_id,
                file_type=file_type,
                storage_area=storage_area,
                size_bytes=len(data),
                creation_time=timestamp,
                last_access_time=timestamp,
                checksum=checksum,
                compression_ratio=compression_ratio,
                encryption_type="AES-256" if encrypt else None,
                attributes=attributes or {}
            )
            
            # Store file and metadata
            partition.files[file_id] = metadata
            partition.used_bytes += len(data)
            self.data_store[file_id] = data
            
            self.logger.info(f"Stored file {file_id} in {storage_area.name}")
            return file_id
            
        except Exception as e:
            self.logger.error(f"Error storing file: {str(e)}")
            return None
            
    def retrieve_file(self, 
                     file_id: str,
                     decrypt: bool = True,
                     decompress: bool = True) -> Optional[Union[bytes, Dict]]:
        """
        Retrieve a file by its ID.
        
        Args:
            file_id: File identifier
            decrypt: Whether to decrypt the data
            decompress: Whether to decompress the data
            
        Returns:
            File data if successful, None otherwise
        """
        try:
            if file_id not in self.data_store:
                raise StorageError(f"File {file_id} not found")
                
            # Get file data and metadata
            data = self.data_store[file_id]
            metadata = None
            for partition in self.storage.values():
                if file_id in partition.files:
                    metadata = partition.files[file_id]
                    break
                    
            if metadata is None:
                raise StorageError(f"Metadata for file {file_id} not found")
                
            # Decrypt if necessary
            if decrypt and metadata.encryption_type:
                data = self._decrypt_data(data)
                
            # Decompress if necessary
            if decompress and metadata.compression_ratio and metadata.compression_ratio < 1.0:
                data = self._decompress_data(data)
                
            # Update last access time
            metadata.last_access_time = datetime.utcnow()
            
            # Convert to dict if it's a JSON file
            if metadata.file_type in [FileType.CONFIG, FileType.TELEMETRY]:
                try:
                    return json.loads(data.decode())
                except json.JSONDecodeError:
                    return data
                    
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving file {file_id}: {str(e)}")
            return None
            
    def delete_file(self, file_id: str) -> bool:
        """Delete a file by its ID."""
        try:
            for storage_area, partition in self.storage.items():
                if file_id in partition.files:
                    if partition.read_only:
                        raise StorageError(f"Cannot delete file from read-only partition {storage_area.name}")
                        
                    metadata = partition.files[file_id]
                    partition.used_bytes -= metadata.size_bytes
                    del partition.files[file_id]
                    del self.data_store[file_id]
                    
                    self.logger.info(f"Deleted file {file_id} from {storage_area.name}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error deleting file {file_id}: {str(e)}")
            return False
            
    def list_files(self, 
                   storage_area: Optional[StorageArea] = None,
                   file_type: Optional[FileType] = None,
                   pattern: Optional[str] = None) -> List[FileMetadata]:
        """
        List files with optional filtering.
        
        Args:
            storage_area: Filter by storage area
            file_type: Filter by file type
            pattern: Filter by filename pattern
            
        Returns:
            List of file metadata
        """
        files = []
        for area, partition in self.storage.items():
            if storage_area and area != storage_area:
                continue
                
            for metadata in partition.files.values():
                if file_type and metadata.file_type != file_type:
                    continue
                if pattern and pattern not in metadata.file_id:
                    continue
                files.append(metadata)
                
        return sorted(files, key=lambda x: x.creation_time, reverse=True)
        
    def get_storage_status(self) -> Dict:
        """Get storage status for all partitions."""
        return {
            area.name: {
                'total_bytes': partition.total_bytes,
                'used_bytes': partition.used_bytes,
                'free_bytes': partition.total_bytes - partition.used_bytes,
                'used_percentage': (partition.used_bytes / partition.total_bytes) * 100,
                'file_count': len(partition.files),
                'max_files': partition.max_files,
                'read_only': partition.read_only,
                'encrypted': partition.encrypted
            }
            for area, partition in self.storage.items()
        }
        
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using zlib."""
        import zlib
        return zlib.compress(data)
        
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using zlib."""
        import zlib
        return zlib.decompress(data)
        
    def _encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using AES-256."""
        # Implementation depends on encryption requirements
        return data
        
    def _decrypt_data(self, data: bytes) -> bytes:
        """Decrypt data using AES-256."""
        # Implementation depends on encryption requirements
        return data

    def get_telemetry(self) -> FMSTelemetry:
        """Generate file management system telemetry packet."""
        def get_storage_stats(area: str) -> FileSystemStats:
            storage = self.storage_areas[area]
            return FileSystemStats(
                total_bytes=storage.get_total_space(),
                used_bytes=storage.get_used_space(),
                free_bytes=storage.get_free_space(),
                file_count=storage.get_file_count(),
                fragmentation=storage.get_fragmentation(),
                last_write_time=storage.get_last_write_time(),
                write_errors=storage.get_write_errors(),
                read_errors=storage.get_read_errors()
            )

        return FMSTelemetry(
            timestamp=datetime.utcnow(),
            system_enabled=self.is_enabled(),
            operating_mode=self.mode.name,
            
            # Storage Areas
            payload_storage=get_storage_stats('payload'),
            telemetry_storage=get_storage_stats('telemetry'),
            software_storage=get_storage_stats('software'),
            configuration_storage=get_storage_stats('configuration'),
            
            # File Operations
            files_written=self.operation_stats['files_written'],
            files_read=self.operation_stats['files_read'],
            files_deleted=self.operation_stats['files_deleted'],
            bytes_written=self.operation_stats['bytes_written'],
            bytes_read=self.operation_stats['bytes_read'],
            failed_operations=self.operation_stats['failed_operations'],
            
            # Transfer Status
            active_transfers=len(self.active_transfers),
            queued_transfers=len(self.transfer_queue),
            failed_transfers=self.transfer_stats['failed'],
            last_transfer_rate_bps=self.transfer_stats['last_rate'],
            transfer_queue_bytes=self.get_queued_bytes(),
            
            # File Types
            image_files=self.get_file_count_by_type('image'),
            telemetry_files=self.get_file_count_by_type('telemetry'),
            log_files=self.get_file_count_by_type('log'),
            config_files=self.get_file_count_by_type('config'),
            software_files=self.get_file_count_by_type('software'),
            
            # System Status
            garbage_collection_running=self.gc_running,
            last_gc_duration_s=self.gc_stats['last_duration'],
            disk_health=self.get_disk_health(),
            fault_flags=self.get_fault_flags(),
            board_temp=self.get_board_temperature(),
            uptime_seconds=int((datetime.utcnow() - self.start_time).total_seconds())
        )

    def publish_telemetry(self):
        """Publish file management system telemetry packet."""
        telemetry = self.get_telemetry()
        packet = telemetry.to_ccsds()
        self.event_bus.publish(
            EventType.TELEMETRY,
            "FMS",
            {"packet": packet.pack()}
        )

    def get_fault_flags(self) -> int:
        """Get file management system fault flags."""
        flags = 0
        
        # Storage space warnings/errors
        for i, area in enumerate(self.storage_areas.values()):
            if area.get_free_space() < self.min_free_space:
                flags |= (1 << i)
            if area.get_fragmentation() > self.max_fragmentation:
                flags |= (1 << (i + 8))
                
        # Operation errors
        if self.operation_stats['failed_operations'] > self.max_failed_ops:
            flags |= 0x10000
            
        # Transfer errors
        if self.transfer_stats['failed'] > self.max_failed_transfers:
            flags |= 0x20000
            
        # System health
        if self.get_disk_health() < 0.5:
            flags |= 0x40000
        if len(self.transfer_queue) >= self.max_queue_size:
            flags |= 0x80000
            
        return flags