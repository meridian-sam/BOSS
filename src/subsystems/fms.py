from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional, List, Union
from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path

class FileType(Enum):
    """Types of files managed by FMS."""
    IMAGE = auto()
    TELEMETRY = auto()
    CONFIG = auto()
    SEQUENCE = auto()  # ATS/RTS sequences
    LOG = auto()
    FIRMWARE = auto()

class StorageArea(Enum):
    """Storage partitions."""
    PAYLOAD = auto()     # For image data
    SYSTEM = auto()      # For ATS/RTS and config files
    TELEMETRY = auto()   # For stored telemetry
    LOG = auto()         # For system logs

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