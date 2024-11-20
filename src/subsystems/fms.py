from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import json
import numpy as np
from typing import Dict, List, Optional, Union
import logging
import hashlib

class FileType(Enum):
    IMAGE = "IMAGE"
    SEQUENCE = "SEQUENCE"
    LOG = "LOG"
    CONFIG = "CONFIG"
    TELEMETRY = "TELEMETRY"

class StorageArea(Enum):
    PAYLOAD = "PAYLOAD"     # For image data
    SYSTEM = "SYSTEM"       # For ATS/RTS and config files
    TELEMETRY = "TELEMETRY" # For stored telemetry
    LOG = "LOG"             # For system logs

@dataclass
class FileMetadata:
    file_id: str
    file_type: FileType
    storage_area: StorageArea
    size_bytes: int
    creation_time: datetime
    last_access_time: datetime
    checksum: str
    attributes: dict  # Additional file-specific metadata

@dataclass
class StoragePartition:
    total_bytes: int
    used_bytes: int
    max_files: int
    files: Dict[str, FileMetadata]

class FileManagementSystem:
    def __init__(self, config):
        self.config = config
        
        # Initialize storage partitions
        self.storage = {
            StorageArea.PAYLOAD: StoragePartition(
                total_bytes=1024*1024*1024,  # 1 GB for payload
                used_bytes=0,
                max_files=1000,
                files={}
            ),
            StorageArea.SYSTEM: StoragePartition(
                total_bytes=64*1024*1024,    # 64 MB for system
                used_bytes=0,
                max_files=100,
                files={}
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
        self.data_store = {}
        
        logging.info("File Management System initialized")

    def store_file(self, data: Union[bytes, dict], 
                  file_type: FileType, 
                  storage_area: StorageArea,
                  attributes: dict = None) -> Optional[str]:
        """Store a new file in the specified storage area"""
        try:
            # Calculate size
            if isinstance(data, bytes):
                size_bytes = len(data)
            else:
                size_bytes = len(json.dumps(data).encode())
            
            # Check storage capacity
            partition = self.storage[storage_area]
            if (partition.used_bytes + size_bytes > partition.total_bytes or
                len(partition.files) >= partition.max_files):
                logging.error(f"Storage area {storage_area.value} full")
                return None
            
            # Generate file ID and checksum
            timestamp = datetime.utcnow()
            file_id = f"{file_type.value}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
            checksum = hashlib.sha256(str(data).encode()).hexdigest()
            
            # Create metadata
            metadata = FileMetadata(
                file_id=file_id,
                file_type=file_type,
                storage_area=storage_area,
                size_bytes=size_bytes,
                creation_time=timestamp,
                last_access_time=timestamp,
                checksum=checksum,
                attributes=attributes or {}
            )
            
            # Store file and metadata
            partition.files[file_id] = metadata
            partition.used_bytes += size_bytes
            self.data_store[file_id] = data
            
            logging.info(f"Stored file {file_id} in {storage_area.value}")
            return file_id
            
        except Exception as e:
            logging.error(f"Error storing file: {str(e)}")
            return None
    
    def retrieve_file(self, file_id: str) -> Optional[Union[bytes, dict]]:
        """Retrieve a file by its ID"""
        try:
            if file_id in self.data_store:
                # Update last access time
                for partition in self.storage.values():
                    if file_id in partition.files:
                        partition.files[file_id].last_access_time = datetime.utcnow()
                        break
                
                return self.data_store[file_id]
            return None
        except Exception as e:
            logging.error(f"Error retrieving file {file_id}: {str(e)}")
            return None
    
    def delete_file(self, file_id: str) -> bool:
        """Delete a file by its ID"""
        try:
            for storage_area, partition in self.storage.items():
                if file_id in partition.files:
                    metadata = partition.files[file_id]
                    partition.used_bytes -= metadata.size_bytes
                    del partition.files[file_id]
                    del self.data_store[file_id]
                    logging.info(f"Deleted file {file_id} from {storage_area.value}")
                    return True
            return False
        except Exception as e:
            logging.error(f"Error deleting file {file_id}: {str(e)}")
            return False
    
    def list_files(self, storage_area: Optional[StorageArea] = None, 
                  file_type: Optional[FileType] = None) -> List[FileMetadata]:
        """List files with optional filtering"""
        files = []
        for area, partition in self.storage.items():
            if storage_area and area != storage_area:
                continue
            for metadata in partition.files.values():
                if file_type and metadata.file_type != file_type:
                    continue
                files.append(metadata)
        return files
    
    def get_storage_status(self) -> Dict:
        """Get storage status for all partitions"""
        return {
            area.value: {
                'total_bytes': partition.total_bytes,
                'used_bytes': partition.used_bytes,
                'free_bytes': partition.total_bytes - partition.used_bytes,
                'used_percentage': (partition.used_bytes / partition.total_bytes) * 100,
                'file_count': len(partition.files),
                'max_files': partition.max_files
            }
            for area, partition in self.storage.items()
        }

class DataStore:
    """High-level interface for data management"""
    def __init__(self, fms: FileManagementSystem):
        self.fms = fms
    
    def store_image(self, image_data: bytes, metadata: dict) -> Optional[str]:
        """Store a camera image"""
        return self.fms.store_file(
            image_data,
            FileType.IMAGE,
            StorageArea.PAYLOAD,
            attributes=metadata
        )
    
    def store_sequence(self, sequence_data: dict, sequence_type: str) -> Optional[str]:
        """Store an ATS or RTS sequence"""
        attributes = {'sequence_type': sequence_type}
        return self.fms.store_file(
            sequence_data,
            FileType.SEQUENCE,
            StorageArea.SYSTEM,
            attributes=attributes
        )
    
    def store_telemetry(self, telemetry_data: dict) -> Optional[str]:
        """Store telemetry data"""
        return self.fms.store_file(
            telemetry_data,
            FileType.TELEMETRY,
            StorageArea.TELEMETRY
        )
    
    def get_latest_images(self, count: int = 10) -> List[Dict]:
        """Get metadata for the most recent images"""
        images = self.fms.list_files(
            storage_area=StorageArea.PAYLOAD,
            file_type=FileType.IMAGE
        )
        return sorted(images, key=lambda x: x.creation_time, reverse=True)[:count]
    
    def get_sequences(self, sequence_type: Optional[str] = None) -> List[Dict]:
        """Get all stored sequences"""
        sequences = self.fms.list_files(
            storage_area=StorageArea.SYSTEM,
            file_type=FileType.SEQUENCE
        )
        if sequence_type:
            sequences = [seq for seq in sequences 
                        if seq.attributes.get('sequence_type') == sequence_type]
        return sequences