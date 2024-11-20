from enum import Enum, auto
from dataclasses import dataclass
import struct
from typing import Optional, List
import logging

class CFDPMessageType(Enum):
    FILE_DIRECTIVE = 0
    FILE_DATA = 1

class DirectiveCode(Enum):
    EOF = 4
    FINISHED = 5
    ACK = 6
    NAK = 7
    PROMPT = 8
    KEEP_ALIVE = 9

@dataclass
class CFDPHeader:
    version: int = 1
    pdu_type: CFDPMessageType = CFDPMessageType.FILE_DATA
    direction: int = 0  # 0: toward receiver, 1: toward sender
    transmission_mode: int = 0  # 0: acknowledged, 1: unacknowledged
    crc_flag: bool = True
    file_size: Optional[int] = None
    source_id: int = 0
    destination_id: int = 0
    sequence_number: int = 0

class CFDPProtocol:
    def __init__(self, entity_id: int):
        self.entity_id = entity_id
        self.transaction_seq = 0
        self.active_transactions = {}
        self.segment_size = 1024  # Default segment size
        
    def create_metadata_pdu(self, filename: str, filesize: int, 
                          destination_id: int) -> bytes:
        """Create metadata PDU for file transfer initiation"""
        header = CFDPHeader(
            pdu_type=CFDPMessageType.FILE_DIRECTIVE,
            file_size=filesize,
            source_id=self.entity_id,
            destination_id=destination_id,
            sequence_number=self.transaction_seq
        )
        
        # Pack header
        header_bytes = self._pack_header(header)
        
        # Pack metadata
        metadata = struct.pack(
            f">II{len(filename)}s",
            filesize,
            len(filename),
            filename.encode()
        )
        
        return header_bytes + metadata
    
    def create_file_data_pdu(self, transaction_id: int, 
                            offset: int, data: bytes) -> bytes:
        """Create file data PDU"""
        header = CFDPHeader(
            pdu_type=CFDPMessageType.FILE_DATA,
            source_id=self.entity_id,
            sequence_number=transaction_id
        )
        
        header_bytes = self._pack_header(header)
        
        # Pack offset and data
        segment = struct.pack(">I", offset) + data
        
        return header_bytes + segment
    
    def create_eof_pdu(self, transaction_id: int, 
                      file_size: int, checksum: bytes) -> bytes:
        """Create EOF PDU"""
        header = CFDPHeader(
            pdu_type=CFDPMessageType.FILE_DIRECTIVE,
            source_id=self.entity_id,
            sequence_number=transaction_id
        )
        
        header_bytes = self._pack_header(header)
        
        # Pack EOF directive
        eof_directive = struct.pack(
            ">BIH16s",
            DirectiveCode.EOF.value,
            file_size,
            0,  # Conditions
            checksum
        )
        
        return header_bytes + eof_directive
    
    def _pack_header(self, header: CFDPHeader) -> bytes:
        """Pack CFDP header into bytes"""
        first_byte = (
            (header.version & 0x03) << 5 |
            (header.pdu_type.value & 0x01) << 4 |
            (header.direction & 0x01) << 3 |
            (header.transmission_mode & 0x01) << 2 |
            (header.crc_flag & 0x01) << 1
        )
        
        return struct.pack(
            ">BHHI",
            first_byte,
            header.source_id,
            header.destination_id,
            header.sequence_number
        )