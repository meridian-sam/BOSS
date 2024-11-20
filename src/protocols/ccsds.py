from enum import Enum, auto
from dataclasses import dataclass
import struct
from typing import Optional, Tuple, List
import numpy as np
from datetime import datetime

class PacketType(Enum):
    """CCSDS packet types."""
    TM = 0  # Telemetry
    TC = 1  # Telecommand

class SequenceFlag(Enum):
    """Segmentation flags for packet sequences."""
    CONTINUATION = 0
    FIRST = 1
    LAST = 2
    STANDALONE = 3

@dataclass
class CCSDSHeader:
    """CCSDS Space Packet Primary Header."""
    packet_version: int = 0
    packet_type: PacketType = PacketType.TM
    secondary_header_flag: bool = True
    apid: int = 0
    sequence_flags: SequenceFlag = SequenceFlag.STANDALONE
    sequence_count: int = 0
    packet_data_length: int = 0

    def pack(self) -> bytes:
        """Pack header into binary format."""
        first_word = (
            (self.packet_version & 0x07) << 13 |
            (self.packet_type.value & 0x01) << 12 |
            (self.secondary_header_flag & 0x01) << 11 |
            (self.apid & 0x7FF)
        )
        second_word = (
            (self.sequence_flags.value & 0x03) << 14 |
            (self.sequence_count & 0x3FFF)
        )
        return struct.pack(">HHH", first_word, second_word, self.packet_data_length)

    @classmethod
    def unpack(cls, data: bytes) -> Tuple['CCSDSHeader', bytes]:
        """Unpack header from binary format."""
        if len(data) < 6:
            raise ValueError("Insufficient data for CCSDS header")
            
        first_word, second_word, length = struct.unpack(">HHH", data[:6])
        
        return cls(
            packet_version=(first_word >> 13) & 0x07,
            packet_type=PacketType((first_word >> 12) & 0x01),
            secondary_header_flag=bool((first_word >> 11) & 0x01),
            apid=first_word & 0x7FF,
            sequence_flags=SequenceFlag((second_word >> 14) & 0x03),
            sequence_count=second_word & 0x3FFF,
            packet_data_length=length
        ), data[6:]

class CCSDSPacket:
    """CCSDS Space Packet implementation."""
    
    def __init__(self, apid: int, packet_type: PacketType, data: Optional[bytes] = None):
        self.header = CCSDSHeader(
            packet_type=packet_type,
            apid=apid
        )
        self.data = bytearray(data) if data else bytearray()
        self.timestamp = datetime.utcnow()
    
    def pack(self) -> bytes:
        """Pack entire packet into binary format."""
        self.header.packet_data_length = len(self.data)
        return self.header.pack() + bytes(self.data)
    
    @classmethod
    def unpack(cls, data: bytes) -> 'CCSDSPacket':
        """Unpack packet from binary format."""
        header, remaining = CCSDSHeader.unpack(data)
        return cls(header.apid, header.packet_type, remaining[:header.packet_data_length])