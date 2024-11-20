from enum import Enum
from dataclasses import dataclass
import struct
from typing import Optional, Tuple, Union
import numpy as np

class PacketType(Enum):
    TM = 0  # Telemetry
    TC = 1  # Telecommand

class SequenceFlag(Enum):
    CONTINUATION = 0
    FIRST = 1
    LAST = 2
    STANDALONE = 3

@dataclass
class CCSDSHeader:
    packet_version: int = 0
    packet_type: PacketType = PacketType.TM
    secondary_header_flag: bool = True
    apid: int = 0
    sequence_flags: SequenceFlag = SequenceFlag.STANDALONE
    sequence_count: int = 0
    packet_data_length: int = 0

class CCSDSPacket:
    def __init__(self, apid: int, packet_type: PacketType):
        self.header = CCSDSHeader(
            packet_type=packet_type,
            apid=apid
        )
        self.secondary_header = None
        self.data = bytearray()
        
    def pack_primary_header(self) -> bytes:
        """Pack CCSDS primary header into bytes"""
        first_word = (
            (self.header.packet_version & 0x07) << 13 |
            (self.header.packet_type.value & 0x01) << 12 |
            (self.header.secondary_header_flag & 0x01) << 11 |
            (self.header.apid & 0x7FF)
        )
        second_word = (
            (self.header.sequence_flags.value & 0x03) << 14 |
            (self.header.sequence_count & 0x3FFF)
        )
        return struct.pack(">HHH", first_word, second_word, self.header.packet_data_length)

    @staticmethod
    def unpack_primary_header(data: bytes) -> Tuple[CCSDSHeader, bytes]:
        """Unpack CCSDS primary header from bytes"""
        if len(data) < 6:
            raise ValueError("Insufficient data for CCSDS header")
            
        first_word, second_word, length = struct.unpack(">HHH", data[:6])
        
        header = CCSDSHeader(
            packet_version=(first_word >> 13) & 0x07,
            packet_type=PacketType((first_word >> 12) & 0x01),
            secondary_header_flag=bool((first_word >> 11) & 0x01),
            apid=first_word & 0x7FF,
            sequence_flags=SequenceFlag((second_word >> 14) & 0x03),
            sequence_count=second_word & 0x3FFF,
            packet_data_length=length
        )
        
        return header, data[6:]