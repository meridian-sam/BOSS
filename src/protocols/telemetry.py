from datetime import datetime
import struct
from typing import Dict, Any

class TelemetryFrameHandler:
    def __init__(self):
        self.virtual_channels = {}
        self.sequence_counters = {}
        
    def create_tm_frame(self, virtual_channel: int, 
                       data: bytes, timestamp: datetime) -> bytes:
        """Create a CCSDS telemetry transfer frame"""
        # Get and increment sequence counter
        sequence_count = self.sequence_counters.get(virtual_channel, 0)
        self.sequence_counters[virtual_channel] = (sequence_count + 1) & 0xFF
        
        # Create primary header
        primary_header = struct.pack(
            ">HBBH",
            0x1000 | (virtual_channel & 0x07),  # Version 1, VC ID
            sequence_count,
            len(data) & 0xFF,
            int(timestamp.timestamp())
        )
        
        return primary_header + data
    
    def pack_telemetry(self, telemetry: Dict[str, Any]) -> List[bytes]:
        """Pack telemetry into CCSDS packets"""
        packets = []
        
        # Create different packets for different telemetry types
        housekeeping = CCSDSPacket(apid=1, packet_type=PacketType.TM)
        payload = CCSDSPacket(apid=2, packet_type=PacketType.TM)
        events = CCSDSPacket(apid=3, packet_type=PacketType.TM)
        
        # Pack housekeeping telemetry
        if 'power' in telemetry:
            housekeeping.data.extend(self._pack_housekeeping(telemetry))
            packets.append(housekeeping)
            
        # Pack payload telemetry
        if 'payload' in telemetry:
            payload.data.extend(self._pack_payload_telemetry(telemetry))
            packets.append(payload)
            
        return packets
    
    def _pack_housekeeping(self, telemetry: Dict[str, Any]) -> bytes:
        """Pack housekeeping telemetry into binary format"""
        # Example packing - adjust based on your needs
        power_data = telemetry.get('power', {})
        return struct.pack(
            ">ffI",
            power_data.get('battery_voltage', 0.0),
            power_data.get('battery_current', 0.0),
            power_data.get('power_consumption_mw', 0)
        )