from dataclasses import dataclass
from datetime import datetime
import struct
from typing import Dict, Any, List, Optional
from .ccsds import CCSDSPacket, PacketType

@dataclass
class TelemetryFrame:
    """CCSDS Telemetry Transfer Frame."""
    virtual_channel: int
    spacecraft_id: int
    virtual_channel_count: int
    data: bytes
    timestamp: datetime
    frame_quality: float = 1.0

class TelemetryFrameHandler:
    """Handles telemetry frame creation and processing."""
    
    def __init__(self, spacecraft_id: int):
        self.spacecraft_id = spacecraft_id
        self.virtual_channels: Dict[int, int] = {}  # VC ID -> sequence count
        self.frame_count = 0
    
    def create_frame(self, 
                    virtual_channel: int, 
                    data: bytes, 
                    timestamp: Optional[datetime] = None) -> TelemetryFrame:
        """Create a telemetry transfer frame."""
        # Get and increment sequence counter
        vc_count = self.virtual_channels.get(virtual_channel, 0)
        self.virtual_channels[virtual_channel] = (vc_count + 1) & 0xFF
        
        return TelemetryFrame(
            virtual_channel=virtual_channel,
            spacecraft_id=self.spacecraft_id,
            virtual_channel_count=vc_count,
            data=data,
            timestamp=timestamp or datetime.utcnow()
        )
    
    def pack_telemetry(self, telemetry: Dict[str, Any]) -> List[CCSDSPacket]:
        """Pack telemetry into CCSDS packets."""
        packets = []
        
        # Create packets for different telemetry types
        if 'housekeeping' in telemetry:
            packets.append(self._pack_housekeeping(telemetry['housekeeping']))
        
        if 'payload' in telemetry:
            packets.append(self._pack_payload(telemetry['payload']))
            
        if 'events' in telemetry:
            packets.append(self._pack_events(telemetry['events']))
            
        return packets
    
    def _pack_housekeeping(self, data: Dict[str, Any]) -> CCSDSPacket:
        """Pack housekeeping telemetry."""
        packet = CCSDSPacket(apid=1, packet_type=PacketType.TM)
        # Pack timestamp
        packet.data.extend(struct.pack('>Q', int(datetime.utcnow().timestamp())))
        # Pack telemetry data
        for key, value in data.items():
            if isinstance(value, float):
                packet.data.extend(struct.pack('>f', value))
            elif isinstance(value, int):
                packet.data.extend(struct.pack('>i', value))
        return packet