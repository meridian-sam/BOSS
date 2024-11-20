import socket
from typing import Optional
from datetime import datetime
import struct

class CommunicationsSubsystem:
    def __init__(self, config):
        self.config = config
        self.tm_handler = TelemetryFrameHandler()
        self.cfdp = CFDPProtocol(entity_id=1)  # Spacecraft ID
        self.uplink_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.downlink_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Setup sockets
        self.uplink_socket.bind(("0.0.0.0", self.config.comms.yamcs_uplink_port))
        self.yamcs_address = (self.config.comms.yamcs_host, 
                            self.config.comms.yamcs_downlink_port)
        
        self._last_telemetry_time = datetime.now()
        
    def process_uplink(self) -> Optional[bytes]:
        """Check for and process any incoming commands"""
        try:
            data, _ = self.uplink_socket.recvfrom(1024)
            return data
        except socket.error:
            return None
            
    def send_telemetry(self, telemetry: dict):
        """Send telemetry using CCSDS packets"""
        # Pack telemetry into CCSDS packets
        packets = self.tm_handler.pack_telemetry(telemetry)
        
        # Create and send frames
        for packet in packets:
            frame = self.tm_handler.create_tm_frame(
                virtual_channel=0,
                data=packet.pack_primary_header() + packet.data,
                timestamp=datetime.utcnow()
            )
            self._transmit_frame(frame)
    
    def initiate_file_transfer(self, file_id: str, 
                             destination_id: int, file_data: bytes):
        """Initiate CFDP file transfer"""
        # Create metadata PDU
        metadata_pdu = self.cfdp.create_metadata_pdu(
            file_id, len(file_data), destination_id
        )
        self._transmit_frame(metadata_pdu)
        
        # Send file data in segments
        offset = 0
        while offset < len(file_data):
            segment = file_data[offset:offset + self.cfdp.segment_size]
            data_pdu = self.cfdp.create_file_data_pdu(
                self.cfdp.transaction_seq,
                offset,
                segment
            )
            self._transmit_frame(data_pdu)
            offset += len(segment)
        
        # Send EOF
        checksum = self._calculate_checksum(file_data)
        eof_pdu = self.cfdp.create_eof_pdu(
            self.cfdp.transaction_seq,
            len(file_data),
            checksum
        )
        self._transmit_frame(eof_pdu)
    
    def process_command(self, command: bytes):
        """Process incoming command to change ADCS mode"""
        # Example command processing
        if command == b'SET_MODE_OFF':
            return ADCS.OFF
        elif command == b'SET_MODE_SUNSAFE':
            return ADCS.SUNSAFE
        elif command == b'SET_MODE_NADIR':
            return ADCS.NADIR
        return None