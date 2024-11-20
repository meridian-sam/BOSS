from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import numpy as np
import logging
from protocols.ccsds import CCSDSPacket, PacketType
from protocols.telemetry import TelemetryFrameHandler
from protocols.cfdp import CFDPProtocol, TransactionStatus

class CommState(Enum):
    """Communication system states."""
    OFF = auto()
    STANDBY = auto()
    RECEIVING = auto()
    TRANSMITTING = auto()
    ERROR = auto()

@dataclass
class LinkBudget:
    """Communication link characteristics."""
    distance_km: float
    elevation_deg: float
    path_loss_db: float
    received_power_dbm: float
    noise_floor_dbm: float
    snr_db: float
    bit_error_rate: float
    link_margin_db: float

@dataclass
class CommStatus:
    """Communication system status."""
    state: CommState
    uplink_enabled: bool
    downlink_enabled: bool
    last_packet_time: Optional[datetime]
    packets_received: int
    packets_transmitted: int
    bit_errors_detected: int
    rssi_dbm: float
    temperature_c: float
    power_consumption_w: float
    timestamp: datetime
    temperature_c: float = 20.0  # Add temperature tracking

class CommunicationsSubsystem:
    """Spacecraft communications subsystem."""
    
    def __init__(self, config):
        """
        Initialize communications subsystem.
        
        Args:
            config: Communications configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize protocols
        self.telemetry_handler = TelemetryFrameHandler(spacecraft_id=1)
        self.cfdp = CFDPProtocol(entity_id=1)
        
        # Initialize state
        self.status = CommStatus(
            state=CommState.STANDBY,
            uplink_enabled=False,
            downlink_enabled=False,
            last_packet_time=None,
            packets_received=0,
            packets_transmitted=0,
            bit_errors_detected=0,
            rssi_dbm=-120.0,
            temperature_c=20.0,
            power_consumption_w=0.0,
            timestamp=datetime.utcnow()
        )
        
        # Radio parameters
        self.tx_power_dbm = 30.0  # 1W transmitter
        self.tx_antenna_gain_db = 3.0
        self.rx_antenna_gain_db = 3.0
        self.system_noise_temp_k = 290.0
        self.implementation_loss_db = 2.0
        
        # Buffer management
        self.uplink_buffer: List[CCSDSPacket] = []
        self.downlink_buffer: List[CCSDSPacket] = []
        self.max_buffer_size = 1000
        
    def update(self, 
              ground_station_position: np.ndarray,
              spacecraft_position: np.ndarray,
              dt: float) -> Dict:
        """
        Update communications system state.
        
        Args:
            ground_station_position: Ground station ECI position [km]
            spacecraft_position: Spacecraft ECI position [km]
            dt: Time step [s]
            
        Returns:
            Communications telemetry dictionary
        """
        # Calculate link budget
        link_budget = self._calculate_link_budget(
            ground_station_position, 
            spacecraft_position
        )
        
        # Update state based on link availability
        if link_budget.link_margin_db > 3.0:  # Minimum margin threshold
            if self.status.state == CommState.STANDBY:
                self._enable_communications()
        else:
            self._disable_communications()
            
        # Process buffers
        self._process_uplink_buffer(dt)
        self._process_downlink_buffer(dt)
        
        # Update CFDP transactions
        self.cfdp.update_transactions()
        
        # Update power consumption based on state
        self._update_power_consumption()
        
        # Update timestamp
        self.status.timestamp = datetime.utcnow()

        if hasattr(self, 'thermal_subsystem'):
            self.status.temperature_c = self.thermal_subsystem.zones[ThermalZone.COMMS].average_temp
        
        return self.get_telemetry()
        
    def send_packet(self, packet: CCSDSPacket) -> bool:
        """
        Queue packet for transmission.
        
        Args:
            packet: CCSDS packet to transmit
            
        Returns:
            True if packet was queued successfully
        """
        if len(self.downlink_buffer) < self.max_buffer_size:
            self.downlink_buffer.append(packet)
            return True
        self.logger.warning("Downlink buffer full, packet dropped")
        return False
        
    def receive_packet(self, data: bytes) -> Optional[CCSDSPacket]:
        """
        Process received packet data.
        
        Args:
            data: Raw packet bytes
            
        Returns:
            Decoded CCSDS packet if successful
        """
        try:
            packet = CCSDSPacket.unpack(data)
            self.status.packets_received += 1
            self.status.last_packet_time = datetime.utcnow()
            self.uplink_buffer.append(packet)
            return packet
        except Exception as e:
            self.logger.error(f"Error processing packet: {str(e)}")
            self.status.bit_errors_detected += 1
            return None
            
    def start_file_transfer(self, 
                           filename: str, 
                           file_size: int, 
                           destination_id: int) -> str:
        """
        Start CFDP file transfer.
        
        Returns:
            Transaction ID
        """
        transaction = self.cfdp.create_transaction(
            filename=filename,
            file_size=file_size,
            destination_id=destination_id
        )
        return str(transaction.transaction_id)
        
    def get_telemetry(self) -> CommsTelemetry:
        """Generate communications subsystem telemetry packet."""
        return CommsTelemetry(
            timestamp=datetime.utcnow(),
            
            # RF Status
            tx_power_dbm=self.transmitter.get_power(),
            rx_power_dbm=self.receiver.get_rssi(),
            tx_current=self.transmitter.get_current(),
            rx_current=self.receiver.get_current(),
            pa_temp=self.transmitter.get_pa_temperature(),
            
            # Link Quality
            link_quality=self.get_link_quality(),
            bit_snr=self.receiver.get_snr(),
            rssi=self.receiver.get_rssi(),
            ber=self.get_bit_error_rate(),
            agc_gain=self.receiver.get_agc_gain(),
            
            # Data Rates
            uplink_rate_bps=self.get_uplink_rate(),
            downlink_rate_bps=self.get_downlink_rate(),
            uplink_rssi=self.receiver.get_rssi(),
            downlink_rssi=self.get_downlink_rssi(),
            
            # Buffer Status
            tx_buffer_used=self.tx_buffer.get_used(),
            rx_buffer_used=self.rx_buffer.get_used(),
            tx_buffer_size=self.tx_buffer.get_size(),
            rx_buffer_size=self.rx_buffer.get_size(),
            
            # Packet Statistics
            packets_sent=self.packet_stats['sent'],
            packets_received=self.packet_stats['received'],
            packets_dropped=self.packet_stats['dropped'],
            crc_errors=self.packet_stats['crc_errors'],
            frame_errors=self.packet_stats['frame_errors'],
            
            # System Status
            tx_enabled=self.transmitter.is_enabled(),
            rx_enabled=self.receiver.is_enabled(),
            mode=self.mode.name,
            fault_flags=self.get_fault_flags(),
            board_temp=self.get_board_temperature()
        )

    def publish_telemetry(self):
        """Publish communications telemetry packet."""
        telemetry = self.get_telemetry()
        packet = telemetry.to_ccsds()
        self.event_bus.publish(
            EventType.TELEMETRY,
            "COMMS",
            {"packet": packet.pack()}
        )

    def get_fault_flags(self) -> int:
        """Get communications subsystem fault flags."""
        flags = 0
        
        # Check transmitter faults
        if self.transmitter.has_fault():
            flags |= 0x01
        if self.transmitter.power_exceeded():
            flags |= 0x02
        if self.transmitter.temperature_exceeded():
            flags |= 0x04
            
        # Check receiver faults
        if self.receiver.has_fault():
            flags |= 0x10
        if self.receiver.signal_lost():
            flags |= 0x20
            
        # Check buffer faults
        if self.tx_buffer.is_full():
            flags |= 0x100
        if self.rx_buffer.is_full():
            flags |= 0x200
            
        # Check link quality
        if self.get_link_quality() < 0.3:
            flags |= 0x1000
            
        return flags
        
    def _calculate_link_budget(self, 
                             ground_station_pos: np.ndarray,
                             spacecraft_pos: np.ndarray) -> LinkBudget:
        """Calculate communication link budget."""
        # Calculate distance and elevation
        relative_pos = spacecraft_pos - ground_station_pos
        distance = np.linalg.norm(relative_pos)
        elevation = self._calculate_elevation(ground_station_pos, relative_pos)
        
        # Calculate losses
        path_loss = 20 * np.log10(distance) + 20 * np.log10(self.config.downlink_frequency) + 32.45
        
        # Calculate received power
        rx_power = (self.tx_power_dbm + 
                   self.tx_antenna_gain_db + 
                   self.rx_antenna_gain_db - 
                   path_loss - 
                   self.implementation_loss_db)
        
        # Calculate noise floor
        noise_floor = -174 + 10 * np.log10(self.config.downlink_rate_bps)
        
        # Calculate SNR and BER
        snr = rx_power - noise_floor
        ber = self._calculate_bit_error_rate(snr)
        
        return LinkBudget(
            distance_km=distance,
            elevation_deg=np.degrees(elevation),
            path_loss_db=path_loss,
            received_power_dbm=rx_power,
            noise_floor_dbm=noise_floor,
            snr_db=snr,
            bit_error_rate=ber,
            link_margin_db=snr - 3.0  # Required SNR threshold
        )
        
    def _calculate_elevation(self, 
                           ground_station_pos: np.ndarray,
                           relative_pos: np.ndarray) -> float:
        """Calculate elevation angle to spacecraft."""
        gs_norm = np.linalg.norm(ground_station_pos)
        return np.arcsin(np.dot(ground_station_pos, relative_pos) / 
                        (gs_norm * np.linalg.norm(relative_pos)))
        
    def _calculate_bit_error_rate(self, snr_db: float) -> float:
        """Calculate bit error rate for given SNR."""
        # Simplified QPSK BER calculation
        snr_linear = 10 ** (snr_db / 10)
        return 0.5 * np.erfc(np.sqrt(snr_linear))
        
    def _process_uplink_buffer(self, dt: float):
        """Process packets in uplink buffer."""
        packets_to_process = min(
            len(self.uplink_buffer),
            int(self.config.uplink_rate_bps * dt / 8 / 1024)  # Packets per timestep
        )
        
        for _ in range(packets_to_process):
            packet = self.uplink_buffer.pop(0)
            # Process packet based on type
            if packet.header.packet_type == PacketType.TC:
                self._handle_telecommand(packet)
                
    def _process_downlink_buffer(self, dt: float):
        """Process packets in downlink buffer."""
        packets_to_transmit = min(
            len(self.downlink_buffer),
            int(self.config.downlink_rate_bps * dt / 8 / 1024)  # Packets per timestep
        )
        
        for _ in range(packets_to_transmit):
            packet = self.downlink_buffer.pop(0)
            # Simulate transmission
            self.status.packets_transmitted += 1
            
    def _handle_telecommand(self, packet: CCSDSPacket):
        """Process telecommand packet."""
        # Implementation depends on command structure
        pass
        
    def _enable_communications(self):
        """Enable communications system."""
        self.status.state = CommState.STANDBY
        self.status.uplink_enabled = True
        self.status.downlink_enabled = True
        self.logger.info("Communications system enabled")
        
    def _disable_communications(self):
        """Disable communications system."""
        self.status.state = CommState.OFF
        self.status.uplink_enabled = False
        self.status.downlink_enabled = False
        self.logger.info("Communications system disabled")
        
    def _update_power_consumption(self):
        """Update power consumption based on state."""
        if self.status.state == CommState.OFF:
            self.status.power_consumption_w = 0.0
        elif self.status.state == CommState.STANDBY:
            self.status.power_consumption_w = 0.5
        elif self.status.state == CommState.RECEIVING:
            self.status.power_consumption_w = 2.0
        elif self.status.state == CommState.TRANSMITTING:
            self.status.power_consumption_w = 5.0