from dataclasses import dataclass
from datetime import datetime
import struct
from typing import Dict, Any, List, Optional, Union
from .ccsds import CCSDSPacket, PacketType
import numpy as np

HOUSEKEEPING_APID = 100
EVENT_APID = 101
PAYLOAD_APID = 102
ADCS_APID = 103
POWER_APID = 104
THERMAL_APID = 105
COMMS_APID = 106
FMS_APID = 107
OBC_APID = 108

@dataclass
class TelemetryValidation:
    """Telemetry validation rules."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    expected_type: type = float
    required: bool = True
    validation_func: Optional[callable] = None

class TelemetryProcessor:
    """Process and validate telemetry data."""
    
    def __init__(self):
        self.validation_rules = {
            'battery_voltage': TelemetryValidation(
                min_value=6.0,
                max_value=8.4,
                required=True
            ),
            'solar_array_current': TelemetryValidation(
                min_value=0.0,
                max_value=2.5,
                required=True
            ),
            'temperature': TelemetryValidation(
                min_value=-40.0,
                max_value=85.0,
                required=True
            ),
            'quaternion': TelemetryValidation(
                validation_func=self._validate_quaternion
            )
        }
        
    def validate_telemetry(self, data: Dict) -> List[str]:
        """Validate telemetry data against rules."""
        MAX_ERRORS = 100
        errors = []
        
        for field, rule in self.validation_rules.items():
            if len(errors) >= MAX_ERRORS:
                errors.append("Maximum error limit reached")
                break
            if field not in data:
                if rule.required:
                    errors.append(f"Missing required field: {field}")
                continue
                
            value = data[field]
            
            # Type validation
            if not isinstance(value, rule.expected_type):
                errors.append(f"Invalid type for {field}: expected {rule.expected_type}")
                continue
                
            # Range validation
            if rule.min_value is not None and value < rule.min_value:
                errors.append(f"{field} below minimum value: {value} < {rule.min_value}")
            
            if rule.max_value is not None and value > rule.max_value:
                errors.append(f"{field} above maximum value: {value} > {rule.max_value}")
                
            # Custom validation
            if rule.validation_func and not rule.validation_func(value):
                errors.append(f"Validation failed for {field}")
                
        return errors

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

@dataclass
class ADCSTelemetry:
    """ADCS telemetry packet structure."""
    # Attitude state
    quaternion: np.ndarray  # [w, x, y, z]
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s
    
    # Sensor readings
    magnetometer: np.ndarray  # [Bx, By, Bz] Tesla
    sun_sensors: List[float]  # Sun sensor readings [0-1]
    gyro_temps: List[float]  # Gyroscope temperatures [°C]
    
    # Actuator states
    reaction_wheel_speeds: np.ndarray  # [rpm]
    reaction_wheel_currents: np.ndarray  # [A]
    magnetorquer_commands: np.ndarray  # [Am²]
    
    # Control state
    control_mode: str  # Current control mode
    control_error: np.ndarray  # Control error vector
    control_effort: np.ndarray  # Control effort vector
    
    # System status
    power_status: bool  # Power state
    fault_flags: int  # Fault indicators
    cpu_temp: float  # CPU temperature [°C]
    
    # Add method to pack into CCSDS packet
    def to_ccsds(self) -> CCSDSPacket:
        packet = CCSDSPacket(ADCS_APID, PacketType.TM)
        
        # Pack timestamp
        packet.data.extend(struct.pack('>Q', int(datetime.utcnow().timestamp())))
        
        # Pack quaternion
        packet.data.extend(struct.pack('>4f', *self.quaternion))
        
        # Pack angular velocity
        packet.data.extend(struct.pack('>3f', *self.angular_velocity))
        
        # Pack sensor data
        packet.data.extend(struct.pack('>3f', *self.magnetometer))
        packet.data.extend(struct.pack(f'>{len(self.sun_sensors)}f', *self.sun_sensors))
        packet.data.extend(struct.pack(f'>{len(self.gyro_temps)}f', *self.gyro_temps))
        
        # Pack actuator states
        packet.data.extend(struct.pack('>3f', *self.reaction_wheel_speeds))
        packet.data.extend(struct.pack('>3f', *self.reaction_wheel_currents))
        packet.data.extend(struct.pack('>3f', *self.magnetorquer_commands))
        
        # Pack control state
        packet.data.extend(self.control_mode.encode().ljust(16, b'\0'))
        packet.data.extend(struct.pack('>3f', *self.control_error))
        packet.data.extend(struct.pack('>3f', *self.control_effort))
        
        # Pack system status
        packet.data.extend(struct.pack('>?If', self.power_status, 
                                     self.fault_flags, self.cpu_temp))
        
        return packet

@dataclass
class PowerTelemetry:
    """Power subsystem telemetry packet structure."""
    # Battery state
    battery_voltage: float  # [V]
    battery_current: float  # [A]
    battery_temperature: float  # [°C]
    battery_charge: float  # State of charge [%]
    battery_health: float  # State of health [%]
    battery_cycles: int  # Charge/discharge cycles
    
    # Solar panel states (per panel)
    solar_voltages: List[float]  # [V]
    solar_currents: List[float]  # [A]
    solar_temperatures: List[float]  # [°C]
    solar_power: List[float]  # [W]
    
    # Power distribution
    bus_voltage_3v3: float  # [V]
    bus_current_3v3: float  # [A]
    bus_voltage_5v: float  # [V]
    bus_current_5v: float  # [A]
    bus_voltage_12v: float  # [V]
    bus_current_12v: float  # [A]
    
    # Subsystem power states
    subsystem_powers: Dict[str, float]  # Power per subsystem [W]
    subsystem_states: Dict[str, bool]  # Power enable states
    
    # System status
    total_power_in: float  # Total power generation [W]
    total_power_out: float  # Total power consumption [W]
    power_mode: str  # Current power mode
    fault_flags: int  # Fault indicators
    board_temp: float  # Board temperature [°C]

    def to_ccsds(self) -> CCSDSPacket:
        """Convert to CCSDS telemetry packet."""
        packet = CCSDSPacket(POWER_APID, PacketType.TM)
        
        # Pack timestamp
        packet.data.extend(struct.pack('>Q', int(datetime.utcnow().timestamp())))
        
        # Pack battery state
        packet.data.extend(struct.pack('>6f', 
            self.battery_voltage,
            self.battery_current,
            self.battery_temperature,
            self.battery_charge,
            self.battery_health,
            float(self.battery_cycles)
        ))
        
        # Pack solar panel arrays
        packet.data.extend(struct.pack(f'>{len(self.solar_voltages)}f', *self.solar_voltages))
        packet.data.extend(struct.pack(f'>{len(self.solar_currents)}f', *self.solar_currents))
        packet.data.extend(struct.pack(f'>{len(self.solar_temperatures)}f', *self.solar_temperatures))
        packet.data.extend(struct.pack(f'>{len(self.solar_power)}f', *self.solar_power))
        
        # Pack power bus states
        packet.data.extend(struct.pack('>6f',
            self.bus_voltage_3v3,
            self.bus_current_3v3,
            self.bus_voltage_5v,
            self.bus_current_5v,
            self.bus_voltage_12v,
            self.bus_current_12v
        ))
        
        # Pack subsystem power states
        num_subsystems = len(self.subsystem_powers)
        packet.data.extend(struct.pack('>H', num_subsystems))
        for name, power in self.subsystem_powers.items():
            packet.data.extend(name.encode().ljust(8, b'\0'))
            packet.data.extend(struct.pack('>f?', power, self.subsystem_states[name]))
        
        # Pack system status
        packet.data.extend(struct.pack('>3f', 
            self.total_power_in,
            self.total_power_out,
            self.board_temp
        ))
        
        # Pack power mode and faults
        packet.data.extend(self.power_mode.encode().ljust(16, b'\0'))
        packet.data.extend(struct.pack('>I', self.fault_flags))
        
        return packet

@dataclass
class ThermalTelemetry:
    """Thermal subsystem telemetry packet structure."""
    # Temperature sensors
    timestamp: datetime
    temperatures: Dict[str, float]  # Temperature readings per zone [°C]
    
    # Heater states
    heater_states: Dict[str, bool]  # On/Off state per heater
    heater_duties: Dict[str, float]  # Duty cycle per heater [%]
    heater_powers: Dict[str, float]  # Power consumption per heater [W]
    
    # Thermal control
    setpoints: Dict[str, float]  # Temperature setpoint per zone [°C]
    control_errors: Dict[str, float]  # Control error per zone [°C]
    control_outputs: Dict[str, float]  # Controller output per zone [%]
    
    # Environmental
    external_temps: Dict[str, float]  # External surface temperatures [°C]
    radiator_temps: Dict[str, float]  # Radiator temperatures [°C]
    heat_flows: Dict[str, float]  # Heat flow per zone [W]
    
    # System status
    control_mode: str  # Current thermal control mode
    fault_flags: int  # Fault indicators
    board_temp: float  # Electronics board temperature [°C]

    def to_ccsds(self) -> CCSDSPacket:
        """Convert to CCSDS telemetry packet."""
        packet = CCSDSPacket(THERMAL_APID, PacketType.TM)
        
        # Pack timestamp
        packet.data.extend(struct.pack('>Q', int(self.timestamp.timestamp())))
        
        # Helper function to pack dictionary data
        def pack_dict(data_dict: Dict, value_format: str = '>f'):
            packet.data.extend(struct.pack('>H', len(data_dict)))
            for name, value in data_dict.items():
                packet.data.extend(name.encode().ljust(8, b'\0'))
                packet.data.extend(struct.pack(value_format, value))
        
        # Pack temperature sensor readings
        pack_dict(self.temperatures)
        
        # Pack heater states
        pack_dict(self.heater_states, '>?')
        pack_dict(self.heater_duties)
        pack_dict(self.heater_powers)
        
        # Pack thermal control data
        pack_dict(self.setpoints)
        pack_dict(self.control_errors)
        pack_dict(self.control_outputs)
        
        # Pack environmental data
        pack_dict(self.external_temps)
        pack_dict(self.radiator_temps)
        pack_dict(self.heat_flows)
        
        # Pack system status
        packet.data.extend(self.control_mode.encode().ljust(16, b'\0'))
        packet.data.extend(struct.pack('>If', 
            self.fault_flags,
            self.board_temp
        ))
        
        return packet

@dataclass
class CommsTelemetry:
    """Communications subsystem telemetry packet structure."""
    # RF Status
    timestamp: datetime
    tx_power_dbm: float  # Transmitter power [dBm]
    rx_power_dbm: float  # Received signal strength [dBm]
    tx_current: float    # Transmitter current [A]
    rx_current: float    # Receiver current [A]
    pa_temp: float      # Power amplifier temperature [°C]
    
    # Link Quality
    link_quality: float  # Link quality metric [0-1]
    bit_snr: float      # Bit SNR [dB]
    rssi: float         # Received Signal Strength Indicator [dBm]
    ber: float          # Bit Error Rate
    agc_gain: float     # Automatic Gain Control level [dB]
    
    # Data Rates
    uplink_rate_bps: float    # Uplink data rate [bps]
    downlink_rate_bps: float  # Downlink data rate [bps]
    uplink_rssi: float        # Uplink RSSI [dBm]
    downlink_rssi: float      # Downlink RSSI [dBm]
    
    # Buffer Status
    tx_buffer_used: int       # Transmit buffer usage [bytes]
    rx_buffer_used: int       # Receive buffer usage [bytes]
    tx_buffer_size: int       # Total transmit buffer size [bytes]
    rx_buffer_size: int       # Total receive buffer size [bytes]
    
    # Packet Statistics
    packets_sent: int         # Total packets sent
    packets_received: int     # Total packets received
    packets_dropped: int      # Dropped packets
    crc_errors: int          # CRC error count
    frame_errors: int        # Frame error count
    
    # System Status
    tx_enabled: bool         # Transmitter enabled state
    rx_enabled: bool         # Receiver enabled state
    mode: str               # Current operating mode
    fault_flags: int        # Fault indicators
    board_temp: float       # Board temperature [°C]

    def to_ccsds(self) -> CCSDSPacket:
        """Convert to CCSDS telemetry packet."""
        packet = CCSDSPacket(COMMS_APID, PacketType.TM)
        
        # Pack timestamp
        packet.data.extend(struct.pack('>Q', int(self.timestamp.timestamp())))
        
        # Pack RF status
        packet.data.extend(struct.pack('>5f',
            self.tx_power_dbm,
            self.rx_power_dbm,
            self.tx_current,
            self.rx_current,
            self.pa_temp
        ))
        
        # Pack link quality metrics
        packet.data.extend(struct.pack('>5f',
            self.link_quality,
            self.bit_snr,
            self.rssi,
            self.ber,
            self.agc_gain
        ))
        
        # Pack data rates
        packet.data.extend(struct.pack('>4f',
            self.uplink_rate_bps,
            self.downlink_rate_bps,
            self.uplink_rssi,
            self.downlink_rssi
        ))
        
        # Pack buffer status
        packet.data.extend(struct.pack('>4I',
            self.tx_buffer_used,
            self.rx_buffer_used,
            self.tx_buffer_size,
            self.rx_buffer_size
        ))
        
        # Pack packet statistics
        packet.data.extend(struct.pack('>5I',
            self.packets_sent,
            self.packets_received,
            self.packets_dropped,
            self.crc_errors,
            self.frame_errors
        ))
        
        # Pack system status
        packet.data.extend(struct.pack('>2?',
            self.tx_enabled,
            self.rx_enabled
        ))
        packet.data.extend(self.mode.encode().ljust(16, b'\0'))
        packet.data.extend(struct.pack('>If',
            self.fault_flags,
            self.board_temp
        ))
        
        return packet

@dataclass
class PayloadTelemetry:
    """Payload (camera) subsystem telemetry packet structure."""
    # Basic Info
    timestamp: datetime
    payload_enabled: bool
    operating_mode: str
    
    # Camera Settings
    exposure_time_ms: float
    gain: float
    binning: int
    region_of_interest: Tuple[int, int, int, int]  # x, y, width, height
    color_mode: str
    
    # Sensor Status
    sensor_temp: float  # [°C]
    sensor_voltage: float  # [V]
    sensor_current: float  # [mA]
    ccd_temp: float  # [°C]
    
    # Image Stats
    images_captured: int
    images_stored: int
    failed_captures: int
    last_image_quality: float  # [0-1]
    last_image_brightness: float  # [0-1]
    last_image_contrast: float  # [0-1]
    
    # Storage Status
    storage_used_bytes: int
    storage_total_bytes: int
    images_queued: int
    compression_ratio: float
    
    # Processing Status
    processing_queue_length: int
    last_processing_time_ms: float
    processing_errors: int
    
    # Power Status
    voltage_3v3: float  # [V]
    voltage_5v: float  # [V]
    current_3v3: float  # [mA]
    current_5v: float  # [mA]
    power_consumption: float  # [W]
    
    # System Status
    fault_flags: int
    board_temp: float  # [°C]
    uptime_seconds: int

    def to_ccsds(self) -> CCSDSPacket:
        """Convert to CCSDS telemetry packet."""
        packet = CCSDSPacket(PAYLOAD_APID, PacketType.TM)
        
        # Pack timestamp
        packet.data.extend(struct.pack('>Q', int(self.timestamp.timestamp())))
        
        # Pack basic info
        packet.data.extend(struct.pack('>?', self.payload_enabled))
        packet.data.extend(self.operating_mode.encode().ljust(16, b'\0'))
        
        # Pack camera settings
        packet.data.extend(struct.pack('>2fI', 
            self.exposure_time_ms,
            self.gain,
            self.binning
        ))
        packet.data.extend(struct.pack('>4I', *self.region_of_interest))
        packet.data.extend(self.color_mode.encode().ljust(8, b'\0'))
        
        # Pack sensor status
        packet.data.extend(struct.pack('>4f',
            self.sensor_temp,
            self.sensor_voltage,
            self.sensor_current,
            self.ccd_temp
        ))
        
        # Pack image stats
        packet.data.extend(struct.pack('>3I3f',
            self.images_captured,
            self.images_stored,
            self.failed_captures,
            self.last_image_quality,
            self.last_image_brightness,
            self.last_image_contrast
        ))
        
        # Pack storage status
        packet.data.extend(struct.pack('>2QIf',
            self.storage_used_bytes,
            self.storage_total_bytes,
            self.images_queued,
            self.compression_ratio
        ))
        
        # Pack processing status
        packet.data.extend(struct.pack('>IfI',
            self.processing_queue_length,
            self.last_processing_time_ms,
            self.processing_errors
        ))
        
        # Pack power status
        packet.data.extend(struct.pack('>5f',
            self.voltage_3v3,
            self.voltage_5v,
            self.current_3v3,
            self.current_5v,
            self.power_consumption
        ))
        
        # Pack system status
        packet.data.extend(struct.pack('>IfI',
            self.fault_flags,
            self.board_temp,
            self.uptime_seconds
        ))
        
        return packet

@dataclass
class FileSystemStats:
    """Statistics for a specific storage area."""
    total_bytes: int
    used_bytes: int
    free_bytes: int
    file_count: int
    fragmentation: float  # [0-1]
    last_write_time: datetime
    write_errors: int
    read_errors: int

@dataclass
class FMSTelemetry:
    """File Management System telemetry packet structure."""
    # Basic Info
    timestamp: datetime
    system_enabled: bool
    operating_mode: str
    
    # Storage Areas (by type)
    payload_storage: FileSystemStats
    telemetry_storage: FileSystemStats
    software_storage: FileSystemStats
    configuration_storage: FileSystemStats
    
    # File Operations
    files_written: int
    files_read: int
    files_deleted: int
    bytes_written: int
    bytes_read: int
    failed_operations: int
    
    # Transfer Status
    active_transfers: int
    queued_transfers: int
    failed_transfers: int
    last_transfer_rate_bps: float
    transfer_queue_bytes: int
    
    # File Types
    image_files: int
    telemetry_files: int
    log_files: int
    config_files: int
    software_files: int
    
    # System Status
    garbage_collection_running: bool
    last_gc_duration_s: float
    disk_health: float  # [0-1]
    fault_flags: int
    board_temp: float  # [°C]
    uptime_seconds: int

    def to_ccsds(self) -> CCSDSPacket:
        """Convert to CCSDS telemetry packet."""
        packet = CCSDSPacket(FMS_APID, PacketType.TM)
        
        # Pack timestamp
        packet.data.extend(struct.pack('>Q', int(self.timestamp.timestamp())))
        
        # Pack basic info
        packet.data.extend(struct.pack('>?', self.system_enabled))
        packet.data.extend(self.operating_mode.encode().ljust(16, b'\0'))
        
        # Helper function to pack storage stats
        def pack_storage_stats(stats: FileSystemStats):
            packet.data.extend(struct.pack('>3Q2If2I',
                stats.total_bytes,
                stats.used_bytes,
                stats.free_bytes,
                stats.file_count,
                stats.fragmentation,
                int(stats.last_write_time.timestamp()),
                stats.write_errors,
                stats.read_errors
            ))
        
        # Pack storage areas
        pack_storage_stats(self.payload_storage)
        pack_storage_stats(self.telemetry_storage)
        pack_storage_stats(self.software_storage)
        pack_storage_stats(self.configuration_storage)
        
        # Pack file operations
        packet.data.extend(struct.pack('>6I',
            self.files_written,
            self.files_read,
            self.files_deleted,
            self.bytes_written,
            self.bytes_read,
            self.failed_operations
        ))
        
        # Pack transfer status
        packet.data.extend(struct.pack('>3IfQ',
            self.active_transfers,
            self.queued_transfers,
            self.failed_transfers,
            self.last_transfer_rate_bps,
            self.transfer_queue_bytes
        ))
        
        # Pack file types
        packet.data.extend(struct.pack('>5I',
            self.image_files,
            self.telemetry_files,
            self.log_files,
            self.config_files,
            self.software_files
        ))
        
        # Pack system status
        packet.data.extend(struct.pack('>?3fI',
            self.garbage_collection_running,
            self.last_gc_duration_s,
            self.disk_health,
            self.board_temp,
            self.uptime_seconds
        ))
        
        # Pack fault flags
        packet.data.extend(struct.pack('>I', self.fault_flags))
        
        return packet

@dataclass
class ProcessStats:
    """Statistics for a running process."""
    name: str
    cpu_percent: float
    memory_bytes: int
    thread_count: int
    uptime_seconds: int
    last_heartbeat: datetime

@dataclass
class OBCTelemetry:
    """On-Board Computer telemetry packet structure."""
    # Basic Info
    timestamp: datetime
    boot_count: int
    uptime_seconds: int
    boot_cause: str
    
    # CPU Status
    cpu_load_1min: float
    cpu_load_5min: float
    cpu_load_15min: float
    cpu_temp: float  # [°C]
    
    # Memory Status
    total_memory_bytes: int
    used_memory_bytes: int
    free_memory_bytes: int
    total_swap_bytes: int
    used_swap_bytes: int
    
    # Process Information
    process_count: int
    thread_count: int
    process_stats: List[ProcessStats]
    
    # Watchdog Status
    watchdog_enabled: bool
    last_watchdog_kick: datetime
    watchdog_timeout_s: float
    
    # Software Status
    software_version: str
    last_update_time: datetime
    pending_updates: int
    failed_updates: int
    
    # Time Synchronization
    time_sync_error_ms: float
    last_time_sync: datetime
    time_source: str
    
    # System Performance
    interrupt_count: int
    context_switches: int
    system_calls: int
    page_faults: int
    
    # Error Counters
    memory_errors: int
    bus_errors: int
    software_errors: int
    hardware_errors: int
    
    # System Status
    operating_mode: str
    fault_flags: int
    board_voltages: Dict[str, float]  # Voltage rails
    board_currents: Dict[str, float]  # Current measurements
    board_temps: Dict[str, float]     # Temperature sensors

    def to_ccsds(self) -> CCSDSPacket:
        """Convert to CCSDS telemetry packet."""
        packet = CCSDSPacket(OBC_APID, PacketType.TM)
        
        # Pack timestamp and basic info
        packet.data.extend(struct.pack('>QIIf',
            int(self.timestamp.timestamp()),
            self.boot_count,
            self.uptime_seconds,
            float(len(self.boot_cause))
        ))
        packet.data.extend(self.boot_cause.encode())
        
        # Pack CPU status
        packet.data.extend(struct.pack('>4f',
            self.cpu_load_1min,
            self.cpu_load_5min,
            self.cpu_load_15min,
            self.cpu_temp
        ))
        
        # Pack memory status
        packet.data.extend(struct.pack('>5Q',
            self.total_memory_bytes,
            self.used_memory_bytes,
            self.free_memory_bytes,
            self.total_swap_bytes,
            self.used_swap_bytes
        ))
        
        # Pack process information
        packet.data.extend(struct.pack('>2I', 
            self.process_count,
            self.thread_count
        ))
        
        # Pack process stats
        packet.data.extend(struct.pack('>I', len(self.process_stats)))
        for proc in self.process_stats:
            packet.data.extend(proc.name.encode().ljust(32, b'\0'))
            packet.data.extend(struct.pack('>fQIIQ',
                proc.cpu_percent,
                proc.memory_bytes,
                proc.thread_count,
                proc.uptime_seconds,
                int(proc.last_heartbeat.timestamp())
            ))
        
        # Pack watchdog status
        packet.data.extend(struct.pack('>?Qf',
            self.watchdog_enabled,
            int(self.last_watchdog_kick.timestamp()),
            self.watchdog_timeout_s
        ))
        
        # Pack software status
        packet.data.extend(self.software_version.encode().ljust(32, b'\0'))
        packet.data.extend(struct.pack('>QII',
            int(self.last_update_time.timestamp()),
            self.pending_updates,
            self.failed_updates
        ))
        
        # Pack time sync status
        packet.data.extend(struct.pack('>fQ',
            self.time_sync_error_ms,
            int(self.last_time_sync.timestamp())
        ))
        packet.data.extend(self.time_source.encode().ljust(16, b'\0'))
        
        # Pack performance counters
        packet.data.extend(struct.pack('>4Q',
            self.interrupt_count,
            self.context_switches,
            self.system_calls,
            self.page_faults
        ))
        
        # Pack error counters
        packet.data.extend(struct.pack('>4I',
            self.memory_errors,
            self.bus_errors,
            self.software_errors,
            self.hardware_errors
        ))
        
        # Pack system status
        packet.data.extend(self.operating_mode.encode().ljust(16, b'\0'))
        packet.data.extend(struct.pack('>I', self.fault_flags))
        
        # Pack board voltages, currents, and temperatures
        for dictionary in [self.board_voltages, self.board_currents, self.board_temps]:
            packet.data.extend(struct.pack('>I', len(dictionary)))
            for name, value in dictionary.items():
                packet.data.extend(name.encode().ljust(16, b'\0'))
                packet.data.extend(struct.pack('>f', value))
        
        return packet

@dataclass
class SubsystemStatus:
    """Basic status information for a subsystem."""
    name: str
    enabled: bool
    mode: str
    uptime_seconds: int
    fault_flags: int
    power_consumption: float  # [W]
    board_temp: float  # [°C]

@dataclass
class HousekeepingTelemetry:
    """System-wide housekeeping telemetry packet structure."""
    # Basic Info
    timestamp: datetime
    spacecraft_mode: str
    eclipse_state: bool
    
    # Power Status
    battery_voltage: float  # [V]
    battery_current: float  # [A]
    battery_charge: float   # [%]
    solar_array_power: float  # [W]
    power_consumption: float  # [W]
    
    # Thermal Status
    min_temperature: float  # [°C]
    max_temperature: float  # [°C]
    avg_temperature: float  # [°C]
    heater_states: Dict[str, bool]
    
    # Attitude Status
    quaternion: List[float]  # [w, x, y, z]
    angular_velocity: List[float]  # [rad/s]
    sun_pointing_error: float  # [deg]
    
    # Orbit Status
    position_eci: List[float]  # [km]
    velocity_eci: List[float]  # [km/s]
    next_aos_time: datetime
    next_los_time: datetime
    
    # Communication Status
    rf_power_dbm: float
    link_quality: float  # [0-1]
    packets_queued: int
    last_ground_contact: datetime
    
    # Memory Status
    memory_usage: float  # [%]
    storage_usage: float  # [%]
    cpu_load: float  # [%]
    
    # Subsystem Status
    subsystem_status: List[SubsystemStatus]
    
    # Error Status
    error_count: int
    warning_count: int
    last_error_time: datetime
    last_error_message: str

    def to_ccsds(self) -> CCSDSPacket:
        """Convert to CCSDS telemetry packet."""
        packet = CCSDSPacket(HOUSEKEEPING_APID, PacketType.TM)
        
        # Pack timestamp and basic info
        packet.data.extend(struct.pack('>Q?', 
            int(self.timestamp.timestamp()),
            self.eclipse_state
        ))
        packet.data.extend(self.spacecraft_mode.encode().ljust(16, b'\0'))
        
        # Pack power status
        packet.data.extend(struct.pack('>5f',
            self.battery_voltage,
            self.battery_current,
            self.battery_charge,
            self.solar_array_power,
            self.power_consumption
        ))
        
        # Pack thermal status
        packet.data.extend(struct.pack('>3f',
            self.min_temperature,
            self.max_temperature,
            self.avg_temperature
        ))
        # Pack heater states
        packet.data.extend(struct.pack('>I', len(self.heater_states)))
        for name, state in self.heater_states.items():
            packet.data.extend(name.encode().ljust(8, b'\0'))
            packet.data.extend(struct.pack('>?', state))
        
        # Pack attitude status
        packet.data.extend(struct.pack('>4f', *self.quaternion))
        packet.data.extend(struct.pack('>3f', *self.angular_velocity))
        packet.data.extend(struct.pack('>f', self.sun_pointing_error))
        
        # Pack orbit status
        packet.data.extend(struct.pack('>3f', *self.position_eci))
        packet.data.extend(struct.pack('>3f', *self.velocity_eci))
        packet.data.extend(struct.pack('>2Q',
            int(self.next_aos_time.timestamp()),
            int(self.next_los_time.timestamp())
        ))
        
        # Pack communication status
        packet.data.extend(struct.pack('>2fIQ',
            self.rf_power_dbm,
            self.link_quality,
            self.packets_queued,
            int(self.last_ground_contact.timestamp())
        ))
        
        # Pack memory status
        packet.data.extend(struct.pack('>3f',
            self.memory_usage,
            self.storage_usage,
            self.cpu_load
        ))
        
        # Pack subsystem status
        packet.data.extend(struct.pack('>I', len(self.subsystem_status)))
        for status in self.subsystem_status:
            packet.data.extend(status.name.encode().ljust(8, b'\0'))
            packet.data.extend(status.mode.encode().ljust(8, b'\0'))
            packet.data.extend(struct.pack('>?I2f',
                status.enabled,
                status.fault_flags,
                status.power_consumption,
                status.board_temp
            ))
        
        # Pack error status
        packet.data.extend(struct.pack('>2IQ',
            self.error_count,
            self.warning_count,
            int(self.last_error_time.timestamp())
        ))
        packet.data.extend(self.last_error_message.encode().ljust(64, b'\0'))
        
        return packet

class EventSeverity(Enum):
    INFO = 0
    WARNING = 1
    ERROR = 2
    CRITICAL = 3

@dataclass
class EventTelemetry:
    """System event telemetry packet structure."""
    # Event Info
    timestamp: datetime
    event_id: int
    severity: EventSeverity
    source: str
    category: str
    message: str
    
    # Additional Data
    subsystem: str
    component: str
    parameter_values: Dict[str, Any]
    related_event_id: Optional[int]
    
    # Context
    spacecraft_mode: str
    orbit_position: List[float]  # [km]
    ground_contact: bool
    
    def to_ccsds(self) -> CCSDSPacket:
        """Convert to CCSDS telemetry packet."""
        packet = CCSDSPacket(EVENT_APID, PacketType.TM)
        
        # Pack event info
        packet.data.extend(struct.pack('>QII',
            int(self.timestamp.timestamp()),
            self.event_id,
            self.severity.value
        ))
        
        # Pack strings
        for text in [self.source, self.category, self.message,
                    self.subsystem, self.component]:
            packet.data.extend(text.encode().ljust(32, b'\0'))
        
        # Pack parameter values
        packet.data.extend(struct.pack('>I', len(self.parameter_values)))
        for key, value in self.parameter_values.items():
            packet.data.extend(key.encode().ljust(16, b'\0'))
            if isinstance(value, (int, bool)):
                packet.data.extend(struct.pack('>?I', True, int(value)))
            elif isinstance(value, float):
                packet.data.extend(struct.pack('>?f', False, value))
            else:
                str_val = str(value)
                packet.data.extend(struct.pack('>I', len(str_val)))
                packet.data.extend(str_val.encode())
        
        # Pack related event
        has_related = self.related_event_id is not None
        packet.data.extend(struct.pack('>?', has_related))
        if has_related:
            packet.data.extend(struct.pack('>I', self.related_event_id))
        
        # Pack context
        packet.data.extend(self.spacecraft_mode.encode().ljust(16, b'\0'))
        packet.data.extend(struct.pack('>3f?',
            *self.orbit_position,
            self.ground_contact
        ))
        
        return packet