from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional, List, Callable
from datetime import datetime
import logging
import json
from src.protocols.events import EventBus, EventType, Event, EventPriority
from .adcs import ADCS, ADCSMode
from .comms import CommunicationsSubsystem, CommState
from .power import PowerSubsystem, PowerState
from .fms import FileManagementSystem, FileType, StorageArea
from .payload import PayloadManager, PayloadState
from .thermal import ThermalSubsystem, ThermalMode
from src.protocols.events import EventBus, EventType, Event, EventPriority, EventSeverity, EventTelemetry

class SystemMode(Enum):
    """Spacecraft system modes."""
    BOOT = auto()
    SAFE = auto()
    NOMINAL = auto()
    SCIENCE = auto()
    MAINTENANCE = auto()
    EMERGENCY = auto()

@dataclass
class SystemState:
    """Overall system state."""
    mode: SystemMode
    boot_count: int
    uptime_seconds: float
    last_command_time: Optional[datetime]
    last_ground_contact: Optional[datetime]
    error_count: int
    warning_count: int
    cpu_usage: float
    memory_usage: float
    temperature_c: float
    timestamp: datetime

@dataclass
class SubsystemStatus:
    name: str
    enabled: bool
    mode: str
    uptime_seconds: float
    fault_flags: int
    power_consumption: float
    board_temp: float

@dataclass
class HousekeepingTelemetry:
    timestamp: datetime
    spacecraft_mode: str
    eclipse_state: bool
    battery_voltage: float
    battery_current: float
    battery_charge: float
    solar_array_power: float
    power_consumption: float
    min_temperature: float
    max_temperature: float
    avg_temperature: float
    heater_states: List[bool]
    quaternion: List[float]
    angular_velocity: List[float]
    sun_pointing_error: float
    position_eci: List[float]
    velocity_eci: List[float]
    next_aos_time: Optional[datetime]
    next_los_time: Optional[datetime]
    rf_power_dbm: float
    link_quality: float
    packets_queued: int
    last_ground_contact: Optional[datetime]
    memory_usage: float
    storage_usage: float
    cpu_load: float
    subsystem_status: List[SubsystemStatus]
    error_count: int
    warning_count: int
    last_error_time: Optional[datetime]
    last_error_message: Optional[str]

@dataclass
class ProcessStats:
    """Process statistics."""
    name: str
    cpu_percent: float
    memory_bytes: int
    thread_count: int
    uptime_seconds: float
    last_heartbeat: datetime

@dataclass
class OBCTelemetry:
    """OBC telemetry data structure."""
    timestamp: datetime
    boot_count: int
    uptime_seconds: int
    boot_cause: str
    cpu_load_1min: float
    cpu_load_5min: float
    cpu_load_15min: float
    cpu_temp: float
    total_memory_bytes: int
    used_memory_bytes: int
    free_memory_bytes: int
    total_swap_bytes: int
    used_swap_bytes: int
    process_count: int
    thread_count: int
    process_stats: List[ProcessStats]
    watchdog_enabled: bool
    last_watchdog_kick: datetime
    watchdog_timeout_s: float
    software_version: str
    last_update_time: Optional[datetime]
    pending_updates: int
    failed_updates: int
    time_sync_error_ms: float
    last_time_sync: Optional[datetime]
    time_source: str
    interrupt_count: int
    context_switches: int
    system_calls: int
    page_faults: int
    memory_errors: int
    bus_errors: int
    software_errors: int
    hardware_errors: int
    operating_mode: str
    fault_flags: int
    board_voltages: List[float]
    board_currents: List[float]
    board_temps: List[float]

@dataclass
class ProcessStats:
    """Process statistics."""
    name: str
    cpu_percent: float
    memory_bytes: int
    thread_count: int
    uptime_seconds: float
    last_heartbeat: datetime

class OBC:
    """On-Board Computer main controller."""
    
    def __init__(self, 
                 comms: CommunicationsSubsystem,
                 adcs: ADCS,
                 power: PowerSubsystem,
                 payload: PayloadManager,
                 fms: FileManagementSystem,
                 thermal: ThermalSubsystem,
                 spacecraft):
        """
        Initialize OBC.
        
        Args:
            comms: Communications subsystem
            adcs: Attitude determination and control system
            power: Power subsystem
            payload: Payload manager
            fms: File management system
        """
        self.comms = comms
        self.adcs = adcs
        self.power = power
        self.payload = payload
        self.fms = fms
        self.thermal = thermal
        self.spacecraft = spacecraft
        self.event_counter = 0
        self.housekeeping_interval = 1.0  # seconds
        self.last_housekeeping_time = datetime.utcnow()
        self.logger = logging.getLogger(__name__)
        
        # Initialize event bus
        self.event_bus = EventBus()
        self._register_event_handlers()
        
        # Initialize state
        self.state = SystemState(
            mode=SystemMode.BOOT,
            boot_count=0,
            uptime_seconds=0.0,
            last_command_time=None,
            last_ground_contact=None,
            error_count=0,
            warning_count=0,
            cpu_usage=0.0,
            memory_usage=0.0,
            temperature_c=20.0,
            timestamp=datetime.utcnow()
        )
        
        # Command handlers
        self.command_handlers: Dict[str, Callable] = {
            'SET_MODE': self._handle_mode_command,
            'RESET': self._handle_reset_command,
            'GET_TELEMETRY': self._handle_telemetry_command,
            'EXECUTE_SEQUENCE': self._handle_sequence_command
        }
        
        # Watchdog configuration
        self.watchdog_timeout = 60.0  # seconds
        self.last_watchdog_reset = datetime.utcnow()
        
        self.logger.info("OBC initialized")

    def collect_housekeeping(self) -> HousekeepingTelemetry:
        """Collect system-wide housekeeping telemetry."""
        current_time = datetime.utcnow()
        
        # Collect subsystem status
        subsystem_status = []
        for name, subsys in self.spacecraft.subsystems.items():
            status = SubsystemStatus(
                name=name,
                enabled=subsys.is_enabled(),
                mode=subsys.mode.name,
                uptime_seconds=subsys.get_uptime(),
                fault_flags=subsys.get_fault_flags(),
                power_consumption=subsys.get_power_consumption(),
                board_temp=subsys.get_board_temperature()
            )
            subsystem_status.append(status)
        
        return HousekeepingTelemetry(
            timestamp=current_time,
            spacecraft_mode=self.spacecraft.mode.name,
            eclipse_state=self.spacecraft.orbit.in_eclipse(),
            
            # Power Status
            battery_voltage=self.spacecraft.power.battery.get_voltage(),
            battery_current=self.spacecraft.power.battery.get_current(),
            battery_charge=self.spacecraft.power.battery.get_state_of_charge(),
            solar_array_power=self.spacecraft.power.get_solar_array_power(),
            power_consumption=self.spacecraft.power.get_total_power_consumption(),
            
            # Thermal Status
            min_temperature=self.spacecraft.thermal.get_min_temperature(),
            max_temperature=self.spacecraft.thermal.get_max_temperature(),
            avg_temperature=self.spacecraft.thermal.get_average_temperature(),
            heater_states=self.spacecraft.thermal.get_heater_states(),
            
            # Attitude Status
            quaternion=self.spacecraft.adcs.state.quaternion.tolist(),
            angular_velocity=self.spacecraft.adcs.state.angular_velocity.tolist(),
            sun_pointing_error=self.spacecraft.adcs.get_sun_pointing_error(),
            
            # Orbit Status
            position_eci=self.spacecraft.orbit.get_position_eci().tolist(),
            velocity_eci=self.spacecraft.orbit.get_velocity_eci().tolist(),
            next_aos_time=self.spacecraft.gsi.get_next_aos_time(),
            next_los_time=self.spacecraft.gsi.get_next_los_time(),
            
            # Communication Status
            rf_power_dbm=self.spacecraft.comms.get_rf_power(),
            link_quality=self.spacecraft.comms.get_link_quality(),
            packets_queued=self.spacecraft.comms.get_queued_packet_count(),
            last_ground_contact=self.spacecraft.gsi.get_last_contact_time(),
            
            # Memory Status
            memory_usage=self.memory_monitor.get_usage_percent(),
            storage_usage=self.spacecraft.fms.get_storage_usage_percent(),
            cpu_load=self.get_cpu_load(),
            
            subsystem_status=subsystem_status,
            
            # Error Status
            error_count=self.error_counter.get_error_count(),
            warning_count=self.error_counter.get_warning_count(),
            last_error_time=self.error_counter.get_last_error_time(),
            last_error_message=self.error_counter.get_last_error_message()
        )

    def publish_housekeeping(self):
        """Publish housekeeping telemetry if interval has elapsed."""
        current_time = datetime.utcnow()
        if (current_time - self.last_housekeeping_time).total_seconds() >= self.housekeeping_interval:
            telemetry = self.collect_housekeeping()
            packet = telemetry.to_ccsds()
            self.event_bus.publish(
                EventType.TELEMETRY,
                "HOUSEKEEPING",
                {"packet": packet.pack()}
            )
            self.last_housekeeping_time = current_time

    def publish_event(self, 
                     severity: EventSeverity,
                     source: str,
                     category: str,
                     message: str,
                     **kwargs):
        """Publish system event."""
        self.event_counter += 1
        
        event = EventTelemetry(
            timestamp=datetime.utcnow(),
            event_id=self.event_counter,
            severity=severity,
            source=source,
            category=category,
            message=message,
            subsystem=kwargs.get('subsystem', ''),
            component=kwargs.get('component', ''),
            parameter_values=kwargs.get('parameters', {}),
            related_event_id=kwargs.get('related_event_id'),
            spacecraft_mode=self.spacecraft.mode.name,
            orbit_position=self.spacecraft.orbit.get_position_eci().tolist(),
            ground_contact=self.spacecraft.gsi.is_in_contact()
        )
        
        packet = event.to_ccsds()
        self.event_bus.publish(
            EventType.TELEMETRY,
            "EVENT",
            {"packet": packet.pack()}
        )
        
        # Log critical events to persistent storage
        if severity == EventSeverity.CRITICAL:
            self.log_critical_event(event)

    def log_critical_event(self, event: EventTelemetry):
        """Log critical events to persistent storage."""
        try:
            log_entry = {
                'timestamp': event.timestamp.isoformat(),
                'event_id': event.event_id,
                'message': event.message,
                'source': event.source,
                'parameters': event.parameter_values
            }
            self.spacecraft.fms.append_to_file(
                'critical_events.log',
                json.dumps(log_entry) + '\n'
            )
        except Exception as e:
            # If logging fails, create a new event but don't recurse
            self.event_bus.publish(
                EventType.TELEMETRY,
                "OBC",
                {"message": f"Failed to log critical event: {str(e)}"}
            )
        
    def update(self, dt: float):
        """
        Update OBC and all subsystems.
        
        Args:
            dt: Time step in seconds
        """
        try:
            # Update system state
            self.state.uptime_seconds += dt
            self.state.timestamp = datetime.utcnow()
            
            # Check watchdog
            self._check_watchdog()

            # Get power state first
            power_telemetry = self.power.update(dt)
            
            # Get current power loads for thermal calculations
            power_loads = {
                'obc': self.state.cpu_usage * 0.5,  # 0.5W at 100% CPU
                'adcs': self.adcs.state.power_consumption,
                'comms': self.comms.status.power_consumption_w,
                'payload': self.payload.status.power_consumption_w
            }

            # Update thermal management with power and attitude information
            thermal_telemetry = self.thermal.update(
                dt,
                power_loads,
                self.adcs.state.quaternion,
                self._check_eclipse_state(),
                self._calculate_solar_intensity()
            )

            # Only proceed with other updates if thermal is not in FAULT mode
            if self.thermal.mode != ThermalMode.FAULT:
                self.adcs.update(dt)
                self.comms.update(dt)
                self.payload.update(dt)
            
            # Update subsystems
            self._update_subsystems(dt)
            
            # Process events
            self.event_bus.process_events()
            
            # Monitor system health
            self._monitor_health()

            # Update housekeeping telemetry
            self.publish_housekeeping()
            
            # Update other OBC functions
            self.monitor_subsystem_health()
            self.check_memory_usage()
            self.update_time_sync()
            self.process_command_queue()
            
            # Store telemetry
            self._store_telemetry()
            
        except Exception as e:
            self.logger.error(f"Error in OBC update: {str(e)}")
            self._handle_error(str(e))
            self.publish_event(
                severity=EventSeverity.ERROR,
                source="OBC",
                category="SYSTEM",
                message=f"OBC update error: {str(e)}",
                subsystem="OBC",
                component="update"
            )
    
    def monitor_subsystem_health(self):
        """Monitor health of all subsystems and generate events for issues."""
        for name, subsys in self.spacecraft.subsystems.items():
            fault_flags = subsys.get_fault_flags()
            if fault_flags != 0:
                self.publish_event(
                    severity=EventSeverity.WARNING,
                    source="OBC",
                    category="HEALTH",
                    message=f"Subsystem fault detected: {name}",
                    subsystem=name,
                    parameters={"fault_flags": fault_flags}
                )
            
    def execute_command(self, command: Dict) -> Dict:
        """
        Execute a command.
        
        Args:
            command: Command dictionary
            
        Returns:
            Command execution result
        """
        try:
            command_type = command.get('type')
            if command_type not in self.command_handlers:
                raise ValueError(f"Unknown command type: {command_type}")
                
            handler = self.command_handlers[command_type]
            result = handler(command)
            
            self.state.last_command_time = datetime.utcnow()
            
            return {
                'status': 'success',
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f"Command execution error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    def get_telemetry(self) -> OBCTelemetry:
        """Generate OBC telemetry packet."""
        current_time = datetime.utcnow()
        
        # Get process statistics
        process_stats = []
        for process in self.process_manager.get_processes():
            stats = ProcessStats(
                name=process.name,
                cpu_percent=process.get_cpu_usage(),
                memory_bytes=process.get_memory_usage(),
                thread_count=process.get_thread_count(),
                uptime_seconds=process.get_uptime(),
                last_heartbeat=process.get_last_heartbeat()
            )
            process_stats.append(stats)
        
        return OBCTelemetry(
            # Basic Info
            timestamp=current_time,
            boot_count=self.boot_counter,
            uptime_seconds=int((current_time - self.boot_time).total_seconds()),
            boot_cause=self.last_boot_cause,
            
            # CPU Status
            cpu_load_1min=self.get_cpu_load(interval=60),
            cpu_load_5min=self.get_cpu_load(interval=300),
            cpu_load_15min=self.get_cpu_load(interval=900),
            cpu_temp=self.get_cpu_temperature(),
            
            # Memory Status
            total_memory_bytes=self.memory_monitor.total,
            used_memory_bytes=self.memory_monitor.used,
            free_memory_bytes=self.memory_monitor.free,
            total_swap_bytes=self.memory_monitor.swap_total,
            used_swap_bytes=self.memory_monitor.swap_used,
            
            # Process Information
            process_count=len(process_stats),
            thread_count=sum(p.thread_count for p in process_stats),
            process_stats=process_stats,
            
            # Watchdog Status
            watchdog_enabled=self.watchdog.is_enabled(),
            last_watchdog_kick=self.watchdog.last_kick_time,
            watchdog_timeout_s=self.watchdog.timeout,
            
            # Software Status
            software_version=self.software_version,
            last_update_time=self.last_software_update,
            pending_updates=len(self.pending_updates),
            failed_updates=self.failed_updates,
            
            # Time Synchronization
            time_sync_error_ms=self.time_sync.get_error_ms(),
            last_time_sync=self.time_sync.last_sync_time,
            time_source=self.time_sync.current_source,
            
            # System Performance
            interrupt_count=self.performance_monitor.get_interrupts(),
            context_switches=self.performance_monitor.get_context_switches(),
            system_calls=self.performance_monitor.get_syscalls(),
            page_faults=self.performance_monitor.get_page_faults(),
            
            # Error Counters
            memory_errors=self.error_counter.memory_errors,
            bus_errors=self.error_counter.bus_errors,
            software_errors=self.error_counter.software_errors,
            hardware_errors=self.error_counter.hardware_errors,
            
            # System Status
            operating_mode=self.mode.name,
            fault_flags=self.get_fault_flags(),
            board_voltages=self.power_monitor.get_voltages(),
            board_currents=self.power_monitor.get_currents(),
            board_temps=self.temperature_monitor.get_temperatures()
        )

    def publish_telemetry(self):
        """Publish OBC telemetry packet."""
        telemetry = self.get_telemetry()
        packet = telemetry.to_ccsds()
        self.event_bus.publish(
            EventType.TELEMETRY,
            "OBC",
            {"packet": packet.pack()}
        )

    def get_fault_flags(self) -> int:
        """Get OBC fault flags."""
        flags = 0
        
        # CPU faults
        if self.get_cpu_temperature() > self.CPU_TEMP_LIMIT:
            flags |= 0x01
        if self.get_cpu_load(60) > self.CPU_LOAD_LIMIT:
            flags |= 0x02
            
        # Memory faults
        if self.memory_monitor.free < self.MIN_FREE_MEMORY:
            flags |= 0x10
        if self.error_counter.memory_errors > self.MAX_MEMORY_ERRORS:
            flags |= 0x20
            
        # Process faults
        if not self.process_manager.all_processes_healthy():
            flags |= 0x100
            
        # Watchdog faults
        if not self.watchdog.is_healthy():
            flags |= 0x1000
            
        # Time sync faults
        if abs(self.time_sync.get_error_ms()) > self.MAX_TIME_ERROR_MS:
            flags |= 0x10000
            
        return flags
        
    def _register_event_handlers(self):
        """Register event handlers."""
        self.event_bus.subscribe(EventType.ERROR, self._handle_error_event)
        self.event_bus.subscribe(EventType.POWER, self._handle_power_event)
        self.event_bus.subscribe(EventType.MODE_CHANGE, self._handle_mode_event)
        self.event_bus.subscribe(EventType.COMMUNICATION, self._handle_comms_event)
        
    def _update_subsystems(self, dt: float):
        """Update all subsystems."""
        # Update power first
        power_telemetry = self.power.update(dt)
        
        # Check if we have enough power for other subsystems
        if self.power.state.battery_state_of_charge > 0.1:  # 10% minimum
            self.adcs.update(dt)
            self.comms.update(dt)
            self.payload.update(dt)
            
        # Update resource usage
        self._update_resource_usage()
        
    def _check_watchdog(self):
        """Check and reset watchdog timer."""
        current_time = datetime.utcnow()
        if (current_time - self.last_watchdog_reset).total_seconds() > self.watchdog_timeout:
            self.logger.warning("Watchdog timeout - initiating reset")
            self._handle_reset_command({'type': 'RESET', 'reset_type': 'soft'})
        else:
            self.last_watchdog_reset = current_time
            
    def _monitor_health(self):
        """Monitor system health and handle anomalies."""
        # Check temperature
        if self.state.temperature_c > 80.0:
            self._handle_error("System temperature critical")
            self.set_mode(SystemMode.SAFE)
            
        # Check power
        if self.power.state.battery_state_of_charge < 0.2:  # 20%
            self.logger.warning("Low battery - entering safe mode")
            self.set_mode(SystemMode.SAFE)
            
        # Check memory usage
        if self.state.memory_usage > 0.9:  # 90%
            self.logger.warning("High memory usage")
            self._cleanup_memory()

        if self.thermal.mode == ThermalMode.FAULT:
            self.logger.error("Thermal system in FAULT mode")
            self.set_mode(SystemMode.SAFE)
        elif self.thermal.mode == ThermalMode.SAFE:
            self.logger.warning("Thermal system in SAFE mode")
            if self.state.mode != SystemMode.SAFE:
                self.set_mode(SystemMode.SAFE)
            
    def _store_telemetry(self):
        """Store system telemetry."""
        telemetry = self.get_telemetry()
        self.fms.store_file(
            telemetry,
            FileType.TELEMETRY,
            StorageArea.TELEMETRY,
            compress=True
        )
        
    def _update_resource_usage(self):
        """Update CPU and memory usage."""
        # Implementation depends on hardware platform
        pass
        
    def _cleanup_memory(self):
        """Perform memory cleanup."""
        # Implementation depends on hardware platform
        pass
        
    def _handle_error(self, error_msg: str):
        """Handle system error."""
        self.state.error_count += 1
        self.logger.error(error_msg)
        self.event_bus.publish(
            EventType.ERROR,
            'OBC',
            {'message': error_msg},
            priority=EventPriority.HIGH
        )
        
    def _handle_error_event(self, event: Event):
        """Handle error events."""
        if event.priority == EventPriority.CRITICAL:
            self.set_mode(SystemMode.SAFE)
            
    def _handle_power_event(self, event: Event):
        """Handle power events."""
        if event.data.get('low_power', False):
            self.set_mode(SystemMode.SAFE)
            
    def _handle_mode_event(self, event: Event):
        """Handle mode change events."""
        new_mode = SystemMode[event.data['mode']]
        self.set_mode(new_mode)
        
    def _handle_comms_event(self, event: Event):
        """Handle communication events."""
        if event.data.get('ground_contact', False):
            self.state.last_ground_contact = datetime.utcnow()
            
    def set_mode(self, mode: SystemMode):
        """Set system mode."""
        if mode == self.state.mode:
            return
            
        self.logger.info(f"System mode changing from {self.state.mode} to {mode}")
        
        # Configure subsystems for new mode
        if mode == SystemMode.SAFE:
            self.adcs.set_mode(ADCSMode.SUNSAFE)
            self.payload.set_state(PayloadState.OFF)
            self.comms.set_state(CommState.STANDBY)
        elif mode == SystemMode.NOMINAL:
            self.adcs.set_mode(ADCSMode.NADIR)
            self.payload.set_state(PayloadState.STANDBY)
            self.comms.set_state(CommState.RECEIVING)
        elif mode == SystemMode.SCIENCE:
            self.adcs.set_mode(ADCSMode.TARGET)
            self.payload.set_state(PayloadState.ACTIVE)
            
        self.state.mode = mode

    def _check_eclipse_state(self) -> bool:
        """Determine if spacecraft is in eclipse."""
        # Implementation depends on orbit propagator
        # Placeholder implementation
        return False
        
    def _calculate_solar_intensity(self) -> float:
        """Calculate solar intensity relative to 1 AU."""
        # Implementation depends on orbit propagator
        # Placeholder implementation
        return 1.0