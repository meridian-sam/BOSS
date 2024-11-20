from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional, List, Callable
from datetime import datetime
import logging
import numpy as np
from events import EventBus, EventType, Event, EventPriority
from .adcs import ADCS, ADCSMode
from .comms import CommunicationsSubsystem, CommState
from .power import PowerSubsystem, PowerState
from .fms import FileManagementSystem, FileType, StorageArea
from .payload import PayloadManager, PayloadState

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

class OBC:
    """On-Board Computer main controller."""
    
    def __init__(self, 
                 comms: CommunicationsSubsystem,
                 adcs: ADCS,
                 power: PowerSubsystem,
                 payload: PayloadManager,
                 fms: FileManagementSystem,
                 thermal: ThermalSubsystem):
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
            
            # Store telemetry
            self._store_telemetry()
            
        except Exception as e:
            self.logger.error(f"Error in OBC update: {str(e)}")
            self._handle_error(str(e))
            
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
            
    def get_telemetry(self) -> Dict:
        """Get system-wide telemetry."""
        return {
            'system': {
                'mode': self.state.mode.name,
                'boot_count': self.state.boot_count,
                'uptime': self.state.uptime_seconds,
                'last_command': self.state.last_command_time.isoformat() 
                    if self.state.last_command_time else None,
                'last_contact': self.state.last_ground_contact.isoformat()
                    if self.state.last_ground_contact else None,
                'errors': self.state.error_count,
                'warnings': self.state.warning_count,
                'cpu_usage': self.state.cpu_usage,
                'memory_usage': self.state.memory_usage,
                'temperature': self.state.temperature_c,
                'timestamp': self.state.timestamp.isoformat()
            },
            'power': self.power.get_telemetry(),
            'adcs': self.adcs.get_telemetry(),
            'comms': self.comms.get_telemetry(),
            'payload': self.payload.get_telemetry(),
            'thermal': self.thermal.get_telemetry()
        }
        
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