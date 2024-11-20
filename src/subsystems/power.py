from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import logging
from src.protocols.events import EventBus, EventType, EventPriority

class PowerState(Enum):
    """Power system states."""
    OFF = auto()
    STARTUP = auto()
    NOMINAL = auto()
    LOW_POWER = auto()
    CRITICAL = auto()
    CHARGING = auto()
    FAULT = auto()

class LoadPriority(Enum):
    """Power load priorities."""
    CRITICAL = 0    # OBC, critical sensors
    HIGH = 1        # COMMS, ADCS
    MEDIUM = 2      # Payload
    LOW = 3         # Non-essential systems

@dataclass
class BatteryState:
    """Battery state information."""
    voltage: float              # Volts
    current: float              # Amperes
    temperature: float          # Celsius
    state_of_charge: float      # 0-1
    charge_cycles: int          # Number of charge cycles
    capacity_degradation: float # 0-1 (1 = new)
    cell_voltages: List[float]  # Individual cell voltages
    cell_temperatures: List[float]  # Individual cell temperatures

@dataclass
class SolarArrayState:
    """Solar array state information."""
    voltage: float              # Volts
    current: float              # Amperes
    temperature: float          # Celsius
    illumination: float         # 0-1
    power_generated: float      # Watts
    degradation: float          # 0-1 (1 = new)

@dataclass
class PowerLoad:
    """Power load information."""
    name: str
    priority: LoadPriority
    nominal_power: float        # Watts
    current_power: float        # Watts
    enabled: bool = True

@dataclass
class PowerTelemetry:
    """Power subsystem telemetry data."""
    battery_voltage: float
    battery_current: float
    battery_temperature: float
    battery_charge: float
    battery_health: float
    battery_cycles: int
    solar_voltages: List[float]
    solar_currents: List[float]
    solar_temperatures: List[float]
    solar_power: List[float]
    bus_voltage_3v3: float
    bus_current_3v3: float
    bus_voltage_5v: float
    bus_current_5v: float
    bus_voltage_12v: float
    bus_current_12v: float
    subsystem_powers: Dict[str, float]
    subsystem_states: Dict[str, bool]
    total_power_in: float
    total_power_out: float
    power_mode: str
    fault_flags: int
    board_temp: float

class PowerSubsystem:
    """Spacecraft power subsystem."""
    
    def __init__(self, config):
        """
        Initialize power subsystem.
        
        Args:
            config: Power system configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.event_bus = EventBus()
        
        # Initialize battery state
        self.battery = BatteryState(
            voltage=7.4,                    # 2S Li-ion nominal
            current=0.0,
            temperature=20.0,
            state_of_charge=1.0,
            charge_cycles=0,
            capacity_degradation=1.0,
            cell_voltages=[3.7, 3.7],      # 2 cells
            cell_temperatures=[20.0, 20.0]
        )
        
        # Initialize solar array state
        self.solar_array = SolarArrayState(
            voltage=12.0,
            current=0.0,
            temperature=20.0,
            illumination=0.0,
            power_generated=0.0,
            degradation=1.0
        )
        
        # Power loads
        self.loads: Dict[str, PowerLoad] = {
            'obc': PowerLoad('obc', LoadPriority.CRITICAL, 0.5, 0.5),
            'adcs': PowerLoad('adcs', LoadPriority.HIGH, 2.0, 0.0),
            'comms': PowerLoad('comms', LoadPriority.HIGH, 5.0, 0.0),
            'payload': PowerLoad('payload', LoadPriority.MEDIUM, 3.0, 0.0)
        }
        
        # System state
        self.state = PowerState.STARTUP
        self.total_power_generated = 0.0
        self.total_power_consumed = 0.0
        self.timestamp = datetime.utcnow()
        
        # Battery protection limits
        self.battery_limits = {
            'voltage_min': 6.0,
            'voltage_max': 8.4,
            'temp_min': 0.0,
            'temp_max': 45.0,
            'soc_low': 0.2,
            'soc_critical': 0.1
        }
        
        self.logger.info("Power subsystem initialized")
        
    def update(self, dt: float) -> Dict:
        """
        Update power subsystem state.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Power telemetry dictionary
        """
        try:
            # Update timestamp
            self.timestamp = datetime.utcnow()
            
            # Update solar array state
            self._update_solar_array(dt)
            
            # Update battery state
            self._update_battery(dt)
            
            # Check battery protection limits
            self._check_battery_limits()
            
            # Update power distribution
            self._update_power_distribution()
            
            # Calculate total power
            self._update_power_totals()

            if hasattr(self, 'thermal_subsystem'):
                thermal_power = self.thermal_subsystem.total_power_consumption
                self.total_power_consumed += thermal_power
            
            return self.get_telemetry()
            
        except Exception as e:
            self.logger.error(f"Error in power update: {str(e)}")
            self.state = PowerState.FAULT
            return self.get_telemetry()
            
    def set_load_state(self, load_name: str, enabled: bool) -> bool:
        """
        Enable or disable a power load.
        
        Args:
            load_name: Name of the load
            enabled: Whether to enable or disable
            
        Returns:
            Success status
        """
        if load_name not in self.loads:
            return False
            
        load = self.loads[load_name]
        
        # Don't allow disabling critical loads
        if not enabled and load.priority == LoadPriority.CRITICAL:
            return False
            
        load.enabled = enabled
        self.logger.info(f"Power load {load_name} {'enabled' if enabled else 'disabled'}")
        return True
        
    def get_telemetry(self) -> Dict:
        """Get power system telemetry."""
        return {
            'state': self.state.name,
            'battery': {
                'voltage': self.battery.voltage,
                'current': self.battery.current,
                'temperature': self.battery.temperature,
                'state_of_charge': self.battery.state_of_charge,
                'charge_cycles': self.battery.charge_cycles,
                'capacity_degradation': self.battery.capacity_degradation,
                'cell_voltages': self.battery.cell_voltages,
                'cell_temperatures': self.battery.cell_temperatures
            },
            'solar_array': {
                'voltage': self.solar_array.voltage,
                'current': self.solar_array.current,
                'temperature': self.solar_array.temperature,
                'illumination': self.solar_array.illumination,
                'power_generated': self.solar_array.power_generated,
                'degradation': self.solar_array.degradation
            },
            'loads': {
                name: {
                    'priority': load.priority.name,
                    'nominal_power': load.nominal_power,
                    'current_power': load.current_power,
                    'enabled': load.enabled
                }
                for name, load in self.loads.items()
            },
            'total_generated': self.total_power_generated,
            'total_consumed': self.total_power_consumed,
            'timestamp': self.timestamp.isoformat()
        }
        
    def _update_solar_array(self, dt: float):
        """Update solar array state."""
        # Solar array power generation model
        # P = P_max * illumination * cos(angle) * degradation
        
        # Update temperature based on illumination
        if self.solar_array.illumination > 0:
            self.solar_array.temperature += (
                (60.0 - self.solar_array.temperature) * 0.1 * dt
            )
        else:
            self.solar_array.temperature += (
                (-10.0 - self.solar_array.temperature) * 0.1 * dt
            )
            
        # Calculate power generation
        max_power = 10.0  # Maximum power at 1AU, normal incidence
        self.solar_array.power_generated = (
            max_power * 
            self.solar_array.illumination * 
            self.solar_array.degradation
        )
        
        # Update voltage and current
        self.solar_array.voltage = 12.0 * self.solar_array.illumination
        self.solar_array.current = (
            self.solar_array.power_generated / self.solar_array.voltage 
            if self.solar_array.voltage > 0 else 0.0
        )
        
    def _update_battery(self, dt: float):
        """Update battery state."""
        # Simple battery model
        power_delta = (
            self.solar_array.power_generated - self.total_power_consumed
        )
        
        # Update state of charge
        capacity_wh = 20.0  # Watt-hours
        dsoc = power_delta * dt / (capacity_wh * 3600.0)  # Convert to hours
        
        self.battery.state_of_charge = np.clip(
            self.battery.state_of_charge + dsoc,
            0.0,
            1.0
        )
        
        # Update battery voltage based on SOC
        # Simple linear approximation
        self.battery.voltage = 6.0 + 2.4 * self.battery.state_of_charge
        
        # Update battery current
        self.battery.current = power_delta / self.battery.voltage
        
        # Update battery temperature
        # Simple thermal model
        ambient_temp = 20.0
        self.battery.temperature += (
            (ambient_temp - self.battery.temperature) * 0.1 * dt +
            abs(self.battery.current) * 0.1 * dt  # Self-heating
        )
        
        # Update cell voltages and temperatures
        for i in range(len(self.battery.cell_voltages)):
            self.battery.cell_voltages[i] = self.battery.voltage / 2.0
            self.battery.cell_temperatures[i] = self.battery.temperature
            
    def _check_battery_limits(self):
        """Check battery protection limits."""
        if self.battery.voltage < self.battery_limits['voltage_min']:
            self.state = PowerState.CRITICAL
            self._handle_critical_battery()
            
        elif self.battery.voltage > self.battery_limits['voltage_max']:
            self.state = PowerState.FAULT
            self._handle_battery_fault()
            
        elif self.battery.temperature < self.battery_limits['temp_min']:
            self.state = PowerState.FAULT
            self._handle_battery_fault()
            
        elif self.battery.temperature > self.battery_limits['temp_max']:
            self.state = PowerState.FAULT
            self._handle_battery_fault()
            
        elif self.battery.state_of_charge < self.battery_limits['soc_critical']:
            self.state = PowerState.CRITICAL
            self._handle_critical_battery()
            
        elif self.battery.state_of_charge < self.battery_limits['soc_low']:
            self.state = PowerState.LOW_POWER
            self._handle_low_power()
            
        elif self.solar_array.power_generated > self.total_power_consumed:
            self.state = PowerState.CHARGING
            
        else:
            self.state = PowerState.NOMINAL
            
    def _update_power_distribution(self):
        """Update power distribution to loads."""
        available_power = (
            self.solar_array.power_generated + 
            (self.battery.voltage * 2.0)  # Maximum battery discharge current
        )
        
        # Distribute power by priority
        remaining_power = available_power
        
        for priority in LoadPriority:
            priority_loads = [
                load for load in self.loads.values()
                if load.priority == priority and load.enabled
            ]
            
            if not priority_loads:
                continue
                
            # Calculate power per load
            power_per_load = remaining_power / len(priority_loads)
            
            for load in priority_loads:
                allocated_power = min(power_per_load, load.nominal_power)
                load.current_power = allocated_power
                remaining_power -= allocated_power
                
            if remaining_power <= 0:
                break
                
    def _update_power_totals(self):
        """Update total power generation and consumption."""
        self.total_power_generated = self.solar_array.power_generated
        self.total_power_consumed = sum(
            load.current_power for load in self.loads.values()
        )
        
    def _handle_critical_battery(self):
        """Handle critical battery state."""
        # Disable all non-critical loads
        for load in self.loads.values():
            if load.priority != LoadPriority.CRITICAL:
                load.enabled = False
                
        self.event_bus.publish(
            EventType.POWER,
            'power',
            {'critical_battery': True},
            priority=EventPriority.CRITICAL
        )
        
    def _handle_low_power(self):
        """Handle low power state."""
        # Disable low priority loads
        for load in self.loads.values():
            if load.priority == LoadPriority.LOW:
                load.enabled = False
                
        self.event_bus.publish(
            EventType.POWER,
            'power',
            {'low_power': True},
            priority=EventPriority.HIGH
        )
        
    def _handle_battery_fault(self):
        """Handle battery fault condition."""
        # Disable all loads except critical
        for load in self.loads.values():
            if load.priority != LoadPriority.CRITICAL:
                load.enabled = False
                
        self.event_bus.publish(
            EventType.ERROR,
            'power',
            {'battery_fault': True},
            priority=EventPriority.CRITICAL
        )

    def get_telemetry(self) -> PowerTelemetry:
        """Generate power subsystem telemetry packet."""
        return PowerTelemetry(
            # Battery state
            battery_voltage=self.battery.get_voltage(),
            battery_current=self.battery.get_current(),
            battery_temperature=self.battery.get_temperature(),
            battery_charge=self.battery.get_state_of_charge(),
            battery_health=self.battery.get_state_of_health(),
            battery_cycles=self.battery.get_cycle_count(),
            
            # Solar panel states
            solar_voltages=[panel.get_voltage() for panel in self.solar_panels],
            solar_currents=[panel.get_current() for panel in self.solar_panels],
            solar_temperatures=[panel.get_temperature() for panel in self.solar_panels],
            solar_power=[panel.get_power() for panel in self.solar_panels],
            
            # Power distribution
            bus_voltage_3v3=self.power_buses['3v3'].get_voltage(),
            bus_current_3v3=self.power_buses['3v3'].get_current(),
            bus_voltage_5v=self.power_buses['5v'].get_voltage(),
            bus_current_5v=self.power_buses['5v'].get_current(),
            bus_voltage_12v=self.power_buses['12v'].get_voltage(),
            bus_current_12v=self.power_buses['12v'].get_current(),
            
            # Subsystem power states
            subsystem_powers={name: bus.get_power() 
                            for name, bus in self.subsystem_buses.items()},
            subsystem_states={name: bus.is_enabled() 
                            for name, bus in self.subsystem_buses.items()},
            
            # System status
            total_power_in=self.get_total_power_in(),
            total_power_out=self.get_total_power_out(),
            power_mode=self.power_mode.name,
            fault_flags=self.get_fault_flags(),
            board_temp=self.get_board_temperature()
        )

    def publish_telemetry(self):
        """Publish power telemetry packet."""
        telemetry = self.get_telemetry()
        packet = telemetry.to_ccsds()
        self.event_bus.publish(
            EventType.TELEMETRY,
            "POWER",
            {"packet": packet.pack()}
        )