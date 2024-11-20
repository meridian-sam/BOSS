from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import logging
from events import EventBus, EventType, EventPriority

class ThermalMode(Enum):
    """Thermal control modes."""
    OFF = auto()
    STARTUP = auto()
    NOMINAL = auto()
    SAFE = auto()
    SURVIVAL = auto()
    FAULT = auto()

class ThermalZone(Enum):
    """Spacecraft thermal zones."""
    BATTERY = auto()
    OBC = auto()
    ADCS = auto()
    COMMS = auto()
    PAYLOAD = auto()
    SOLAR_ARRAYS = auto()
    PROPULSION = auto()

@dataclass
class HeaterConfig:
    """Heater configuration."""
    max_power: float          # Watts
    setpoint: float          # Celsius
    hysteresis: float        # Celsius
    enabled: bool = True
    duty_cycle: float = 0.0  # 0-1

@dataclass
class ThermistorData:
    """Thermistor measurement data."""
    temperature: float       # Celsius
    raw_value: int          # ADC counts
    valid: bool = True
    last_valid_time: Optional[datetime] = None

@dataclass
class ThermalZoneState:
    """Thermal zone state."""
    zone: ThermalZone
    temperatures: List[ThermistorData]  # Multiple sensors per zone
    heater: HeaterConfig
    average_temp: float
    power_consumption: float
    last_update: datetime

class ThermalSubsystem:
    """Spacecraft thermal management subsystem."""
    
    def __init__(self, config):
        """
        Initialize thermal subsystem.
        
        Args:
            config: Thermal system configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.event_bus = EventBus()
        
        # Initialize thermal zones
        self.zones: Dict[ThermalZone, ThermalZoneState] = self._initialize_zones()
        
        # System state
        self.mode = ThermalMode.STARTUP
        self.total_power_consumption = 0.0
        self.eclipse_state = False
        self.solar_intensity = 1.0  # 1 AU nominal
        self.beta_angle = 0.0       # Sun angle to orbit plane
        self.timestamp = datetime.utcnow()
        
        # Thermal model parameters
        self.thermal_mass = {
            ThermalZone.BATTERY: 2.0,
            ThermalZone.OBC: 0.5,
            ThermalZone.ADCS: 1.0,
            ThermalZone.COMMS: 1.0,
            ThermalZone.PAYLOAD: 1.5,
            ThermalZone.SOLAR_ARRAYS: 3.0,
            ThermalZone.PROPULSION: 2.0
        }
        
        self.thermal_coupling = {
            (ThermalZone.BATTERY, ThermalZone.OBC): 0.2,
            (ThermalZone.OBC, ThermalZone.ADCS): 0.3,
            (ThermalZone.COMMS, ThermalZone.OBC): 0.2,
            # Add more thermal couplings as needed
        }
        
        # Temperature limits
        self.temperature_limits = {
            ThermalZone.BATTERY: (-10.0, 40.0, -5.0, 35.0),  # (min, max, warning_min, warning_max)
            ThermalZone.OBC: (-20.0, 60.0, -15.0, 55.0),
            ThermalZone.ADCS: (-20.0, 50.0, -15.0, 45.0),
            ThermalZone.COMMS: (-30.0, 70.0, -25.0, 65.0),
            ThermalZone.PAYLOAD: (-30.0, 60.0, -25.0, 55.0),
            ThermalZone.SOLAR_ARRAYS: (-100.0, 100.0, -90.0, 90.0),
            ThermalZone.PROPULSION: (-30.0, 50.0, -25.0, 45.0)
        }
        
        self.logger.info("Thermal subsystem initialized")

    def update(self, dt: float, 
               power_loads: Dict[str, float],
               attitude: np.ndarray,
               eclipse: bool,
               solar_intensity: float) -> Dict:
        """
        Update thermal subsystem state.
        
        Args:
            dt: Time step in seconds
            power_loads: Power consumption of each subsystem
            attitude: Spacecraft attitude quaternion
            eclipse: Whether spacecraft is in eclipse
            solar_intensity: Solar intensity relative to 1 AU
            
        Returns:
            Thermal telemetry dictionary
        """
        try:
            # Update timestamp and environmental conditions
            self.timestamp = datetime.utcnow()
            self.eclipse_state = eclipse
            self.solar_intensity = solar_intensity
            
            # Calculate solar heat input for each zone
            solar_heat = self._calculate_solar_heat(attitude)
            
            # Update temperatures for each zone
            for zone, state in self.zones.items():
                # Get internal heat from power consumption
                internal_heat = power_loads.get(zone.name.lower(), 0.0)
                
                # Update zone temperatures
                self._update_zone_temperature(
                    zone, 
                    dt, 
                    internal_heat, 
                    solar_heat.get(zone, 0.0)
                )
                
                # Control heaters
                self._control_heater(zone)
                
            # Check temperature limits
            self._check_temperature_limits()
            
            # Update total power consumption
            self._update_power_consumption()
            
            return self.get_telemetry()
            
        except Exception as e:
            self.logger.error(f"Error in thermal update: {str(e)}")
            self.mode = ThermalMode.FAULT
            return self.get_telemetry()

    def set_heater_state(self, zone: ThermalZone, enabled: bool) -> bool:
        """
        Enable or disable heater for a thermal zone.
        
        Args:
            zone: Thermal zone
            enabled: Whether to enable heater
            
        Returns:
            Success status
        """
        if zone not in self.zones:
            return False
            
        self.zones[zone].heater.enabled = enabled
        self.logger.info(f"Heater for zone {zone.name} {'enabled' if enabled else 'disabled'}")
        return True

    def set_heater_setpoint(self, zone: ThermalZone, setpoint: float) -> bool:
        """
        Set heater setpoint for a thermal zone.
        
        Args:
            zone: Thermal zone
            setpoint: Temperature setpoint in Celsius
            
        Returns:
            Success status
        """
        if zone not in self.zones:
            return False
            
        min_temp, max_temp, _, _ = self.temperature_limits[zone]
        if not min_temp <= setpoint <= max_temp:
            return False
            
        self.zones[zone].heater.setpoint = setpoint
        self.logger.info(f"Heater setpoint for zone {zone.name} set to {setpoint}°C")
        return True

    def set_mode(self, mode: ThermalMode):
        """Set thermal control mode."""
        if mode == self.mode:
            return
            
        self.logger.info(f"Thermal mode changing from {self.mode} to {mode}")
        
        if mode == ThermalMode.SAFE:
            # Set conservative setpoints
            for zone in self.zones.values():
                _, _, warning_min, _ = self.temperature_limits[zone.zone]
                zone.heater.setpoint = warning_min + 5.0
                
        elif mode == ThermalMode.SURVIVAL:
            # Set minimum survival temperatures
            for zone in self.zones.values():
                min_temp, _, _, _ = self.temperature_limits[zone.zone]
                zone.heater.setpoint = min_temp + 2.0
                
        self.mode = mode

    def get_telemetry(self) -> ThermalTelemetry:
        """Generate thermal subsystem telemetry packet."""
        current_time = datetime.utcnow()
        
        return ThermalTelemetry(
            timestamp=current_time,
            
            # Temperature sensor readings
            temperatures={
                sensor.name: sensor.get_temperature()
                for sensor in self.temperature_sensors
            },
            
            # Heater states
            heater_states={
                heater.name: heater.is_enabled()
                for heater in self.heaters
            },
            heater_duties={
                heater.name: heater.get_duty_cycle()
                for heater in self.heaters
            },
            heater_powers={
                heater.name: heater.get_power()
                for heater in self.heaters
            },
            
            # Thermal control data
            setpoints={
                zone.name: zone.get_setpoint()
                for zone in self.thermal_zones
            },
            control_errors={
                zone.name: zone.get_control_error()
                for zone in self.thermal_zones
            },
            control_outputs={
                zone.name: zone.get_control_output()
                for zone in self.thermal_zones
            },
            
            # Environmental data
            external_temps={
                sensor.name: sensor.get_temperature()
                for sensor in self.external_sensors
            },
            radiator_temps={
                rad.name: rad.get_temperature()
                for rad in self.radiators
            },
            heat_flows={
                zone.name: zone.get_heat_flow()
                for zone in self.thermal_zones
            },
            
            # System status
            control_mode=self.control_mode.name,
            fault_flags=self.get_fault_flags(),
            board_temp=self.get_board_temperature()
        )

    def publish_telemetry(self):
        """Publish thermal telemetry packet."""
        telemetry = self.get_telemetry()
        packet = telemetry.to_ccsds()
        self.event_bus.publish(
            EventType.TELEMETRY,
            "THERMAL",
            {"packet": packet.pack()}
        )

    def get_fault_flags(self) -> int:
        """Get thermal subsystem fault flags."""
        flags = 0
        
        # Check for temperature out of range
        for i, sensor in enumerate(self.temperature_sensors):
            if not sensor.is_temperature_valid():
                flags |= (1 << i)
                
        # Check for heater faults
        for i, heater in enumerate(self.heaters):
            if heater.has_fault():
                flags |= (1 << (i + 16))
                
        # Check for control loop faults
        for i, zone in enumerate(self.thermal_zones):
            if zone.control_error_exceeded():
                flags |= (1 << (i + 24))
                
        return flags

    def _initialize_zones(self) -> Dict[ThermalZone, ThermalZoneState]:
        """Initialize thermal zones with sensors and heaters."""
        zones = {}
        for zone in ThermalZone:
            # Create multiple thermistors per zone
            thermistors = [
                ThermistorData(20.0, 2048, True, datetime.utcnow())
                for _ in range(2)  # 2 sensors per zone
            ]
            
            # Create heater configuration
            heater = HeaterConfig(
                max_power=10.0,  # 10W heater
                setpoint=20.0,   # 20°C setpoint
                hysteresis=2.0   # 2°C hysteresis
            )
            
            zones[zone] = ThermalZoneState(
                zone=zone,
                temperatures=thermistors,
                heater=heater,
                average_temp=20.0,
                power_consumption=0.0,
                last_update=datetime.utcnow()
            )
            
        return zones

    def _calculate_solar_heat(self, attitude: np.ndarray) -> Dict[ThermalZone, float]:
        """Calculate solar heat input for each zone based on attitude."""
        # Convert quaternion to rotation matrix
        R = self._quaternion_to_matrix(attitude)
        
        # Define surface normal vectors for each zone
        normals = {
            ThermalZone.SOLAR_ARRAYS: np.array([0, 0, 1]),  # Assuming solar arrays track sun
            ThermalZone.PAYLOAD: np.array([1, 0, 0]),
            # Add more surfaces as needed
        }
        
        # Calculate sun vector in body frame
        sun_vector = np.array([1, 0, 0])  # Assuming sun along X in inertial frame
        sun_body = R.T @ sun_vector
        
        # Calculate heat input for each zone
        solar_heat = {}
        solar_constant = 1361.0  # W/m²
        
        for zone in ThermalZone:
            if zone in normals:
                # Calculate cosine of incidence angle
                cos_theta = np.dot(normals[zone], sun_body)
                
                # Calculate heat input
                if not self.eclipse_state and cos_theta > 0:
                    heat = solar_constant * self.solar_intensity * cos_theta
                else:
                    heat = 0.0
                    
                solar_heat[zone] = heat
            else:
                solar_heat[zone] = 0.0
                
        return solar_heat

    def _update_zone_temperature(self, 
                               zone: ThermalZone, 
                               dt: float,
                               internal_heat: float,
                               solar_heat: float):
        """Update temperature for a thermal zone."""
        state = self.zones[zone]
        
        # Calculate total heat input
        heater_power = (state.heater.enabled * 
                       state.heater.max_power * 
                       state.heater.duty_cycle)
                       
        total_heat = internal_heat + solar_heat + heater_power
        
        # Calculate heat exchange with coupled zones
        for (zone1, zone2), coupling in self.thermal_coupling.items():
            if zone == zone1:
                other_zone = zone2
            elif zone == zone2:
                other_zone = zone1
            else:
                continue
                
            temp_diff = (self.zones[other_zone].average_temp - 
                        state.average_temp)
            total_heat += coupling * temp_diff
        
        # Update temperature using simple thermal model
        dT = total_heat * dt / self.thermal_mass[zone]
        
        # Update each sensor with some random variation
        for sensor in state.temperatures:
            noise = np.random.normal(0, 0.1)  # 0.1°C standard deviation
            sensor.temperature += dT + noise
            sensor.raw_value = int(sensor.temperature * 100)  # Simple ADC conversion
            
        # Update average temperature
        state.average_temp = np.mean([
            sensor.temperature for sensor in state.temperatures
        ])
        
        state.last_update = datetime.utcnow()

    def _control_heater(self, zone: ThermalZone):
        """Control heater for a thermal zone."""
        state = self.zones[zone]
        
        if not state.heater.enabled:
            state.heater.duty_cycle = 0.0
            state.power_consumption = 0.0
            return
            
        # Simple bang-bang control with hysteresis
        if state.average_temp < (state.heater.setpoint - state.heater.hysteresis):
            state.heater.duty_cycle = 1.0
        elif state.average_temp > (state.heater.setpoint + state.heater.hysteresis):
            state.heater.duty_cycle = 0.0
            
        state.power_consumption = state.heater.max_power * state.heater.duty_cycle

    def _check_temperature_limits(self):
        """Check temperature limits and update mode if necessary."""
        for zone, state in self.zones.items():
            min_temp, max_temp, warning_min, warning_max = self.temperature_limits[zone]
            
            if (state.average_temp < min_temp or 
                state.average_temp > max_temp):
                self.mode = ThermalMode.FAULT
                self.event_bus.publish(
                    EventType.ERROR,
                    'thermal',
                    {
                        'zone': zone.name,
                        'temperature': state.average_temp,
                        'limit_violation': True
                    },
                    priority=EventPriority.CRITICAL
                )
                
            elif (state.average_temp < warning_min or 
                  state.average_temp > warning_max):
                if self.mode != ThermalMode.FAULT:
                    self.mode = ThermalMode.SAFE
                self.event_bus.publish(
                    EventType.WARNING,
                    'thermal',
                    {
                        'zone': zone.name,
                        'temperature': state.average_temp,
                        'warning': True
                    },
                    priority=EventPriority.HIGH
                )

    def _update_power_consumption(self):
        """Update total power consumption from all heaters."""
        self.total_power_consumption = sum(
            state.power_consumption for state in self.zones.values()
        )

    @staticmethod
    def _quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])