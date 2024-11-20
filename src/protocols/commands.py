from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from datetime import datetime

class CommandType(Enum):
    """Command types for different subsystems."""
    SYSTEM = auto()      # OBC/general system commands
    ADCS = auto()        # Attitude control commands
    POWER = auto()       # Power system commands
    THERMAL = auto()     # Thermal control commands
    COMMS = auto()       # Communications commands
    PAYLOAD = auto()     # Payload operation commands
    FMS = auto()         # File management commands

class FileType(Enum):
    IMAGE = auto()
    TELEMETRY = auto()
    CONFIG = auto()
    SOFTWARE = auto()
    LOG = auto()

class StorageArea(Enum):
    PAYLOAD = auto()
    SYSTEM = auto()
    CONFIG = auto()
    SOFTWARE = auto()

@dataclass
class Command:
    """Spacecraft command structure."""
    type: CommandType
    subtype: str
    parameters: Dict[str, Any]
    timestamp: datetime
    sequence_number: int
    source: str = "ground"
    timeout_s: Optional[float] = None
    verify_execution: bool = True

@dataclass
class CommandValidation:
    """Command validation rules and constraints."""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    required_params: Optional[List[str]] = None
    param_types: Optional[Dict[str, type]] = None

class CommandError(Exception):
    """Base class for command-related errors."""
    pass

class ValidationError(CommandError):
    """Raised when command validation fails."""
    pass

class ExecutionError(CommandError):
    """Raised when command execution fails."""
    pass

class CommandHandler:
    """Central command handling system."""
    
    def __init__(self, obc):
        """
        Initialize command handler.
        
        Args:
            obc: On-board computer instance
        """
        self.obc = obc
        self.sequence_counter = 0
        self.command_history: List[Command] = []
        
        # Command handlers for each subsystem
        self.handlers = {
            CommandType.SYSTEM: self._handle_system_command,
            CommandType.ADCS: self._handle_adcs_command,
            CommandType.POWER: self._handle_power_command,
            CommandType.THERMAL: self._handle_thermal_command,
            CommandType.COMMS: self._handle_comms_command,
            CommandType.PAYLOAD: self._handle_payload_command,
            CommandType.FMS: self._handle_fms_command
        }

        self.validation_rules = {
            CommandType.POWER: {
                'SET_POWER_MODE': CommandValidation(
                    allowed_values=['NORMAL', 'LOW_POWER', 'SAFE', 'EMERGENCY'],
                    required_params=['mode']
                )
            },
            CommandType.THERMAL: {
                'SET_HEATER_SETPOINT': CommandValidation(
                    min_value=-20.0,
                    max_value=50.0,
                    required_params=['zone', 'setpoint'],
                    param_types={'setpoint': float}
                )
            }
        }
        
    def process_command(self, command_dict: Dict) -> Dict:
        """Process incoming command with enhanced error handling."""
        try:
            # Validate command structure
            if not validate_dict_keys(command_dict, ['type', 'subtype']):
                raise ValidationError("Missing required command fields")

            # Create command object
            command = Command(
                type=CommandType[command_dict['type']],
                subtype=command_dict['subtype'],
                parameters=command_dict.get('parameters', {}),
                timestamp=datetime.utcnow(),
                sequence_number=self.sequence_counter,
                source=command_dict.get('source', 'ground'),
                timeout_s=command_dict.get('timeout_s'),
                verify_execution=command_dict.get('verify_execution', True)
            )

            # Validate command parameters
            validation_result = self._validate_command(command)
            if validation_result:
                raise ValidationError(validation_result)

            # Execute command
            result = self.handlers[command.type](command)
            
            # Log command execution
            self.logger.info(f"Command executed: {command.type.name}:{command.subtype} "
                           f"(seq: {command.sequence_number})")
            
            return result

        except ValidationError as e:
            self.logger.error(f"Command validation failed: {str(e)}")
            return {'success': False, 'error': f'Validation error: {str(e)}'}
            
        except ExecutionError as e:
            self.logger.error(f"Command execution failed: {str(e)}")
            return {'success': False, 'error': f'Execution error: {str(e)}'}
            
        except Exception as e:
            self.logger.error(f"Unexpected error in command processing: {str(e)}")
            return {'success': False, 'error': f'Internal error: {str(e)}'}

        except KeyError:
            return {
                'success': False,
                'error': f"Invalid command type: {command_dict.get('type')}"
            }
            
    def _handle_thermal_command(self, command: Command) -> Dict:
        """Handle thermal subsystem commands."""
        try:
            if command.subtype == 'SET_HEATER_STATE':
                zone = command.parameters['zone']
                enabled = command.parameters['enabled']
                success = self.obc.thermal.set_heater_state(zone, enabled)
                return {'success': success}
                
            elif command.subtype == 'SET_HEATER_SETPOINT':
                zone = command.parameters['zone']
                setpoint = command.parameters['setpoint']
                success = self.obc.thermal.set_heater_setpoint(zone, setpoint)
                return {'success': success}
                
            elif command.subtype == 'SET_THERMAL_MODE':
                mode = command.parameters['mode']
                success = self.obc.thermal.set_mode(mode)
                return {'success': success}
                
            return {'success': False, 'error': 'Unknown thermal command subtype'}
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Thermal command error: {str(e)}'
            }
            
    def _handle_system_command(self, command: Command) -> Dict:
        """Handle system-level commands."""
        try:
            if command.subtype == 'REBOOT':
                success = self.obc.reboot()
                return {'success': success}
            elif command.subtype == 'SET_MODE':
                mode = command.parameters['mode']
                success = self.obc.set_mode(mode)
                return {'success': success}
            return {'success': False, 'error': 'Unknown system command subtype'}
        except Exception as e:
            return {'success': False, 'error': f'System command error: {str(e)}'}

    def _handle_adcs_command(self, command: Command) -> Dict:
        """Handle ADCS commands."""
        try:
            if command.subtype == 'SET_ATTITUDE':
                quaternion = command.parameters['quaternion']
                success = self.obc.adcs.set_attitude(quaternion)
                return {'success': success}
            return {'success': False, 'error': 'Unknown ADCS command subtype'}
        except Exception as e:
            return {'success': False, 'error': f'ADCS command error: {str(e)}'}

    def _handle_power_command(self, command: Command) -> Dict:
        """Handle power system commands."""
        try:
            if command.subtype == 'SET_POWER_MODE':
                mode = command.parameters['mode']
                success = self.obc.power.set_mode(mode)
                return {'success': success}
            return {'success': False, 'error': 'Unknown power command subtype'}
        except Exception as e:
            return {'success': False, 'error': f'Power command error: {str(e)}'}

    def _handle_comms_command(self, command: Command) -> Dict:
        """Handle communications commands."""
        try:
            if command.subtype == 'SET_TX_POWER':
                power = command.parameters['power_dbm']
                success = self.obc.comms.set_tx_power(power)
                return {'success': success}
            return {'success': False, 'error': 'Unknown comms command subtype'}
        except Exception as e:
            return {'success': False, 'error': f'Comms command error: {str(e)}'}

    def _handle_fms_command(self, command: Command) -> Dict:
        """Handle file management commands."""
        try:
            if command.subtype == 'DELETE_FILE':
                path = command.parameters['path']
                success = self.obc.fms.delete_file(path)
                return {'success': success}
            return {'success': False, 'error': 'Unknown FMS command subtype'}
        except Exception as e:
            return {'success': False, 'error': f'FMS command error: {str(e)}'}
        
    def _handle_payload_command(self, command: Command) -> Dict:
        """Handle payload commands."""
        try:
            if command.subtype == 'CAPTURE_IMAGE':
                # Get spacecraft state
                position = self.obc.get_position()
                attitude = self.obc.adcs.state.quaternion
                
                # Capture image
                image_path = self.obc.payload.capture_image(
                    lat=position['latitude'],
                    lon=position['longitude'],
                    alt=position['altitude'],
                    quaternion=attitude,
                    timestamp=datetime.utcnow()
                )
                
                if image_path:
                    # Store in file management system
                    self.obc.fms.store_file(
                        image_path,
                        FileType.IMAGE,
                        StorageArea.PAYLOAD
                    )
                    return {'success': True, 'image_path': image_path}
                    
                return {'success': False, 'error': 'Image capture failed'}
            return {'success': False, 'error': 'Unknown payload command subtype'}
        except Exception as e:
            return {'success': False, 'error': f'Payload command error: {str(e)}'}

