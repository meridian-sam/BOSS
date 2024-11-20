from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import queue

@dataclass
class Command:
    """Base class for spacecraft commands"""
    command_id: str
    target_subsystem: str
    command_data: dict

@dataclass
class TimeTaggedCommand:
    """Command with execution time information"""
    command: Command
    execution_time: datetime  # For ATS
    relative_time: Optional[float] = None  # For RTS (seconds from sequence start)
    executed: bool = False

class SequenceType(Enum):
    ATS = "ATS"
    RTS = "RTS"

class CommandSequence:
    def __init__(self, sequence_id: str, sequence_type: SequenceType):
        self.sequence_id = sequence_id
        self.sequence_type = sequence_type
        self.commands: List[TimeTaggedCommand] = []
        self.active = False
        self.start_time: Optional[datetime] = None
    
    def add_command(self, command: TimeTaggedCommand):
        self.commands.append(command)
        # Sort commands by execution time or relative time
        if self.sequence_type == SequenceType.ATS:
            self.commands.sort(key=lambda x: x.execution_time)
        else:
            self.commands.sort(key=lambda x: x.relative_time)

class OBC:
    def __init__(self, comms, adcs, power, orbit, environment, payload):
        self.comms = comms
        self.adcs = adcs
        self.power = power
        self.orbit = orbit
        self.environment = environment
        self.payload = payload
        
        self.command_queue = queue.Queue()
        self.current_mode = 'NORMAL'

        # Initialize FMS and Data Store
        self.fms = FileManagementSystem(config)
        self.data_store = DataStore(self.fms)

        # Command sequences
        self.ats_sequences: Dict[str, CommandSequence] = {}
        self.rts_sequences: Dict[str, CommandSequence] = {}
        self.active_sequences: List[CommandSequence] = []
        
        # Command history
        self.command_history: List[Tuple[datetime, Command]] = []
        
        # Setup logging
        logging.basicConfig(filename='obc.log', level=logging.INFO)
        
    def add_ats(self, sequence_id: str, commands: List[TimeTaggedCommand]):
        """Add a new ATS sequence"""
        sequence = CommandSequence(sequence_id, SequenceType.ATS)
        for cmd in commands:
            sequence.add_command(cmd)
        self.ats_sequences[sequence_id] = sequence
        
        # Store sequence in FMS
        sequence_data = {
            'sequence_id': sequence_id,
            'commands': [{'command_id': cmd.command.command_id,
                         'execution_time': cmd.execution_time.isoformat(),
                         'command_data': cmd.command.command_data}
                        for cmd in commands]
        }
        self.data_store.store_sequence(sequence_data, 'ATS')
        
        logging.info(f"Added ATS sequence {sequence_id} with {len(commands)} commands")
    
    def add_rts(self, sequence_id: str, commands: List[TimeTaggedCommand]):
        """Add a new RTS sequence"""
        sequence = CommandSequence(sequence_id, SequenceType.RTS)
        for cmd in commands:
            sequence.add_command(cmd)
        self.rts_sequences[sequence_id] = sequence
        logging.info(f"Added RTS sequence {sequence_id} with {len(commands)} commands")
    
    def start_sequence(self, sequence_id: str, sequence_type: SequenceType):
        """Start a command sequence"""
        sequences = self.ats_sequences if sequence_type == SequenceType.ATS else self.rts_sequences
        if sequence_id in sequences:
            sequence = sequences[sequence_id]
            sequence.active = True
            sequence.start_time = datetime.utcnow()
            self.active_sequences.append(sequence)
            logging.info(f"Started {sequence_type.value} sequence {sequence_id}")
    
    def stop_sequence(self, sequence_id: str):
        """Stop a command sequence"""
        self.active_sequences = [seq for seq in self.active_sequences 
                               if seq.sequence_id != sequence_id]
        logging.info(f"Stopped sequence {sequence_id}")
    
    def process_sequences(self):
        """Process all active command sequences"""
        current_time = datetime.utcnow()
        
        for sequence in self.active_sequences:
            for cmd in sequence.commands:
                if cmd.executed:
                    continue
                
                execute = False
                if sequence.sequence_type == SequenceType.ATS:
                    execute = current_time >= cmd.execution_time
                else:  # RTS
                    elapsed_time = (current_time - sequence.start_time).total_seconds()
                    execute = elapsed_time >= cmd.relative_time
                
                if execute:
                    self.execute_command(cmd.command)
                    cmd.executed = True
                    self.command_history.append((current_time, cmd.command))
    
    def execute_command(self, command: Command):
        """Execute a single command"""
        try:
            if command.target_subsystem == "ADCS":
                if command.command_data.get("type") == "SET_MODE":
                    self.adcs.set_control_mode(command.command_data["mode"])
            
            if command.target_subsystem == "PAYLOAD":
                if command.command_data.get("type") == "CAMERA_POWER":
                    self.payload.set_state(CameraState[command.command_data["state"]])
                elif command.command_data.get("type") == "CAPTURE_IMAGE":
                    # Get current position and attitude
                    pos = self.orbit.get_position()
                    attitude = self.adcs.attitude_dynamics.get_attitude()
                    
                    # Check if sun is in camera FOV
                    sun_direction = self.environment._cached_bodies.sun_direction
                    sun_in_fov = self._check_sun_in_fov(sun_direction, attitude)
                    
                    # Attempt to capture image
                    self.payload.capture_image(pos, attitude, sun_in_fov)
                elif command.command_data.get("type") == "DELETE_IMAGE":
                    self.payload.delete_image(command.command_data["image_id"])
            
            logging.info(f"Executed command {command.command_id}")
        except Exception as e:
            logging.error(f"Error executing command {command.command_id}: {str(e)}")
    
    def process_commands(self):
        """Process incoming commands and command sequences"""
        # Process real-time commands
        command = self.comms.process_uplink()
        if command is not None:
            if isinstance(command, Command):
                self.command_queue.put(command)
            elif isinstance(command, dict) and command.get("type") == "SEQUENCE":
                # Handle sequence commands
                if command.get("action") == "START":
                    self.start_sequence(command["sequence_id"], 
                                     SequenceType[command["sequence_type"]])
                elif command.get("action") == "STOP":
                    self.stop_sequence(command["sequence_id"])
        
        # Process command queue
        while not self.command_queue.empty():
            command = self.command_queue.get()
            self.execute_command(command)
        
        # Process sequences
        self.process_sequences()

    def process_payload_data(self, image: Image):
        """Process and store payload data"""
        # Store image data
        image_id = self.data_store.store_image(
            image.data,
            {
                'timestamp': image.timestamp.isoformat(),
                'size_bytes': image.size_bytes,
                'compressed': image.compressed,
                **image.metadata
            }
        )
        
        if image_id:
            logging.info(f"Stored image {image_id}")
            return image_id
        return None

    def _check_sun_in_fov(self, sun_direction: np.ndarray, 
                         attitude: np.ndarray) -> bool:
        """Check if the sun is in the camera's field of view"""
        # Transform sun direction to camera frame
        camera_quaternion = self._combine_quaternions(
            attitude, self.payload.config.mounting_quaternion)
        sun_in_camera_frame = self._rotate_vector(sun_direction, camera_quaternion)
        
        # Check if sun is within FOV
        camera_boresight = np.array([0, 0, 1])  # Assuming +Z is boresight
        angle = np.arccos(np.dot(sun_in_camera_frame, camera_boresight))
        return angle < (self.payload.fov_rad / 2)

    def collect_telemetry(self) -> dict:
        telemetry = super().collect_telemetry()
        telemetry['payload'] = self.payload.get_telemetry()
        # Add storage status to telemetry
        telemetry['storage'] = self.fms.get_storage_status()
        # Store telemetry
        self.data_store.store_telemetry(telemetry)
        return telemetry
    
    def collect_telemetry(self) -> dict:
        """Collect telemetry data from all subsystems"""
        pos, vel = self.orbit.get_position(), self.orbit.get_velocity()
        eclipse_state = self.environment.calculate_eclipse(pos)
        magnetic_field = self.environment.calculate_magnetic_field(pos, self.orbit.current_time)
        
        adcs_telemetry = self.adcs.update(
            self.orbit.config.dt, magnetic_field, 
            self.environment._cached_bodies.sun_direction, 
            -pos / np.linalg.norm(pos)
        )
        
        power_telemetry = self.power.update(self.orbit.config.dt, eclipse_state, adcs_telemetry)
        
        telemetry = {
            'timestamp': self.orbit.current_time.isoformat(),
            'position_km': pos.tolist(),
            'velocity_kps': vel.tolist(),
            'eclipse': {
                'type': eclipse_state.eclipse_type,
                'illumination': eclipse_state.illumination,
                'sun_angle_deg': np.degrees(eclipse_state.sun_angle)
            },
            'magnetic_field': magnetic_field.tolist(),
            'adcs': adcs_telemetry,
            'power': power_telemetry
        }
        
        logging.info(f"Telemetry collected: {telemetry}")
        return telemetry
    
    def monitor_health(self):
        """Monitor subsystem health and perform recovery actions if needed"""
        # Example: Check battery level
        battery_percentage = self.power.battery_charge / self.power.battery_capacity_wh * 100
        if battery_percentage < 20:
            logging.warning("Battery level low, entering power-saving mode")
            self.current_mode = 'POWER_SAVING'
            self.adcs.set_control_mode(ADCS.OFF)
    
    def manage_modes(self):
        """Manage spacecraft modes and transitions"""
        if self.current_mode == 'POWER_SAVING':
            # Example: Reduce power consumption
            self.adcs.set_control_mode(ADCS.OFF)
        elif self.current_mode == 'NORMAL':
            # Example: Normal operations
            pass
    
    def run(self):
        """Main OBC loop"""
        try:
            while True:
                self.process_commands()
                self.monitor_health()
                self.manage_modes()
                telemetry = self.collect_telemetry()
                self.comms.send_telemetry(telemetry)
                time.sleep(self.orbit.config.dt)
        except KeyboardInterrupt:
            logging.info("OBC terminated by user")
            print("OBC terminated by user")