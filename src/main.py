#!/usr/bin/env python3
"""
BOSS (Basic Open-source Satellite Simulator)
Main entry point for satellite simulation.

This module initializes and runs the complete satellite simulation, including all subsystems,
environment models, and telemetry handling.
"""

import argparse
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np

from config import Config, load_config
from models.attitude import AttitudeDynamics
from models.environment import Environment
from models.orbit import OrbitPropagator
from models.magnetic_field import MagneticFieldModel
from subsystems.obc import OBC
from subsystems.comms import CommunicationsSubsystem
from subsystems.adcs import ADCS
from subsystems.power import PowerSubsystem
from subsystems.thermal import ThermalSubsystem
from subsystems.payload import PayloadSubsystem
from utils.logging import SpacecraftLogger
from utils.time import SimulationClock
# from visualization.dashboard import SimulationDashboard

class SatelliteSimulator:
    """Main satellite simulation controller."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize satellite simulation.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else Config()
        
        # Initialize logging
        self.logger = SpacecraftLogger(
            "BOSS",
            log_file=self.config.simulation.logging_level.lower() + ".log"
        )
        self.logger.info("Initializing BOSS simulation...")
        
        # Initialize simulation clock
        self.clock = SimulationClock(
            self.config.simulation.start_time,
            self.config.simulation.time_factor
        )
        
        # Initialize environment models
        self.environment = Environment(self.config.environment)
        self.magnetic_field = MagneticFieldModel(self.config.environment.magnetic_model)
        
        # Initialize dynamics
        self.orbit = OrbitPropagator(self.config.orbit, self.environment)
        self.attitude = AttitudeDynamics(
            inertia_matrix=np.diag([0.1, 0.1, 0.1])  # Example inertia matrix
        )
        
        # Initialize subsystems
        self.init_subsystems()
        
        # Initialize visualization if enabled
        self.dashboard = None
        # if self.config.simulation.enable_visualization:
        #     self.dashboard = SimulationDashboard(self)
        
        # Simulation control
        self.running = False
        self.step_count = 0
        self.setup_signal_handlers()
        
    def init_subsystems(self):
        """Initialize all spacecraft subsystems."""
        try:
            # Initialize subsystems with dependencies
            self.power = PowerSubsystem(self.config)
            self.thermal = ThermalSubsystem(self.config)
            self.comms = CommunicationsSubsystem(self.config)
            self.adcs = ADCS(self.config, self.attitude)
            self.payload = PayloadSubsystem(self.config)
            
            # Initialize OBC last as it depends on other subsystems
            self.obc = OBC(
                comms=self.comms,
                adcs=self.adcs,
                power=self.power,
                thermal=self.thermal,
                payload=self.payload,
                orbit=self.orbit,
                environment=self.environment
            )
            
            self.logger.info("All subsystems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize subsystems: {str(e)}")
            raise
            
    def setup_signal_handlers(self):
        """Setup handlers for system signals."""
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)
        
    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info("Shutdown signal received")
        self.stop()
        
    def run(self):
        """Run the main simulation loop."""
        self.running = True
        self.logger.info("Starting simulation...")
        
        try:
            while self.running:
                # Get current simulation time
                current_time = self.clock.now()
                
                # Check simulation duration
                if (current_time - self.config.simulation.start_time).total_seconds() > \
                   self.config.simulation.max_duration_s:
                    self.logger.info("Maximum simulation duration reached")
                    break
                
                # Update environment and dynamics
                self.update_environment(current_time)
                self.update_dynamics()
                
                # Update subsystems
                self.update_subsystems(current_time)
                
                # Generate and store telemetry
                self.handle_telemetry()
                
                # Update visualization
                if self.dashboard:
                    self.dashboard.update()
                
                # Periodic logging
                if self.step_count % self.config.simulation.log_interval == 0:
                    self.log_state()
                
                self.step_count += 1
                
        except Exception as e:
            self.logger.error(f"Simulation error: {str(e)}")
            raise
        finally:
            self.cleanup()
            
    def update_environment(self, current_time: datetime):
        """Update environmental models."""
        self.environment.update(current_time)
        
    def update_dynamics(self):
        """Update orbital and attitude dynamics."""
        # Propagate orbit
        self.orbit.propagate(self.config.simulation.dt)
        
        # Update attitude dynamics with current torques
        torques = self.adcs.get_control_torques()
        self.attitude.update(torques, self.config.simulation.dt)
        
    def update_subsystems(self, current_time: datetime):
        """Update all spacecraft subsystems."""
        # Update OBC first as it coordinates other subsystems
        self.obc.update(current_time)
        
        # Update other subsystems
        self.power.update(current_time)
        self.thermal.update(current_time)
        self.comms.update(current_time)
        self.adcs.update(current_time)
        self.payload.update(current_time)
        
    def handle_telemetry(self):
        """Generate and store telemetry data."""
        if self.step_count % self.config.simulation.telemetry_interval == 0:
            telemetry = self.obc.generate_telemetry()
            self.obc.store_telemetry(telemetry)
            
    def log_state(self):
        """Log current simulation state."""
        orbit_state = self.orbit.get_state()
        attitude_state = self.attitude.get_attitude()
        
        self.logger.info(
            f"Simulation step {self.step_count}\n"
            f"Time: {self.clock.now()}\n"
            f"Position (km): {orbit_state.position}\n"
            f"Velocity (km/s): {orbit_state.velocity}\n"
            f"Attitude quaternion: {attitude_state}\n"
            f"Power state: {self.power.get_state()}\n"
            f"Battery level: {self.power.get_battery_level():.1f}%"
        )
        
    def cleanup(self):
        """Cleanup resources before exit."""
        self.logger.info("Cleaning up simulation resources...")
        if self.dashboard:
            self.dashboard.close()
        self.running = False
        
    def stop(self):
        """Stop the simulation."""
        self.running = False
        self.logger.info("Simulation stopped")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BOSS Satellite Simulator")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Simulation duration in seconds"
    )
    parser.add_argument(
        "--time-factor",
        type=float,
        help="Simulation time acceleration factor"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize and run simulation
        simulator = SatelliteSimulator(args.config)
        
        # Override config with command line arguments if provided
        if args.duration:
            simulator.config.simulation.max_duration_s = args.duration
        if args.time_factor:
            simulator.config.simulation.time_factor = args.time_factor
        
        # Run simulation
        simulator.run()
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()