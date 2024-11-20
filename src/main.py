from config import SimConfig
from models.attitude import AttitudeDynamics
from models.environment import Environment
from models.orbit import OrbitPropagator
from models.magnetic_field import MagneticFieldModel
from subsystems.obc import OBC
from subsystems.comms import CommunicationsSubsystem
from subsystems.adcs import ADCS

def main():
    # Initialize configuration
    config = SimConfig()
    
    # Initialize environment and subsystems
    env = Environment()
    orbit = OrbitPropagator(config, env)
    magnetic_field_model = MagneticFieldModel()
    attitude_dynamics = AttitudeDynamics(np.diag([0.1, 0.1, 0.1]))
    adcs = ADCS(config, attitude_dynamics)
    power = PowerSubsystem(config)
    comms = CommunicationsSubsystem(config)
    
    # Initialize OBC
    obc = OBC(comms, adcs, power, orbit, env)
    
    # Run OBC
    obc.run()

if __name__ == "__main__":
    main()