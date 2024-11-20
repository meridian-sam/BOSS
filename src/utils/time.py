from datetime import datetime, timedelta
import time

class SimulationClock:
    """Simulation time management."""
    
    def __init__(self, start_time: datetime, time_factor: float = 1.0):
        """
        Initialize simulation clock.
        
        Args:
            start_time: Simulation start time
            time_factor: Time acceleration factor (>1 for faster than real-time)
        """
        self.start_time = start_time
        self.time_factor = time_factor
        self.sim_start = time.time()
        
    def now(self) -> datetime:
        """Get current simulation time."""
        elapsed = (time.time() - self.sim_start) * self.time_factor
        return self.start_time + timedelta(seconds=elapsed)
        
    def get_timestamp(self) -> float:
        """Get current time as UNIX timestamp."""
        return self.now().timestamp()