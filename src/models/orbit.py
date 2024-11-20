from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
from scipy.integrate import solve_ivp
from datetime import datetime, timedelta

@dataclass
class OrbitState:
    """Spacecraft orbital state."""
    position: np.ndarray      # Position vector [km]
    velocity: np.ndarray      # Velocity vector [km/s]
    timestamp: datetime       # State epoch
    elements: Optional[Dict]  # Keplerian elements if calculated

@dataclass
class OrbitForces:
    """Forces acting on spacecraft."""
    gravity_2body: np.ndarray  # Two-body gravitational force [N]
    gravity_j2: np.ndarray     # J2 perturbation force [N]
    drag: np.ndarray          # Atmospheric drag force [N]
    srp: np.ndarray           # Solar radiation pressure force [N]
    third_body: np.ndarray    # Third-body perturbations [N]
    total: np.ndarray         # Total force [N]

class OrbitPropagator:
    def __init__(self, config, environment):
        self.config = config
        self.env = environment

        # Solar radiation pressure constant at 1 AU
        self.SOLAR_PRESSURE_1AU = 4.56e-6  # N/m²
        
        self.current_time = datetime.utcnow()
    
    def _initialize_state(self) -> OrbitState:
        """Initialize orbit state from Keplerian elements."""
        state_vector = self._kepler_to_state(
            self.config.orbit.semi_major_axis_km,
            self.config.orbit.eccentricity,
            np.radians(self.config.orbit.inclination_deg),
            np.radians(self.config.orbit.raan_deg),
            np.radians(self.config.orbit.arg_perigee_deg),
            np.radians(self.config.orbit.true_anomaly_deg)
        )
        
        return OrbitState(
            position=state_vector[:3],
            velocity=state_vector[3:],
            timestamp=datetime.utcnow(),
            elements=self._calculate_elements(state_vector)
        )

    def _calculate_forces(self, pos: np.ndarray, vel: np.ndarray) -> OrbitForces:
        """
        Calculate all forces acting on the spacecraft.
        
        Args:
            pos: Position vector [km]
            vel: Velocity vector [km/s]
            
        Returns:
            OrbitForces object containing all force components
        """
        # Two-body gravitational force
        r = np.linalg.norm(pos)
        f_2body = -self.env.EARTH_MU * self.config.orbit.mass_kg * pos / (r**3)
        
        # J2 perturbation
        z2 = pos[2]**2
        r2 = r**2
        tx = pos[0] / r * (5*z2/r2 - 1)
        ty = pos[1] / r * (5*z2/r2 - 1)
        tz = pos[2] / r * (5*z2/r2 - 3)
        
        j2_factor = 1.5 * self.env.J2 * self.env.EARTH_MU * \
                   self.env.EARTH_RADIUS**2 / (r**4)
        f_j2 = j2_factor * np.array([tx, ty, tz]) * self.config.orbit.mass_kg
        
        # Atmospheric drag
        v_rel = vel  # Simplified - should account for Earth's rotation
        v_rel_mag = np.linalg.norm(v_rel)
        rho = self.env.calculate_atmospheric_density(pos)
        
        drag_factor = -0.5 * rho * v_rel_mag * self.config.orbit.drag_coefficient * \
                     self.config.orbit.drag_area_m2
        f_drag = drag_factor * v_rel
        
        # Solar radiation pressure
        f_srp = self._calculate_srp_force(pos)
        
        # Third-body perturbations
        f_third_body = self._calculate_third_body_forces(pos)
        
        # Total force
        f_total = f_2body + f_j2 + f_drag + f_srp + f_third_body
        
        return OrbitForces(
            gravity_2body=f_2body,
            gravity_j2=f_j2,
            drag=f_drag,
            srp=f_srp,
            third_body=f_third_body,
            total=f_total
        )

    def _calculate_srp_force(self, pos: np.ndarray) -> np.ndarray:
        """Calculate solar radiation pressure force."""
        bodies = self.env.update(self.state.timestamp)
        sun_pos = bodies.position[1]
        satellite_to_sun = sun_pos - pos
        sun_distance = np.linalg.norm(satellite_to_sun)
        sun_direction = satellite_to_sun / sun_distance
        
        # Check eclipse
        eclipse_factor = self.env.calculate_eclipse(pos)
        if eclipse_factor == 0.0:
            return np.zeros(3)
            
        # Calculate pressure at spacecraft distance
        au_in_km = 149597870.7
        distance_ratio = au_in_km / sun_distance
        solar_pressure = self.SOLAR_PRESSURE_1AU * (distance_ratio ** 2)
        
        # Calculate force
        srp_force = -solar_pressure * self.config.orbit.srp_area_m2 * \
                   (1 + self.config.orbit.reflectivity) * \
                   sun_direction
        
        return srp_force * eclipse_factor
        
    def _kepler_to_state(self, a: float, e: float, i: float, 
                        raan: float, arg_peri: float, true_anom: float) -> np.ndarray:
        """Convert Keplerian elements to Cartesian state vector [x, y, z, vx, vy, vz]"""
        # Calculate parameter and radius
        p = a * (1 - e**2)
        r = p / (1 + e * np.cos(true_anom))
        
        # Position in perifocal frame
        x_pqw = r * np.cos(true_anom)
        y_pqw = r * np.sin(true_anom)
        z_pqw = 0
        
        # Velocity in perifocal frame
        mu = self.env.EARTH_MU
        v_factor = np.sqrt(mu / p)
        vx_pqw = -v_factor * np.sin(true_anom)
        vy_pqw = v_factor * (e + np.cos(true_anom))
        vz_pqw = 0
        
        # Rotation matrices
        R_W = np.array([[np.cos(raan), -np.sin(raan), 0],
                       [np.sin(raan), np.cos(raan), 0],
                       [0, 0, 1]])
        
        R_I = np.array([[1, 0, 0],
                       [0, np.cos(i), -np.sin(i)],
                       [0, np.sin(i), np.cos(i)]])
        
        R_P = np.array([[np.cos(arg_peri), -np.sin(arg_peri), 0],
                       [np.sin(arg_peri), np.cos(arg_peri), 0],
                       [0, 0, 1]])
        
        # Combined rotation matrix
        R = R_W @ R_I @ R_P
        
        # Transform to ECI frame
        pos_eci = R @ np.array([x_pqw, y_pqw, z_pqw])
        vel_eci = R @ np.array([vx_pqw, vy_pqw, vz_pqw])
        
        return np.concatenate([pos_eci, vel_eci])
    
def _calculate_srp_acceleration(self, pos: np.ndarray, bodies) -> np.ndarray:
        """Calculate solar radiation pressure acceleration"""
        # Get sun direction and distance
        sun_pos = bodies.position[1]
        satellite_to_sun = sun_pos - pos
        sun_distance = np.linalg.norm(satellite_to_sun)
        sun_direction = satellite_to_sun / sun_distance
        
        # Check if satellite is in eclipse
        eclipse_factor = self.env.calculate_eclipse(pos)
        if eclipse_factor == 0.0:
            return np.zeros(3)
            
        # Calculate solar pressure at satellite distance
        # Pressure drops with square of distance ratio
        au_in_km = 149597870.7  # 1 AU in kilometers
        distance_ratio = au_in_km / sun_distance
        solar_pressure = self.SOLAR_PRESSURE_1AU * (distance_ratio ** 2)
        
        # Calculate SRP acceleration
        # F = -P * A * (1 + r) * sun_direction / m
        # where P is solar pressure, A is area, r is reflectivity coefficient
        srp_acceleration = -solar_pressure * self.config.orbit.srp_area_m2 * \
                         (1 + self.config.orbit.reflectivity) * \
                         sun_direction / self.config.orbit.mass_kg
        
        # Apply eclipse factor
        return srp_acceleration * eclipse_factor
    
    def _acceleration(self, t: float, state: np.ndarray) -> np.ndarray:
        """Calculate total acceleration for numerical integration"""
        pos = state[:3]
        vel = state[3:]
        r = np.linalg.norm(pos)
        
        # Get celestial body positions
        bodies = self.env.update(self.current_time + timedelta(seconds=t))
        
        # Previous acceleration calculations...
        a_2body = -self.env.EARTH_MU * pos / (r**3)
        
        # J2 perturbation
        z2 = pos[2]**2
        r2 = r**2
        tx = pos[0] / r * (5*z2/r2 - 1)
        ty = pos[1] / r * (5*z2/r2 - 1)
        tz = pos[2] / r * (5*z2/r2 - 3)
        
        j2_factor = 1.5 * self.env.J2 * self.env.EARTH_MU * self.env.EARTH_RADIUS**2 / (r**4)
        a_j2 = j2_factor * np.array([tx, ty, tz])
        
        # Atmospheric drag
        v_rel = vel
        v_rel_mag = np.linalg.norm(v_rel)
        
        rho = self.env.calculate_atmospheric_density(pos)
        drag_factor = -0.5 * rho * v_rel_mag * self.config.orbit.drag_coefficient * \
                     self.config.orbit.drag_area_m2 / self.config.orbit.mass_kg
        a_drag = drag_factor * v_rel
        
        # Third-body perturbations
        def third_body_accel(body_pos, body_mu):
            r_sat_body = body_pos - pos
            r_sat_body_mag = np.linalg.norm(r_sat_body)
            return body_mu * (r_sat_body / r_sat_body_mag**3 - body_pos / np.linalg.norm(body_pos)**3)
        
        moon_mu = 4902.8
        sun_mu = 1.32712440018e11
        a_moon = third_body_accel(bodies.position[0], moon_mu)
        a_sun = third_body_accel(bodies.position[1], sun_mu)
        
        # Solar radiation pressure
        a_srp = self._calculate_srp_acceleration(pos, bodies)
        
        # Total acceleration
        accel = a_2body + a_j2 + a_drag + a_moon + a_sun + a_srp
        
        return np.concatenate([vel, accel])
    
    def propagate(self, dt: float) -> OrbitState:
        """
        Propagate orbit by specified time step.
        
        Args:
            dt: Time step [s]
            
        Returns:
            Updated orbit state
        """
        solution = solve_ivp(
            self._acceleration,
            (0, dt),
            np.concatenate([self.state.position, self.state.velocity]),
            method='RK45',
            rtol=1e-10,
            atol=1e-10
        )
        
        # Update state
        new_state_vector = solution.y[:,-1]
        self.state = OrbitState(
            position=new_state_vector[:3],
            velocity=new_state_vector[3:],
            timestamp=self.state.timestamp + timedelta(seconds=dt),
            elements=self._calculate_elements(new_state_vector)
        )
        
        return self.state

    def get_state(self) -> OrbitState:
        """Get current orbit state."""
        return self.state
    
    def get_forces(self) -> OrbitForces:
        """Get current forces acting on spacecraft."""
        return self._calculate_forces(self.state.position, self.state.velocity)

    def _calculate_elements(self, state_vector: np.ndarray) -> Dict:
        """Calculate Keplerian elements from state vector."""
        pos = state_vector[:3]
        vel = state_vector[3:]
        
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        # Angular momentum
        h = np.cross(pos, vel)
        h_mag = np.linalg.norm(h)
        
        # Node vector
        n = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n)
        
        # Eccentricity vector
        e = ((v**2 - self.env.EARTH_MU/r) * pos - np.dot(pos, vel) * vel) / self.env.EARTH_MU
        ecc = np.linalg.norm(e)
        
        # Semi-major axis
        a = -self.env.EARTH_MU / (2 * (v**2/2 - self.env.EARTH_MU/r))
        
        # Inclination
        inc = np.arccos(h[2]/h_mag)
        
        # RAAN
        raan = np.arccos(n[0]/n_mag) if n[1] >= 0 else 2*np.pi - np.arccos(n[0]/n_mag)
        
        # Argument of perigee
        arg_p = np.arccos(np.dot(n, e)/(n_mag * ecc))
        if e[2] < 0:
            arg_p = 2*np.pi - arg_p
            
        # True anomaly
        true_anom = np.arccos(np.dot(e, pos)/(ecc * r))
        if np.dot(pos, vel) < 0:
            true_anom = 2*np.pi - true_anom
            
        return {
            'semi_major_axis_km': a,
            'eccentricity': ecc,
            'inclination_deg': np.degrees(inc),
            'raan_deg': np.degrees(raan),
            'arg_perigee_deg': np.degrees(arg_p),
            'true_anomaly_deg': np.degrees(true_anom)
        }
    
    def get_position(self) -> np.ndarray:
        """Get current position vector in km"""
        return self.state[:3]
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity vector in km/s"""
        return self.state[3:]
    
    def get_orbital_elements(self) -> dict:
        """Calculate and return current orbital elements"""
        pos = self.state[:3]
        vel = self.state[3:]
        
        r = np.linalg.norm(pos)
        v = np.linalg.norm(vel)
        
        # Angular momentum
        h = np.cross(pos, vel)
        h_mag = np.linalg.norm(h)
        
        # Node vector
        n = np.cross([0, 0, 1], h)
        n_mag = np.linalg.norm(n)
        
        # Eccentricity vector
        e = ((v**2 - self.env.EARTH_MU/r) * pos - np.dot(pos, vel) * vel) / self.env.EARTH_MU
        ecc = np.linalg.norm(e)
        
        # Semi-major axis
        a = -self.env.EARTH_MU / (2 * (v**2/2 - self.env.EARTH_MU/r))
        
        # Inclination
        inc = np.arccos(h[2]/h_mag)
        
        # RAAN
        raan = np.arccos(n[0]/n_mag) if n[1] >= 0 else 2*np.pi - np.arccos(n[0]/n_mag)
        
        # Argument of perigee
        arg_p = np.arccos(np.dot(n, e)/(n_mag * ecc))
        if e[2] < 0:
            arg_p = 2*np.pi - arg_p
            
        # True anomaly
        true_anom = np.arccos(np.dot(e, pos)/(ecc * r))
        if np.dot(pos, vel) < 0:
            true_anom = 2*np.pi - true_anom
            
        return {
            'semi_major_axis_km': a,
            'eccentricity': ecc,
            'inclination_deg': np.degrees(inc),
            'raan_deg': np.degrees(raan),
            'arg_perigee_deg': np.degrees(arg_p),
            'true_anomaly_deg': np.degrees(true_anom)
        }

    def get_srp_force(self) -> Tuple[float, float]:
        """
        Returns current solar radiation pressure force magnitude (N) and 
        eclipse factor (0-1)
        """
        pos = self.state[:3]
        bodies = self.env.update(self.current_time)
        
        # Calculate SRP acceleration
        a_srp = self._calculate_srp_acceleration(pos, bodies)
        
        # Convert acceleration to force (F = ma)
        srp_force = np.linalg.norm(a_srp) * self.config.orbit.mass_kg
        eclipse_factor = self.env.calculate_eclipse(pos)
        
        return srp_force, eclipse_factor
