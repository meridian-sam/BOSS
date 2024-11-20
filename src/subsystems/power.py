from typing import Dict
from models.eclipse import EclipseState

class PowerSubsystem:
    def __init__(self, config):
        self.config = config
        self.solar_panel_power = 10.0  # Watts at 1 AU in full sun
        self.battery_capacity_wh = 30.0  # Watt-hours
        self.battery_charge = self.battery_capacity_wh  # Start fully charged
        self.battery_voltage = 7.4  # Volts (2S Li-ion)
        
    def update(self, dt: float, eclipse_state: EclipseState) -> Dict:
        """
        Update power system state
        Returns power telemetry dictionary
        """
        # Calculate solar panel power considering eclipse
        solar_power = self.solar_panel_power * eclipse_state.illumination
        
        # Basic power consumption model
        base_power_consumption = 2.0  # Watts

        # Total power consumption
        total_power_consumption = base_power_consumption + \
                                  adcs_power['reaction_wheel_power_w'] + \
                                  adcs_power['magnetorquer_power_w']
        
        # Power balance
        power_balance = solar_power - total_power_consumption
        
        # Update battery charge
        energy_delta = power_balance * (dt / 3600.0)  # Convert to watt-hours
        self.battery_charge = min(
            self.battery_capacity_wh,
            max(0.0, self.battery_charge + energy_delta)
        )
        
        return {
            'battery_charge_wh': self.battery_charge,
            'battery_percentage': (self.battery_charge / self.battery_capacity_wh) * 100,
            'solar_power_w': solar_power,
            'power_consumption_w': total_power_consumption,
            'eclipse_state': eclipse_state.eclipse_type,
            'illumination': eclipse_state.illumination
        }