"""
Spacecraft subsystem implementations.
"""

from .adcs import ADCS, ADCSMode, ADCSState
from .comms import CommunicationsSubsystem, CommState
from .fms import FileManagementSystem, StorageArea
from .obc import OBC, SystemMode
from .payload import PayloadManager, PayloadState
from .power import PowerSubsystem, PowerState
from .thermal import ThermalSubsystem, ThermalMode, ThermalZone  # New import

__all__ = [
    'ADCS',
    'ADCSMode',
    'ADCSState',
    'CommunicationsSubsystem',
    'CommState',
    'FileManagementSystem',
    'StorageArea',
    'OBC',
    'SystemMode',
    'PayloadManager',
    'PayloadState',
    'PowerSubsystem',
    'PowerState',
    'ThermalSubsystem',
    'ThermalMode',
    'ThermalZone'
]