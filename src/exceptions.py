"""
Custom exceptions for satellite operations.
"""

class SatelliteError(Exception):
    """Base exception for all satellite-related errors."""
    pass

class SubsystemError(SatelliteError):
    """Base exception for subsystem-specific errors."""
    def __init__(self, subsystem: str, message: str):
        self.subsystem = subsystem
        self.message = message
        super().__init__(f"[{subsystem}] {message}")

class CommandError(SatelliteError):
    """Exception for command processing errors."""
    def __init__(self, command_id: str, reason: str):
        self.command_id = command_id
        self.reason = reason
        super().__init__(f"Command {command_id} failed: {reason}")

class StorageError(SatelliteError):
    """Exception for storage-related errors."""
    pass

class CommunicationError(SatelliteError):
    """Exception for communication-related errors."""
    pass

class ValidationError(SatelliteError):
    """Exception for data validation errors."""
    pass