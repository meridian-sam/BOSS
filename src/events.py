"""
Event system for inter-subsystem communication.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable
from queue import Queue
import uuid

class EventType(Enum):
    """Types of events that can occur in the system."""
    COMMAND = auto()
    TELEMETRY = auto()
    STATUS = auto()
    ERROR = auto()
    MODE_CHANGE = auto()
    POWER = auto()
    ATTITUDE = auto()
    ORBITAL = auto()
    PAYLOAD = auto()
    COMMUNICATION = auto()
    FILE_TRANSFER = auto()
    SCHEDULE = auto()

class EventPriority(Enum):
    """Priority levels for events."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class Event:
    """
    Represents a system event.
    
    Attributes:
        event_id: Unique identifier for the event
        type: Type of event
        source: Source subsystem/component
        destination: Target subsystem/component (None for broadcast)
        priority: Event priority level
        timestamp: Time the event was created
        data: Event payload data
        metadata: Additional event metadata
    """
    event_id: str
    type: EventType
    source: str
    destination: Optional[str]
    priority: EventPriority
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class EventBus:
    """
    Central event bus for managing system-wide events.
    """
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {
            event_type: [] for event_type in EventType
        }
        self._event_queue: Queue[Event] = Queue()
        self._event_history: List[Event] = []
        self.max_history = 1000  # Maximum number of events to keep in history
        
    def publish(self, 
                event_type: EventType,
                source: str,
                data: Dict[str, Any],
                destination: Optional[str] = None,
                priority: EventPriority = EventPriority.MEDIUM,
                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Publish an event to the event bus.
        
        Args:
            event_type: Type of event
            source: Source of the event
            data: Event data payload
            destination: Optional destination subsystem
            priority: Event priority level
            metadata: Optional additional metadata

        Returns:
            str: Event ID
        """
        event = Event(
            event_id=str(uuid.uuid4()),
            type=event_type,
            source=source,
            destination=destination,
            priority=priority,
            timestamp=datetime.utcnow(),
            data=data,
            metadata=metadata or {}
        )
        
        self._event_queue.put(event)
        self._event_history.append(event)
        
        # Trim history if needed
        if len(self._event_history) > self.max_history:
            self._event_history = self._event_history[-self.max_history:]
            
        return event.event_id
    
    def subscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of events to subscribe to
            callback: Function to call when event occurs
        """
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: EventType, callback: Callable[[Event], None]):
        """
        Unsubscribe from events of a specific type.
        
        Args:
            event_type: Type of events to unsubscribe from
            callback: Function to remove from subscribers
        """
        if callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
    
    def process_events(self):
        """Process all pending events in the queue."""
        while not self._event_queue.empty():
            event = self._event_queue.get()
            
            # Notify all subscribers for this event type
            for callback in self._subscribers[event.type]:
                try:
                    callback(event)
                except Exception as e:
                    print(f"Error processing event {event.event_id}: {str(e)}")
    
    def get_history(self, 
                   event_type: Optional[EventType] = None,
                   source: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[Event]:
        """
        Get event history with optional filtering.
        
        Args:
            event_type: Filter by event type
            source: Filter by source
            start_time: Filter by start time
            end_time: Filter by end time

        Returns:
            List[Event]: Filtered event history
        """
        filtered_history = self._event_history
        
        if event_type:
            filtered_history = [e for e in filtered_history if e.type == event_type]
        if source:
            filtered_history = [e for e in filtered_history if e.source == source]
        if start_time:
            filtered_history = [e for e in filtered_history if e.timestamp >= start_time]
        if end_time:
            filtered_history = [e for e in filtered_history if e.timestamp <= end_time]
            
        return filtered_history

# Global event bus instance
event_bus = EventBus()

def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    return event_bus