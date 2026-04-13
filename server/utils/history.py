"""
Reading History Management - Circular Buffer for Sensor Data.

Maintains a fixed-size circular buffer of the most recent sensor readings
and model predictions. Provides methods to retrieve recent readings and
compute statistics over the history window.

The history is stored in memory (no persistence) and is used for:
  - Displaying historical trends in the web dashboard
  - Computing aggregate statistics (class distribution, etc.)
  - Feeding data to visualization components
"""

from collections import deque, Counter
from datetime import datetime
from typing import Dict, List, Any, Optional


class ReadingHistory:
    """
    Circular buffer for storing sensor readings and predictions.
    
    Maintains a fixed-size rolling window of readings with FIFO eviction
    when the buffer is full. Newest readings are added to the front.
    
    Attributes
    ----------
    _buffer : deque
        Circular buffer of reading dictionaries.
    
    Parameters
    ----------
    maxlen : int, optional
        Maximum number of readings to keep in memory (default: 500).
    
    Examples
    --------
    >>> history = ReadingHistory(maxlen=100)
    >>> reading = {"mq135": 45.2, "temp": 22.5}
    >>> prediction = {"gas_class": "Normal", "confidence": 0.92}
    >>> history.append({"reading": reading, "prediction": prediction})
    >>> recent = history.get_recent(n=50)
    >>> stats = history.get_statistics()
    """
    
    def __init__(self, maxlen: int = 500) -> None:
        """
        Initialize the reading history buffer.
        
        Parameters
        ----------
        maxlen : int
            Maximum number of readings to store.
        """
        self._buffer = deque(maxlen=maxlen)
        self._maxlen = maxlen
    
    def append(self, reading_entry: Dict[str, Any]) -> None:
        """
        Add a sensor reading to the history.
        
        Appends a new reading to the front of the buffer.
        Oldest reading is automatically removed if buffer is full.
        
        Parameters
        ----------
        reading_entry : dict
            Reading dictionary containing timestamp, sensor values,
            and prediction results.
        
        Examples
        --------
        >>> entry = {
        ...     "timestamp": "2024-04-13T10:30:45Z",
        ...     "mq135": 42.5,
        ...     "prediction": {"gas_class": "Normal"}
        ... }
        >>> history.append(entry)
        """
        self._buffer.appendleft(reading_entry)
    
    def get_recent(self, n: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieve the most recent N readings.
        
        Returns readings in reverse chronological order
        (most recent first).
        
        Parameters
        ----------
        n : int
            Number of recent readings to retrieve (default: 100).
        
        Returns
        -------
        list
            List of reading dictionaries (newest first).
        
        Examples
        --------
        >>> recent_readings = history.get_recent(n=50)
        >>> latest = recent_readings[0]  # Most recent reading
        """
        return list(self._buffer)[:n]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute aggregate statistics over the history window.
        
        Calculates summary statistics including total count, class
        distribution, severity distribution, and timestamp of last update.
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'total': Total number of readings in buffer
            - 'class_counts': Counter of gas class occurrences
            - 'severity_counts': Counter of severity levels
            - 'last_updated': ISO timestamp of most recent reading
            - 'classes': List of unique gas classes seen
        
        Examples
        --------
        >>> stats = history.get_statistics()
        >>> print(f"Total readings: {stats['total']}")
        >>> print(f"Most common class: {stats['class_counts'].most_common(1)}")
        """
        readings = list(self._buffer)
        
        if not readings:
            return {
                "total": 0,
                "class_counts": {},
                "severity_counts": {},
                "last_updated": None,
                "classes": []
            }
        
        # Extract data from all readings
        gas_classes = [
            reading.get("prediction", {}).get("gas_class") 
            for reading in readings
        ]
        gas_classes = [c for c in gas_classes if c is not None]
        
        severities = [
            reading.get("prediction", {}).get("aqi_severity")
            for reading in readings
        ]
        severities = [s for s in severities if s is not None]
        
        statistics = {
            "total": len(readings),
            "class_counts": dict(Counter(gas_classes)),
            "severity_counts": dict(Counter(severities)),
            "last_updated": readings[0].get("timestamp"),
            "classes": list(set(gas_classes))
        }
        
        return statistics
    
    def __len__(self) -> int:
        """Return the number of readings currently in buffer."""
        return len(self._buffer)
    
    def __repr__(self) -> str:
        """Return string representation of history."""
        return f"ReadingHistory(size={len(self._buffer)}/{self._maxlen})"

