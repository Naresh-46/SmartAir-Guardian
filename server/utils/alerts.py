"""
Alert Management System - Hazardous Gas Detection Alerts.

Monitors model predictions for hazardous gas conditions and generates
alerts when dangerous levels are detected. Provides a pluggable alert
delivery system supporting multiple backends (logging, MQTT, webhooks, etc.).

Features:
  - Real-time hazard detection from predictions
  - Configurable alert thresholds and conditions
  - Alert history with recent alerts retrieval
  - Extensible delivery mechanisms (email, SMS, webhooks, MQTT)
  - Duplicate alert suppression via cooldown periods
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Set
import logging

logger = logging.getLogger(__name__)


class AlertManager:
    """
    Manager for generating and delivering hazard alerts.
    
    Monitors predictions for dangerous gas conditions and creates alerts.
    Configurable thresholds determine what constitutes a hazardous state.
    
    Attributes
    ----------
    alert_classes : set
        Gas classes that trigger alerts (all non-clean air classes).
    alert_severities : set
        Severity levels that trigger alerts.
    alert_cooldown : int
        Minimum seconds between alerts for same gas class.
    
    Parameters
    ----------
    alert_cooldown : int, optional
        Minimum seconds between duplicate alerts (default: 60).
    max_history : int, optional
        Maximum number of alerts to keep in memory (default: 100).
    
    Examples
    --------
    >>> alert_mgr = AlertManager()
    >>> prediction = {
    ...     "gas_class": "Smoke",
    ...     "aqi_severity": "Unhealthy",
    ...     "hazard_level": 0.85
    ... }
    >>> alert = alert_mgr.check_alert(prediction)
    >>> if alert:
    ...     print(f"Alert triggered: {alert['message']}")
    """
    
    # Gas classes that should trigger alerts (all except clean air)
    ALERT_CLASSES = {"Smoke", "Alcohol", "NH3", "Fire", "LPG"}
    
    # Severity levels that warrant alerts
    ALERT_SEVERITIES = {"Unhealthy", "Very Unhealthy", "Hazardous"}
    
    def __init__(self, alert_cooldown: int = 60, max_history: int = 100) -> None:
        """
        Initialize the alert manager.
        
        Parameters
        ----------
        alert_cooldown : int
            Minimum seconds between alerts for the same gas.
        max_history : int
            Maximum alerts to store in history.
        """
        self._alert_log = deque(maxlen=max_history)
        self._last_alert_time: Dict[str, datetime] = {}
        self._alert_cooldown = timedelta(seconds=alert_cooldown)
        self._max_history = max_history
        
        logger.debug(
            f"AlertManager initialized with cooldown={alert_cooldown}s, "
            f"max_history={max_history}"
        )
    
    def check_alert(self, prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if prediction triggers an alert.
        
        Evaluates prediction against alert thresholds. If triggered and
        not in cooldown period, generates and logs an alert.
        
        Parameters
        ----------
        prediction : dict
            Model prediction dictionary containing:
            - 'gas_class': str, name of detected gas
            - 'aqi_severity': str, severity level
            - 'hazard_level': float, confidence (0-1)
            - 'concentration_ppm': float, PPM estimate
        
        Returns
        -------
        dict or None
            Alert dictionary if triggered, None otherwise.
        
        Examples
        --------
        >>> prediction = {
        ...     "gas_class": "Smoke",
        ...     "aqi_severity": "Unhealthy",
        ...     "hazard_level": 0.92,
        ...     "concentration_ppm": 450
        ... }
        >>> alert = alert_mgr.check_alert(prediction)
        >>> if alert:
        ...     print(alert)  # {'timestamp': ..., 'message': ...}
        """
        # Check if conditions warrant an alert
        if not self._should_alert(prediction):
            return None
        
        # Check cooldown period
        gas_class = prediction.get("gas_class", "Unknown")
        if self._in_cooldown(gas_class):
            logger.debug(f"Alert for {gas_class} in cooldown period")
            return None
        
        # Create alert
        alert = self._create_alert(prediction)
        
        # Update cooldown
        self._last_alert_time[gas_class] = datetime.utcnow()
        
        # Store in history
        self._alert_log.appendleft(alert)
        
        # Deliver alert
        self._deliver_alert(alert)
        
        logger.warning(f"Alert generated: {alert['message']}")
        
        return alert
    
    def _should_alert(self, prediction: Dict[str, Any]) -> bool:
        """
        Determine if prediction triggers alert conditions.
        
        Parameters
        ----------
        prediction : dict
            Model prediction.
        
        Returns
        -------
        bool
            True if alert conditions met, False otherwise.
        """
        gas_class = prediction.get("gas_class", "Unknown")
        severity = prediction.get("aqi_severity", "Good")
        hazard_level = prediction.get("hazard_level", 0.0)
        
        # Check if gas class and severity warrant alerting
        is_dangerous_class = gas_class in self.ALERT_CLASSES
        is_dangerous_severity = severity in self.ALERT_SEVERITIES
        is_high_confidence = hazard_level > 0.5
        
        return is_dangerous_class and is_dangerous_severity and is_high_confidence
    
    def _in_cooldown(self, gas_class: str) -> bool:
        """
        Check if alert for this gas is still in cooldown period.
        
        Parameters
        ----------
        gas_class : str
            Gas class identifier.
        
        Returns
        -------
        bool
            True if in cooldown, False otherwise.
        """
        if gas_class not in self._last_alert_time:
            return False
        
        last_alert = self._last_alert_time[gas_class]
        return datetime.utcnow() - last_alert < self._alert_cooldown
    
    def _create_alert(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construct alert dictionary from prediction.
        
        Parameters
        ----------
        prediction : dict
            Model prediction.
        
        Returns
        -------
        dict
            Alert dictionary with all relevant information.
        """
        alert = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "gas_class": prediction.get("gas_class", "Unknown"),
            "severity": prediction.get("aqi_severity", "Unknown"),
            "hazard_level": prediction.get("hazard_level", 0.0),
            "concentration_ppm": prediction.get("concentration_ppm", 0.0),
            "message": self._build_message(prediction),
        }
        return alert
    
    def _build_message(self, prediction: Dict[str, Any]) -> str:
        """
        Build human-readable alert message.
        
        Parameters
        ----------
        prediction : dict
            Model prediction.
        
        Returns
        -------
        str
            Formatted alert message.
        """
        gas = prediction.get("gas_class", "Unknown gas")
        severity = prediction.get("aqi_severity", "Unknown")
        ppm = prediction.get("concentration_ppm", 0.0)
        confidence = prediction.get("hazard_level", 0.0) * 100
        
        message = (
            f"⚠️ ALERT: {severity} air quality detected!\n"
            f"   Gas: {gas}\n"
            f"   Level: ~{ppm:.1f} PPM\n"
            f"   Confidence: {confidence:.0f}%"
        )
        return message
    
    def _deliver_alert(self, alert: Dict[str, Any]) -> None:
        """
        Deliver alert via all configured channels.
        
        This is the main extensibility point for connecting to external
        services. Override or extend to add:
          - Email notifications
          - SMS alerts
          - Slack/Discord webhooks
          - MQTT publishing
          - Database logging
        
        Parameters
        ----------
        alert : dict
            Alert to deliver.
        
        Examples
        --------
        To add webhook delivery:
        
        >>> def _deliver_alert(self, alert):
        ...     super()._deliver_alert(alert)
        ...     requests.post("https://webhook.example.com", json=alert)
        """
        # Log to application logger
        logger.critical(alert["message"])
        
        # Uncomment below to add MQTT delivery:
        # self._deliver_via_mqtt(alert)
        
        # Uncomment below to add webhook delivery:
        # self._deliver_via_webhook(alert)
        
        # Uncomment below to add email delivery:
        # self._deliver_via_email(alert)
    
    def get_recent_alerts(self, n: int = 20) -> list:
        """
        Retrieve recent alerts.
        
        Parameters
        ----------
        n : int
            Number of recent alerts to retrieve (default: 20).
        
        Returns
        -------
        list
            Recent alert dictionaries (newest first).
        
        Examples
        --------
        >>> alerts = alert_mgr.get_recent_alerts(n=10)
        >>> for alert in alerts:
        ...     print(f"{alert['timestamp']}: {alert['gas_class']}")
        """
        return list(self._alert_log)[:n]
    
    def clear_history(self) -> int:
        """
        Clear all alert history.
        
        Returns
        -------
        int
            Number of alerts cleared.
        """
        count = len(self._alert_log)
        self._alert_log.clear()
        logger.info(f"Cleared {count} alerts from history")
        return count
    
    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"AlertManager(history_size={len(self._alert_log)}/{self._max_history}, "
            f"cooldown={self._alert_cooldown.total_seconds()}s)"
        )

