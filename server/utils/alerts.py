# ============================================================
#  SmartAir-Guardian — Alert Manager
#  server/utils/alerts.py
# ============================================================

from collections import deque
from datetime import datetime


class AlertManager:
    """Sends alerts when hazardous gas conditions are detected."""

    ALERT_CLASSES    = {1, 2, 3, 4, 5}          # everything except clean air
    ALERT_SEVERITIES = {"WARNING", "DANGER"}

    def __init__(self):
        self.alert_log        = deque(maxlen=100)
        self._last_alert_time = {}               # gas_id → last alert timestamp

    def should_alert(self, prediction: dict) -> bool:
        gas_id   = prediction.get("gas_class_id", 0)
        severity = prediction.get("severity", "SAFE")
        return (gas_id in self.ALERT_CLASSES and
                severity in self.ALERT_SEVERITIES)

    def process(self, prediction: dict) -> dict | None:
        if not self.should_alert(prediction):
            return None

        alert = {
            "timestamp":  datetime.utcnow().isoformat() + "Z",
            "gas_name":   prediction["gas_name"],
            "gas_class":  prediction["gas_class_id"],
            "severity":   prediction["severity"],
            "confidence": prediction["confidence"],
            "ppm":        prediction["ppm_estimate"],
            "message":    self._build_message(prediction),
        }
        self.alert_log.appendleft(alert)
        self._send(alert)
        return alert

    def _build_message(self, p: dict) -> str:
        return (
            f"⚠ {p['severity']} — {p['gas_name']} detected "
            f"({p['confidence']:.1f}% confidence, ~{p['ppm_estimate']:.0f} ppm)"
        )

    def _send(self, alert: dict):
        """
        Hook for real alert delivery.
        Extend to add MQTT, webhook, email, or SMS.
        """
        print(f"[ALERT] {alert['message']}")

        # ── MQTT example (uncomment + pip install paho-mqtt) ──
        # import paho.mqtt.client as mqtt
        # import json
        # client = mqtt.Client()
        # client.connect("broker.hivemq.com", 1883)
        # client.publish("smartair/alerts", json.dumps(alert))
        # client.disconnect()

    def get_recent_alerts(self, n: int = 20) -> list:
        return list(self.alert_log)[:n]
