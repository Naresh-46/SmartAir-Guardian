# ============================================================
#  SmartAir-Guardian — Reading History
#  server/utils/history.py
# ============================================================

from collections import deque, Counter
from datetime import datetime


class ReadingHistory:
    """Circular buffer of sensor readings (last N entries)."""

    def __init__(self, maxlen=500):
        self._buf = deque(maxlen=maxlen)

    def add(self, reading: dict, prediction: dict) -> dict:
        entry = {
            "timestamp":  datetime.utcnow().isoformat() + "Z",
            "sensors":    reading,
            "prediction": prediction,
        }
        self._buf.appendleft(entry)
        return entry

    def get_recent(self, n: int = 100) -> list:
        return list(self._buf)[:n]

    def get_stats(self) -> dict:
        buf = list(self._buf)
        if not buf:
            return {"total": 0}
        classes    = [e["prediction"]["gas_class_id"] for e in buf]
        severities = [e["prediction"]["severity"]     for e in buf]
        return {
            "total":           len(buf),
            "class_counts":    dict(Counter(classes)),
            "severity_counts": dict(Counter(severities)),
            "last_updated":    buf[0]["timestamp"],
        }
