# ============================================================
#  SmartAir-Guardian — History Routes
#  server/routes/history_route.py
#
#  GET /api/history          → last N readings
#  GET /api/history?n=50     → last 50 readings (max 500)
#  GET /api/stats            → class + severity counts
#  GET /api/alerts           → last 20 triggered alerts
# ============================================================

from flask import Blueprint, jsonify, request, current_app

history_bp = Blueprint("history", __name__)


@history_bp.route("/api/history")
def history():
    try:
        n = int(request.args.get("n", 100))
        n = min(max(n, 1), 500)
    except ValueError:
        n = 100

    data = current_app.reading_hist.get_recent(n)
    return jsonify({"count": len(data), "readings": data})


@history_bp.route("/api/stats")
def stats():
    return jsonify(current_app.reading_hist.get_stats())


@history_bp.route("/api/alerts")
def alerts():
    try:
        n = int(request.args.get("n", 20))
        n = min(max(n, 1), 100)
    except ValueError:
        n = 20

    data = current_app.alert_manager.get_recent_alerts(n)
    return jsonify({"count": len(data), "alerts": data})
