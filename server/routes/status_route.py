# ============================================================
#  SmartAir-Guardian — Status Route
#  server/routes/status_route.py
#
#  GET /api/status  → model info + reading stats + server health
# ============================================================

import os
from flask import Blueprint, jsonify, current_app

status_bp = Blueprint("status", __name__)


@status_bp.route("/api/status")
def status():
    loader = current_app.model_loader
    hist   = current_app.reading_hist

    model_info = {
        "loaded": loader.loaded,
        "path":   loader.model_path,
        "exists": os.path.exists(loader.model_path),
    }
    if loader.loaded and loader.model:
        try:
            model_info["parameters"] = loader.model.count_params()
        except Exception:
            pass

    return jsonify({
        "status":  "ok",
        "model":   model_info,
        "history": hist.get_stats(),
        "version": "1.0.0",
        "project": "SmartAir-Guardian",
    })
