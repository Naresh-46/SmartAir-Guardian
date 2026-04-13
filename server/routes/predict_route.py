# ============================================================
#  SmartAir-Guardian — Predict Route
#  server/routes/predict_route.py
#
#  POST /api/predict
#  Body (JSON):
#    { "mq135": float, "mq3": float, "mq7": float, "mq4": float,
#      "temp": float, "hum": float, "flame": 0|1 }
#
#  Response:
#    { "timestamp", "prediction": {...}, "alert": {...}|null }
# ============================================================

from flask import Blueprint, request, jsonify, current_app

predict_bp = Blueprint("predict", __name__)

REQUIRED_FIELDS = ["mq135", "mq3", "mq7", "mq4", "temp", "hum", "flame"]


@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "JSON body required"}), 400

    missing = [k for k in REQUIRED_FIELDS if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        mq135 = float(data["mq135"])
        mq3   = float(data["mq3"])
        mq7   = float(data["mq7"])
        mq4   = float(data["mq4"])
        temp  = float(data["temp"])
        hum   = float(data["hum"])
        flame = float(data["flame"])
    except (ValueError, TypeError) as exc:
        return jsonify({"error": f"Invalid value: {exc}"}), 400

    loader   = current_app.model_loader
    hist     = current_app.reading_hist
    alertmgr = current_app.alert_manager

    prediction = loader.predict(mq135, mq3, mq7, mq4, temp, hum, flame)

    reading = {
        "mq135": mq135, "mq3": mq3, "mq7": mq7, "mq4": mq4,
        "temp":  temp,  "hum": hum, "flame": flame,
    }
    entry = hist.add(reading, prediction)
    alert = alertmgr.process(prediction)

    return jsonify({
        "timestamp":  entry["timestamp"],
        "prediction": prediction,
        "alert":      alert,
    }), 200
