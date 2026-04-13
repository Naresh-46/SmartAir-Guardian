# ============================================================
#  SmartAir-Guardian — Flask Server Entry Point
#  server/app.py
#
#  Run from project root:
#    python server/app.py
#
#  Endpoints:
#    GET  /              → Dashboard UI
#    POST /api/predict   → Single inference (ESP32 → server)
#    GET  /api/history   → Last N readings  (default 100)
#    GET  /api/stats     → Class + severity counts
#    GET  /api/alerts    → Recent triggered alerts
#    GET  /api/status    → Model + server health
# ============================================================

import os
import sys

from flask import Flask, render_template, jsonify

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server.utils.load_model import ModelLoader
from server.utils.history    import ReadingHistory
from server.utils.alerts     import AlertManager
from server.routes.predict_route import predict_bp
from server.routes.history_route import history_bp
from server.routes.status_route  import status_bp

# ── App setup ────────────────────────────────────────────────
app = Flask(__name__,
            template_folder="templates",
            static_folder="static")

app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "smartair-dev-key")
app.config["MODEL_PATH"] = os.getenv(
    "MODEL_PATH",
    "model/outputs/smartair_model.keras"
)

# ── Shared instances (attached to app for blueprint access) ──
app.model_loader  = ModelLoader(app.config["MODEL_PATH"])
app.reading_hist  = ReadingHistory(maxlen=500)
app.alert_manager = AlertManager()

# ── Blueprints ───────────────────────────────────────────────
app.register_blueprint(predict_bp)
app.register_blueprint(history_bp)
app.register_blueprint(status_bp)

# ── Dashboard ────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

# ── Error handlers ───────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "internal server error", "detail": str(e)}), 500

# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    print(f"\n[SmartAir] Server starting → http://localhost:{port}")
    print(f"[SmartAir] Model path : {app.config['MODEL_PATH']}")
    print(f"[SmartAir] Debug mode : {debug}\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
