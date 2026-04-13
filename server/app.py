"""
SmartAir Guardian Flask Server - Main Application Entry Point.

This module provides the main Flask application for the SmartAir Guardian
system, handling HTTP routes, dashboard serving, and integration with
the trained ML model for real-time air quality predictions.

The server provides:
  - Dashboard UI for visualizing air quality data
  - REST API for model predictions
  - Historical data retrieval and analysis
  - Real-time alert management

Usage
-----
Run from project root:
    python server/app.py

Configuration
-------------
Set environment variables:
  - PORT: Server port (default: 5000)
  - MODEL_PATH: Path to trained model (default: model/outputs/smartair_model.keras)
  - SECRET_KEY: Flask secret key for session management
  - FLASK_DEBUG: Enable debug mode (default: true)
"""

import os
import sys
from pathlib import Path

from flask import Flask, render_template, jsonify

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logger import get_logger
from config.settings import (
    SERVER_HOST, SERVER_PORT, SERVER_DEBUG, SECRET_KEY, MODEL_PATH,
    MAX_HISTORY_SIZE
)
from server.utils.load_model import ModelLoader
from server.utils.history import ReadingHistory
from server.utils.alerts import AlertManager
from server.routes.predict_route import predict_bp
from server.routes.history_route import history_bp
from server.routes.status_route import status_bp


logger = get_logger(__name__)


def create_app(config: dict = None) -> Flask:
    """
    Application factory for creating and configuring the Flask app.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary to override defaults.
    
    Returns
    -------
    Flask
        Configured Flask application instance.
    
    Examples
    --------
    >>> app = create_app()
    >>> app.run(debug=True)
    """
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )
    
    # Configuration
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", SECRET_KEY)
    app.config["MODEL_PATH"] = os.getenv("MODEL_PATH", str(MODEL_PATH))
    app.config["MAX_HISTORY_SIZE"] = MAX_HISTORY_SIZE
    
    # Override with provided config if any
    if config:
        app.config.update(config)
    
    logger.info(f"Flask app created with MODEL_PATH: {app.config['MODEL_PATH']}")
    
    # Initialize shared application context
    _initialize_app_context(app)
    
    # Register blueprints
    _register_blueprints(app)
    
    # Register error handlers
    _register_error_handlers(app)
    
    # Log initialization complete
    logger.info("Flask application initialized successfully")
    
    return app


def _initialize_app_context(app: Flask) -> None:
    """
    Initialize shared application context instances.
    
    Attaches model loader, reading history, and alert manager
    to the Flask app for use by blueprints.
    
    Parameters
    ----------
    app : Flask
        Flask application instance.
    """
    try:
        app.model_loader = ModelLoader(app.config["MODEL_PATH"])
        logger.info("Model loader initialized")
    except Exception as e:
        logger.error(f"Failed to initialize model loader: {e}")
        raise
    
    app.reading_history = ReadingHistory(maxlen=app.config["MAX_HISTORY_SIZE"])
    app.alert_manager = AlertManager()
    
    logger.info("Application context initialized with model, history, and alerts")


def _register_blueprints(app: Flask) -> None:
    """
    Register Flask blueprints for routes.
    
    Parameters
    ----------
    app : Flask
        Flask application instance.
    """
    app.register_blueprint(predict_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(status_bp)
    logger.info("Route blueprints registered")


def _register_error_handlers(app: Flask) -> None:
    """
    Register custom error handlers.
    
    Parameters
    ----------
    app : Flask
        Flask application instance.
    """
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 Not Found errors."""
        logger.warning(f"404 error: {error}")
        return jsonify({"error": "Endpoint not found"}), 404
    
    @app.errorhandler(500)
    def handle_server_error(error):
        """Handle 500 Internal Server Error."""
        logger.error(f"500 error: {error}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "detail": str(error)
        }), 500
    
    logger.info("Error handlers registered")


@app.route("/")
def dashboard():
    """
    Serve the main dashboard HTML page.
    
    Returns
    -------
    str
        Rendered HTML dashboard template.
    """
    logger.debug("Dashboard page requested")
    return render_template("dashboard.html")


def run_server(host: str = SERVER_HOST, port: int = SERVER_PORT, 
               debug: bool = SERVER_DEBUG) -> None:
    """
    Start the Flask development server.
    
    Parameters
    ----------
    host : str
        Server host address (default: 0.0.0.0).
    port : int
        Server port number (default: 5000).
    debug : bool
        Enable debug mode (default: True).
    """
    logger.info(f"Starting SmartAir Guardian server")
    logger.info(f"  Server: http://{host}:{port}")
    logger.info(f"  Model: {app.config['MODEL_PATH']}")
    logger.info(f"  Debug mode: {debug}")
    
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    # Create app instance
    app = create_app()
    
    # Get configuration from environment
    port = int(os.getenv("PORT", SERVER_PORT))
    debug = os.getenv("FLASK_DEBUG", str(SERVER_DEBUG)).lower() == "true"
    host = os.getenv("SERVER_HOST", SERVER_HOST)
    
    # Run server
    try:
        run_server(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)


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
