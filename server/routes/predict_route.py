"""
Prediction Route - Single Model Inference Endpoint.

Handles POST requests for air quality predictions based on sensor input.
Receives sensor readings from ESP32 devices and returns predictions along
with any triggered alerts.

Endpoints
---------
POST /api/predict
    Submit sensor readings and get model predictions.
    
    Request Body (JSON):
        {
            "mq135": float,    # MQ-135 sensor reading (ppm)
            "mq3": float,      # MQ-3 sensor reading (alcohol ppm)
            "mq7": float,      # MQ-7 sensor reading (CO ppm)
            "mq4": float,      # MQ-4 sensor reading (methane ppm)
            "temp": float,     # Temperature in Celsius
            "humidity": float, # Relative humidity percentage
            "flame": 0|1       # Flame detector (0=no flame, 1=flame detected)
        }
    
    Response (JSON):
        {
            "timestamp": "2024-04-13T10:30:45Z",
            "prediction": {
                "gas_class": "Normal",
                "concentration_ppm": 45.2,
                "aqi_severity": "Good",
                "hazard_level": 0.15
            },
            "alert": null | {...}  # Alert object if threshold exceeded
        }
"""

from datetime import datetime
from typing import Dict, Any, Tuple

from flask import Blueprint, request, jsonify, current_app

from config.logger import get_logger


logger = get_logger(__name__)


predict_bp = Blueprint("predict", __name__)

# Expected sensor fields that must be present in request
REQUIRED_SENSOR_FIELDS = ["mq135", "mq3", "mq7", "mq4", "temp", "humidity", "flame"]


@predict_bp.route("/api/predict", methods=["POST"])
def predict_endpoint() -> Tuple[Dict[str, Any], int]:
    """
    Handle incoming prediction requests from sensor nodes.
    
    Returns
    -------
    Tuple[Dict, int]
        Response JSON and HTTP status code.
    
    Raises
    ------
    ValueError
        If required fields are missing or have invalid types.
    
    Examples
    --------
    Request:
        POST /api/predict
        Content-Type: application/json
        {
            "mq135": 42.5,
            "mq3": 15.0,
            "mq7": 8.0,
            "mq4": 25.0,
            "temp": 22.5,
            "humidity": 65.0,
            "flame": 0
        }
    """
    # Parse JSON request
    sensor_data = request.get_json(silent=True)
    
    if not sensor_data:
        logger.warning("Prediction request received with empty JSON body")
        return jsonify({"error": "JSON body required"}), 400
    
    # Validate required fields
    missing_fields = [field for field in REQUIRED_SENSOR_FIELDS 
                      if field not in sensor_data]
    if missing_fields:
        logger.warning(f"Missing required fields: {missing_fields}")
        return jsonify({"error": f"Missing fields: {missing_fields}"}), 400
    
    # Parse and validate sensor values
    try:
        sensor_values = _parse_sensor_values(sensor_data)
    except (ValueError, TypeError) as error:
        logger.warning(f"Invalid sensor values: {error}")
        return jsonify({"error": f"Invalid sensor value: {error}"}), 400
    
    # Get application components
    model_loader = current_app.model_loader
    reading_history = current_app.reading_history
    alert_manager = current_app.alert_manager
    
    # Make prediction
    try:
        prediction = model_loader.predict(**sensor_values)
        logger.debug(f"Prediction completed: {prediction['gas_class']}")
    except Exception as error:
        logger.error(f"Prediction failed: {error}", exc_info=True)
        return jsonify({"error": "Model prediction failed", "detail": str(error)}), 500
    
    # Create reading record
    timestamp = datetime.utcnow().isoformat() + "Z"
    reading_record = {
        "timestamp": timestamp,
        **sensor_values,
        "prediction": prediction
    }
    
    # Store in history
    reading_history.append(reading_record)
    logger.debug(f"Reading stored. History size: {len(reading_history)}")
    
    # Check for alerts
    alert = alert_manager.check_alert(prediction)
    if alert:
        logger.info(f"Alert triggered: {alert}")
    
    # Return response
    response = {
        "timestamp": timestamp,
        "prediction": prediction,
        "alert": alert
    }
    
    return jsonify(response), 200


def _parse_sensor_values(sensor_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Parse and validate sensor values from request.
    
    Parameters
    ----------
    sensor_data : dict
        Raw sensor data from request JSON.
    
    Returns
    -------
    dict
        Parsed sensor values as floats.
    
    Raises
    ------
    ValueError
        If any value cannot be converted to float or is out of valid range.
    TypeError
        If value type is invalid.
    """
    try:
        parsed_values = {
            "mq135": float(sensor_data["mq135"]),
            "mq3": float(sensor_data["mq3"]),
            "mq7": float(sensor_data["mq7"]),
            "mq4": float(sensor_data["mq4"]),
            "temp": float(sensor_data["temp"]),
            "humidity": float(sensor_data["humidity"]),
            "flame": float(sensor_data["flame"])
        }
    except (ValueError, TypeError) as error:
        raise ValueError(f"Cannot convert sensor value to float: {error}")
    
    # Validate reasonable ranges
    if not (-40 < parsed_values["temp"] < 60):
        raise ValueError(f"Temperature out of range: {parsed_values['temp']}")
    
    if not (0 <= parsed_values["humidity"] <= 100):
        raise ValueError(f"Humidity out of range: {parsed_values['humidity']}")
    
    if parsed_values["flame"] not in [0, 1]:
        raise ValueError(f"Flame value must be 0 or 1: {parsed_values['flame']}")
    
    return parsed_values

        "temp":  temp,  "hum": hum, "flame": flame,
    }
    entry = hist.add(reading, prediction)
    alert = alertmgr.process(prediction)

    return jsonify({
        "timestamp":  entry["timestamp"],
        "prediction": prediction,
        "alert":      alert,
    }), 200
