"""
Configuration settings for SmartAir Guardian application.

This module centralizes all configuration constants including paths,
serial communication parameters, model settings, and data collection configurations.
"""

import os
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Directory Configuration
# ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"
SERVER_DIR = BASE_DIR / "server"
LOG_DIR = BASE_DIR / "logs"

# Ensure log directory exists
LOG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Dataset Configuration
# ─────────────────────────────────────────────────────────────

DATASET_PATH = DATASET_DIR / "gas_dataset.csv"
TRAIN_DATASET_PATH = DATASET_DIR / "processed" / "smartair_train.csv"
TEST_DATASET_PATH = DATASET_DIR / "processed" / "smartair_test.csv"
BACKUP_DIR = DATASET_DIR / "backups"

# Ensure dataset directories exist
DATASET_DIR.mkdir(exist_ok=True)
BACKUP_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Model Configuration
# ─────────────────────────────────────────────────────────────

MODEL_PATH = MODEL_DIR / "outputs" / "smartair_model.keras"
MODEL_BACKUP_PATH = MODEL_DIR / "outputs" / "smartair_model.backup.keras"
MODEL_HISTORY_PATH = MODEL_DIR / "outputs" / "training_history.json"

# Ensure model directories exist
(MODEL_DIR / "outputs").mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Serial Communication Configuration
# ─────────────────────────────────────────────────────────────

# Default serial port (override via environment variable for cross-platform compatibility)
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM3")
SERIAL_BAUD_RATE = int(os.getenv("SERIAL_BAUD_RATE", 115200))
SERIAL_TIMEOUT = 2  # seconds


# ─────────────────────────────────────────────────────────────
# Data Collection Configuration
# ─────────────────────────────────────────────────────────────

SAMPLE_INTERVAL_MS = 500  # Milliseconds between sensor samples
SAMPLES_PER_CLASS = 500   # Target number of samples per gas class
NUM_SENSOR_CLASSES = 6    # Number of different gas types to classify


# ─────────────────────────────────────────────────────────────
# Server Configuration
# ─────────────────────────────────────────────────────────────

SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", 5000))
SERVER_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY", "smartair-development-key-change-in-production")

# Maximum number of readings kept in memory
MAX_HISTORY_SIZE = 500

# Alert thresholds
ALERT_AQI_THRESHOLD = 150  # Unhealthy air quality threshold
ALERT_GAS_PPM_THRESHOLD = 100  # Parts per million threshold for gas concentration


# ─────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOG_DIR / "smartair.log"