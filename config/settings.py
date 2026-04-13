import os

# Base directory of project
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# Dataset path
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "gas_dataset.csv")

# Model path (future use)
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

# Serial configuration
SERIAL_PORT = "COM3"   # 🔁 change this to your actual COM port
BAUD_RATE = 115200

# Collection settings
SAMPLE_INTERVAL_MS = 500
TARGET_PER_CLASS = 500

# Backup directory
BACKUP_DIR = os.path.join(BASE_DIR, "dataset_backups")