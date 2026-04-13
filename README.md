# SmartAir Guardian

IoT Multi-Gas Monitoring System with Deep Learning Classification

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)]()
[![Flask](https://img.shields.io/badge/Flask-2.x-green)]()
[![License](https://img.shields.io/badge/License-MIT-purple)](LICENSE)

## Overview

SmartAir Guardian is an intelligent air quality monitoring system that uses IoT sensors and deep learning to detect and classify harmful gases. The system combines:

- **Multiple Gas Sensors**: MQ-135, MQ-3, MQ-7, MQ-4 for comprehensive gas detection
- **Environmental Sensors**: Temperature and humidity monitoring
- **Deep Learning Model**: Multi-task DNN for gas classification, concentration estimation, and hazard severity detection
- **Web Dashboard**: Real-time visualization of air quality metrics
- **ESP32 Integration**: Wireless sensor data collection from remote locations
- **Alert System**: Automatic notifications when hazardous conditions are detected

## Features

### Core Functionality
- 🔍 **Multi-Class Gas Detection**: Classifies 6 different gas types with high accuracy
- 📊 **Real-Time Analysis**: Processes sensor readings instantly and provides predictions
- ⚠️ **Intelligent Alerting**: Triggers alerts based on gas concentration and air quality index (AQI)
- 📈 **Historical Data**: Maintains and visualizes historical readings and trends
- 🌐 **Web Interface**: Beautiful dashboard for monitoring air quality metrics
- 📡 **IoT Integration**: Seamless communication with ESP32 sensor nodes

### Technical Highlights
- Multi-task learning architecture with 3 output heads:
  - Gas type classification (6 classes)
  - Concentration PPM estimation
  - Hazard severity detection
- Balanced class weights to handle imbalanced training data
- Early stopping and learning rate reduction callbacks
- Comprehensive performance metrics and visualizations

## Quick Start

### Requirements

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SmartAir-Guardian.git
cd SmartAir-Guardian
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure settings (optional):
Edit `config/settings.py` to customize:
- Serial port for sensor communication
- Model path
- Server host and port
- Alert thresholds

### Running the System

#### Train the Model

```bash
python run_training.py
```

This will:
- Load training and test datasets
- Build the multi-task DNN model
- Train on sensor data
- Evaluate performance and save the model

#### Start the Web Server

```bash
python server/app.py
```

The dashboard will be available at `http://localhost:5000`

#### Collect Sensor Data

```bash
python data_collection/scripts/collect_dataset.py
```

## Project Structure

```
SmartAir-Guardian/
├── config/                          # Configuration modules
│   ├── __init__.py
│   ├── settings.py                  # Global application settings
│   └── logger.py                    # Logging configuration
│
├── data_collection/                 # Data collection and fusion
│   ├── firmware/                    # Arduino/ESP32 firmware
│   │   ├── gas_data_collector/
│   │   ├── smartair_http_client/
│   │   └── smartair_labeling_esp8266/
│   └── scripts/
│       ├── collect_dataset.py       # Main data collection
│       ├── validate_dataset.py      # Data validation
│       └── fusion_pipeline.py       # Multi-source data fusion
│
├── dataset/                         # Training data
│   ├── raw/                         # Original datasets
│   ├── processed/                   # Preprocessed data
│   │   ├── smartair_train.csv
│   │   └── smartair_test.csv
│   └── esp32_collected/             # ESP32 sensor dumps
│
├── model/                           # ML model components
│   ├── train_model.py              # Training script
│   ├── evaluate.py                 # Evaluation and metrics
│   ├── predict.py                  # Inference interface
│   ├── configs/                    # Model configuration
│   │   └── model_config.yaml       # Hyperparameters
│   ├── baselines/                  # Baseline models
│   │   ├── random_forest.py
│   │   ├── svm.py
│   │   └── single_task_dnn.py
│   ├── utils/
│   │   ├── preprocessing.py        # Data loading & transformation
│   │   ├── metrics.py              # Evaluation metrics & plots
│   │   └── fusion_pipeline.py      # Advanced preprocessing
│   └── outputs/                    # Trained models & results
│       ├── smartair_model.keras    # Production model
│       ├── training_history.json
│       └── evaluation_report.txt
│
├── server/                         # Flask web application
│   ├── app.py                      # Main Flask app
│   ├── routes/
│   │   ├── predict_route.py        # /api/predict endpoint
│   │   ├── history_route.py        # /api/history endpoint
│   │   └── status_route.py         # /api/status endpoint
│   ├── utils/
│   │   ├── load_model.py           # Model loading and caching
│   │   ├── history.py              # Reading history management
│   │   ├── alerts.py               # Alert triggering logic
│   ├── templates/
│   │   └── dashboard.html          # Web UI
│   └── static/
│       ├── css/
│       │   └── style.css
│       └── js/
│           ├── api.js
│           ├── dashboard.js
│           └── chart_config.js
│
├── run_training.py                 # Training monitor script
├── check_model.py                  # Model validation utility
├── test_imports.py                 # Dependency validation
├── test_server.py                  # Server integration tests
├── requirements.txt                # Python dependencies
├── requirements_server.txt         # Server-specific dependencies
└── README.md                       # This file
```

## Configuration

### Main Settings (config/settings.py)

```python
# Serial Communication
SERIAL_PORT = "COM3"           # Adjust to your port
SERIAL_BAUD_RATE = 115200

# Data Collection
SAMPLE_INTERVAL_MS = 500       # 500ms between samples
SAMPLES_PER_CLASS = 500        # Target per gas type

# Server
SERVER_HOST = "0.0.0.0"        # Listen on all interfaces
SERVER_PORT = 5000
SERVER_DEBUG = True            # Set to False in production

# Alert Thresholds
ALERT_AQI_THRESHOLD = 150      # Unhealthy air quality
ALERT_GAS_PPM_THRESHOLD = 100  # PPM concentration limit
```

### Model Configuration (model/configs/model_config.yaml)

The YAML file defines:
- Dataset paths
- Feature columns (sensor, engineered, missingness)
- Model architecture (dense units, dropout, activation)
- Training parameters (batch size, epochs, learning rate)
- Class definitions and mappings

## API Reference

### Prediction Endpoint

**POST /api/predict**

Submit sensor readings for classification:

```json
{
  "mq135": 42.5,
  "mq3": 15.0,
  "mq7": 8.0,
  "mq4": 25.0,
  "temp": 22.5,
  "humidity": 65.0,
  "flame": 0
}
```

Response:
```json
{
  "timestamp": "2024-04-13T10:30:45Z",
  "prediction": {
    "gas_class": "Normal",
    "concentration_ppm": 45.2,
    "aqi_severity": "Good",
    "hazard_level": 0.15
  },
  "alert": null
}
```

### History Endpoint

**GET /api/history?limit=100**

Retrieve historical readings (default: last 100)

### Status Endpoint

**GET /api/status**

Check model and server health status

## Model Architecture

The DNN consists of:

```
Input Layer (7 features)
    ↓
Shared Backbone:
  - Dense(128, ReLU) → BatchNorm → Dropout(0.3)
  - Dense(64, ReLU) → BatchNorm → Dropout(0.15)
    ↓
Three Output Heads:
  ├─ Classification Head → Dense(64) → Dense(6, Softmax)  [Gas Type]
  ├─ Regression Head    → Dense(32) → Dense(1, Linear)    [PPM]
  └─ Severity Head      → Dense(32) → Dense(1, Sigmoid)   [Hazard]
```

## Performance

Expected performance on test set:
- **Gas Classification Accuracy**: ~92%
- **Concentration MAE**: ±15 PPM
- **Hazard Detection F1**: ~0.88
- **Inference Time**: <5ms per sample

## Logging

Configure logging via environment variable:

```bash
export LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR, CRITICAL
python server/app.py
```

Logs are saved to `logs/smartair.log` with automatic rotation (10MB max, 5 backups).

## Testing

Run tests to validate setup:

```bash
python test_imports.py      # Check all dependencies
python test_server.py       # Test server endpoints
python check_model.py       # Validate trained model
```

## Environment Variables

```bash
# Server Configuration
export SERVER_HOST=0.0.0.0
export SERVER_PORT=5000
export FLASK_DEBUG=true
export SECRET_KEY=your-secret-key

# Model Configuration
export MODEL_PATH=model/outputs/smartair_model.keras

# Data Collection
export SERIAL_PORT=/dev/ttyUSB0  # Linux/Mac
export SERIAL_PORT=COM3          # Windows

# Logging
export LOG_LEVEL=INFO
```

## Troubleshooting

### Serial Port Issues
- Linux/Mac: Use `/dev/ttyUSB0` or `/dev/ttyACM0`
- Windows: Use `COM3`, `COM4`, etc.
- Find available ports: `python -m serial.tools.list_ports`

### Model Not Loading
- Verify `MODEL_PATH` points to existing file
- Check TensorFlow installation: `python -c "import tensorflow; print(tensorflow.__version__)"`

### Dashboard Not Displaying
- Check server is running: `http://localhost:5000`
- Clear browser cache
- Check browser console for JavaScript errors

### Low Prediction Accuracy
- Ensure dataset is balanced across gas classes
- Check sensor calibration
- Verify feature normalization in preprocessing

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use SmartAir Guardian in your research, please cite:

```bibtex
@software{smartair2024,
  title={SmartAir Guardian: IoT Multi-Gas Monitoring with Deep Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/SmartAir-Guardian}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review the troubleshooting section

## Changelog

### Version 1.0.0 (2024-04-13)
- Initial release
- Multi-task DNN model
- Web dashboard
- Real-time prediction API
- Alert system
- Data collection scripts

## Roadmap

- [ ] Mobile app for iOS/Android
- [ ] Cloud integration (Azure IoT Hub)
- [ ] Advanced time-series forecasting
- [ ] Multi-model ensemble
- [ ] WebSocket for live updates
- [ ] Database backend for long-term storage
- [ ] Docker containerization
- [ ] Kubernetes deployment

---

**SmartAir Guardian** - Making air quality monitoring intelligent and accessible.
