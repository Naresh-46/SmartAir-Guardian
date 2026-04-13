# ============================================================
#  SmartAir-Guardian — Single Inference Script
#  model/predict.py
#
#  Used by: server/app.py (Flask API)
#  Also usable standalone for quick testing.
#
#  Usage:
#    python model/predict.py
#    python model/predict.py --mq135 420 --mq3 380 --mq7 500 \
#           --mq4 450 --temp 28.5 --hum 55.0 --flame 0
# ============================================================

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.utils.preprocessing import load_config


# ── Class and severity labels ────────────────────────────────
SEV_LABELS = {0: "SAFE", 1: "WARNING", 2: "DANGER"}


def build_feature_vector(mq135, mq3, mq7, mq4,
                          temp, hum, flame,
                          cfg):
    """
    Build normalized feature vector from raw sensor readings.

    NOTE: In production, normalization parameters (mean, std)
    must be saved during training and loaded here.
    For now, uses reasonable default scaling based on ADC range.
    Replace with joblib.load("scaler.pkl") after training.
    """
    eps = 1e-6

    # Raw sensors
    raw = np.array([mq135, mq3, mq7, mq4, temp, hum, flame],
                   dtype=np.float32)

    # Normalize (replace with saved StandardScaler in production)
    # These are approximate means/stds from Dataset 2 (MultimodalGas)
    means = np.array([416.7, 462.0, 566.0, 450.0, 28.0, 55.0, 0.0])
    stds  = np.array([76.7,  70.3,  83.1,  30.0,  3.0,  8.0, 0.5])
    norm  = (raw - means) / (stds + eps)

    mq135_n, mq3_n, mq7_n, mq4_n, temp_n, hum_n, flame_n = norm

    # Engineered features
    mq135_mq7_ratio = mq135_n / (mq7_n + eps)
    mq3_mq4_ratio   = mq3_n   / (mq4_n + eps)
    mq135_mq3_ratio = mq135_n / (mq3_n + eps)
    temp_x_mq135    = temp_n  * mq135_n
    hum_x_mq7       = hum_n   * mq7_n
    aqi_score       = mq135_n*0.4 + mq7_n*0.3 + mq3_n*0.2 + mq4_n*0.1
    fire_risk       = flame_n*0.5 + mq135_n*0.3 + temp_n*0.2

    # Missingness flags (0 = real reading from ESP32)
    miss = np.zeros(6, dtype=np.float32)

    feature_vec = np.concatenate([
        norm,
        [mq135_mq7_ratio, mq3_mq4_ratio, mq135_mq3_ratio,
         temp_x_mq135, hum_x_mq7, aqi_score, fire_risk],
        miss
    ])

    return feature_vec.reshape(1, -1).astype(np.float32)


def predict(mq135, mq3, mq7, mq4, temp, hum, flame,
            model=None, cfg=None):
    """
    Run inference on one reading.
    Returns dict with all three head predictions.
    """
    if cfg is None:
        cfg = load_config("model/configs/model_config.yaml")

    if model is None:
        try:
            import tensorflow as tf
            model_path = cfg["paths"]["model_out"]
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model not found at {model_path}. "
                    "Run train_model.py first."
                )
            model = tf.keras.models.load_model(model_path)
        except ImportError:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    X = build_feature_vector(mq135, mq3, mq7, mq4, temp, hum, flame, cfg)

    preds = model.predict(X, verbose=0)

    # Head 1 — gas classification
    class_probs  = preds[0][0]
    gas_class_id = int(np.argmax(class_probs))
    gas_name     = cfg["classes"]["names"][gas_class_id]
    confidence   = float(class_probs[gas_class_id])

    # Head 2 — PPM regression
    ppm_estimate = float(preds[1][0][0])

    # Head 3 — severity
    sev_score = float(preds[2][0][0])
    if sev_score < 0.33:
        severity = "SAFE"
    elif sev_score < 0.66:
        severity = "WARNING"
    else:
        severity = "DANGER"

    result = {
        "gas_class_id":  gas_class_id,
        "gas_name":      gas_name,
        "confidence":    round(confidence * 100, 2),
        "ppm_estimate":  round(ppm_estimate, 2),
        "severity":      severity,
        "severity_score": round(sev_score, 4),
        "all_class_probs": {
            cfg["classes"]["names"][i]: round(float(p) * 100, 2)
            for i, p in enumerate(class_probs)
        }
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="SmartAir single inference")
    parser.add_argument("--mq135", type=float, default=420.0)
    parser.add_argument("--mq3",   type=float, default=380.0)
    parser.add_argument("--mq7",   type=float, default=500.0)
    parser.add_argument("--mq4",   type=float, default=450.0)
    parser.add_argument("--temp",  type=float, default=28.5)
    parser.add_argument("--hum",   type=float, default=55.0)
    parser.add_argument("--flame", type=float, default=0.0)
    args = parser.parse_args()

    print("\n[PREDICT] Input readings:")
    print(f"  MQ135={args.mq135}  MQ3={args.mq3}  MQ7={args.mq7}  MQ4={args.mq4}")
    print(f"  Temp={args.temp}°C  Humidity={args.hum}%  Flame={args.flame}")

    result = predict(
        args.mq135, args.mq3, args.mq7, args.mq4,
        args.temp, args.hum, args.flame
    )

    print("\n[PREDICT] ── RESULT ──────────────────────────────")
    print(f"  Gas Type    : {result['gas_name']} (class {result['gas_class_id']})")
    print(f"  Confidence  : {result['confidence']}%")
    print(f"  PPM Estimate: {result['ppm_estimate']} ppm")
    print(f"  Severity    : {result['severity']} (score={result['severity_score']})")
    print("\n  All class probabilities:")
    for name, prob in result["all_class_probs"].items():
        bar = "█" * int(prob / 5)
        print(f"    {name:<18} {bar} {prob:.1f}%")
    print("──────────────────────────────────────────────────\n")

    return result


if __name__ == "__main__":
    main()
