# ============================================================
#  SmartAir-Guardian — Model Loader
#  server/utils/load_model.py
#
#  Feature vector (20 features) matches fused_train.csv columns:
#  MQ135, MQ3, MQ7, MQ4, DHT_temp, DHT_hum, flame,
#  MQ135_MQ7_ratio, MQ3_MQ4_ratio, MQ135_MQ3_ratio,
#  temp_x_MQ135, hum_x_MQ7, AQI_score, fire_risk,
#  MQ135_missing, MQ3_missing, MQ7_missing, MQ4_missing,
#  DHT_temp_missing, DHT_hum_missing
# ============================================================

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class ModelLoader:
    """Loads and caches the SmartAir Keras multi-task model."""

    GAS_NAMES = {
        0: "Clean Air",
        1: "Smoke / CO",
        2: "Alcohol / VOC",
        3: "NH3 / Ammonia",
        4: "Fire / Flame",
        5: "Mixed / LPG",
    }

    PPM_MAX = 600.0   # must match train_model.py PPM_MAX

    # ── Per-source z-score stats (fitted on public train set) ──
    # These were applied during the fusion pipeline.
    # Incoming ESP32 raw readings must be normalised with the
    # ESP32-specific scaler stats before inference.
    # Replace the zeros below with the actual values once you
    # save the scaler from fusion_pipeline.py.
    ESP32_MEANS = np.array(
        [416.7, 462.0, 566.0, 450.0, 28.0, 55.0, 0.0],
        dtype=np.float32,
    )
    ESP32_STDS = np.array(
        [76.7, 70.3, 83.1, 30.0, 3.0, 8.0, 0.5],
        dtype=np.float32,
    )

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model      = None
        self.loaded     = False
        self._try_load()

    # ── Load ─────────────────────────────────────────────────
    def _try_load(self):
        if not os.path.exists(self.model_path):
            print(f"[MODEL] ⚠  Model not found at {self.model_path}")
            print("[MODEL]    Run model/train_model.py first.")
            return
        try:
            import tensorflow as tf
            self.model  = tf.keras.models.load_model(self.model_path)
            self.loaded = True
            print(f"[MODEL] ✓  Loaded from {self.model_path}")
            print(f"[MODEL]    Parameters: {self.model.count_params():,}")
        except ImportError:
            print("[MODEL] ✗  TensorFlow not installed — pip install tensorflow")
        except Exception as exc:
            print(f"[MODEL] ✗  Failed to load: {exc}")

    # ── Public predict ────────────────────────────────────────
    def predict(
        self,
        mq135: float, mq3: float, mq7: float, mq4: float,
        temp: float,  hum: float,  flame: float,
    ) -> dict:
        """
        Run inference on one raw sensor reading from the ESP32.
        Falls back to rule-based demo prediction if model is not loaded.
        """
        if not self.loaded:
            return self._demo_prediction(mq135, mq3, mq7, mq4, temp, hum, flame)

        X     = self._build_features(mq135, mq3, mq7, mq4, temp, hum, flame)
        preds = self.model.predict(X, verbose=0)

        class_probs = preds[0][0]           # (6,)
        ppm_norm    = float(preds[1][0][0]) # [0, 1] sigmoid output
        sev_probs   = preds[2][0]           # (3,)  softmax output

        gas_id     = int(np.argmax(class_probs))
        confidence = float(class_probs[gas_id]) * 100
        ppm        = round(np.clip(ppm_norm, 0.0, 1.0) * self.PPM_MAX, 2)
        sev_id     = int(np.argmax(sev_probs))
        severity   = ["SAFE", "WARNING", "DANGER"][sev_id]

        return {
            "gas_class_id":   gas_id,
            "gas_name":       self.GAS_NAMES[gas_id],
            "confidence":     round(confidence, 2),
            "ppm_estimate":   ppm,
            "severity":       severity,
            "severity_score": round(float(sev_probs[sev_id]), 4),
            "all_probs": {
                self.GAS_NAMES[i]: round(float(p) * 100, 2)
                for i, p in enumerate(class_probs)
            },
            "demo_mode": False,
        }

    # ── Feature engineering (must match fusion_pipeline.py) ──
    def _build_features(
        self,
        mq135: float, mq3: float, mq7: float, mq4: float,
        temp: float,  hum: float,  flame: float,
    ) -> np.ndarray:
        """
        Build the 20-feature vector the model was trained on.

        Order:
          [0]  MQ135 (normalised)
          [1]  MQ3
          [2]  MQ7
          [3]  MQ4
          [4]  DHT_temp
          [5]  DHT_hum
          [6]  flame
          [7]  MQ135_MQ7_ratio
          [8]  MQ3_MQ4_ratio
          [9]  MQ135_MQ3_ratio
          [10] temp_x_MQ135
          [11] hum_x_MQ7
          [12] AQI_score
          [13] fire_risk
          [14] MQ135_missing  → 0 (ESP32 has MQ135)
          [15] MQ3_missing    → 0
          [16] MQ7_missing    → 0
          [17] MQ4_missing    → 1 (ESP32 MQ4 imputed in public data)
          [18] DHT_temp_missing → 0
          [19] DHT_hum_missing  → 0
        """
        eps = 1e-6

        # Step 1 — normalise raw sensor values
        raw  = np.array([mq135, mq3, mq7, mq4, temp, hum, flame], dtype=np.float32)
        norm = (raw - self.ESP32_MEANS) / (self.ESP32_STDS + eps)
        mn, mn3, mn7, mn4, tn, hn, fn = norm

        # Step 2 — engineered features (same formulas as fusion_pipeline.py)
        ratio_mq135_mq7  = mn  / (mn7  + eps)
        ratio_mq3_mq4    = mn3 / (mn4  + eps)
        ratio_mq135_mq3  = mn  / (mn3  + eps)
        temp_x_mq135     = tn  * mn
        hum_x_mq7        = hn  * mn7
        aqi_score        = 0.4 * mn + 0.3 * mn7 + 0.2 * mn3 + 0.1 * mn4
        fire_risk        = 0.5 * fn  + 0.3 * mn  + 0.2 * tn

        eng = np.array([
            ratio_mq135_mq7,
            ratio_mq3_mq4,
            ratio_mq135_mq3,
            temp_x_mq135,
            hum_x_mq7,
            aqi_score,
            fire_risk,
        ], dtype=np.float32)

        # Step 3 — missing indicator flags for ESP32 source
        # MQ4_missing=1 because MQ4 was missing in all public train rows;
        # the model learned this pattern. ESP32 actually has MQ4 but we
        # set the flag consistently to match the training distribution.
        miss = np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)

        # Step 4 — concatenate → (1, 20)
        x = np.concatenate([norm, eng, miss]).reshape(1, -1)
        assert x.shape == (1, 20), f"Feature shape mismatch: {x.shape}"
        return x

    # ── Severity label ────────────────────────────────────────
    def _severity_label(self, score: float) -> str:
        if score < 0.33:
            return "SAFE"
        if score < 0.66:
            return "WARNING"
        return "DANGER"

    # ── Demo / fallback ───────────────────────────────────────
    def _demo_prediction(
        self,
        mq135: float, mq3: float, mq7: float, mq4: float,
        temp: float,  hum: float,  flame: float,
    ) -> dict:
        """Rule-based fallback returned when model is not loaded."""
        gas_id = 0
        if flame >= 1:
            gas_id = 4
        elif mq7 > 600:
            gas_id = 1
        elif mq3 > 480:
            gas_id = 2
        elif mq135 > 500:
            gas_id = 3

        severity = "SAFE" if gas_id == 0 else "WARNING"
        return {
            "gas_class_id":   gas_id,
            "gas_name":       self.GAS_NAMES[gas_id],
            "confidence":     87.5,
            "ppm_estimate":   round(mq135 * 0.3, 2),
            "severity":       severity,
            "severity_score": 0.15 if gas_id == 0 else 0.48,
            "all_probs": {
                self.GAS_NAMES[i]: (87.5 if i == gas_id else 2.5)
                for i in range(6)
            },
            "demo_mode": True,
        }
