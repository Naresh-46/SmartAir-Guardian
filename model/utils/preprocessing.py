# ============================================================
#  SmartAir-Guardian — Preprocessing Utilities
#  model/utils/preprocessing.py
#  Used by train_model.py, evaluate.py, and baselines/
# ============================================================

import numpy as np
import pandas as pd
import yaml
import os


def load_config(config_path="model/configs/model_config.yaml"):
    """Load central config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_feature_columns(cfg):
    """Return ordered list of all feature column names."""
    feats = cfg["features"]
    return (
        feats["sensor_cols"]
        + feats["engineered_cols"]
        + feats["missingness_cols"]
    )


def load_data(cfg):
    """
    Load train and test CSVs.
    Returns X_train, y_train, X_test, y_test as numpy arrays.
    """
    feat_cols = get_feature_columns(cfg)

    train_df = pd.read_csv(cfg["paths"]["train_csv"])
    test_df  = pd.read_csv(cfg["paths"]["test_csv"])

    # Verify all feature columns exist
    missing_train = [c for c in feat_cols if c not in train_df.columns]
    missing_test  = [c for c in feat_cols if c not in test_df.columns]
    if missing_train:
        raise ValueError(f"Missing columns in train CSV: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing columns in test CSV: {missing_test}")

    X_train = train_df[feat_cols].values.astype(np.float32)
    y_train = train_df["gas_class"].values.astype(np.int32)

    X_test  = test_df[feat_cols].values.astype(np.float32)
    y_test  = test_df["gas_class"].values.astype(np.int32)

    print(f"[DATA] Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"[DATA] Train classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"[DATA] Test  classes: {dict(zip(*np.unique(y_test,  return_counts=True)))}")

    return X_train, y_train, X_test, y_test


def make_severity_labels(y_gas):
    """
    Derive binary severity label from gas class.
    0 (clean air) → safe (0)
    Any other class → hazard (1)
    Used as target for Head 3.
    """
    return (y_gas != 0).astype(np.float32)


def make_ppm_labels(y_gas):
    """
    Derive approximate PPM regression target from gas class.
    Since we don't have real PPM for all rows, we use
    class-based median estimates for training.
    Replace with real PPM column when ESP32 data is available.
    """
    ppm_map = {
        0: 0.0,    # clean — near zero
        1: 150.0,  # smoke/CO
        2: 80.0,   # alcohol/VOC
        3: 100.0,  # NH3
        4: 200.0,  # fire — high CO
        5: 180.0,  # mixed/LPG
    }
    return np.array([ppm_map.get(int(c), 0.0) for c in y_gas], dtype=np.float32)


def prepare_targets(y_gas, cfg):
    """
    Build all three target arrays from gas class labels.
    Returns dict matching Keras model output names.
    """
    n_classes = cfg["classes"]["num_classes"]

    # Head 1: one-hot encoded gas class
    from tensorflow.keras.utils import to_categorical
    y_class = to_categorical(y_gas, num_classes=n_classes)

    # Head 2: PPM regression
    y_ppm = make_ppm_labels(y_gas)

    # Head 3: binary severity
    y_severity = make_severity_labels(y_gas)

    return {
        cfg["model"]["head_classification"]["output_name"]: y_class,
        cfg["model"]["head_regression"]["output_name"]:     y_ppm,
        cfg["model"]["head_severity"]["output_name"]:       y_severity,
    }


def class_weights(y_train):
    """
    Compute balanced class weights for imbalanced training data.
    Pass result to model.fit(class_weight=...).
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes.tolist(), weights.tolist()))


def print_data_summary(X_train, y_train, X_test, y_test, cfg):
    """Print a clean data summary before training."""
    class_names = cfg["classes"]["names"]
    print("\n" + "="*55)
    print("  SMARTAIR DATA SUMMARY")
    print("="*55)
    print(f"  Input features : {X_train.shape[1]}")
    print(f"  Train samples  : {len(X_train)}")
    print(f"  Test  samples  : {len(X_test)}")
    print(f"  Gas classes    : {cfg['classes']['num_classes']}")
    print("-"*55)
    print("  Train class distribution:")
    for cls, cnt in zip(*np.unique(y_train, return_counts=True)):
        pct = cnt / len(y_train) * 100
        bar = "█" * int(pct / 3)
        print(f"    {cls} {class_names[cls]:<18} {cnt:>5}  {bar} {pct:.1f}%")
    print("-"*55)
    print("  Test class distribution:")
    for cls, cnt in zip(*np.unique(y_test, return_counts=True)):
        pct = cnt / len(y_test) * 100
        bar = "█" * int(pct / 3)
        print(f"    {cls} {class_names[cls]:<18} {cnt:>5}  {bar} {pct:.1f}%")
    print("="*55 + "\n")
