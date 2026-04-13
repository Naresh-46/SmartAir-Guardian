"""
Data Preprocessing and Feature Engineering Utilities.

This module provides functions for loading, validating, and preprocessing
sensor data for the SmartAir Guardian ML model. It handles:
    - Loading training and test datasets from CSV files
    - Feature extraction and validation
    - Target variable preparation (classification, regression, severity)
    - Class weighting for imbalanced datasets
    - Data summary reporting and visualization

Used by train_model.py, evaluate.py, and baseline models.
"""

from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def load_config(config_path: str = "model/configs/model_config.yaml") -> Dict[str, Any]:
    """
    Load and parse the YAML configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the model configuration file (YAML format).
    
    Returns
    -------
    dict
        Parsed configuration dictionary containing all model and data settings.
    
    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    yaml.YAMLError
        If the YAML file is malformed.
    
    Examples
    --------
    >>> config = load_config("model/configs/model_config.yaml")
    >>> print(config["training"]["batch_size"])
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, "r") as file_handle:
        config = yaml.safe_load(file_handle)
    
    return config


def get_feature_columns(config: Dict[str, Any]) -> List[str]:
    """
    Extract the ordered list of feature column names from configuration.
    
    Combines sensor columns, engineered features, and missingness indicators
    into a single ordered feature list.
    
    Parameters
    ----------
    config : dict
        Model configuration dictionary.
    
    Returns
    -------
    list
        Ordered list of feature column names.
    
    Examples
    --------
    >>> config = load_config()
    >>> features = get_feature_columns(config)
    >>> print(f"Using {len(features)} features: {features[:3]} ...")
    """
    features_config = config["features"]
    
    feature_list = (
        features_config["sensor_cols"]
        + features_config["engineered_cols"]
        + features_config["missingness_cols"]
    )
    
    return feature_list


def load_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and validate training and test datasets.
    
    Reads CSV files, extracts features and labels, performs validation,
    and returns standardized numpy arrays for model training.
    
    Parameters
    ----------
    config : dict
        Model configuration dictionary with paths and feature definitions.
    
    Returns
    -------
    tuple
        A tuple of (X_train, y_train, X_test, y_test) as numpy arrays.
        - X_train, X_test: shape (n_samples, n_features), dtype float32
        - y_train, y_test: shape (n_samples,), dtype int32
    
    Raises
    ------
    FileNotFoundError
        If training or test CSV files do not exist.
    ValueError
        If required feature columns are missing from the data.
    
    Examples
    --------
    >>> config = load_config()
    >>> X_train, y_train, X_test, y_test = load_data(config)
    >>> print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    """
    feature_columns = get_feature_columns(config)
    
    # Load datasets
    train_path = config["paths"]["train_csv"]
    test_path = config["paths"]["test_csv"]
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Validate feature columns exist in both datasets
    missing_train = [col for col in feature_columns if col not in train_df.columns]
    missing_test = [col for col in feature_columns if col not in test_df.columns]
    
    if missing_train:
        raise ValueError(f"Missing columns in training CSV: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing columns in test CSV: {missing_test}")
    
    # Extract features and labels
    X_train = train_df[feature_columns].values.astype(np.float32)
    y_train = train_df["gas_class"].values.astype(np.int32)
    
    X_test = test_df[feature_columns].values.astype(np.float32)
    y_test = test_df["gas_class"].values.astype(np.int32)
    
    print(f"[DATA] Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"[DATA] Train classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"[DATA] Test  classes: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    return X_train, y_train, X_test, y_test


def make_severity_labels(gas_class_labels: np.ndarray) -> np.ndarray:
    """
    Derive binary severity labels from gas class labels.
    
    Converts multi-class gas classification to binary hazard severity:
    - Class 0 (clean air) → 0 (safe)
    - Any other class → 1 (hazard detected)
    
    Used as the target for the severity detection head in multi-task learning.
    
    Parameters
    ----------
    gas_class_labels : np.ndarray
        Array of gas class labels (0-5).
    
    Returns
    -------
    np.ndarray
        Binary severity labels (0=safe, 1=hazard), dtype float32.
    
    Examples
    --------
    >>> y_gas = np.array([0, 1, 2, 0, 3])
    >>> severity = make_severity_labels(y_gas)
    >>> print(severity)  # [0. 1. 1. 0. 1.]
    """
    return (gas_class_labels != 0).astype(np.float32)


def make_ppm_labels(gas_class_labels: np.ndarray) -> np.ndarray:
    """
    Derive gas concentration (PPM) labels from gas class labels.
    
    Maps gas classes to approximate PPM (parts per million) concentrations
    based on typical exposure levels. Since the current dataset doesn't
    include real PPM measurements, class-based median estimates are used.
    
    Future: Replace with real PPM column when ESP32 sensor provides it.
    
    Parameters
    ----------
    gas_class_labels : np.ndarray
        Array of gas class labels (0-5).
    
    Returns
    -------
    np.ndarray
        PPM concentration targets, dtype float32.
    
    Notes
    -----
    Mapping:
        0 (clean)    → 0 PPM
        1 (smoke/CO) → 150 PPM
        2 (alcohol)  → 80 PPM
        3 (NH3)      → 100 PPM
        4 (fire)     → 200 PPM
        5 (LPG mix)  → 180 PPM
    
    Examples
    --------
    >>> y_gas = np.array([0, 1, 4])
    >>> ppm = make_ppm_labels(y_gas)
    >>> print(ppm)  # [0. 150. 200.]
    """
    # Class to approximate PPM mapping
    class_to_ppm = {
        0: 0.0,    # clean air
        1: 150.0,  # smoke/CO
        2: 80.0,   # alcohol/VOC
        3: 100.0,  # ammonia (NH3)
        4: 200.0,  # fire (high CO)
        5: 180.0,  # LPG/mixed gases
    }
    
    ppm_array = np.array(
        [class_to_ppm.get(int(label), 0.0) for label in gas_class_labels],
        dtype=np.float32
    )
    
    return ppm_array


def prepare_targets(
    gas_class_labels: np.ndarray,
    config: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Prepare all three target arrays for multi-task learning model.
    
    Generates targets for the three output heads:
    1. Gas classification (one-hot encoded)
    2. Concentration regression (PPM estimates)
    3. Severity detection (binary)
    
    Parameters
    ----------
    gas_class_labels : np.ndarray
        Array of gas class labels (0-5).
    config : dict
        Model configuration dictionary.
    
    Returns
    -------
    dict
        Dictionary mapping output head names to target arrays:
        {
            "gas_class": one-hot array (n_samples, n_classes),
            "concentration_ppm": regression targets (n_samples,),
            "hazard_severity": binary targets (n_samples,)
        }
    
    Examples
    --------
    >>> y_gas = np.array([0, 1, 2, 0, 3])
    >>> config = load_config()
    >>> targets = prepare_targets(y_gas, config)
    >>> print(targets["gas_class"].shape)  # (5, 6)
    """
    num_classes = config["classes"]["num_classes"]
    
    # Import here to avoid TensorFlow import at module level
    from tensorflow.keras.utils import to_categorical
    
    # Head 1: One-hot encoded gas classification
    y_class_onehot = to_categorical(gas_class_labels, num_classes=num_classes)
    
    # Head 2: PPM regression targets
    y_ppm = make_ppm_labels(gas_class_labels)
    
    # Head 3: Binary severity targets
    y_severity = make_severity_labels(gas_class_labels)
    
    # Map to output names from config
    targets = {
        config["model"]["head_classification"]["output_name"]: y_class_onehot,
        config["model"]["head_regression"]["output_name"]: y_ppm,
        config["model"]["head_severity"]["output_name"]: y_severity,
    }
    
    return targets


def class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights for imbalanced training data.
    
    Calculates class weights to handle imbalanced datasets by giving
    higher weight to underrepresented classes during training.
    
    Parameters
    ----------
    y_train : np.ndarray
        Training labels array.
    
    Returns
    -------
    dict
        Dictionary mapping class indices to weight values.
    
    Examples
    --------
    >>> y_train = np.array([0, 0, 1, 1, 1, 2])
    >>> weights = class_weights(y_train)
    >>> print(weights)  # {0: 0.75, 1: 0.5, 2: 1.5}
    
    Notes
    -----
    Pass the result to model.fit():
        weights = class_weights(y_train)
        model.fit(X, y, class_weight=weights)
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    calculated_weights = compute_class_weight(
        "balanced",
        classes=classes,
        y=y_train
    )
    
    weight_dict = dict(zip(classes.tolist(), calculated_weights.tolist()))
    return weight_dict


def print_data_summary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any]
) -> None:
    """
    Print a formatted data summary before model training.
    
    Displays dataset statistics including shape, class distribution,
    and relative proportions with visual progress bars.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features array.
    y_train : np.ndarray
        Training labels array.
    X_test : np.ndarray
        Test features array.
    y_test : np.ndarray
        Test labels array.
    config : dict
        Model configuration dictionary with class names.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> X_train, y_train, X_test, y_test = load_data(config)
    >>> print_data_summary(X_train, y_train, X_test, y_test, config)
    """
    class_names = config["classes"]["names"]
    num_classes = config["classes"]["num_classes"]
    
    print("\n" + "="*60)
    print("  SMARTAIR Guardian — DATA SUMMARY")
    print("="*60)
    print(f"  Input Features          : {X_train.shape[1]}")
    print(f"  Training Samples        : {len(X_train):,}")
    print(f"  Test Samples            : {len(X_test):,}")
    print(f"  Gas Classes             : {num_classes}")
    print("-"*60)
    print("  Training Class Distribution:")
    
    for class_idx, sample_count in zip(*np.unique(y_train, return_counts=True)):
        percentage = (sample_count / len(y_train)) * 100
        bar_length = int(percentage / 3)
        progress_bar = "█" * bar_length
        class_name = class_names.get(class_idx, f"Unknown-{class_idx}")
        print(f"    {class_idx} {class_name:<20} {sample_count:>6}  {progress_bar:20} {percentage:>5.1f}%")
    
    print("-"*60)
    print("  Test Class Distribution:")
    
    for class_idx, sample_count in zip(*np.unique(y_test, return_counts=True)):
        percentage = (sample_count / len(y_test)) * 100
        bar_length = int(percentage / 3)
        progress_bar = "█" * bar_length
        class_name = class_names.get(class_idx, f"Unknown-{class_idx}")
        print(f"    {class_idx} {class_name:<20} {sample_count:>6}  {progress_bar:20} {percentage:>5.1f}%")
    
    print("="*60 + "\n")

