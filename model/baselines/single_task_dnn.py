# ============================================================
#  SmartAir-Guardian — Baseline: Single-Task DNN
#  model/baselines/single_task_dnn.py
#
#  Proves multi-task architecture beats 3 separate single-task
#  networks — mandatory comparison for the paper.
#
#  Run from project root:
#    python model/baselines/single_task_dnn.py
# ============================================================

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.utils.preprocessing import (
    load_config, load_data, print_data_summary,
    make_severity_labels, make_ppm_labels, class_weights
)
from model.utils.metrics import (
    plot_confusion_matrix, print_classification_report
)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("[ERROR] TensorFlow not installed. Run: pip install tensorflow")
    sys.exit(1)


def build_single_task_dnn(input_dim, n_classes, cfg):
    """Single-task DNN — classification only, no shared backbone."""
    st_cfg = cfg["baselines"]["single_task_dnn"]
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = inputs
    for i, units in enumerate(st_cfg["hidden_units"]):
        x = layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Dropout(st_cfg["dropout_rate"], name=f"drop_{i}")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)
    model = keras.Model(inputs, outputs, name="SingleTask_DNN")
    return model


def train_single_task_dnn():
    cfg = load_config("model/configs/model_config.yaml")
    st_cfg = cfg["baselines"]["single_task_dnn"]

    X_train, y_train, X_test, y_test = load_data(cfg)
    print_data_summary(X_train, y_train, X_test, y_test, cfg)

    n_classes = cfg["classes"]["num_classes"]
    inp_dim   = cfg["features"]["input_dim"]

    # One-hot targets
    y_train_cat = to_categorical(y_train, num_classes=n_classes)
    y_test_cat  = to_categorical(y_test,  num_classes=n_classes)

    tf.random.set_seed(cfg["training"]["random_seed"])

    print(f"\n[ST-DNN] Building single-task DNN...")
    model = build_single_task_dnn(inp_dim, n_classes, cfg)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=cfg["training"]["learning_rate"]
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    cw = class_weights(y_train)
    callbacks = [
        EarlyStopping(monitor="val_loss",
                      patience=cfg["training"]["early_stopping_patience"],
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss",
                          factor=cfg["training"]["reduce_lr_factor"],
                          patience=cfg["training"]["reduce_lr_patience"],
                          min_lr=1e-6, verbose=1),
    ]

    print("\n[ST-DNN] Training...")
    model.fit(
        X_train, y_train_cat,
        epochs=st_cfg["epochs"],
        batch_size=st_cfg["batch_size"],
        validation_split=0.15,
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n[ST-DNN] Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"[ST-DNN] Test F1 Score  : {f1:.4f}")
    print(f"[ST-DNN] Test Precision : {prec:.4f}")
    print(f"[ST-DNN] Test Recall    : {rec:.4f}")

    class_names = [cfg["classes"]["names"][i] for i in range(n_classes)]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    plot_confusion_matrix(
        y_test, y_pred, class_names,
        save_path=os.path.join(results_dir, "singletask_confusion_matrix.png")
    )
    metrics = print_classification_report(
        y_test, y_pred, class_names,
        save_path=os.path.join(results_dir, "singletask_evaluation_report.txt")
    )

    model.save(os.path.join(results_dir, "singletask_dnn.keras"))
    print(f"\n[ST-DNN] Model saved → {results_dir}")

    results = {"Single-Task DNN": metrics}
    with open(os.path.join(results_dir, "singletask_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return metrics


if __name__ == "__main__":
    train_single_task_dnn()
