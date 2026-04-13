# ============================================================
#  SmartAir-Guardian — Main Model Training Script
#  model/train_model.py
#
#  Architecture: Multi-task DNN
#   Shared backbone → 3 output heads:
#     Head 1: Gas type classification  (Softmax, 6 classes)
#     Head 2: Concentration regression (Linear, PPM estimate)
#     Head 3: AQI severity             (Sigmoid, binary hazard)
#
#  Run from project root:
#    python model/train_model.py
# ============================================================

import os
import sys
import numpy as np
import yaml

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.utils.preprocessing import (
    load_config, load_data, prepare_targets,
    class_weights, print_data_summary
)
from model.utils.metrics import (
    plot_training_history, plot_confusion_matrix,
    print_classification_report, compare_models_table
)


# ── LOAD CONFIG ──────────────────────────────────────────────
cfg = load_config("model/configs/model_config.yaml")
np.random.seed(cfg["training"]["random_seed"])

# ── LOAD DATA ────────────────────────────────────────────────
X_train, y_train, X_test, y_test = load_data(cfg)
print_data_summary(X_train, y_train, X_test, y_test, cfg)

# ── BUILD TARGETS ────────────────────────────────────────────
# Import TF here so error is clear if not installed
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    )
    print(f"[TF] TensorFlow version: {tf.__version__}")
except ImportError:
    print("[ERROR] TensorFlow not installed.")
    print("        Run: pip install tensorflow")
    sys.exit(1)

tf.random.set_seed(cfg["training"]["random_seed"])

train_targets = prepare_targets(y_train, cfg)
test_targets  = prepare_targets(y_test,  cfg)

# ── BUILD MODEL ──────────────────────────────────────────────
def build_smartair_model(cfg):
    """
    Multi-task DNN with shared backbone and 3 output heads.
    """
    mc  = cfg["model"]
    bb  = mc["backbone"]
    h1  = mc["head_classification"]
    h2  = mc["head_regression"]
    h3  = mc["head_severity"]
    n_c = cfg["classes"]["num_classes"]
    inp_dim = cfg["features"]["input_dim"]

    # ── Input ──
    inputs = keras.Input(shape=(inp_dim,), name="sensor_input")

    # ── Shared backbone ──
    x = layers.Dense(bb["dense1_units"], activation=bb["activation"],
                      name="backbone_dense1")(inputs)
    x = layers.BatchNormalization(name="backbone_bn1")(x)
    x = layers.Dropout(bb["dropout_rate"], name="backbone_drop1")(x)
    x = layers.Dense(bb["dense2_units"], activation=bb["activation"],
                      name="backbone_dense2")(x)
    x = layers.BatchNormalization(name="backbone_bn2")(x)
    x = layers.Dropout(bb["dropout_rate"] * 0.5, name="backbone_drop2")(x)

    # ── Head 1: Gas classification ──
    cls = layers.Dense(h1["dense_units"], activation="relu", name="cls_dense")(x)
    cls_out = layers.Dense(n_c, activation=h1["activation"],
                            name=h1["output_name"])(cls)

    # ── Head 2: Concentration regression ──
    reg = layers.Dense(h2["dense_units"], activation="relu", name="reg_dense")(x)
    reg_out = layers.Dense(1, activation=h2["activation"],
                            name=h2["output_name"])(reg)

    # ── Head 3: AQI severity ──
    sev = layers.Dense(h3["dense_units"], activation="relu", name="sev_dense")(x)
    sev_out = layers.Dense(1, activation=h3["activation"],
                            name=h3["output_name"])(sev)

    model = keras.Model(inputs=inputs,
                        outputs=[cls_out, reg_out, sev_out],
                        name="SmartAir_MultiTask")
    return model


model = build_smartair_model(cfg)

# ── COMPILE ──────────────────────────────────────────────────
lw = cfg["training"]["loss_weights"]
cls_name = cfg["model"]["head_classification"]["output_name"]
reg_name = cfg["model"]["head_regression"]["output_name"]
sev_name = cfg["model"]["head_severity"]["output_name"]

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=cfg["training"]["learning_rate"]
    ),
    loss={
        cls_name: "categorical_crossentropy",
        reg_name: "mse",
        sev_name: "binary_crossentropy",
    },
    loss_weights={
        cls_name: lw["gas_class"],
        reg_name: lw["ppm"],
        sev_name: lw["severity"],
    },
    metrics={
        cls_name: ["accuracy"],
        reg_name: ["mae"],
        sev_name: ["accuracy"],
    }
)

model.summary()

# ── CALLBACKS ────────────────────────────────────────────────
os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=cfg["training"]["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=cfg["training"]["reduce_lr_factor"],
        patience=cfg["training"]["reduce_lr_patience"],
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=cfg["paths"]["model_out"],
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
]

# ── CLASS WEIGHTS ────────────────────────────────────────────
# Note: For multi-task models, class_weight is not supported in Keras
# Instead, we rely on loss_weights and balanced loss functions
# to handle class imbalance for the gas_class head
cw = class_weights(y_train)
print(f"\n[TRAIN] Class balance info (for reference): {cw}")
print("[TRAIN] Note: Using loss_weights instead of class_weight for multi-output model")

# ── TRAIN ────────────────────────────────────────────────────
print("\n[TRAIN] Starting training...\n")
history = model.fit(
    X_train,
    train_targets,
    epochs=cfg["training"]["epochs"],
    batch_size=cfg["training"]["batch_size"],
    validation_split=cfg["training"]["validation_split"],
    callbacks=callbacks,
    verbose=1,
)

# ── SAVE PLOTS ───────────────────────────────────────────────
plot_training_history(
    history,
    save_path=os.path.join(cfg["paths"]["results_dir"], "training_history.png")
)

# ── EVALUATE ─────────────────────────────────────────────────
print("\n[EVAL] Evaluating on test set...\n")
test_results = model.evaluate(X_test, test_targets, verbose=1)
print(f"\n[EVAL] Test loss: {test_results[0]:.4f}")

# Get classification predictions
preds = model.predict(X_test, verbose=0)
y_pred_class = np.argmax(preds[0], axis=1)

class_names = [cfg["classes"]["names"][i]
               for i in range(cfg["classes"]["num_classes"])]

plot_confusion_matrix(
    y_test, y_pred_class, class_names,
    save_path=os.path.join(cfg["paths"]["results_dir"], "confusion_matrix.png")
)

metrics = print_classification_report(
    y_test, y_pred_class, class_names,
    save_path=os.path.join(cfg["paths"]["results_dir"], "evaluation_report.txt")
)

# Save results for comparison table (used by evaluate.py)
import json
results_path = os.path.join(cfg["paths"]["results_dir"], "smartair_dnn_results.json")
with open(results_path, "w") as f:
    json.dump({"SmartAir (Multi-task DNN)": metrics}, f, indent=2)

print(f"\n[DONE] Model saved → {cfg['paths']['model_out']}")
print(f"[DONE] All outputs  → {cfg['paths']['results_dir']}")
