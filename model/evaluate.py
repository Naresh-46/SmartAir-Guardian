# ============================================================
#  SmartAir-Guardian — Full Evaluation Script
#  model/evaluate.py
#
#  Loads trained model, runs on test set, produces all plots,
#  builds comparison table with baselines.
#
#  Run AFTER training all models:
#    python model/evaluate.py
# ============================================================

import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.utils.preprocessing import load_config, load_data
from model.utils.metrics import (
    plot_confusion_matrix, print_classification_report,
    compare_models_table
)


def load_all_results(results_dir):
    """Load all saved *_results.json files from outputs/."""
    all_results = {}
    for path in glob.glob(os.path.join(results_dir, "*_results.json")):
        with open(path) as f:
            data = json.load(f)
            all_results.update(data)
    return all_results


def evaluate_main_model(cfg, X_test, y_test):
    """Load and evaluate the main SmartAir multi-task model."""
    model_path = cfg["paths"]["model_out"]

    if not os.path.exists(model_path):
        print(f"[WARN] Main model not found at {model_path}")
        print("       Run train_model.py first.")
        return None

    try:
        import tensorflow as tf
        from tensorflow.keras.utils import to_categorical
        model = tf.keras.models.load_model(model_path)
        print(f"[EVAL] Loaded model from {model_path}")
    except ImportError:
        print("[ERROR] TensorFlow not installed.")
        return None

    from model.utils.preprocessing import prepare_targets
    test_targets = prepare_targets(y_test, cfg)

    # Evaluate all heads
    results_raw = model.evaluate(X_test, test_targets, verbose=1)
    print(f"\n[EVAL] Combined test loss: {results_raw[0]:.4f}")

    # Classification predictions
    preds = model.predict(X_test, verbose=0)
    y_pred = np.argmax(preds[0], axis=1)

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    class_names = [cfg["classes"]["names"][i]
                   for i in range(cfg["classes"]["num_classes"])]
    results_dir = cfg["paths"]["results_dir"]

    plot_confusion_matrix(
        y_test, y_pred, class_names,
        save_path=os.path.join(results_dir, "confusion_matrix.png")
    )
    print_classification_report(
        y_test, y_pred, class_names,
        save_path=os.path.join(results_dir, "evaluation_report.txt")
    )

    # Regression head — MAE on test set
    y_ppm_pred = preds[1].flatten()
    from model.utils.preprocessing import make_ppm_labels
    y_ppm_true = make_ppm_labels(y_test)
    mae = np.mean(np.abs(y_ppm_true - y_ppm_pred))
    print(f"\n[EVAL] Head 2 (PPM regression) MAE: {mae:.2f} ppm")

    # Severity head — binary accuracy
    y_sev_pred = (preds[2].flatten() > 0.5).astype(int)
    from model.utils.preprocessing import make_severity_labels
    y_sev_true = make_severity_labels(y_test).astype(int)
    sev_acc = np.mean(y_sev_pred == y_sev_true)
    print(f"[EVAL] Head 3 (Severity) Accuracy: {sev_acc:.4f}")

    metrics["ppm_mae"] = float(mae)
    metrics["severity_accuracy"] = float(sev_acc)

    return metrics


def main():
    cfg = load_config("model/configs/model_config.yaml")
    _, _, X_test, y_test = load_data(cfg)
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "="*55)
    print("  SMARTAIR — FULL EVALUATION")
    print("="*55)

    # Evaluate main model
    main_metrics = evaluate_main_model(cfg, X_test, y_test)

    # Load all baseline results
    all_results = load_all_results(results_dir)

    # Add main model to comparison
    if main_metrics:
        all_results["SmartAir (Multi-task DNN)"] = main_metrics

    if not all_results:
        print("\n[WARN] No results found. Run train_model.py and all baseline scripts first.")
        return

    # Build comparison table
    # Sort by accuracy descending
    sorted_results = dict(
        sorted(all_results.items(),
               key=lambda x: x[1].get("accuracy", 0),
               reverse=True)
    )

    compare_models_table(
        sorted_results,
        save_path=os.path.join(results_dir, "model_comparison.txt")
    )

    # Print paper-ready summary
    print("\n" + "="*55)
    print("  PAPER-READY RESULT SUMMARY")
    print("="*55)
    if "SmartAir (Multi-task DNN)" in sorted_results:
        mt = sorted_results["SmartAir (Multi-task DNN)"]
        print(f"\n  SmartAir Multi-Task DNN:")
        print(f"    Accuracy  : {mt.get('accuracy',0):.4f}")
        print(f"    F1 Score  : {mt.get('f1',0):.4f}")
        if "ppm_mae" in mt:
            print(f"    PPM MAE   : {mt.get('ppm_mae',0):.2f} ppm")
        if "severity_accuracy" in mt:
            print(f"    Severity Acc: {mt.get('severity_accuracy',0):.4f}")

    print("\n  Baseline comparison:")
    for name, m in sorted_results.items():
        if name != "SmartAir (Multi-task DNN)":
            print(f"    {name:<28} Acc={m.get('accuracy',0):.4f}  F1={m.get('f1',0):.4f}")

    print("\n[DONE] All evaluation outputs saved to:", results_dir)


if __name__ == "__main__":
    main()
