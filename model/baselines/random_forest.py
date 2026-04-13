# ============================================================
#  SmartAir-Guardian — Baseline: Random Forest
#  model/baselines/random_forest.py
#
#  Run from project root:
#    python model/baselines/random_forest.py
# ============================================================

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.utils.preprocessing import load_config, load_data, print_data_summary
from model.utils.metrics import (
    plot_confusion_matrix, print_classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib


def train_random_forest():
    cfg = load_config("model/configs/model_config.yaml")
    rf_cfg = cfg["baselines"]["random_forest"]

    X_train, y_train, X_test, y_test = load_data(cfg)
    print_data_summary(X_train, y_train, X_test, y_test, cfg)

    print("\n[RF] Training Random Forest...")
    print(f"     n_estimators : {rf_cfg['n_estimators']}")
    print(f"     max_depth    : {rf_cfg['max_depth']}")

    rf = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        random_state=rf_cfg["random_state"],
        n_jobs=rf_cfg["n_jobs"],
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n[RF] Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"[RF] Test F1 Score  : {f1:.4f}")
    print(f"[RF] Test Precision : {prec:.4f}")
    print(f"[RF] Test Recall    : {rec:.4f}")

    # Feature importance
    from model.utils.preprocessing import get_feature_columns
    feat_cols = get_feature_columns(cfg)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\n[RF] Top 10 Feature Importances:")
    for i in top_idx:
        print(f"     {feat_cols[i]:<25} {importances[i]:.4f}")

    # Save plots and report
    class_names = [cfg["classes"]["names"][i]
                   for i in range(cfg["classes"]["num_classes"])]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    plot_confusion_matrix(
        y_test, y_pred, class_names,
        save_path=os.path.join(results_dir, "rf_confusion_matrix.png")
    )
    metrics = print_classification_report(
        y_test, y_pred, class_names,
        save_path=os.path.join(results_dir, "rf_evaluation_report.txt")
    )

    # Save model
    model_path = os.path.join(results_dir, "random_forest_model.joblib")
    joblib.dump(rf, model_path)
    print(f"\n[RF] Model saved → {model_path}")

    # Save results JSON for comparison table
    results = {"Random Forest": metrics}
    with open(os.path.join(results_dir, "rf_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return metrics


if __name__ == "__main__":
    train_random_forest()
