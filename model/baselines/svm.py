# ============================================================
#  SmartAir-Guardian — Baseline: Support Vector Machine
#  model/baselines/svm.py
#
#  Run from project root:
#    python model/baselines/svm.py
# ============================================================

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.utils.preprocessing import load_config, load_data, print_data_summary
from model.utils.metrics import plot_confusion_matrix, print_classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib


def train_svm():
    cfg = load_config("model/configs/model_config.yaml")
    svm_cfg = cfg["baselines"]["svm"]

    X_train, y_train, X_test, y_test = load_data(cfg)
    print_data_summary(X_train, y_train, X_test, y_test, cfg)

    # SVM is slow on large datasets — sample if needed
    MAX_TRAIN = 5000
    if len(X_train) > MAX_TRAIN:
        print(f"\n[SVM] Subsampling train to {MAX_TRAIN} rows (SVM is O(n²))")
        idx = np.random.choice(len(X_train), MAX_TRAIN, replace=False)
        X_tr = X_train[idx]
        y_tr = y_train[idx]
    else:
        X_tr, y_tr = X_train, y_train

    print(f"\n[SVM] Training SVM (kernel={svm_cfg['kernel']}, C={svm_cfg['C']})...")
    print("      This may take a few minutes on large datasets...")

    svm = SVC(
        kernel=svm_cfg["kernel"],
        C=svm_cfg["C"],
        gamma=svm_cfg["gamma"],
        class_weight="balanced",
        random_state=svm_cfg["random_state"],
        probability=True,
    )
    svm.fit(X_tr, y_tr)

    y_pred = svm.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n[SVM] Test Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"[SVM] Test F1 Score  : {f1:.4f}")
    print(f"[SVM] Test Precision : {prec:.4f}")
    print(f"[SVM] Test Recall    : {rec:.4f}")

    class_names = [cfg["classes"]["names"][i]
                   for i in range(cfg["classes"]["num_classes"])]
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    plot_confusion_matrix(
        y_test, y_pred, class_names,
        save_path=os.path.join(results_dir, "svm_confusion_matrix.png")
    )
    metrics = print_classification_report(
        y_test, y_pred, class_names,
        save_path=os.path.join(results_dir, "svm_evaluation_report.txt")
    )

    model_path = os.path.join(results_dir, "svm_model.joblib")
    joblib.dump(svm, model_path)
    print(f"\n[SVM] Model saved → {model_path}")

    results = {"SVM (RBF kernel)": metrics}
    with open(os.path.join(results_dir, "svm_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return metrics


if __name__ == "__main__":
    train_svm()
