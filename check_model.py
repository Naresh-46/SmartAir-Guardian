#!/usr/bin/env python3
"""Check SmartAir Model Training Status"""

import os
import glob

print("\n" + "="*70)
print("  SMARTAIR MODEL TRAINING STATUS CHECK")
print("="*70 + "\n")

# Check 1: Main model file
print("[CHECK 1] Main Model File")
main_model = "model/outputs/smartair_model.keras"
if os.path.exists(main_model):
    size_mb = os.path.getsize(main_model) / (1024*1024)
    print(f"  ✅ smartair_model.keras EXISTS")
    print(f"     Size: {size_mb:.2f} MB")
    print(f"     Status: TRAINED ✓")
else:
    print(f"  ❌ smartair_model.keras NOT FOUND")
    print(f"     Status: NOT TRAINED")

print()

# Check 2: Output directory
print("[CHECK 2] Model Outputs Directory")
output_dir = "model/outputs/"
if os.path.isdir(output_dir):
    print(f"  ✅ model/outputs/ exists")
    files = os.listdir(output_dir)
    print(f"     Files count: {len(files)}")
    if files:
        print(f"     Contents:")
        for f in sorted(files):
            fpath = os.path.join(output_dir, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                if size > 1024*1024:
                    print(f"       • {f} ({size/(1024*1024):.2f} MB)")
                elif size > 1024:
                    print(f"       • {f} ({size/1024:.2f} KB)")
                else:
                    print(f"       • {f} ({size} bytes)")
else:
    print(f"  ⚠️  model/outputs/ directory not found")

print()

# Check 3: Baseline models
print("[CHECK 3] Baseline Models")
baselines = [
    ("Random Forest", "model/outputs/random_forest_model.joblib"),
    ("SVM", "model/outputs/svm_model.joblib"),
    ("Single-task DNN", "model/outputs/singletask_dnn.keras"),
]
for name, path in baselines:
    if os.path.exists(path):
        size = os.path.getsize(path) / (1024*1024)
        print(f"  ✅ {name}: {size:.2f} MB")
    else:
        print(f"  ❌ {name}: Not found")

print()

# Check 4: Training artifacts
print("[CHECK 4] Training Artifacts & Reports")
artifacts = glob.glob("model/outputs/*.txt") + glob.glob("model/outputs/*.png") + glob.glob("model/outputs/*.json")
if artifacts:
    for artifact in sorted(artifacts):
        print(f"  ✅ {os.path.basename(artifact)}")
else:
    print(f"  ⚠️  No training artifacts found")

print()
print("="*70)

# Final verdict
model_trained = os.path.exists("model/outputs/smartair_model.keras")
if model_trained:
    print("  ✅ VERDICT: MODEL IS TRAINED")
    print("  The server is using real ML predictions")
    print("="*70)
else:
    print("  ❌ VERDICT: MODEL IS NOT TRAINED YET")
    print("  The server is using demo/fallback predictions")
    print()
    print("  To train the model, run:")
    print("    python -m model.train_model")
    print()
    print("  This will:")
    print("    1. Load training/test data")
    print("    2. Build multi-task DNN architecture")
    print("    3. Train for up to 100 epochs")
    print("    4. Save model to: model/outputs/smartair_model.keras")
    print("    5. Generate evaluation plots")
    print("="*70)

print()
