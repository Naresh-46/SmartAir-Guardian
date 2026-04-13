#!/usr/bin/env python3
"""Monitor model training progress"""

import subprocess
import time
import os
import sys

print("\n" + "="*70)
print("  SMARTAIR MODEL TRAINING - BACKGROUND MONITOR")
print("="*70 + "\n")

print("Starting training process...")
print("This may take 10-15 minutes depending on your CPU.\n")

# Start training
proc = subprocess.Popen(
    [sys.executable, "-m", "model.train_model"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

start_time = time.time()
model_created = False
training_started = False

# Monitor output in real-time
print("[TRAINING OUTPUT]")
print("-" * 70)

for line in proc.stdout:
    print(line, end='')
    
    # Check for training start
    if "Epoch" in line and not training_started:
        training_started = True
        print("\n✅ TRAINING STARTED!\n")
    
    # Check if model file was created
    if not model_created and os.path.exists("model/outputs/smartair_model.keras"):
        model_created = True
        size_mb = os.path.getsize("model/outputs/smartair_model.keras") / (1024*1024)
        print(f"\n✅ MODEL FILE CREATED: {size_mb:.2f} MB\n")

# Wait for completion
proc.wait()
elapsed = time.time() - start_time

print("-" * 70)
print(f"\n[COMPLETED] Training finished in {elapsed/60:.1f} minutes\n")

# Check results
if os.path.exists("model/outputs/smartair_model.keras"):
    size_mb = os.path.getsize("model/outputs/smartair_model.keras") / (1024*1024)
    print(f"✅ TRAINING SUCCESSFUL!")
    print(f"   Model saved: model/outputs/smartair_model.keras ({size_mb:.2f} MB)")
    print(f"   Training time: {elapsed/60:.1f} minutes")
    print("\nThe server will now use real ML predictions!")
else:
    print(f"❌ TRAINING FAILED - Model file not created")
    print(f"   Exit code: {proc.returncode}")

print()
