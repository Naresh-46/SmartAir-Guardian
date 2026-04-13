#!/usr/bin/env python3
"""
SmartAir Guardian Model Training Monitor.

This script monitors the training process of the multi-task DNN model,
providing real-time feedback on training progress, model creation,
and training completion metrics.

Usage
-----
    python run_training.py

Output
------
    Displays training progress and saves results to model/outputs/.
"""

import subprocess
import time
import os
import sys
from pathlib import Path
from typing import Optional

from config.logger import get_logger


logger = get_logger(__name__)


def run_training_monitor() -> int:
    """
    Start and monitor the model training process.
    
    Returns
    -------
    int
        Exit code from the training process.
    
    Raises
    ------
    FileNotFoundError
        If the training module cannot be found.
    """
    logger.info("="*70)
    logger.info("SmartAir Guardian - Model Training Started")
    logger.info("="*70)
    logger.info("Starting training process (estimated 10-15 minutes)...")
    
    # Start training subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "model.train_model"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    start_time = time.time()
    model_created = False
    training_started = False
    
    logger.info("Monitoring training output...")
    logger.info("-" * 70)
    
    # Monitor training output in real-time
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            
            # Check for training start indicator
            if "Epoch" in line and not training_started:
                training_started = True
                logger.info("\n✅ TRAINING STARTED!\n")
            
            # Check if model file was created
            if not model_created and _check_model_created():
                model_created = True
                model_size_mb = _get_model_size_mb()
                logger.info(f"\n✅ MODEL FILE CREATED: {model_size_mb:.2f} MB\n")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        process.terminate()
        return 1
    
    # Wait for training completion
    exit_code = process.wait()
    elapsed_seconds = time.time() - start_time
    
    logger.info("-" * 70)
    
    if exit_code == 0:
        logger.info(f"✅ Training completed successfully in {elapsed_seconds:.1f} seconds")
        if model_created:
            model_size = _get_model_size_mb()
            logger.info(f"   Final model size: {model_size:.2f} MB")
    else:
        logger.error(f"❌ Training failed with exit code {exit_code}")
    
    logger.info("="*70)
    
    return exit_code


def _check_model_created() -> bool:
    """
    Check if the trained model file exists.
    
    Returns
    -------
    bool
        True if model file exists, False otherwise.
    """
    model_path = Path("model/outputs/smartair_model.keras")
    return model_path.exists()


def _get_model_size_mb() -> float:
    """
    Get the size of the trained model in megabytes.
    
    Returns
    -------
    float
        Model size in MB, or 0 if file doesn't exist.
    """
    model_path = Path("model/outputs/smartair_model.keras")
    if model_path.exists():
        return model_path.stat().st_size / (1024 * 1024)
    return 0.0


if __name__ == "__main__":
    try:
        sys.exit(run_training_monitor())
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


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
