#!/usr/bin/env python3
"""
SmartAir Guardian — Dataset Validator & Summary
================================================
Run this after collection to check your CSV is
clean and ready for model training.

Usage:
    python validate_dataset.py
    python validate_dataset.py --file my_data.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
from pathlib import Path

LABEL_NAMES = {0:"Normal", 1:"LPG", 2:"Smoke", 3:"CO", 4:"Methane"}
TARGET      = 500

def bar(count, target=TARGET, width=25):
    pct  = min(100, int(count/target*100))
    done = int(pct/100*width)
    return f"{'█'*done}{'░'*(width-done)} {pct:3d}%  ({count}/{target})"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="gas_dataset.csv")
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"File not found: {args.file}")
        return

    print(f"\n{'='*55}")
    print(f"  SmartAir Dataset Validator")
    print(f"  File: {args.file}  ({os.path.getsize(args.file)//1024} KB)")
    print(f"{'='*55}")

    df = pd.read_csv(args.file)
    total = len(df)
    print(f"\n  Total rows loaded : {total}")

    # ── Missing values ────────────────────────────────────────
    print(f"\n── Missing / NaN values ──────────────────────────────")
    missing = df.isnull().sum()
    has_missing = missing[missing > 0]
    if has_missing.empty:
        print("  No missing values found.")
    else:
        print(has_missing.to_string())
        df.dropna(inplace=True)
        print(f"  Dropped {total - len(df)} rows with NaN. Remaining: {len(df)}")

    # ── Range check ───────────────────────────────────────────
    print(f"\n── Sensor value ranges ───────────────────────────────")
    mq_cols = ["mq135","mq2","mq7","mq4","mq3"]
    for col in mq_cols:
        mn,mx = int(df[col].min()), int(df[col].max())
        bad = ((df[col] < 0) | (df[col] > 1023)).sum()
        flag = "  WARN — out-of-range rows: "+str(bad) if bad else ""
        print(f"  {col:8s}: {mn:4d} – {mx:4d}{flag}")

    for col, lo, hi in [("temperature",-10,80),("humidity",0,100)]:
        mn = round(df[col].min(),1)
        mx = round(df[col].max(),1)
        bad = ((df[col] < lo) | (df[col] > hi)).sum()
        flag = f"  WARN — {bad} out-of-range" if bad else ""
        print(f"  {col:12s}: {mn} – {mx}{flag}")

    # ── Per-class counts ──────────────────────────────────────
    print(f"\n── Per-class sample counts ───────────────────────────")
    counts = df["label"].value_counts().sort_index()
    all_ok = True
    for lbl in range(5):
        count = int(counts.get(lbl, 0))
        if count < TARGET:
            all_ok = False
        print(f"  [{lbl}] {LABEL_NAMES[lbl]:8s}  {bar(count)}")

    print(f"\n  Total valid rows: {len(df)}")
    if all_ok:
        print("  All classes have >= 500 samples.")
    else:
        short = [LABEL_NAMES[i] for i in range(5) if int(counts.get(i,0)) < TARGET]
        print(f"  WARNING: Need more data for: {', '.join(short)}")

    # ── Class balance ─────────────────────────────────────────
    print(f"\n── Class balance check ───────────────────────────────")
    cnt_vals = [int(counts.get(i,1)) for i in range(5)]
    ratio = max(cnt_vals) / min(cnt_vals)
    print(f"  Max/min ratio: {ratio:.1f}x", end="")
    if ratio > 3:
        print("  (WARN: imbalanced — consider collecting more for smaller classes)")
    else:
        print("  (Good balance)")

    # ── Feature stats ─────────────────────────────────────────
    print(f"\n── Feature statistics ────────────────────────────────")
    feature_cols = mq_cols + ["temperature","humidity"]
    stats = df[feature_cols].describe().round(1)
    print(stats.to_string())

    # ── Save cleaned file ─────────────────────────────────────
    cleaned_file = args.file.replace(".csv","_cleaned.csv")
    df.to_csv(cleaned_file, index=False)
    print(f"\n── Cleaned file saved ────────────────────────────────")
    print(f"  {cleaned_file}  ({len(df)} rows)")

    # ── Training readiness ────────────────────────────────────
    print(f"\n{'='*55}")
    if all_ok and ratio <= 5:
        print("  READY FOR TRAINING")
        print(f"  Use: {cleaned_file}")
    else:
        print("  NOT READY — collect more data first")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    main()
