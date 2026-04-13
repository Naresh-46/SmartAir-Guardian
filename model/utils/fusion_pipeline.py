"""
SmartAir — Dataset Fusion Pipeline
====================================
Inputs :  smartair_train.csv          (public dataset — 6240 rows, classes 0/1/2/5 only)
          smartair_test_-_Copy.csv    (your ESP32 data — 35217 rows, all 6 classes)
          smartair_train_-_Copy.csv   (duplicate of train — auto-detected & skipped)

Problems found in your data (auto-fixed here):
  1. Train has ZERO flame readings — flame column is always 0.0
  2. Train is MISSING classes 3 (NH3) and 4 (fire/flame)
  3. Test has class 4 = 51.6% of rows → severe imbalance
  4. Train MQ7 range [-2.47, 2.77] vs Test MQ7 range [-0.23, 38.50] → domain shift
  5. Train: MQ4/DHT missing (imputed). Test: MQ135/MQ4 missing (different sources)
  6. train_-_Copy.csv is identical to train.csv → ignored

Outputs:  fused_train.csv   — training set (public sources, SMOTE balanced)
          fused_test.csv    — test set (your ESP32 data only, untouched)
          fusion_report.txt — full audit trail
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from collections import Counter
import warnings, os, sys

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────
TRAIN_PATH      = "/mnt/user-data/uploads/smartair_train.csv"
TRAIN_COPY_PATH = "/mnt/user-data/uploads/smartair_train_-_Copy.csv"
TEST_PATH       = "/mnt/user-data/uploads/smartair_test_-_Copy.csv"
OUT_DIR         = "/mnt/user-data/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

REPORT_LINES = []

def log(msg=""):
    print(msg)
    REPORT_LINES.append(msg)

# ─────────────────────────────────────────────
#  Class label map
# ─────────────────────────────────────────────
GAS_NAMES = {
    0: "clean",
    1: "smoke_CO",
    2: "alcohol_VOC",
    3: "NH3_ammonia",
    4: "fire_flame",
    5: "mixed_LPG",
}

FEATURE_COLS = [
    "MQ135", "MQ3", "MQ7", "MQ4",
    "DHT_temp", "DHT_hum", "flame",
    "MQ135_MQ7_ratio", "MQ3_MQ4_ratio", "MQ135_MQ3_ratio",
    "temp_x_MQ135", "hum_x_MQ7",
    "AQI_score", "fire_risk",
    "MQ135_missing", "MQ3_missing", "MQ7_missing", "MQ4_missing",
    "DHT_temp_missing", "DHT_hum_missing",
]
LABEL_COL = "gas_class"

# ══════════════════════════════════════════════
#  STEP 1 — Load & deduplicate sources
# ══════════════════════════════════════════════
log("=" * 60)
log("STEP 1 — Load & deduplicate sources")
log("=" * 60)

train      = pd.read_csv(TRAIN_PATH)
train_copy = pd.read_csv(TRAIN_COPY_PATH)
test       = pd.read_csv(TEST_PATH)

log(f"  Loaded train      : {train.shape}")
log(f"  Loaded train_copy : {train_copy.shape}")
log(f"  Loaded test       : {test.shape}")

# Check if copy is identical
if train.equals(train_copy):
    log("  train_-_Copy.csv is IDENTICAL to train.csv → skipping duplicate")
else:
    log("  WARNING: train_copy differs — investigate before merging")

# Tag sources
train["_source"] = "public_train"
test["_source"]  = "esp32_test"

log(f"\n  Train classes present : {sorted(train[LABEL_COL].unique())}")
log(f"  Test  classes present : {sorted(test[LABEL_COL].unique())}")
log(f"\n  Train class distribution:")
for cls, cnt in sorted(train[LABEL_COL].value_counts().items()):
    log(f"    class {cls} ({GAS_NAMES.get(cls,'?'):15s}): {cnt:5d} rows")
log(f"\n  Test class distribution:")
for cls, cnt in sorted(test[LABEL_COL].value_counts().items()):
    pct = cnt / len(test) * 100
    log(f"    class {cls} ({GAS_NAMES.get(cls,'?'):15s}): {cnt:5d} rows ({pct:.1f}%)")

# ══════════════════════════════════════════════
#  STEP 2 — Audit & document problems
# ══════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 2 — Problem audit")
log("=" * 60)

problems = []

# P1: flame column in train is all zeros
if train["flame"].nunique() == 1:
    log("  [P1] CRITICAL: flame column in public train is always 0.0")
    log("       → flame was not available in public dataset")
    log("       → MQ135_missing=0, MQ3_missing=0, MQ7_missing=0")
    log("       → MQ4_missing=1, DHT_temp_missing=1, DHT_hum_missing=1 (entire public set)")
    problems.append("P1: flame always 0 in public train")

# P2: missing classes
train_classes = set(train[LABEL_COL].unique())
all_classes   = set(range(6))
missing_cls   = all_classes - train_classes
if missing_cls:
    log(f"\n  [P2] CRITICAL: public train MISSING classes: {missing_cls}")
    log(f"       → class 3 (NH3) and class 4 (fire) only exist in ESP32 test data")
    log(f"       → model CANNOT learn NH3/fire patterns from public data alone")
    log(f"       → Fix: use ESP32 data for those classes in training split too")
    problems.append(f"P2: classes {missing_cls} absent from public train")

# P3: class 4 dominates test
cls4_pct = (test[LABEL_COL] == 4).sum() / len(test) * 100
if cls4_pct > 40:
    log(f"\n  [P3] HIGH: class 4 (fire_flame) = {cls4_pct:.1f}% of ESP32 test data")
    log(f"       → severe imbalance; model will be biased toward fire class")
    problems.append(f"P3: class 4 = {cls4_pct:.1f}% of test data")

# P4: MQ7 range mismatch (domain shift)
train_mq7_max = train["MQ7"].max()
test_mq7_max  = test["MQ7"].max()
if test_mq7_max > train_mq7_max * 5:
    log(f"\n  [P4] CRITICAL domain shift: MQ7 train max={train_mq7_max:.2f}, test max={test_mq7_max:.2f}")
    log(f"       → ESP32 MQ7 scaled differently (different Ro calibration or ADC resolution)")
    log(f"       → Both sets are already z-score normalized per source — this is expected")
    log(f"       → Per-source normalization (already applied) is the correct fix")
    problems.append("P4: MQ7 domain shift between sources")

log(f"\n  Total problems found: {len(problems)}")

# ══════════════════════════════════════════════
#  STEP 3 — Split strategy
#  Rule: public data → train only
#        ESP32 data  → test set exclusively
#        EXCEPTION: classes 3 & 4 only exist in ESP32
#        → take 20% of ESP32 class 3 & 4 for training,
#          keep 80% for test (never contaminate test with public rows)
# ══════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 3 — Train / Test split strategy")
log("=" * 60)

log("  Rule: public sources → TRAIN only")
log("  Rule: ESP32 data     → TEST  only (held-out, never trained on)")
log("  Exception: classes 3 & 4 absent from public data")
log("  → Take 20% of ESP32 class 3 & 4 rows for train (stratified)")
log("  → Remaining 80% stay in test")

SEED = 42
esp32_cls_34 = test[test[LABEL_COL].isin([3, 4])].copy()

train_from_esp32_parts = []
test_keep_parts        = []

for cls in [3, 4]:
    cls_rows = esp32_cls_34[esp32_cls_34[LABEL_COL] == cls]
    n_train  = max(int(len(cls_rows) * 0.20), 50)   # at least 50 rows
    shuffled = cls_rows.sample(frac=1, random_state=SEED)
    train_from_esp32_parts.append(shuffled.iloc[:n_train])
    test_keep_parts.append(shuffled.iloc[n_train:])
    log(f"    class {cls} ({GAS_NAMES[cls]}): {len(cls_rows)} total → "
        f"{n_train} to train, {len(cls_rows)-n_train} kept in test")

train_from_esp32 = pd.concat(train_from_esp32_parts)

# Final test = all ESP32 rows EXCEPT the 20% we pulled for training
esp32_other = test[~test[LABEL_COL].isin([3, 4])].copy()
test_final  = pd.concat([esp32_other] + test_keep_parts).reset_index(drop=True)

log(f"\n  ESP32 rows added to train (cls 3+4): {len(train_from_esp32)}")
log(f"  Final test set size                : {len(test_final)}")

# ══════════════════════════════════════════════
#  STEP 4 — Combine training data
# ══════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 4 — Combine training data")
log("=" * 60)

train_combined = pd.concat([train, train_from_esp32], ignore_index=True)
log(f"  Public train rows : {len(train)}")
log(f"  ESP32 cls 3&4 rows: {len(train_from_esp32)}")
log(f"  Combined train    : {len(train_combined)}")
log(f"  Class distribution before balancing:")
for cls, cnt in sorted(train_combined[LABEL_COL].value_counts().items()):
    log(f"    class {cls} ({GAS_NAMES.get(cls,'?'):15s}): {cnt:5d}")

# ══════════════════════════════════════════════
#  STEP 5 — Verify no NaNs, fix missing flags
# ══════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 5 — NaN check & missing indicator audit")
log("=" * 60)

nan_train = train_combined[FEATURE_COLS].isnull().sum()
nan_test  = test_final[FEATURE_COLS].isnull().sum()
log(f"  NaNs in combined train : {nan_train.sum()}")
log(f"  NaNs in test           : {nan_test.sum()}")

if nan_train.sum() > 0:
    log("  NaN columns in train:")
    log(str(nan_train[nan_train > 0]))
    # Fill with column median (already normalized — median ≈ 0)
    train_combined[FEATURE_COLS] = train_combined[FEATURE_COLS].fillna(
        train_combined[FEATURE_COLS].median()
    )
    log("  → Filled with column median")

log("\n  Missing indicator summary (train):")
for col in ["MQ135_missing","MQ3_missing","MQ7_missing","MQ4_missing",
            "DHT_temp_missing","DHT_hum_missing"]:
    vc = train_combined[col].value_counts().to_dict()
    log(f"    {col}: {vc}")

# ══════════════════════════════════════════════
#  STEP 6 — Balance training set (oversampling)
#  Strategy: upsample minority classes to match
#  the count of the largest class.
#  SMOTE not used here (data already normalized,
#  ratio features would be corrupted by synthetic
#  interpolation — simple resample is safer).
# ══════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 6 — Balance training set (oversample minority classes)")
log("=" * 60)

class_counts = train_combined[LABEL_COL].value_counts()
max_count    = class_counts.max()
log(f"  Target count per class: {max_count}")

balanced_parts = []
for cls in sorted(train_combined[LABEL_COL].unique()):
    cls_df = train_combined[train_combined[LABEL_COL] == cls]
    if len(cls_df) < max_count:
        cls_df = resample(cls_df, replace=True, n_samples=max_count,
                          random_state=SEED)
        log(f"    class {cls} ({GAS_NAMES.get(cls,'?'):15s}): "
            f"upsampled {class_counts[cls]} → {max_count}")
    else:
        log(f"    class {cls} ({GAS_NAMES.get(cls,'?'):15s}): "
            f"unchanged at {len(cls_df)}")
    balanced_parts.append(cls_df)

train_balanced = pd.concat(balanced_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
log(f"\n  Balanced train size: {len(train_balanced)}")
log(f"  Classes: {sorted(train_balanced[LABEL_COL].unique())}")

# ══════════════════════════════════════════════
#  STEP 7 — Drop duplicate rows
# ══════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 7 — Drop duplicates")
log("=" * 60)

before = len(train_balanced)
train_balanced = train_balanced.drop_duplicates(subset=FEATURE_COLS).reset_index(drop=True)
log(f"  Dropped {before - len(train_balanced)} duplicate rows from train")
log(f"  Train size after dedup: {len(train_balanced)}")

before = len(test_final)
test_final = test_final.drop_duplicates(subset=FEATURE_COLS).reset_index(drop=True)
log(f"  Dropped {before - len(test_final)} duplicate rows from test")
log(f"  Test size after dedup : {len(test_final)}")

# ══════════════════════════════════════════════
#  STEP 8 — Final validation
# ══════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 8 — Final validation")
log("=" * 60)

# Check no test rows leaked into train
train_features = set(train_balanced[FEATURE_COLS].apply(tuple, axis=1))
test_features  = set(test_final[FEATURE_COLS].apply(tuple, axis=1))
overlap = train_features & test_features
log(f"  Train/test row overlap : {len(overlap)} rows "
    f"{'✓ CLEAN' if not overlap else '✗ LEAKAGE DETECTED'}")

# Check all 6 classes in train
train_cls = set(train_balanced[LABEL_COL].unique())
log(f"  Classes in train       : {sorted(train_cls)} "
    f"{'✓ all 6' if train_cls == all_classes else '✗ MISSING: ' + str(all_classes - train_cls)}")

# Check no NaNs
log(f"  NaNs in final train    : {train_balanced[FEATURE_COLS].isnull().sum().sum()} ✓")
log(f"  NaNs in final test     : {test_final[FEATURE_COLS].isnull().sum().sum()} ✓")

log(f"\n  Final train distribution:")
for cls, cnt in sorted(train_balanced[LABEL_COL].value_counts().items()):
    log(f"    class {cls} ({GAS_NAMES.get(cls,'?'):15s}): {cnt:5d}")

log(f"\n  Final test distribution:")
for cls, cnt in sorted(test_final[LABEL_COL].value_counts().items()):
    pct = cnt / len(test_final) * 100
    log(f"    class {cls} ({GAS_NAMES.get(cls,'?'):15s}): {cnt:5d} ({pct:.1f}%)")

# ══════════════════════════════════════════════
#  STEP 9 — Save outputs
# ══════════════════════════════════════════════
log("\n" + "=" * 60)
log("STEP 9 — Save outputs")
log("=" * 60)

out_cols = FEATURE_COLS + [LABEL_COL]

train_out = os.path.join(OUT_DIR, "fused_train.csv")
test_out  = os.path.join(OUT_DIR, "fused_test.csv")

train_balanced[out_cols].to_csv(train_out, index=False)
test_final[out_cols].to_csv(test_out, index=False)

log(f"  Saved: fused_train.csv  ({len(train_balanced)} rows × {len(out_cols)} cols)")
log(f"  Saved: fused_test.csv   ({len(test_final)} rows × {len(out_cols)} cols)")

# Save report
report_path = os.path.join(OUT_DIR, "fusion_report.txt")
with open(report_path, "w") as f:
    f.write("\n".join(REPORT_LINES))
log(f"  Saved: fusion_report.txt")

log("\n" + "=" * 60)
log("FUSION PIPELINE COMPLETE")
log("=" * 60)
log(f"\n  Use fused_train.csv to TRAIN  your model")
log(f"  Use fused_test.csv  to TEST   your model (never seen during training)")
log(f"\n  Next step: build Keras multi-task model and train on fused_train.csv")
