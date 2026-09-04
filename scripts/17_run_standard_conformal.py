from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = ROOT / "results"


# ============================================================
# Settings
# ============================================================

ALPHA = 0.10
TARGET_COVERAGE = 1.0 - ALPHA


# ============================================================
# Load CNN probabilities
# ============================================================

cal_probs = np.load(
    RESULT_DIR / "standard_cnn_calibration_probs.npy"
)

cal_labels = np.load(
    RESULT_DIR / "standard_cnn_calibration_labels.npy"
)

test_probs = np.load(
    RESULT_DIR / "standard_cnn_test_probs.npy"
)

test_labels = np.load(
    RESULT_DIR / "standard_cnn_test_labels.npy"
)


print("=" * 60)
print("STANDARD CONFORMAL PREDICTION")
print("=" * 60)

print(
    f"Calibration samples: {len(cal_labels):,}"
)

print(
    f"Test samples: {len(test_labels):,}"
)

print(
    f"Calibration positives: {int(cal_labels.sum()):,}"
)

print(
    f"Test positives: {int(test_labels.sum()):,}"
)

print(
    f"Alpha: {ALPHA}"
)

print(
    f"Target coverage: {TARGET_COVERAGE:.1%}"
)


# ============================================================
# LAC
#
# For binary classification:
#
#   if y = 1:
#       score = 1 - P(y=1)
#
#   if y = 0:
#       score = P(y=1)
#
# The score measures how badly the model fits the true class.
# ============================================================

cal_scores_lac = np.where(
    cal_labels == 1,
    1.0 - cal_probs,
    cal_probs,
)


# Finite-sample conformal quantile.
n = len(cal_scores_lac)

rank = int(
    np.ceil(
        (n + 1) * (1.0 - ALPHA)
    )
)

rank = min(
    max(rank, 1),
    n,
)

qhat_lac = np.sort(
    cal_scores_lac
)[rank - 1]


# ------------------------------------------------------------
# Build LAC prediction sets.
#
# Class 0 is included if:
#     P(y=1) <= qhat
#
# Class 1 is included if:
#     1-P(y=1) <= qhat
# ------------------------------------------------------------

include_0 = (
    test_probs <= qhat_lac
)

include_1 = (
    1.0 - test_probs <= qhat_lac
)


# ------------------------------------------------------------
# Coverage
# ------------------------------------------------------------

covered_lac = np.where(
    test_labels == 0,
    include_0,
    include_1,
)

coverage_lac = covered_lac.mean()

set_size_lac = (
    include_0.astype(np.int8)
    + include_1.astype(np.int8)
)

avg_size_lac = set_size_lac.mean()

singleton_lac = (
    set_size_lac == 1
).mean()

doubleton_lac = (
    set_size_lac == 2
).mean()

empty_lac = (
    set_size_lac == 0
).mean()


# ------------------------------------------------------------
# Class conditional coverage
# ------------------------------------------------------------

healthy_mask = (
    test_labels == 0
)

failure_mask = (
    test_labels == 1
)

healthy_coverage_lac = (
    covered_lac[healthy_mask].mean()
    if healthy_mask.any()
    else np.nan
)

failure_coverage_lac = (
    covered_lac[failure_mask].mean()
    if failure_mask.any()
    else np.nan
)


# ============================================================
# APS
#
# For binary classification, sort class probabilities from
# highest to lowest and calculate the cumulative probability
# needed to include the true class.
# ============================================================

p1 = cal_probs
p0 = 1.0 - cal_probs

# Highest-probability class
cal_top_is_1 = p1 >= p0

cal_scores_aps = np.where(
    cal_labels == 1,

    # True class is class 1.
    np.where(
        cal_top_is_1,
        p1,
        p0 + p1
    ),

    # True class is class 0.
    np.where(
        cal_top_is_1,
        p1 + p0,
        p0
    ),
)


# ------------------------------------------------------------
# APS quantile
# ------------------------------------------------------------

rank = int(
    np.ceil(
        (n + 1) * (1.0 - ALPHA)
    )
)

rank = min(
    max(rank, 1),
    n,
)

qhat_aps = np.sort(
    cal_scores_aps
)[rank - 1]


# ------------------------------------------------------------
# APS prediction sets
# ------------------------------------------------------------

test_p1 = test_probs
test_p0 = 1.0 - test_probs

test_top_is_1 = (
    test_p1 >= test_p0
)

# If class 1 is the top class:
#
#   class 1 needs p1
#   class 0 needs p1 + p0 = 1
#
# If class 0 is the top class:
#
#   class 0 needs p0
#   class 1 needs p0 + p1 = 1

include_1_aps = np.where(
    test_top_is_1,
    test_p1 <= qhat_aps,
    1.0 <= qhat_aps,
)

include_0_aps = np.where(
    test_top_is_1,
    1.0 <= qhat_aps,
    test_p0 <= qhat_aps,
)


# ------------------------------------------------------------
# Coverage
# ------------------------------------------------------------

covered_aps = np.where(
    test_labels == 0,
    include_0_aps,
    include_1_aps,
)

coverage_aps = covered_aps.mean()

set_size_aps = (
    include_0_aps.astype(np.int8)
    + include_1_aps.astype(np.int8)
)

avg_size_aps = set_size_aps.mean()

singleton_aps = (
    set_size_aps == 1
).mean()

doubleton_aps = (
    set_size_aps == 2
).mean()

empty_aps = (
    set_size_aps == 0
).mean()


healthy_coverage_aps = (
    covered_aps[healthy_mask].mean()
    if healthy_mask.any()
    else np.nan
)

failure_coverage_aps = (
    covered_aps[failure_mask].mean()
    if failure_mask.any()
    else np.nan
)


# ============================================================
# Save results
# ============================================================

np.save(
    RESULT_DIR / "standard_lac_covered.npy",
    covered_lac,
)

np.save(
    RESULT_DIR / "standard_lac_set_size.npy",
    set_size_lac,
)

np.save(
    RESULT_DIR / "standard_aps_covered.npy",
    covered_aps,
)

np.save(
    RESULT_DIR / "standard_aps_set_size.npy",
    set_size_aps,
)


# ============================================================
# Print LAC results
# ============================================================

print()
print("=" * 60)
print("LAC RESULTS")
print("=" * 60)

print(
    f"Target coverage       : "
    f"{TARGET_COVERAGE:.4f}"
)

print(
    f"Empirical coverage    : "
    f"{coverage_lac:.4f}"
)

print(
    f"Healthy coverage      : "
    f"{healthy_coverage_lac:.4f}"
)

print(
    f"Failure coverage      : "
    f"{failure_coverage_lac:.4f}"
)

print(
    f"Average set size      : "
    f"{avg_size_lac:.4f}"
)

print(
    f"Singleton sets        : "
    f"{singleton_lac:.4%}"
)

print(
    f"Doubleton sets        : "
    f"{doubleton_lac:.4%}"
)

print(
    f"Empty sets            : "
    f"{empty_lac:.4%}"
)

print(
    f"qhat                  : "
    f"{qhat_lac:.6f}"
)


# ============================================================
# Print APS results
# ============================================================

print()
print("=" * 60)
print("APS RESULTS")
print("=" * 60)

print(
    f"Target coverage       : "
    f"{TARGET_COVERAGE:.4f}"
)

print(
    f"Empirical coverage    : "
    f"{coverage_aps:.4f}"
)

print(
    f"Healthy coverage      : "
    f"{healthy_coverage_aps:.4f}"
)

print(
    f"Failure coverage      : "
    f"{failure_coverage_aps:.4f}"
)

print(
    f"Average set size      : "
    f"{avg_size_aps:.4f}"
)

print(
    f"Singleton sets        : "
    f"{singleton_aps:.4%}"
)

print(
    f"Doubleton sets        : "
    f"{doubleton_aps:.4%}"
)

print(
    f"Empty sets            : "
    f"{empty_aps:.4%}"
)

print(
    f"qhat                  : "
    f"{qhat_aps:.6f}"
)


# ============================================================
# Finish
# ============================================================

print()
print("=" * 60)
print("STANDARD CONFORMAL PREDICTION COMPLETE")
print("=" * 60)