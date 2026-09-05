import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.conformal.split import (
    SplitConformal,
    coverage_report,
)


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results"

ALPHA = 0.10
SEED = 42


def run_method(score_fn):

    print()
    print("=" * 60)
    print(f"SPLIT CONFORMAL — {score_fn.upper()}")
    print("=" * 60)

    # Load CNN probabilities generated earlier.
    cal_probs = np.load(
        RESULTS / "cnn_calibration_probs.npy"
    )

    cal_labels = np.load(
        RESULTS / "cnn_calibration_labels.npy"
    )

    test_probs = np.load(
        RESULTS / "cnn_test_probs.npy"
    )

    test_labels = np.load(
        RESULTS / "cnn_test_labels.npy"
    )

    print(f"Calibration samples: {len(cal_labels):,}")
    print(f"Test samples       : {len(test_labels):,}")

    # Fit ONLY on calibration data.
    cp = SplitConformal(
        alpha=ALPHA,
        score_fn=score_fn,
        seed=SEED,
        allow_empty=False,
    )

    cp.fit(
        cal_probs,
        cal_labels,
    )

    # Predict on completely untouched test data.
    result = cp.predict(test_probs)

    report = coverage_report(
        result,
        test_labels,
    )

    print()
    print("RESULTS")
    print("-" * 40)

    print(
        f"Target coverage       : "
        f"{report['target_coverage']:.3f}"
    )

    print(
        f"Empirical coverage    : "
        f"{report['empirical_coverage']:.4f}"
    )

    print(
        f"Healthy coverage      : "
        f"{report['coverage_healthy']:.4f}"
    )

    print(
        f"Failure coverage      : "
        f"{report['coverage_failure']:.4f}"
    )

    print(
        f"Average set size      : "
        f"{report['avg_set_size']:.4f}"
    )

    print(
        f"Singleton rate        : "
        f"{report['singleton_rate']:.4f}"
    )

    print(
        f"Doubleton rate        : "
        f"{report['doubleton_rate']:.4f}"
    )

    print(
        f"Empty rate            : "
        f"{report['empty_rate']:.4f}"
    )

    print(
        f"qhat                  : "
        f"{report['qhat']:.6f}"
    )

    print(
        f"Calibration size      : "
        f"{report['n_cal']:,}"
    )

    print(
        f"Test size             : "
        f"{report['n_test']:,}"
    )

    print(
        f"Test failures         : "
        f"{report['n_failures_test']:,}"
    )

    print()
    print("Example prediction sets:")

    for i in range(10):
        labels = result.as_labels()[i]

        print(
            f"  sample {i}: "
            f"P(failure)={test_probs[i]:.4f} "
            f"-> set={labels}"
        )

    return report


print("=" * 60)
print("SSD FAILURE PREDICTION — CONFORMAL PREDICTION")
print("=" * 60)

lac_report = run_method("lac")
aps_report = run_method("aps")

print()
print("=" * 60)
print("CONFORMAL ANALYSIS COMPLETE")
print("=" * 60)