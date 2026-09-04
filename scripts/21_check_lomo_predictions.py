from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

RESULTS = Path("results/lomo")

for lomo in ["A", "B", "C"]:

    print("\n" + "=" * 60)
    print(f"LOMO-{lomo} SANITY CHECK")
    print("=" * 60)

    cal_probs = np.load(
        RESULTS / f"lomo_{lomo}_calibration_probs.npy"
    )
    cal_y = np.load(
        RESULTS / f"lomo_{lomo}_calibration_labels.npy"
    )

    test_probs = np.load(
        RESULTS / f"lomo_{lomo}_test_probs.npy"
    )
    test_y = np.load(
        RESULTS / f"lomo_{lomo}_test_labels.npy"
    )

    print("\nCALIBRATION")
    print(f"Samples: {len(cal_y):,}")
    print(f"Failures: {cal_y.sum():,}")
    print(f"Failure rate: {cal_y.mean():.6%}")
    print(f"Min probability: {cal_probs.min():.6f}")
    print(f"Max probability: {cal_probs.max():.6f}")
    print(f"Mean probability: {cal_probs.mean():.6f}")
    print(f"Median probability: {np.median(cal_probs):.6f}")
    print(
        f"Failure mean probability: "
        f"{cal_probs[cal_y == 1].mean():.6f}"
    )
    print(
        f"Healthy mean probability: "
        f"{cal_probs[cal_y == 0].mean():.6f}"
    )

    print("\nTEST")
    print(f"Samples: {len(test_y):,}")
    print(f"Failures: {test_y.sum():,}")
    print(f"Failure rate: {test_y.mean():.6%}")
    print(f"Min probability: {test_probs.min():.6f}")
    print(f"Max probability: {test_probs.max():.6f}")
    print(f"Mean probability: {test_probs.mean():.6f}")
    print(f"Median probability: {np.median(test_probs):.6f}")
    print(
        f"Failure mean probability: "
        f"{test_probs[test_y == 1].mean():.6f}"
    )
    print(
        f"Healthy mean probability: "
        f"{test_probs[test_y == 0].mean():.6f}"
    )

    # Metrics
    cal_roc = roc_auc_score(cal_y, cal_probs)
    cal_pr = average_precision_score(cal_y, cal_probs)

    test_roc = roc_auc_score(test_y, test_probs)
    test_pr = average_precision_score(test_y, test_probs)

    print("\nMETRICS")
    print(f"Calibration ROC-AUC: {cal_roc:.4f}")
    print(f"Calibration PR-AUC : {cal_pr:.4f}")
    print(f"Test ROC-AUC       : {test_roc:.4f}")
    print(f"Test PR-AUC        : {test_pr:.4f}")

    # Check whether probabilities are almost constant
    print("\nPREDICTION VARIABILITY")
    print(
        f"Calibration std: {cal_probs.std():.8f}"
    )
    print(
        f"Test std       : {test_probs.std():.8f}"
    )

    if cal_probs.std() < 0.001:
        print("WARNING: Calibration predictions are nearly constant.")

    if test_probs.std() < 0.001:
        print("WARNING: Test predictions are nearly constant.")

    # Check inverted prediction
    inverted_test_roc = roc_auc_score(
        test_y,
        1.0 - test_probs
    )

    print(
        f"\nInverted test ROC-AUC: "
        f"{inverted_test_roc:.4f}"
    )

    if test_roc < 0.5:
        print(
            "NOTE: Inverting the prediction would improve ROC-AUC."
        )

print("\n" + "=" * 60)
print("LOMO SANITY CHECK COMPLETE")
print("=" * 60)