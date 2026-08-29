"""
Split (inductive) conformal prediction.

Guarantee: if calibration and test data are exchangeable, then

    P(Y_test in C(X_test)) >= 1 - alpha

marginally over the draw of calibration and test points. Distribution-free
and finite-sample — no assumption that the underlying model is calibrated,
well-specified, or good.

The exchangeability precondition is what this project stress-tests. Under
LOMO the calibration drives come from two vendors and the test drives from
a third, so exchangeability fails by construction and the guarantee is not
expected to hold. Measuring that gap is the paper's primary result.

Validate on an exchangeable split BEFORE running LOMO. Otherwise
undercoverage cannot be attributed to shift rather than to a bug here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scores import SCORE_FNS

HEALTHY, FAILURE = 0, 1


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """
    The finite-sample corrected quantile:

        qhat = the ceil((n+1)(1-alpha))/n empirical quantile of scores

    The (n+1) term is not cosmetic. Using the plain (1-alpha) quantile
    undercovers by roughly 1/n, which on a small calibration set looks
    exactly like a mild distribution-shift effect — precisely the
    confusion this project must avoid.

    Returns +inf when n is too small for the level, which yields
    all-inclusive sets. That is the honest answer: with n < 1/alpha - 1
    the data cannot support the requested coverage.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    scores = scores[np.isfinite(scores)]
    n = scores.shape[0]
    if n == 0:
        raise ValueError("empty calibration set")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


@dataclass
class ConformalResult:
    """Prediction sets as a boolean (n, 2) mask over [healthy, failure]."""
    sets: np.ndarray
    qhat: float
    alpha: float
    score_fn: str
    n_cal: int

    @property
    def set_sizes(self) -> np.ndarray:
        return self.sets.sum(axis=1)

    @property
    def is_singleton(self) -> np.ndarray:
        return self.set_sizes == 1

    @property
    def is_doubleton(self) -> np.ndarray:
        """Both labels included — the explicit 'I don't know'."""
        return self.set_sizes == 2

    @property
    def is_empty(self) -> np.ndarray:
        """
        Possible under LAC when qhat < 0.5 and the model is confident.
        Empty sets are informative (both labels are surprising) but
        operationally awkward; report their rate rather than hiding it.
        """
        return self.set_sizes == 0

    def as_labels(self) -> list[list[int]]:
        return [[k for k in (HEALTHY, FAILURE) if row[k]]
                for row in self.sets]


class SplitConformal:
    """
    Fit on calibration scores, emit prediction sets on new points.

        cp = SplitConformal(alpha=0.10).fit(cal_probs, cal_labels)
        result = cp.predict(test_probs)

    The calibration set must be disjoint from training data, and under
    LOMO must be drawn only from the training vendors. Including
    held-out-vendor drives restores exchangeability and voids the
    experiment.
    """

    def __init__(self, alpha: float = 0.10, score_fn: str = "lac",
                 seed: int = 0, allow_empty: bool = True):
        if score_fn not in SCORE_FNS:
            raise ValueError(
                f"unknown score_fn {score_fn!r}; "
                f"choose from {sorted(SCORE_FNS)}"
            )
        self.alpha = alpha
        self.score_fn = score_fn
        self.seed = seed
        self.allow_empty = allow_empty
        self.qhat_: float | None = None
        self.n_cal_: int | None = None

    def fit(self, cal_probs: np.ndarray,
            cal_labels: np.ndarray) -> "SplitConformal":
        true_fn, _ = SCORE_FNS[self.score_fn]
        if self.score_fn == "aps":
            scores = true_fn(cal_probs, cal_labels,
                             rng=np.random.default_rng(self.seed))
        else:
            scores = true_fn(cal_probs, cal_labels)

        self.qhat_ = conformal_quantile(scores, self.alpha)
        self.n_cal_ = len(np.asarray(cal_probs).ravel())
        return self

    def predict(self, probs: np.ndarray) -> ConformalResult:
        if self.qhat_ is None:
            raise RuntimeError("call fit() before predict()")

        _, cand_fn = SCORE_FNS[self.score_fn]
        if self.score_fn == "aps":
            cand = cand_fn(probs, rng=np.random.default_rng(self.seed + 1))
        else:
            cand = cand_fn(probs)
        sets = cand <= self.qhat_

        if not self.allow_empty:
            # Force the argmax label into any empty set. Conservative:
            # coverage can only increase. An empty set is theoretically
            # meaningful (both labels are surprising) but not actionable
            # for an operator deciding whether to replace a drive.
            empty = ~sets.any(axis=1)
            if empty.any():
                best = np.argmin(cand[empty], axis=1)
                sets[np.flatnonzero(empty), best] = True

        return ConformalResult(
            sets=sets,
            qhat=self.qhat_,
            alpha=self.alpha,
            score_fn=self.score_fn,
            n_cal=self.n_cal_,
        )


# ---------------------------------------------------------------
# Coverage diagnostics
# ---------------------------------------------------------------

def empirical_coverage(result: ConformalResult,
                       labels: np.ndarray) -> float:
    """Fraction of test points whose true label is in the set."""
    labels = np.asarray(labels, dtype=int).ravel()
    return float(result.sets[np.arange(len(labels)), labels].mean())


def class_conditional_coverage(result: ConformalResult,
                               labels: np.ndarray) -> dict[int, float]:
    """
    Coverage per true class.

    Report this alongside marginal coverage. Under the ~1% failure rate
    of the real fleet, marginal coverage is dominated by healthy drives
    and can sit at nominal while failure-class coverage is far below it.
    """
    labels = np.asarray(labels, dtype=int).ravel()
    hit = result.sets[np.arange(len(labels)), labels]
    out = {}
    for k in (HEALTHY, FAILURE):
        mask = labels == k
        out[k] = float(hit[mask].mean()) if mask.any() else float("nan")
    return out


def coverage_report(result: ConformalResult,
                    labels: np.ndarray) -> dict:
    """Everything needed for one row of the LOMO results table."""
    labels = np.asarray(labels, dtype=int).ravel()
    cc = class_conditional_coverage(result, labels)
    return {
        "alpha": result.alpha,
        "target_coverage": 1.0 - result.alpha,
        "empirical_coverage": empirical_coverage(result, labels),
        "coverage_healthy": cc[HEALTHY],
        "coverage_failure": cc[FAILURE],
        "avg_set_size": float(result.set_sizes.mean()),
        "singleton_rate": float(result.is_singleton.mean()),
        "doubleton_rate": float(result.is_doubleton.mean()),
        "empty_rate": float(result.is_empty.mean()),
        "qhat": result.qhat,
        "n_cal": result.n_cal,
        "n_test": int(len(labels)),
        "n_failures_test": int((labels == FAILURE).sum()),
        "score_fn": result.score_fn,
    }
