"""
End-to-end pipeline smoke test.

Runs synthetic -> labels -> splits -> windows -> normalise -> dummy
model -> conformal -> coverage report, for the standard split and all
three LOMO folds.

This is the test that catches interface mismatches between modules.
Every mismatch found here is one not found during a training run.

The model is a deliberate stand-in: logistic regression on the last
timestep. Nothing here is a claim about accuracy. What is being checked
is that the pieces compose, that leakage guards survive the full
pipeline, and that the LOMO harness produces a populated results table.

    python3 -m pytest tests/test_pipeline.py -v -s
"""

import numpy as np
import pytest

from src.conformal.split import SplitConformal, coverage_report
from src.data.labels import build_labels
from src.data.splits import make_all_lomo_splits, make_standard_split
from src.data.synthetic import make_synthetic
from src.data.windowing import Normalizer, flatten, make_windows

STRIDE = 7


# ---------------------------------------------------------------
# Dummy model — a stand-in, not a baseline
# ---------------------------------------------------------------

class DummyModel:
    """
    Logistic regression by gradient descent on the final timestep.

    Deliberately minimal and dependency-free. Its only job is to emit
    probabilities so the conformal layer has something to wrap.
    """

    def __init__(self, lr=0.5, epochs=300, seed=0):
        self.lr, self.epochs, self.seed = lr, epochs, seed
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.w = rng.normal(0, 0.01, d)
        self.b = 0.0

        for _ in range(self.epochs):
            p = self._sigmoid(X @ self.w + self.b)
            err = p - y
            self.w -= self.lr * (X.T @ err) / n
            self.b -= self.lr * err.mean()
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return self._sigmoid(X @ self.w + self.b)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def run_fold(labelled, split, alpha=0.10, score_fn="lac"):
    """One fold end to end. Returns the coverage report."""
    parts = {}
    for name in ("train", "cal", "test"):
        parts[name] = make_windows(split.apply(labelled, name),
                                   stride=STRIDE)

    # Normalisation fit on TRAIN ONLY. Fitting on pooled data leaks into
    # both the model and the conformal calibration scores.
    norm = Normalizer.fit(parts["train"])
    parts = {k: norm.transform(v) for k, v in parts.items()}

    Xtr, _ = flatten(parts["train"], "last")
    Xca, _ = flatten(parts["cal"], "last")
    Xte, _ = flatten(parts["test"], "last")

    model = DummyModel().fit(Xtr, parts["train"].y)

    cp = SplitConformal(alpha=alpha, score_fn=score_fn).fit(
        model.predict_proba(Xca), parts["cal"].y
    )
    result = cp.predict(model.predict_proba(Xte))

    rep = coverage_report(result, parts["test"].y)
    rep["split"] = split.name
    rep["n_train_windows"] = len(parts["train"])
    rep["test_prevalence"] = float(parts["test"].y.mean())
    return rep


@pytest.fixture(scope="module")
def pipeline():
    df, _ = make_synthetic()
    labelled, _ = build_labels(df)
    standard = make_standard_split(labelled)
    lomo = make_all_lomo_splits(labelled)
    return labelled, standard, lomo


# ---------------------------------------------------------------
# The pipeline composes
# ---------------------------------------------------------------

def test_standard_fold_runs_end_to_end(pipeline):
    labelled, standard, _ = pipeline
    rep = run_fold(labelled, standard)

    assert rep["n_test"] > 0
    assert rep["n_cal"] > 0
    assert rep["n_failures_test"] > 0
    assert 0.0 <= rep["empirical_coverage"] <= 1.0
    assert rep["target_coverage"] == 0.90


@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_lomo_fold_runs_end_to_end(pipeline, vendor):
    labelled, _, lomo = pipeline
    rep = run_fold(labelled, lomo[vendor])

    assert rep["split"] == f"lomo_{vendor}"
    assert rep["n_test"] > 0
    assert rep["n_failures_test"] > 0


def test_all_folds_produce_a_full_results_row(pipeline):
    """Every field the LOMO results table needs must be populated."""
    labelled, standard, lomo = pipeline
    required = [
        "split", "alpha", "target_coverage", "empirical_coverage",
        "coverage_healthy", "coverage_failure", "avg_set_size",
        "singleton_rate", "doubleton_rate", "empty_rate", "qhat",
        "n_cal", "n_test", "n_failures_test", "test_prevalence",
    ]
    for split in [standard, *lomo.values()]:
        rep = run_fold(labelled, split)
        for k in required:
            assert k in rep, f"{split.name} missing {k}"
            assert rep[k] is not None


# ---------------------------------------------------------------
# Leakage guards survive the full pipeline
# ---------------------------------------------------------------

@pytest.mark.parametrize("vendor", ["A", "B", "C"])
def test_no_held_out_vendor_windows_in_train_or_cal(pipeline, vendor):
    """
    The split-level guard is asserted in validate_split. This checks the
    property survives windowing, which is where a join could reintroduce
    it.
    """
    labelled, _, lomo = pipeline
    split = lomo[vendor]
    for part in ("train", "cal"):
        ws = make_windows(split.apply(labelled, part), stride=STRIDE)
        assert vendor not in set(np.unique(ws.vendors))

    te = make_windows(split.apply(labelled, "test"), stride=STRIDE)
    assert set(np.unique(te.vendors)) == {vendor}


def test_no_drive_appears_in_two_parts_after_windowing(pipeline):
    labelled, standard, _ = pipeline
    keys = {}
    for part in ("train", "cal", "test"):
        ws = make_windows(standard.apply(labelled, part), stride=STRIDE)
        keys[part] = {tuple(k) for k in ws.drives}

    assert not (keys["train"] & keys["cal"])
    assert not (keys["train"] & keys["test"])
    assert not (keys["cal"] & keys["test"])


def test_normalizer_stats_come_from_train_only(pipeline):
    labelled, standard, _ = pipeline
    tr = make_windows(standard.apply(labelled, "train"), stride=STRIDE)
    te = make_windows(standard.apply(labelled, "test"), stride=STRIDE)

    from_train = Normalizer.fit(tr)
    from_pooled = Normalizer.fit(
        make_windows(labelled, stride=STRIDE)
    )
    # If these were identical the fit would be pooling silently.
    assert not np.allclose(from_train.mean, from_pooled.mean)
    assert np.isfinite(from_train.transform(te).X).all()


# ---------------------------------------------------------------
# Sanity properties, not accuracy claims
# ---------------------------------------------------------------

def test_dummy_model_learns_the_injected_signal(pipeline):
    """
    The fixture injects drift into r_5, r_187 and r_197. If even a
    linear model on the last timestep cannot beat chance, the signal is
    not reaching the model and every downstream number is meaningless.
    """
    labelled, standard, _ = pipeline
    tr = make_windows(standard.apply(labelled, "train"), stride=STRIDE)
    te = make_windows(standard.apply(labelled, "test"), stride=STRIDE)
    norm = Normalizer.fit(tr)
    tr, te = norm.transform(tr), norm.transform(te)

    Xtr, _ = flatten(tr, "last")
    Xte, _ = flatten(te, "last")
    p = DummyModel().fit(Xtr, tr.y).predict_proba(Xte)

    pos, neg = p[te.y == 1], p[te.y == 0]
    assert len(pos) > 0 and len(neg) > 0
    assert pos.mean() > neg.mean(), (
        "model scores positives no higher than negatives; "
        "the injected signal is not reaching it"
    )


def test_coverage_is_reported_per_class(pipeline):
    """
    Class-conditional coverage must be present in every results row.
    Reporting only marginal coverage would repeat the exact mistake the
    paper criticises.
    """
    labelled, standard, _ = pipeline
    rep = run_fold(labelled, standard)
    assert not np.isnan(rep["coverage_healthy"])
    assert not np.isnan(rep["coverage_failure"])


def test_set_rates_sum_to_one(pipeline):
    labelled, standard, _ = pipeline
    rep = run_fold(labelled, standard)
    total = (rep["singleton_rate"] + rep["doubleton_rate"]
             + rep["empty_rate"])
    assert abs(total - 1.0) < 1e-9


def test_pipeline_is_deterministic(pipeline):
    labelled, standard, _ = pipeline
    a = run_fold(labelled, standard)
    b = run_fold(labelled, standard)
    assert a["empirical_coverage"] == b["empirical_coverage"]
    assert a["qhat"] == b["qhat"]


# ---------------------------------------------------------------
# Reporting helper — prints the table the paper needs
# ---------------------------------------------------------------

def test_print_results_table(pipeline, capsys):
    """
    Not an assertion of results, a shape check on the table. Run with
    -s to see it.
    """
    labelled, standard, lomo = pipeline
    rows = [run_fold(labelled, s)
            for s in [standard, *lomo.values()]]

    with capsys.disabled():
        print()
        print(f"{'split':<10} {'n_test':>7} {'prev':>7} {'cov':>7} "
              f"{'cov_hlt':>8} {'cov_fail':>9} {'size':>6} {'empty':>7}")
        for r in rows:
            print(f"{r['split']:<10} {r['n_test']:>7,} "
                  f"{r['test_prevalence']:>7.4f} "
                  f"{r['empirical_coverage']:>7.4f} "
                  f"{r['coverage_healthy']:>8.4f} "
                  f"{r['coverage_failure']:>9.4f} "
                  f"{r['avg_set_size']:>6.3f} "
                  f"{r['empty_rate']:>7.4f}")
        print("\nDummy model on synthetic data. Shape check only — "
              "these are not results.")

    assert len(rows) == 4
