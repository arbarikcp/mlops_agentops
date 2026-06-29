"""ML testing helpers: data contract, behavioral, and training smoke tests.

Day 55 — pure-Python implementations so all tests run without sklearn/pandas
in the CI environment. Uses only stdlib + the feature-engineering logic
already in the platform.

Classes:
  DataContractResult  — outcome of one data contract check
  DataContractChecker — schema, null-rate, and label-distribution checks
  BehavioralResult    — outcome of one behavioral invariant check
  BehavioralChecker   — monotonicity, robustness, directional, invariance
  SmokeResult         — outcome of a training smoke run
  SmokeTrainer        — 100-row synthetic train; convergence + reproducibility
  AUCGuardResult      — AUC regression guard outcome
  AUCGuard            — compares current AUC to stored baseline

See: docs/phase8/day55_ml_testing.md
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable


# ── DataContractChecker ────────────────────────────────────────────────────────

@dataclass
class DataContractResult:
    """Outcome of one data contract check.

    Attributes:
        check_name:    Identifier for the check.
        passed:        True if the contract was satisfied.
        message:       Human-readable summary.
        details:       Numeric or extra metadata.
    """

    check_name: str
    passed: bool
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class DataContractChecker:
    """Validates an incoming data batch against a schema and label contract.

    Args:
        schema: dict mapping column name → {"dtype", "min"?, "max"?, "null_rate"?}
        label_column: Name of the binary label column.
        min_positive_rate: Minimum fraction of positive labels (default 0.05).
        max_positive_rate: Maximum fraction of positive labels (default 0.40).
    """

    def __init__(
        self,
        schema: dict[str, dict],
        label_column: str = "default",
        min_positive_rate: float = 0.05,
        max_positive_rate: float = 0.40,
    ) -> None:
        if not schema:
            raise ValueError("schema cannot be empty")
        self.schema = schema
        self.label_column = label_column
        self.min_positive_rate = min_positive_rate
        self.max_positive_rate = max_positive_rate

    def check_schema(self, data: list[dict]) -> DataContractResult:
        """Check that all expected columns are present in each row."""
        if not data:
            return DataContractResult("schema", False, "data is empty")

        missing: list[str] = []
        for col in self.schema:
            if col not in data[0]:
                missing.append(col)

        passed = len(missing) == 0
        return DataContractResult(
            check_name="schema",
            passed=passed,
            message="ok" if passed else f"missing columns: {missing}",
            details={"missing": missing},
        )

    def check_null_rates(self, data: list[dict]) -> DataContractResult:
        """Check that null rates are within the declared maximum per column."""
        if not data:
            return DataContractResult("null_rates", False, "data is empty")

        failures: list[str] = []
        n = len(data)
        for col, spec in self.schema.items():
            max_null = spec.get("null_rate", 0.0)
            actual_null = sum(1 for row in data if row.get(col) is None) / n
            if actual_null > max_null:
                failures.append(f"{col}: {actual_null:.3f} > {max_null:.3f}")

        passed = len(failures) == 0
        return DataContractResult(
            check_name="null_rates",
            passed=passed,
            message="ok" if passed else f"null rate violations: {failures}",
            details={"violations": failures},
        )

    def check_label_dist(self, data: list[dict]) -> DataContractResult:
        """Check that the label positive rate is within [min, max]."""
        if not data:
            return DataContractResult("label_dist", False, "data is empty")

        labels = [row.get(self.label_column) for row in data if row.get(self.label_column) is not None]
        if not labels:
            return DataContractResult("label_dist", False, "no label values found")

        pos_rate = sum(1 for v in labels if v == 1) / len(labels)
        passed = self.min_positive_rate <= pos_rate <= self.max_positive_rate
        return DataContractResult(
            check_name="label_dist",
            passed=passed,
            message=(
                "ok"
                if passed
                else f"positive rate {pos_rate:.3f} outside [{self.min_positive_rate}, {self.max_positive_rate}]"
            ),
            details={"positive_rate": pos_rate},
        )

    def run_all(self, data: list[dict]) -> list[DataContractResult]:
        """Run schema, null_rates, and label_dist checks. Returns all results."""
        return [
            self.check_schema(data),
            self.check_null_rates(data),
            self.check_label_dist(data),
        ]


# ── BehavioralChecker ─────────────────────────────────────────────────────────

@dataclass
class BehavioralResult:
    """Outcome of one behavioral invariant check.

    Attributes:
        check_name: Short identifier.
        passed:     True if the invariant holds.
        message:    Human-readable explanation of pass or fail.
        violations: Count of samples that violate the invariant.
    """

    check_name: str
    passed: bool
    message: str = ""
    violations: int = 0


class BehavioralChecker:
    """Checks model behavioral invariants: monotonicity, robustness, directional, invariance.

    Args:
        predict_fn: Callable that takes a list[dict] of feature dicts → list[float] of scores.
    """

    def __init__(self, predict_fn: Callable[[list[dict]], list[float]]) -> None:
        self.predict_fn = predict_fn

    def check_monotonicity(
        self,
        base_row: dict,
        feature: str,
        low_value: float,
        high_value: float,
        direction: str = "higher_score_for_higher_value",
    ) -> BehavioralResult:
        """Check that changing `feature` from low_value to high_value changes score in expected direction.

        Args:
            base_row: Template row with all features populated.
            feature:  Feature name to vary.
            low_value: Lower feature value.
            high_value: Higher feature value.
            direction: "higher_score_for_higher_value" or "lower_score_for_higher_value".
        """
        row_low = {**base_row, feature: low_value}
        row_high = {**base_row, feature: high_value}
        score_low, score_high = self.predict_fn([row_low])[0], self.predict_fn([row_high])[0]

        if direction == "higher_score_for_higher_value":
            passed = score_high >= score_low
        else:
            passed = score_high <= score_low

        return BehavioralResult(
            check_name=f"monotonicity:{feature}",
            passed=passed,
            message=(
                "ok"
                if passed
                else f"score_low={score_low:.4f} score_high={score_high:.4f} direction={direction}"
            ),
        )

    def check_robustness(
        self, rows: list[dict], noise_pct: float = 0.01, max_delta: float = 0.05
    ) -> BehavioralResult:
        """Check that small input noise doesn't cause large score swings.

        Args:
            rows:      List of feature dicts to evaluate.
            noise_pct: Fraction of each numeric value to perturb (default 1%).
            max_delta: Maximum allowed score change per sample (default 0.05).
        """
        rng = random.Random(42)
        violations = 0

        for row in rows:
            score_orig = self.predict_fn([row])[0]
            noisy = {}
            for k, v in row.items():
                if isinstance(v, (int, float)):
                    noisy[k] = v * (1 + rng.uniform(-noise_pct, noise_pct))
                else:
                    noisy[k] = v
            score_noisy = self.predict_fn([noisy])[0]
            if abs(score_noisy - score_orig) > max_delta:
                violations += 1

        passed = violations == 0
        return BehavioralResult(
            check_name="robustness",
            passed=passed,
            message="ok" if passed else f"{violations}/{len(rows)} samples exceeded delta {max_delta}",
            violations=violations,
        )

    def check_invariance(
        self, rows: list[dict], feature: str, value_a: Any, value_b: Any, tolerance: float = 0.02
    ) -> BehavioralResult:
        """Check that swapping a protected-attribute feature doesn't change the score.

        Args:
            rows:      List of feature dicts.
            feature:   Feature name to swap (e.g., "gender").
            value_a:   First value (e.g., "M").
            value_b:   Second value (e.g., "F").
            tolerance: Max allowed absolute score difference (default 0.02).
        """
        violations = 0
        for row in rows:
            row_a = {**row, feature: value_a}
            row_b = {**row, feature: value_b}
            s_a = self.predict_fn([row_a])[0]
            s_b = self.predict_fn([row_b])[0]
            if abs(s_a - s_b) > tolerance:
                violations += 1

        passed = violations == 0
        return BehavioralResult(
            check_name=f"invariance:{feature}",
            passed=passed,
            message="ok" if passed else f"{violations}/{len(rows)} samples exceeded tolerance {tolerance}",
            violations=violations,
        )

    def check_confidence(self, rows: list[dict], min_stdev: float = 0.05) -> BehavioralResult:
        """Check that the model produces diverse scores (not stuck at 0.5).

        Args:
            rows:      List of feature dicts.
            min_stdev: Minimum required standard deviation of scores (default 0.05).
        """
        scores = self.predict_fn(rows)
        mean = sum(scores) / len(scores) if scores else 0.0
        variance = sum((s - mean) ** 2 for s in scores) / len(scores) if scores else 0.0
        stdev = math.sqrt(variance)
        passed = stdev >= min_stdev
        return BehavioralResult(
            check_name="confidence",
            passed=passed,
            message=f"stdev={stdev:.4f}" + ("" if passed else f" < min_stdev={min_stdev}"),
        )


# ── SmokeTrainer ──────────────────────────────────────────────────────────────

@dataclass
class SmokeResult:
    """Outcome of a training smoke run.

    Attributes:
        passed:      True if all smoke assertions were satisfied.
        auc:         AUC on held-out 20% split.
        n_rows:      Number of rows used.
        reproducible: True if two identical seeded runs gave same AUC.
        message:     Human-readable summary.
    """

    passed: bool
    auc: float
    n_rows: int
    reproducible: bool = True
    message: str = ""


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _mann_whitney_auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Compute AUC via Mann-Whitney U (no sklearn needed)."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    wins = sum(1 for p in pos_scores for n in neg_scores if p > n)
    ties = sum(0.5 for p in pos_scores for n in neg_scores if p == n)
    return (wins + ties) / (n_pos * n_neg)


class SmokeTrainer:
    """Fast training smoke test using logistic regression on synthetic data.

    Trains a simple logistic regression with gradient descent in pure Python
    on 100 synthetic rows. No external libraries required.

    Args:
        n_rows:    Number of synthetic rows to generate (default 100).
        n_features: Number of input features (default 5).
        max_iter:  Gradient descent iterations (default 50).
        lr:        Learning rate (default 0.1).
        seed:      Random seed for reproducibility.
        val_split: Fraction for validation (default 0.2).
    """

    def __init__(
        self,
        n_rows: int = 100,
        n_features: int = 5,
        max_iter: int = 50,
        lr: float = 0.1,
        seed: int = 42,
        val_split: float = 0.2,
    ) -> None:
        self.n_rows = n_rows
        self.n_features = n_features
        self.max_iter = max_iter
        self.lr = lr
        self.seed = seed
        self.val_split = val_split

    def _generate_data(self, seed: int) -> tuple[list[list[float]], list[int]]:
        rng = random.Random(seed)
        X, y = [], []
        for i in range(self.n_rows):
            features = [rng.gauss(0, 1) for _ in range(self.n_features)]
            # Label correlated with feature sum so AUC > 0.5 is achievable
            log_odds = sum(features) * 0.5
            prob = _logistic(log_odds)
            label = 1 if rng.random() < prob else 0
            X.append(features)
            y.append(label)
        return X, y

    def _train(self, X: list[list[float]], y: list[int]) -> list[float]:
        """Train logistic regression by gradient descent. Returns weights (bias last)."""
        n, d = len(X), len(X[0])
        w = [0.0] * (d + 1)  # last element is bias
        for _ in range(self.max_iter):
            for i in range(n):
                xi = X[i] + [1.0]
                pred = _logistic(sum(w[j] * xi[j] for j in range(d + 1)))
                err = pred - y[i]
                for j in range(d + 1):
                    w[j] -= self.lr * err * xi[j]
        return w

    def _predict(self, X: list[list[float]], w: list[float]) -> list[float]:
        d = len(w) - 1
        return [_logistic(sum(w[j] * (x[j] if j < d else 1.0) for j in range(d + 1))) for x in X]

    def run(self) -> SmokeResult:
        """Train on synthetic data, compute AUC, and check reproducibility."""
        X, y = self._generate_data(self.seed)

        n_val = max(1, int(self.n_rows * self.val_split))
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val,   y_val   = X[-n_val:], y[-n_val:]

        w = self._train(X_train, y_train)
        scores = self._predict(X_val, w)

        pos_scores = [scores[i] for i, lbl in enumerate(y_val) if lbl == 1]
        neg_scores = [scores[i] for i, lbl in enumerate(y_val) if lbl == 0]
        auc = _mann_whitney_auc(pos_scores, neg_scores)

        # Reproducibility: train again with same seed
        X2, y2 = self._generate_data(self.seed)
        w2 = self._train(X2[:-n_val], y2[:-n_val])
        scores2 = self._predict(X2[-n_val:], w2)
        pos2 = [scores2[i] for i, lbl in enumerate(y2[-n_val:]) if lbl == 1]
        neg2 = [scores2[i] for i, lbl in enumerate(y2[-n_val:]) if lbl == 0]
        auc2 = _mann_whitney_auc(pos2, neg2)
        reproducible = abs(auc - auc2) <= 0.001

        issues = []
        if auc <= 0.5:
            issues.append(f"AUC {auc:.4f} not better than random")
        if not reproducible:
            issues.append(f"AUC not reproducible: {auc:.4f} vs {auc2:.4f}")
        if len(w) != self.n_features + 1:
            issues.append(f"weight count mismatch: {len(w)} != {self.n_features + 1}")

        passed = len(issues) == 0
        return SmokeResult(
            passed=passed,
            auc=auc,
            n_rows=self.n_rows,
            reproducible=reproducible,
            message="ok" if passed else "; ".join(issues),
        )


# ── AUCGuard ──────────────────────────────────────────────────────────────────

@dataclass
class AUCGuardResult:
    """Outcome of the AUC regression guard check.

    Attributes:
        passed:      True if current AUC >= baseline - tolerance.
        current_auc: AUC from the current CI run.
        baseline_auc: Stored reference AUC.
        delta:       current_auc - baseline_auc (positive = improvement).
        message:     Human-readable outcome.
    """

    passed: bool
    current_auc: float
    baseline_auc: float
    delta: float
    message: str = ""


class AUCGuard:
    """Prevents training code changes from silently degrading model AUC.

    Args:
        baseline_auc: The stored reference AUC (from last known-good run).
        tolerance:    Allowed regression before failing (default 0.01).
    """

    def __init__(self, baseline_auc: float, tolerance: float = 0.01) -> None:
        if not (0.0 <= baseline_auc <= 1.0):
            raise ValueError(f"baseline_auc must be in [0, 1]; got {baseline_auc}")
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        self.baseline_auc = baseline_auc
        self.tolerance = tolerance

    def check(self, current_auc: float) -> AUCGuardResult:
        """Compare current_auc to the stored baseline.

        Returns:
            AUCGuardResult with passed=True if current_auc >= baseline - tolerance.
        """
        delta = current_auc - self.baseline_auc
        passed = current_auc >= self.baseline_auc - self.tolerance
        if passed and delta > 0:
            msg = f"improved by {delta:.4f} (baseline={self.baseline_auc:.4f})"
        elif passed:
            msg = f"within tolerance (delta={delta:.4f})"
        else:
            msg = f"regression: {delta:.4f} below tolerance -{self.tolerance:.4f}"
        return AUCGuardResult(
            passed=passed,
            current_auc=current_auc,
            baseline_auc=self.baseline_auc,
            delta=delta,
            message=msg,
        )

    def update_baseline(self, new_auc: float) -> None:
        """Store a new baseline (called when CI passes and AUC improved)."""
        self.baseline_auc = new_auc
