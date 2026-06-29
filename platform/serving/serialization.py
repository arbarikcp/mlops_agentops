"""Model serialization: ONNX export, pickle risk, checksum verification.

Production rule: ONNX for serving, joblib for registry/debug only.

Pickle risk:
    pickle.loads() executes arbitrary Python code. Never deserialise a
    pickle received from an untrusted source. ONNX runtime has no such
    risk — it is a pure graph execution engine.

ONNX export flow:
    LightGBM Booster → onnxmltools.convert_lightgbm() → .onnx file
    Then validate parity: sklearn predict_proba vs ORT session.run()

See: docs/phase4/day22_serialization.md for full theory.

Usage:
    from serving.serialization import ModelSerializer

    serializer = ModelSerializer()
    result = serializer.export_to_onnx(lgb_model, feature_names, path)
    report = serializer.validate_parity(result.onnx_path, lgb_model, X_test)
    assert report.passed, f"Parity failed: max_diff={report.max_diff}"
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Parity tolerance: float32 conversion introduces ~1e-4 error
_PARITY_TOLERANCE = 1e-3


@dataclass
class OnnxExportResult:
    """Result of an ONNX export operation.

    Attributes:
        onnx_path:    Path to the written .onnx file.
        checksum:     SHA-256 hex digest of the file.
        opset:        ONNX op-set version used.
        n_features:   Number of input features.
        input_name:   ONNX graph input name (to pass to session.run).
        output_names: ONNX graph output names (label, probabilities).
    """

    onnx_path: Path
    checksum: str
    opset: int
    n_features: int
    input_name: str
    output_names: list[str]


@dataclass
class ParityReport:
    """Numeric parity result between sklearn and ONNX model outputs.

    Attributes:
        max_diff:       Maximum absolute probability difference across all rows.
        mean_diff:      Mean absolute probability difference.
        n_rows_tested:  Number of test rows evaluated.
        passed:         True if max_diff <= threshold.
        threshold:      Tolerance used for the pass/fail decision.
    """

    max_diff: float
    mean_diff: float
    n_rows_tested: int
    passed: bool
    threshold: float


@dataclass
class PickleRiskReport:
    """Result of a pickle risk assessment.

    Attributes:
        path:           Path assessed.
        is_pickle:      True if the file is a pickle (joblib) artifact.
        risk_level:     "none" | "low" | "high"
        recommendation: Human-readable guidance.
    """

    path: Path
    is_pickle: bool
    risk_level: str
    recommendation: str


class ModelSerializer:
    """Handles model serialization to ONNX and pickle risk assessment.

    All methods are stateless — create one instance per application and reuse.
    """

    def export_to_onnx(
        self,
        model: Any,
        feature_names: list[str],
        output_path: Path,
        *,
        opset: int = 17,
    ) -> OnnxExportResult:
        """Export a LightGBM (or sklearn) model to ONNX format.

        Args:
            model:          Trained model with a predict_proba method.
            feature_names:  Ordered list of feature column names.
            output_path:    Destination path for the .onnx file.
            opset:          ONNX operator-set version (default 17).

        Returns:
            OnnxExportResult with path, checksum, and graph metadata.

        Raises:
            ImportError: If onnxmltools or onnxruntime are not installed.
            RuntimeError: If the conversion fails.
        """
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
        except ImportError as exc:
            raise ImportError(
                "onnxmltools is required for ONNX export. "
                "Install with: uv add onnxmltools skl2onnx onnxruntime"
            ) from exc

        n_features = len(feature_names)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]

        log.info("Converting model to ONNX (opset=%d, n_features=%d)", opset, n_features)
        try:
            onnx_model = convert_lightgbm(
                model,
                initial_types=initial_type,
                target_opset=opset,
            )
        except Exception as exc:
            raise RuntimeError(f"ONNX conversion failed: {exc}") from exc

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        checksum = self.compute_sha256(output_path)
        checksum_path = output_path.with_suffix(output_path.suffix + ".sha256")
        checksum_path.write_text(checksum)

        # Inspect the exported graph for input/output names
        import onnx as _onnx
        graph = _onnx.load(str(output_path)).graph
        input_name = graph.input[0].name
        output_names = [o.name for o in graph.output]

        log.info("Exported ONNX model to %s (sha256=%s)", output_path, checksum[:12])
        return OnnxExportResult(
            onnx_path=output_path,
            checksum=checksum,
            opset=opset,
            n_features=n_features,
            input_name=input_name,
            output_names=output_names,
        )

    def load_onnx_session(self, path: Path) -> Any:
        """Load an ONNX model into an OnnxRuntime InferenceSession.

        Args:
            path: Path to the .onnx file.

        Returns:
            onnxruntime.InferenceSession ready for inference.

        Raises:
            ImportError: If onnxruntime is not installed.
            FileNotFoundError: If the .onnx file does not exist.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required. Install with: uv add onnxruntime"
            ) from exc

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {path}")

        return ort.InferenceSession(str(path))

    def validate_parity(
        self,
        onnx_path: Path,
        sklearn_model: Any,
        X_test: pd.DataFrame | np.ndarray,
        *,
        threshold: float = _PARITY_TOLERANCE,
    ) -> ParityReport:
        """Compare ONNX and sklearn model outputs on the same test set.

        Verifies that float32 conversion and operator mapping have not
        introduced unacceptable numeric drift.

        Args:
            onnx_path:      Path to the .onnx file.
            sklearn_model:  Original trained model (with predict_proba).
            X_test:         Test input data (DataFrame or ndarray).
            threshold:      Maximum tolerated absolute probability difference.

        Returns:
            ParityReport with max_diff, mean_diff, passed status.
        """
        session = self.load_onnx_session(onnx_path)

        if isinstance(X_test, pd.DataFrame):
            X_np = X_test.to_numpy(dtype=np.float32)
        else:
            X_np = np.asarray(X_test, dtype=np.float32)

        # ONNX inference
        input_name = session.get_inputs()[0].name
        ort_outputs = session.run(None, {input_name: X_np})

        # Prefer 2D [N, 2] probability output over 1D label output
        proba_ort = None
        fallback_1d = None
        for out in ort_outputs:
            arr = np.array(out)
            if arr.ndim == 2 and arr.shape[1] == 2:
                proba_ort = arr[:, 1]  # positive class probability
                break
            elif arr.ndim == 1 and len(arr) == len(X_np) and fallback_1d is None:
                fallback_1d = arr  # keep as fallback, keep scanning for 2D

        if proba_ort is None:
            proba_ort = fallback_1d if fallback_1d is not None else np.array(ort_outputs[-1]).ravel()

        # sklearn inference
        proba_sklearn = sklearn_model.predict_proba(X_test)[:, 1].astype(np.float32)

        diffs = np.abs(proba_ort - proba_sklearn)
        max_diff = float(diffs.max())
        mean_diff = float(diffs.mean())
        passed = max_diff <= threshold

        if not passed:
            log.warning("Parity check FAILED: max_diff=%.6f > threshold=%.6f", max_diff, threshold)
        else:
            log.info("Parity check passed: max_diff=%.6f", max_diff)

        return ParityReport(
            max_diff=max_diff,
            mean_diff=mean_diff,
            n_rows_tested=len(X_np),
            passed=passed,
            threshold=threshold,
        )

    def compute_sha256(self, path: Path) -> str:
        """Compute the SHA-256 hex digest of a file.

        Args:
            path: File to hash.

        Returns:
            Lowercase hex string (64 characters).
        """
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def verify_sha256(self, path: Path, expected: str) -> bool:
        """Verify that a file's SHA-256 checksum matches the expected value.

        Args:
            path:     File to verify.
            expected: Expected SHA-256 hex digest.

        Returns:
            True if checksums match, False otherwise.
        """
        actual = self.compute_sha256(path)
        match = actual == expected.lower()
        if not match:
            log.error(
                "Checksum mismatch for %s: expected=%s, actual=%s",
                path, expected[:12], actual[:12],
            )
        return match

    def assess_pickle_risk(self, path: Path) -> PickleRiskReport:
        """Assess whether a file is a pickle and report the risk level.

        Production rule: pickle is "high" risk when loaded from any path
        that could be influenced by external input (network, user upload, etc.).
        Internal CI-produced artifacts are "low" risk but should still be
        migrated to ONNX.

        Args:
            path: Path to the model artifact to inspect.

        Returns:
            PickleRiskReport with risk_level and recommendation.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        is_pickle = suffix in {".pkl", ".pickle", ".joblib"}

        if not is_pickle:
            return PickleRiskReport(
                path=path,
                is_pickle=False,
                risk_level="none",
                recommendation="No pickle detected. Safe to serve.",
            )

        # Additional check: peek at magic bytes
        try:
            with open(path, "rb") as f:
                magic = f.read(2)
            # Pickle magic bytes: \x80\x04 (protocol 4), \x80\x05 (protocol 5)
            is_confirmed_pickle = magic[0:1] == b"\x80"
        except (OSError, IndexError):
            is_confirmed_pickle = is_pickle

        if is_confirmed_pickle:
            return PickleRiskReport(
                path=path,
                is_pickle=True,
                risk_level="high",
                recommendation=(
                    "Pickle artifact detected. Export to ONNX before serving. "
                    "Never deserialise pickle from untrusted sources — "
                    "it executes arbitrary code on load."
                ),
            )

        return PickleRiskReport(
            path=path,
            is_pickle=True,
            risk_level="low",
            recommendation=(
                "File has pickle extension. Verify it was produced by your own CI pipeline "
                "and consider migrating to ONNX for production serving."
            ),
        )
