"""Tests for serving/serialization.py."""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from serving.serialization import (
    ModelSerializer,
    OnnxExportResult,
    ParityReport,
    PickleRiskReport,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def serializer() -> ModelSerializer:
    return ModelSerializer()


@pytest.fixture
def dummy_model():
    """Minimal sklearn-like model with predict_proba."""
    model = MagicMock()
    rng = np.random.default_rng(0)
    n = 50
    proba = rng.uniform(0.1, 0.9, n)
    model.predict_proba.return_value = np.column_stack([1 - proba, proba])
    return model


@pytest.fixture
def feature_names() -> list[str]:
    return [f"feat_{i}" for i in range(10)]


def _write_temp_file(content: bytes, suffix: str = ".pkl") -> Path:
    """Write bytes to a temp file and return its path."""
    import tempfile as _tmp
    fd, path = _tmp.mkstemp(suffix=suffix)
    with open(fd, "wb") as f:
        f.write(content)
    return Path(path)


# ── compute_sha256 ────────────────────────────────────────────────────────────

class TestComputeSha256:
    def test_returns_64_char_hex(self, serializer, tmp_path) -> None:
        p = tmp_path / "file.bin"
        p.write_bytes(b"hello")
        result = serializer.compute_sha256(p)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_matches_known_hash(self, serializer, tmp_path) -> None:
        p = tmp_path / "file.bin"
        p.write_bytes(b"hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert serializer.compute_sha256(p) == expected

    def test_different_content_different_hash(self, serializer, tmp_path) -> None:
        p1 = tmp_path / "a.bin"
        p2 = tmp_path / "b.bin"
        p1.write_bytes(b"aaa")
        p2.write_bytes(b"bbb")
        assert serializer.compute_sha256(p1) != serializer.compute_sha256(p2)

    def test_empty_file_has_known_hash(self, serializer, tmp_path) -> None:
        p = tmp_path / "empty.bin"
        p.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert serializer.compute_sha256(p) == expected


# ── verify_sha256 ─────────────────────────────────────────────────────────────

class TestVerifySha256:
    def test_correct_checksum_returns_true(self, serializer, tmp_path) -> None:
        p = tmp_path / "file.bin"
        p.write_bytes(b"data")
        digest = serializer.compute_sha256(p)
        assert serializer.verify_sha256(p, digest) is True

    def test_wrong_checksum_returns_false(self, serializer, tmp_path) -> None:
        p = tmp_path / "file.bin"
        p.write_bytes(b"data")
        assert serializer.verify_sha256(p, "a" * 64) is False

    def test_uppercase_input_accepted(self, serializer, tmp_path) -> None:
        p = tmp_path / "file.bin"
        p.write_bytes(b"data")
        digest = serializer.compute_sha256(p).upper()
        assert serializer.verify_sha256(p, digest) is True


# ── assess_pickle_risk ────────────────────────────────────────────────────────

class TestAssessPickleRisk:
    def test_onnx_file_no_risk(self, serializer, tmp_path) -> None:
        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00\x00")
        report = serializer.assess_pickle_risk(p)
        assert report.risk_level == "none"
        assert report.is_pickle is False

    def test_pkl_file_high_risk(self, serializer, tmp_path) -> None:
        p = tmp_path / "model.pkl"
        # Write valid pickle magic bytes
        p.write_bytes(b"\x80\x05" + b"\x00" * 10)
        report = serializer.assess_pickle_risk(p)
        assert report.is_pickle is True
        assert report.risk_level == "high"
        assert "ONNX" in report.recommendation

    def test_joblib_extension_detected(self, serializer, tmp_path) -> None:
        p = tmp_path / "model.joblib"
        p.write_bytes(b"\x80\x04" + b"\x00" * 10)
        report = serializer.assess_pickle_risk(p)
        assert report.is_pickle is True

    def test_pickle_extension_detected(self, serializer, tmp_path) -> None:
        p = tmp_path / "model.pickle"
        p.write_bytes(b"\x80\x04" + b"\x00" * 5)
        report = serializer.assess_pickle_risk(p)
        assert report.is_pickle is True

    def test_report_has_path(self, serializer, tmp_path) -> None:
        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00")
        report = serializer.assess_pickle_risk(p)
        assert report.path == p

    def test_returns_pickle_risk_report_type(self, serializer, tmp_path) -> None:
        p = tmp_path / "model.pkl"
        p.write_bytes(b"\x80\x05\x00")
        report = serializer.assess_pickle_risk(p)
        assert isinstance(report, PickleRiskReport)


# ── export_to_onnx (mocked) ───────────────────────────────────────────────────

class TestExportToOnnx:
    def test_raises_import_error_without_onnxmltools(self, serializer, dummy_model, feature_names, tmp_path) -> None:
        with patch.dict("sys.modules", {"onnxmltools": None}):
            with pytest.raises(ImportError, match="onnxmltools"):
                serializer.export_to_onnx(dummy_model, feature_names, tmp_path / "out.onnx")

    def test_export_creates_file_and_checksum(self, serializer, dummy_model, feature_names, tmp_path) -> None:
        """Test with full mock of onnxmltools + onnx."""
        mock_onnx_model = MagicMock()
        mock_onnx_model.SerializeToString.return_value = b"\x08\x07" + b"\x00" * 20  # fake onnx bytes

        mock_graph_input = MagicMock()
        mock_graph_input.name = "float_input"
        mock_graph_output = MagicMock()
        mock_graph_output.name = "probabilities"
        mock_onnx_model.graph.input = [mock_graph_input]
        mock_onnx_model.graph.output = [mock_graph_output]

        mock_onnx_load = MagicMock(return_value=mock_onnx_model)
        mock_convert = MagicMock(return_value=mock_onnx_model)
        mock_float_type = MagicMock()

        with patch.dict("sys.modules", {
            "onnxmltools": MagicMock(convert_lightgbm=mock_convert),
            "onnxmltools.convert.common.data_types": MagicMock(FloatTensorType=mock_float_type),
            "onnx": MagicMock(load=mock_onnx_load),
        }):
            out_path = tmp_path / "model.onnx"
            result = serializer.export_to_onnx(dummy_model, feature_names, out_path)

        assert out_path.exists()
        assert out_path.with_suffix(".onnx.sha256").exists()
        assert isinstance(result, OnnxExportResult)
        assert result.n_features == len(feature_names)
        assert result.opset == 17

    def test_export_with_custom_opset(self, serializer, dummy_model, feature_names, tmp_path) -> None:
        mock_onnx_model = MagicMock()
        mock_onnx_model.SerializeToString.return_value = b"\x08\x07" + b"\x00" * 20
        mock_onnx_model.graph.input = [MagicMock(name="float_input")]
        mock_onnx_model.graph.output = [MagicMock(name="output")]

        with patch.dict("sys.modules", {
            "onnxmltools": MagicMock(convert_lightgbm=MagicMock(return_value=mock_onnx_model)),
            "onnxmltools.convert.common.data_types": MagicMock(FloatTensorType=MagicMock()),
            "onnx": MagicMock(load=MagicMock(return_value=mock_onnx_model)),
        }):
            out_path = tmp_path / "model.onnx"
            result = serializer.export_to_onnx(dummy_model, feature_names, out_path, opset=12)

        assert result.opset == 12


# ── load_onnx_session (mocked) ────────────────────────────────────────────────

class TestLoadOnnxSession:
    def test_raises_import_error_without_onnxruntime(self, serializer, tmp_path) -> None:
        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00")
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(ImportError, match="onnxruntime"):
                serializer.load_onnx_session(p)

    def test_raises_file_not_found(self, serializer, tmp_path) -> None:
        with patch.dict("sys.modules", {"onnxruntime": MagicMock()}):
            with pytest.raises(FileNotFoundError):
                serializer.load_onnx_session(tmp_path / "nonexistent.onnx")

    def test_returns_session_on_valid_file(self, serializer, tmp_path) -> None:
        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00")
        mock_session = MagicMock()
        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            result = serializer.load_onnx_session(p)

        assert result is mock_session


# ── validate_parity ───────────────────────────────────────────────────────────

class TestValidateParity:
    def _make_session_mock(self, proba: np.ndarray) -> MagicMock:
        """Mock ORT session that returns [labels, probabilities]."""
        session = MagicMock()
        n = len(proba)
        session.get_inputs.return_value = [MagicMock(name="float_input")]
        labels = (proba > 0.5).astype(np.int64)
        proba_2d = np.column_stack([1 - proba, proba])
        session.run.return_value = [labels, proba_2d]
        return session

    def test_passes_for_identical_outputs(self, serializer, tmp_path) -> None:
        rng = np.random.default_rng(0)
        proba = rng.uniform(0.1, 0.9, 50)

        model = MagicMock()
        model.predict_proba.return_value = np.column_stack([1 - proba, proba])
        session = self._make_session_mock(proba)

        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00")

        with patch.object(serializer, "load_onnx_session", return_value=session):
            report = serializer.validate_parity(p, model, np.random.rand(50, 5).astype(np.float32))

        assert report.passed is True
        assert isinstance(report, ParityReport)

    def test_fails_for_large_divergence(self, serializer, tmp_path) -> None:
        rng = np.random.default_rng(1)
        proba_sklearn = rng.uniform(0.1, 0.9, 50)
        proba_onnx = 1 - proba_sklearn   # completely inverted → max diff ≈ 1

        model = MagicMock()
        model.predict_proba.return_value = np.column_stack([1 - proba_sklearn, proba_sklearn])
        session = self._make_session_mock(proba_onnx)

        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00")

        with patch.object(serializer, "load_onnx_session", return_value=session):
            report = serializer.validate_parity(p, model, np.random.rand(50, 5).astype(np.float32))

        assert report.passed is False
        assert report.max_diff > 0.5

    def test_report_has_correct_n_rows(self, serializer, tmp_path) -> None:
        n = 30
        rng = np.random.default_rng(2)
        proba = rng.uniform(0.1, 0.9, n)
        model = MagicMock()
        model.predict_proba.return_value = np.column_stack([1 - proba, proba])
        session = self._make_session_mock(proba)
        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00")

        with patch.object(serializer, "load_onnx_session", return_value=session):
            report = serializer.validate_parity(p, model, np.random.rand(n, 5).astype(np.float32))

        assert report.n_rows_tested == n

    def test_custom_threshold_respected(self, serializer, tmp_path) -> None:
        rng = np.random.default_rng(3)
        proba = rng.uniform(0.1, 0.9, 50)
        X = np.random.default_rng(99).random((50, 5)).astype(np.float32)

        model = MagicMock()
        model.predict_proba.return_value = np.column_stack([1 - proba, proba])
        p = tmp_path / "model.onnx"
        p.write_bytes(b"\x00")

        # Perturbation = 0.0005: fails tight (0.0001) but passes loose (0.01)
        proba_onnx_perturbed = proba + 0.0005
        session_perturbed = self._make_session_mock(proba_onnx_perturbed)

        with patch.object(serializer, "load_onnx_session", return_value=session_perturbed):
            report_tight = serializer.validate_parity(p, model, X, threshold=0.0001)

        with patch.object(serializer, "load_onnx_session", return_value=session_perturbed):
            report_loose = serializer.validate_parity(p, model, X, threshold=0.01)

        assert report_tight.passed is False
        assert report_loose.passed is True
