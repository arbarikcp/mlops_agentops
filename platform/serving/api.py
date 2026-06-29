"""FastAPI application for credit risk online inference.

Endpoints:
    GET  /health           — liveness probe (always 200 if process is alive)
    GET  /ready            — readiness probe (503 until model is loaded)
    POST /v1/predict       — single-row inference
    POST /v1/predict/batch — multi-row inference (≤ 1000 rows)
    GET  /v1/model/info    — model version and configuration metadata

Design decisions:
    - Model is loaded once in the `lifespan` startup handler, not per-request.
    - Pydantic v2 strict validation rejects NaN/Inf at the boundary.
    - /health vs /ready are separate so K8s can distinguish liveness from readiness.
    - URL versioning (/v1/) enables parallel deployment of v1 and v2 during migration.

See: docs/phase4/day24_fastapi.md for theory.

Usage:
    uvicorn serving.api:app --host 0.0.0.0 --port 8080

    # With model path override:
    MODEL_PATH=models/credit_risk_model.onnx uvicorn serving.api:app
"""
from __future__ import annotations

import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from serving.inference import ModelRunner

log = logging.getLogger(__name__)

_MAX_BATCH_SIZE = 1000

# ── Application state ──────────────────────────────────────────────────────────

_runner: ModelRunner | None = None
_model_ready: bool = False
_startup_time: float = time.time()


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Single-row prediction request.

    Attributes:
        applicant_id: Unique applicant identifier (must be > 0).
        features:     Feature dict mapping column name to float value.
    """

    model_config = ConfigDict(strict=True)

    applicant_id: int = Field(gt=0, description="Applicant identifier")
    features: dict[str, float] = Field(min_length=1, description="Feature name → value")

    @model_validator(mode="after")
    def check_no_nan_inf(self) -> "PredictRequest":
        for k, v in self.features.items():
            if math.isnan(v):
                raise ValueError(f"NaN not allowed in feature '{k}'")
            if math.isinf(v):
                raise ValueError(f"Inf not allowed in feature '{k}'")
        return self


class PredictResponse(BaseModel):
    """Single-row prediction response."""

    applicant_id: int
    score: float = Field(ge=0.0, le=1.0, description="Positive-class probability")
    label: int = Field(ge=0, le=1, description="Binary decision (1=default predicted)")
    model_version: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    """Multi-row prediction request (up to MAX_BATCH_SIZE rows)."""

    model_config = ConfigDict(strict=True)

    rows: list[PredictRequest] = Field(min_length=1, max_length=_MAX_BATCH_SIZE)


class BatchPredictResponse(BaseModel):
    """Multi-row prediction response."""

    predictions: list[PredictResponse]
    n_rows: int
    model_version: str
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Liveness probe response."""

    status: str
    uptime_seconds: float


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    ready: bool
    model_version: str | None


class ModelInfoResponse(BaseModel):
    """Model metadata response."""

    model_version: str
    model_path: str
    threshold: float
    feature_names: list[str] | None
    ready: bool


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the model on startup; release on shutdown."""
    global _runner, _model_ready

    model_path = Path(os.environ.get("MODEL_PATH", "models/credit_risk_model.onnx"))
    model_version = os.environ.get("MODEL_VERSION", "unknown")
    threshold = float(os.environ.get("MODEL_THRESHOLD", "0.5"))

    try:
        _runner = ModelRunner(
            model_path=model_path,
            model_version=model_version,
            threshold=threshold,
        )
        _runner.load()
        _model_ready = True
        log.info("Model loaded: version=%s path=%s", model_version, model_path)
    except Exception as exc:
        log.error("Failed to load model: %s", exc)
        _model_ready = False

    yield

    _model_ready = False
    log.info("Shutting down API")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Credit Risk Inference API",
    description="Online inference service for the credit-default risk model.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Middleware ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> Any:
    """Log method, path, status code and latency for every request."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    log.info(
        "%s %s → %d  (%.1fms)",
        request.method, request.url.path, response.status_code, elapsed_ms,
    )
    return response


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness probe — always returns 200 while the process is alive."""
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.time() - _startup_time, 1),
    )


@app.get("/ready", response_model=ReadinessResponse, tags=["ops"])
async def ready() -> JSONResponse:
    """Readiness probe — returns 503 until the model is fully loaded."""
    if not _model_ready or _runner is None:
        return JSONResponse(
            status_code=503,
            content=ReadinessResponse(ready=False, model_version=None).model_dump(),
        )
    return JSONResponse(
        status_code=200,
        content=ReadinessResponse(ready=True, model_version=_runner.model_version).model_dump(),
    )


@app.get("/v1/model/info", response_model=ModelInfoResponse, tags=["v1"])
async def model_info() -> ModelInfoResponse:
    """Return model version, path, and configuration metadata."""
    if _runner is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return ModelInfoResponse(
        model_version=_runner.model_version,
        model_path=str(_runner.model_path),
        threshold=_runner.threshold,
        feature_names=_runner.feature_names,
        ready=_model_ready,
    )


@app.post("/v1/predict", response_model=PredictResponse, tags=["v1"])
async def predict(request: PredictRequest) -> PredictResponse:
    """Score a single applicant and return probability + binary label.

    Returns 503 if model is not yet loaded, 422 if input is invalid.
    """
    if not _model_ready or _runner is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    try:
        result = _runner.predict_single(
            request.features,
            applicant_id=request.applicant_id,
        )
    except Exception as exc:
        log.exception("Inference error for applicant_id=%d", request.applicant_id)
        raise HTTPException(status_code=500, detail="Inference failed") from exc

    return PredictResponse(
        applicant_id=result.applicant_id,
        score=result.score,
        label=result.label,
        model_version=result.model_version,
        latency_ms=result.latency_ms,
    )


@app.post("/v1/predict/batch", response_model=BatchPredictResponse, tags=["v1"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """Score multiple applicants in one request (max 1000 rows).

    Returns results in the same order as the input rows.
    """
    if not _model_ready or _runner is None:
        raise HTTPException(status_code=503, detail="Model not ready")

    start = time.perf_counter()
    predictions: list[PredictResponse] = []

    try:
        for row in request.rows:
            result = _runner.predict_single(row.features, applicant_id=row.applicant_id)
            predictions.append(PredictResponse(
                applicant_id=result.applicant_id,
                score=result.score,
                label=result.label,
                model_version=result.model_version,
                latency_ms=result.latency_ms,
            ))
    except Exception as exc:
        log.exception("Batch inference error")
        raise HTTPException(status_code=500, detail="Batch inference failed") from exc

    total_ms = (time.perf_counter() - start) * 1000
    model_ver = _runner.model_version if _runner else "unknown"

    return BatchPredictResponse(
        predictions=predictions,
        n_rows=len(predictions),
        model_version=model_ver,
        total_latency_ms=total_ms,
    )
