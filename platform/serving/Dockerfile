# Credit Risk Inference API — Multi-stage Docker build
#
# Stage 1 (builder): install all Python deps into a virtual env
# Stage 2 (runtime): copy only the .venv and app code; run as non-root
#
# Build:
#   docker buildx build -f serving/Dockerfile -t credit-risk-api:latest .
#
# Run locally:
#   docker run -p 8080:8080 \
#     -e MODEL_PATH=/app/models/credit_risk_model.onnx \
#     -e MODEL_VERSION=v1.0.0 \
#     -v $(pwd)/models:/app/models:ro \
#     credit-risk-api:latest
#
# Security:
#   - Non-root user (appuser, UID 1001)
#   - No build tools in runtime image
#   - Model artifact mounted at runtime, not baked in
#   - Scan: trivy image --severity HIGH,CRITICAL credit-risk-api:latest

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv==0.4.28

# Copy dependency manifests only (cache layer invalidated only when deps change)
COPY pyproject.toml uv.lock ./

# Install production dependencies into .venv
# --no-dev excludes test, lint, notebook packages
RUN uv sync --no-dev --frozen

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root user to limit container escape blast radius (PCI-DSS, SOC 2)
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid 1001 --no-create-home --shell /bin/false appuser

# Copy virtual environment from builder (no pip, no build tools)
COPY --from=builder /app/.venv /app/.venv

# Copy application source code
COPY serving/ /app/serving/
COPY data/ /app/data/
COPY monitoring/ /app/monitoring/

# Model artifact is mounted at runtime — not baked into the image.
# Mounting keeps model updates independent of image rebuilds.
# Override with: -v /host/models:/app/models:ro
RUN mkdir -p /app/models && chown appuser:appgroup /app/models

# Activate the virtual environment for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Model configuration — override via K8s env vars or -e flags
    MODEL_PATH="/app/models/credit_risk_model.onnx" \
    MODEL_VERSION="unknown" \
    MODEL_THRESHOLD="0.5"

# Switch to non-root user before starting the server
USER appuser

EXPOSE 8080

# Health check so Docker/K8s can detect when the service is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

# Start the FastAPI server
CMD ["uvicorn", "serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--timeout-keep-alive", "30"]
