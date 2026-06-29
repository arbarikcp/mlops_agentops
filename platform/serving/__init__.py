"""Serving module — Phase 4: Packaging & Serving (Days 22–30).

Modules:
    serialization   — ONNX export, pickle-risk checks, checksum verification
    inference       — Online / batch / streaming inference abstractions
    api             — FastAPI application (versioned endpoints, health/readiness)
    batch_inference — Idempotent batch scoring jobs with manifest
    api_contract    — Request/response schema versioning and compatibility
    security        — API key auth, rate limiting, secrets management
    load_test       — Locust scenarios and latency profiling utilities
"""
