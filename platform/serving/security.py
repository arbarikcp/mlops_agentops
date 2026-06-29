"""Serving security: API key authentication, rate limiting, secrets config.

Production security layers:
    1. Authentication — every request carries a valid API key
    2. Authorization  — keys have roles; endpoints check required role
    3. Rate limiting  — per-client token bucket (429 on excess)
    4. Secrets config — loads from env vars, never from code

Usage in FastAPI:
    from serving.security import ApiKeyStore, RateLimiter, SecurityConfig

    store = ApiKeyStore.from_env()
    limiter = RateLimiter(max_requests=100, window_seconds=60)

    @app.post("/v1/predict")
    async def predict(
        request: PredictRequest,
        key: ApiKey = Depends(store.fastapi_dependency()),
    ):
        if not limiter.check(key.key_id):
            raise HTTPException(429, "Rate limit exceeded")
        ...

See: docs/phase4/day30_serving_security.md for theory.
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_HASH_ALGORITHM = "sha256"


def _hash_key(raw_key: str) -> str:
    """One-way SHA-256 hash of a raw API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── ApiKey ─────────────────────────────────────────────────────────────────────

@dataclass
class ApiKey:
    """An API key record stored in the ApiKeyStore.

    Attributes:
        key_id:     Unique identifier (human-readable, e.g. "service-a").
        key_hash:   SHA-256 hash of the raw key — never stored as plaintext.
        roles:      List of roles this key is granted.
        created_at: ISO-8601 UTC creation timestamp.
        expires_at: ISO-8601 UTC expiry, or None for non-expiring keys.
    """

    key_id: str
    key_hash: str
    roles: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now_iso)
    expires_at: str | None = None

    def is_valid(self) -> bool:
        """Return True if this key has not expired."""
        if self.expires_at is None:
            return True
        expiry = datetime.fromisoformat(self.expires_at)
        return datetime.now(timezone.utc) < expiry

    def has_role(self, role: str) -> bool:
        """Return True if this key has the given role."""
        return role in self.roles


# ── ApiKeyStore ────────────────────────────────────────────────────────────────

class ApiKeyStore:
    """In-memory store of ApiKey records, keyed by key_id.

    Keys are looked up by hashing the raw key and comparing to stored hashes.
    This means raw keys are never held in memory after hashing.

    Args:
        require_auth: If False, validate() always returns a guest key (dev mode).
    """

    def __init__(self, *, require_auth: bool = True) -> None:
        self._keys: dict[str, ApiKey] = {}
        self.require_auth = require_auth

    @classmethod
    def from_env(cls) -> "ApiKeyStore":
        """Build a store from environment variables.

        Environment variable format:
            API_KEYS=<key_id>:<raw_key>:<roles>, ...
            Example: API_KEYS=service-a:secret123:predictor,reader

        If API_KEYS is not set, returns a store with require_auth=False (dev mode).
        """
        raw = os.environ.get("API_KEYS", "")
        if not raw:
            log.warning("API_KEYS not set — authentication disabled (dev mode)")
            return cls(require_auth=False)

        store = cls(require_auth=True)
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":", 2)
            if len(parts) != 3:
                log.error("Invalid API_KEYS entry: %r — expected key_id:raw_key:roles", entry)
                continue
            key_id, raw_key, roles_str = parts
            roles = [r.strip() for r in roles_str.split(";") if r.strip()]
            store.add_key(key_id, raw_key, roles)

        return store

    def add_key(
        self,
        key_id: str,
        raw_key: str,
        roles: list[str],
        *,
        expires_at: str | None = None,
    ) -> None:
        """Add a new key to the store.

        Args:
            key_id:     Unique key identifier.
            raw_key:    The raw API key string (will be hashed immediately).
            roles:      List of role strings.
            expires_at: Optional ISO-8601 expiry timestamp.
        """
        if key_id in self._keys:
            raise ValueError(f"key_id {key_id!r} already exists")
        self._keys[key_id] = ApiKey(
            key_id=key_id,
            key_hash=_hash_key(raw_key),
            roles=roles,
            expires_at=expires_at,
        )
        log.info("Added API key: key_id=%s roles=%s", key_id, roles)

    def validate(self, raw_key: str) -> ApiKey | None:
        """Validate a raw key and return its ApiKey if valid.

        Args:
            raw_key: The raw key from the Authorization header.

        Returns:
            ApiKey if valid and not expired; None otherwise.
        """
        if not self.require_auth:
            return ApiKey(key_id="anonymous", key_hash="", roles=["predictor", "reader"])

        candidate_hash = _hash_key(raw_key)
        for api_key in self._keys.values():
            if api_key.key_hash == candidate_hash:
                if not api_key.is_valid():
                    log.warning("Rejected expired key: key_id=%s", api_key.key_id)
                    return None
                return api_key

        return None

    def revoke(self, key_id: str) -> bool:
        """Remove a key from the store. Returns True if removed."""
        if key_id in self._keys:
            del self._keys[key_id]
            log.info("Revoked key: key_id=%s", key_id)
            return True
        return False

    def list_key_ids(self) -> list[str]:
        """Return all key IDs (not raw keys or hashes)."""
        return list(self._keys.keys())

    @staticmethod
    def generate_raw_key(prefix: str = "sk") -> str:
        """Generate a cryptographically secure random API key."""
        return f"{prefix}-{secrets.token_urlsafe(32)}"


# ── RateLimiter ────────────────────────────────────────────────────────────────

class RateLimiter:
    """Per-client sliding-window rate limiter (in-process, not Redis).

    Uses a sliding window: counts requests in the last `window_seconds`.
    Thread-safety: not thread-safe. Use one instance per worker process.

    Args:
        max_requests:    Maximum requests allowed per window.
        window_seconds:  Length of the time window in seconds.
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 60.0) -> None:
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # client_id → list of request timestamps
        self._windows: dict[str, list[float]] = {}

    def check(self, client_id: str) -> bool:
        """Return True if the client is within the rate limit; False if exceeded.

        Also records this request (side effect: consumes a token if allowed).

        Args:
            client_id: Unique identifier for the client (API key ID or IP).

        Returns:
            True if allowed, False if rate limited.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        if client_id not in self._windows:
            self._windows[client_id] = []

        # Evict expired timestamps
        self._windows[client_id] = [t for t in self._windows[client_id] if t > cutoff]

        if len(self._windows[client_id]) >= self.max_requests:
            log.warning("Rate limit exceeded for client=%s", client_id)
            return False

        self._windows[client_id].append(now)
        return True

    def get_remaining(self, client_id: str) -> int:
        """Return how many more requests this client can make in the current window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        used = sum(1 for t in self._windows.get(client_id, []) if t > cutoff)
        return max(0, self.max_requests - used)

    def reset(self, client_id: str | None = None) -> None:
        """Clear rate limit state for a client or all clients."""
        if client_id:
            self._windows.pop(client_id, None)
        else:
            self._windows.clear()


# ── SecurityConfig ─────────────────────────────────────────────────────────────

@dataclass
class SecurityConfig:
    """Serving-layer security policy configuration.

    Loaded from environment variables or constructed in code.

    Attributes:
        require_auth:               Reject requests without valid API key.
        rate_limit_per_minute:      Requests per minute per client.
        allowed_roles_for_predict:  Roles that can call /v1/predict.
        allowed_roles_for_admin:    Roles that can call admin endpoints.
        log_request_body:           Log request body (NEVER in production).
        redact_auth_header:         Replace Authorization value in logs with ***.
    """

    require_auth: bool = True
    rate_limit_per_minute: int = 100
    allowed_roles_for_predict: list[str] = field(default_factory=lambda: ["predictor", "admin"])
    allowed_roles_for_admin: list[str] = field(default_factory=lambda: ["admin"])
    log_request_body: bool = False
    redact_auth_header: bool = True

    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Load configuration from environment variables."""
        return cls(
            require_auth=os.environ.get("REQUIRE_AUTH", "true").lower() == "true",
            rate_limit_per_minute=int(os.environ.get("RATE_LIMIT_PER_MIN", "100")),
            log_request_body=os.environ.get("LOG_REQUEST_BODY", "false").lower() == "true",
        )
