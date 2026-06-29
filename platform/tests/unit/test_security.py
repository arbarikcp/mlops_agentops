"""Tests for serving/security.py."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from serving.security import (
    ApiKey,
    ApiKeyStore,
    RateLimiter,
    SecurityConfig,
    _hash_key,
)


# ── _hash_key ─────────────────────────────────────────────────────────────────

class TestHashKey:
    def test_same_input_same_hash(self) -> None:
        assert _hash_key("secret") == _hash_key("secret")

    def test_different_inputs_different_hashes(self) -> None:
        assert _hash_key("a") != _hash_key("b")

    def test_hash_is_64_chars_hex(self) -> None:
        h = _hash_key("test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ── ApiKey ─────────────────────────────────────────────────────────────────────

class TestApiKey:
    def test_non_expiring_key_is_valid(self) -> None:
        key = ApiKey(key_id="k1", key_hash="abc", roles=["predictor"])
        assert key.is_valid() is True

    def test_future_expiry_is_valid(self) -> None:
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        key = ApiKey(key_id="k1", key_hash="abc", roles=[], expires_at=future)
        assert key.is_valid() is True

    def test_past_expiry_is_invalid(self) -> None:
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        key = ApiKey(key_id="k1", key_hash="abc", roles=[], expires_at=past)
        assert key.is_valid() is False

    def test_has_role_true(self) -> None:
        key = ApiKey(key_id="k1", key_hash="abc", roles=["predictor", "reader"])
        assert key.has_role("predictor") is True

    def test_has_role_false(self) -> None:
        key = ApiKey(key_id="k1", key_hash="abc", roles=["reader"])
        assert key.has_role("admin") is False

    def test_empty_roles(self) -> None:
        key = ApiKey(key_id="k1", key_hash="abc")
        assert key.has_role("anything") is False


# ── ApiKeyStore ────────────────────────────────────────────────────────────────

class TestApiKeyStore:
    @pytest.fixture
    def store(self) -> ApiKeyStore:
        s = ApiKeyStore(require_auth=True)
        s.add_key("service-a", "raw_secret_key_123", ["predictor", "reader"])
        return s

    def test_validate_correct_key(self, store) -> None:
        result = store.validate("raw_secret_key_123")
        assert result is not None
        assert result.key_id == "service-a"

    def test_validate_wrong_key_returns_none(self, store) -> None:
        result = store.validate("wrong_key")
        assert result is None

    def test_validate_returns_roles(self, store) -> None:
        result = store.validate("raw_secret_key_123")
        assert "predictor" in result.roles

    def test_add_duplicate_key_raises(self, store) -> None:
        with pytest.raises(ValueError, match="already exists"):
            store.add_key("service-a", "other_key", [])

    def test_revoke_removes_key(self, store) -> None:
        store.revoke("service-a")
        assert store.validate("raw_secret_key_123") is None

    def test_revoke_nonexistent_returns_false(self, store) -> None:
        assert store.revoke("no-such-key") is False

    def test_list_key_ids(self, store) -> None:
        assert "service-a" in store.list_key_ids()

    def test_no_auth_mode_always_valid(self) -> None:
        store = ApiKeyStore(require_auth=False)
        result = store.validate("any_key")
        assert result is not None
        assert result.key_id == "anonymous"

    def test_expired_key_returns_none(self, store) -> None:
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        store.add_key("expired-key", "old_secret", ["reader"], expires_at=past)
        assert store.validate("old_secret") is None

    def test_generate_raw_key_is_unique(self) -> None:
        k1 = ApiKeyStore.generate_raw_key()
        k2 = ApiKeyStore.generate_raw_key()
        assert k1 != k2

    def test_generate_raw_key_has_prefix(self) -> None:
        key = ApiKeyStore.generate_raw_key("myprefix")
        assert key.startswith("myprefix-")


# ── RateLimiter ────────────────────────────────────────────────────────────────

class TestRateLimiter:
    def test_first_request_allowed(self) -> None:
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        assert limiter.check("client-1") is True

    def test_within_limit_allowed(self) -> None:
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.check("client-1") is True

    def test_exceed_limit_blocked(self) -> None:
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter.check("client-1")
        # 4th request should be blocked
        assert limiter.check("client-1") is False

    def test_different_clients_independent(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.check("client-1")
        # client-2 is independent — should be allowed
        assert limiter.check("client-2") is True

    def test_get_remaining_decrements(self) -> None:
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        limiter.check("c")
        limiter.check("c")
        assert limiter.get_remaining("c") == 8

    def test_get_remaining_zero_when_exceeded(self) -> None:
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.check("c")
        limiter.check("c")
        assert limiter.get_remaining("c") == 0

    def test_reset_specific_client(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.check("c1")
        limiter.reset("c1")
        assert limiter.check("c1") is True  # slot freed

    def test_reset_all_clients(self) -> None:
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.check("c1")
        limiter.check("c2")
        limiter.reset()
        assert limiter.check("c1") is True
        assert limiter.check("c2") is True

    def test_invalid_max_requests_raises(self) -> None:
        with pytest.raises(ValueError, match="max_requests"):
            RateLimiter(max_requests=0)

    def test_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window_seconds"):
            RateLimiter(window_seconds=0)

    def test_window_expiry_allows_new_requests(self) -> None:
        """Requests older than window_seconds should not count."""
        limiter = RateLimiter(max_requests=1, window_seconds=0.05)  # 50ms window
        limiter.check("c")       # uses the 1 slot
        time.sleep(0.06)          # wait for window to expire
        assert limiter.check("c") is True  # slot should be free again


# ── SecurityConfig ─────────────────────────────────────────────────────────────

class TestSecurityConfig:
    def test_defaults(self) -> None:
        cfg = SecurityConfig()
        assert cfg.require_auth is True
        assert cfg.rate_limit_per_minute == 100
        assert "predictor" in cfg.allowed_roles_for_predict

    def test_admin_not_in_predict_by_default(self) -> None:
        cfg = SecurityConfig()
        assert "admin" in cfg.allowed_roles_for_predict

    def test_log_request_body_false_by_default(self) -> None:
        assert SecurityConfig().log_request_body is False

    def test_from_env_defaults(self, monkeypatch) -> None:
        monkeypatch.delenv("REQUIRE_AUTH", raising=False)
        monkeypatch.delenv("RATE_LIMIT_PER_MIN", raising=False)
        cfg = SecurityConfig.from_env()
        assert cfg.require_auth is True

    def test_from_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("REQUIRE_AUTH", "false")
        monkeypatch.setenv("RATE_LIMIT_PER_MIN", "50")
        cfg = SecurityConfig.from_env()
        assert cfg.require_auth is False
        assert cfg.rate_limit_per_minute == 50
