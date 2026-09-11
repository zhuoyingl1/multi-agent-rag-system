from datetime import UTC, datetime

import pytest

from multi_agent_rag.auth import (
    AuthenticationConfigurationError,
    decode_access_token,
    hash_password,
    issue_access_token,
    normalize_email,
    validate_authentication_configuration,
    verify_password,
)
from multi_agent_rag.persistence import UserRecord


def fake_user(password_hash: str = "stored-hash") -> UserRecord:
    now = datetime.now(UTC)
    return UserRecord(
        user_id="user-123",
        email="researcher@example.com",
        display_name="Researcher",
        password_hash=password_hash,
        active=True,
        created_at=now,
        updated_at=now,
    )


def test_password_hash_round_trip_uses_random_salts() -> None:
    first = hash_password("correct-horse-battery")
    second = hash_password("correct-horse-battery")

    assert first != second
    assert verify_password("correct-horse-battery", first) is True
    assert verify_password("wrong-password", first) is False
    assert "correct-horse-battery" not in first


def test_access_token_round_trip(monkeypatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "test-secret-that-is-at-least-32-characters")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_MINUTES", "30")

    token, expires_in = issue_access_token(fake_user())
    claims = decode_access_token(token)

    assert expires_in == 1800
    assert claims.user_id == "user-123"
    assert claims.email == "researcher@example.com"


def test_production_auth_requires_explicit_secret(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    monkeypatch.delenv("JWT_SECRET", raising=False)

    with pytest.raises(AuthenticationConfigurationError, match="JWT_SECRET"):
        validate_authentication_configuration()


def test_email_normalization_rejects_invalid_value() -> None:
    assert normalize_email(" Researcher@Example.COM ") == "researcher@example.com"

    with pytest.raises(ValueError, match="valid email"):
        normalize_email("not-an-email")
