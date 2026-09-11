"""Password hashing and signed access tokens for API authentication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt

from multi_agent_rag.persistence.models import UserRecord

EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
PASSWORD_SCHEME = "scrypt"
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_LENGTH = 32
DEVELOPMENT_SECRET = "local-development-secret-change-before-production"


class AuthenticationError(ValueError):
    """Raised when credentials or access tokens are invalid."""


class AuthenticationConfigurationError(RuntimeError):
    """Raised when required authentication settings are unsafe or missing."""


@dataclass(frozen=True)
class TokenClaims:
    """Validated identity claims extracted from an access token."""

    user_id: str
    email: str
    display_name: str


def authentication_required() -> bool:
    return _env_flag("AUTH_REQUIRED", False)


def registration_enabled() -> bool:
    return _env_flag("AUTH_ALLOW_REGISTRATION", True)


def validate_authentication_configuration() -> None:
    _jwt_secret()
    _token_lifetime_seconds()


def normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if not EMAIL_PATTERN.fullmatch(normalized):
        raise AuthenticationError("A valid email address is required.")
    return normalized


def hash_password(password: str) -> str:
    _validate_password(password)
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        dklen=SCRYPT_LENGTH,
    )
    return "$".join(
        (
            PASSWORD_SCHEME,
            str(SCRYPT_N),
            str(SCRYPT_R),
            str(SCRYPT_P),
            _encode(salt),
            _encode(digest),
        )
    )


def verify_password(password: str, encoded_password: str) -> bool:
    try:
        scheme, n, r, p, salt, expected = encoded_password.split("$", 5)
        if scheme != PASSWORD_SCHEME:
            return False
        expected_digest = _decode(expected)
        actual_digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=_decode(salt),
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected_digest),
        )
        return hmac.compare_digest(actual_digest, expected_digest)
    except (TypeError, ValueError):
        return False


def issue_access_token(user: UserRecord, now: datetime | None = None) -> tuple[str, int]:
    issued_at = now or datetime.now(UTC)
    expires_seconds = _token_lifetime_seconds()
    expires_at = issued_at + timedelta(seconds=expires_seconds)
    payload = {
        "sub": user.user_id,
        "email": user.email,
        "name": user.display_name,
        "iat": issued_at,
        "exp": expires_at,
        "iss": os.getenv("JWT_ISSUER") or "multi-agent-rag",
        "aud": os.getenv("JWT_AUDIENCE") or "multi-agent-rag-api",
    }
    token = jwt.encode(payload, _jwt_secret(), algorithm="HS256")
    return token, expires_seconds


def decode_access_token(token: str) -> TokenClaims:
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=["HS256"],
            issuer=os.getenv("JWT_ISSUER") or "multi-agent-rag",
            audience=os.getenv("JWT_AUDIENCE") or "multi-agent-rag-api",
        )
        user_id = str(payload.get("sub") or "").strip()
        email = normalize_email(str(payload.get("email") or ""))
        if not user_id:
            raise AuthenticationError("Access token subject is missing.")
        return TokenClaims(user_id, email, str(payload.get("name") or "").strip())
    except jwt.PyJWTError as exc:
        raise AuthenticationError("Access token is invalid or expired.") from exc


def _validate_password(password: str) -> None:
    if len(password) < 8:
        raise AuthenticationError("Password must contain at least 8 characters.")
    if len(password) > 128:
        raise AuthenticationError("Password must not exceed 128 characters.")


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET") or DEVELOPMENT_SECRET
    if authentication_required() and secret == DEVELOPMENT_SECRET:
        raise AuthenticationConfigurationError("JWT_SECRET must be configured when authentication is required.")
    if len(secret) < 32:
        raise AuthenticationConfigurationError("JWT_SECRET must contain at least 32 characters.")
    return secret


def _token_lifetime_seconds() -> int:
    try:
        minutes = int(os.getenv("JWT_ACCESS_TOKEN_MINUTES") or "60")
    except ValueError as exc:
        raise AuthenticationConfigurationError("JWT access token lifetime must be an integer.") from exc
    if minutes <= 0:
        raise AuthenticationConfigurationError("JWT access token lifetime must be greater than zero.")
    return minutes * 60


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
