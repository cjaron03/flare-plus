"""tests for AdminConfig login behaviour."""

import time

import pytest

from src.config import AdminConfig


@pytest.fixture(autouse=True)
def reset_admin_config(monkeypatch):
    """ensure AdminConfig session state is reset between tests."""
    monkeypatch.setattr(AdminConfig, "_session_granted", False)
    AdminConfig.revoke_session_access()
    AdminConfig._failed_attempts = []
    AdminConfig._locked_until = 0.0
    yield
    AdminConfig.revoke_session_access()
    AdminConfig._failed_attempts = []
    AdminConfig._locked_until = 0.0


def test_admin_login_flow(monkeypatch):
    """UI credentials should unlock and re-lock admin access."""
    monkeypatch.setattr(AdminConfig, "ACCESS_TOKEN", None)
    monkeypatch.setattr(AdminConfig, "RUNTIME_TOKEN", None)
    monkeypatch.setattr(AdminConfig, "LOGIN_ENABLED", True)

    assert AdminConfig.has_access() is False
    assert "Login" in AdminConfig.disabled_reason()

    success, _ = AdminConfig.validate_credentials("wrong", "bad")
    assert success is False
    assert AdminConfig.has_access() is False

    success, message = AdminConfig.validate_credentials("plncake", "12345")
    assert success is True
    assert "successful" in message.lower()
    assert AdminConfig.has_access() is True
    assert "UI session" in AdminConfig.status_indicator()

    AdminConfig.revoke_session_access()
    assert AdminConfig.has_access() is False


def test_admin_login_toggle(monkeypatch):
    """UI login can be globally disabled."""
    monkeypatch.setattr(AdminConfig, "LOGIN_ENABLED", False)

    success, message = AdminConfig.validate_credentials("plncake", "12345")
    assert success is False
    assert "disabled" in message.lower()


def test_admin_login_rate_limit(monkeypatch):
    """too many failures trigger a temporary lockout."""
    monkeypatch.setattr(AdminConfig, "ACCESS_TOKEN", None)
    monkeypatch.setattr(AdminConfig, "RUNTIME_TOKEN", None)
    monkeypatch.setattr(AdminConfig, "LOGIN_ENABLED", True)
    monkeypatch.setattr(AdminConfig, "MAX_LOGIN_ATTEMPTS", 2)
    monkeypatch.setattr(AdminConfig, "LOGIN_LOCKOUT_SECONDS", 1)
    monkeypatch.setattr(AdminConfig, "LOGIN_WINDOW_SECONDS", 60)
    AdminConfig._failed_attempts = []
    AdminConfig._locked_until = 0.0

    AdminConfig.validate_credentials("wrong", "bad")
    AdminConfig.validate_credentials("wrong", "bad")
    success, message = AdminConfig.validate_credentials("plncake", "12345")
    assert success is False
    assert "try again" in message.lower()

    AdminConfig._locked_until = time.time() - 1
    success, _ = AdminConfig.validate_credentials("plncake", "12345")
    assert success is True
