"""tests for AdminConfig login behaviour."""

import pytest

from src.config import AdminConfig


@pytest.fixture(autouse=True)
def reset_admin_config(monkeypatch):
    """ensure AdminConfig session state is reset between tests."""
    monkeypatch.setattr(AdminConfig, "_session_granted", False)
    yield
    AdminConfig.revoke_session_access()


def test_admin_login_flow(monkeypatch):
    """UI credentials should unlock and re-lock admin access."""
    monkeypatch.setattr(AdminConfig, "ACCESS_TOKEN", None)
    monkeypatch.setattr(AdminConfig, "RUNTIME_TOKEN", None)

    assert AdminConfig.has_access() is False
    assert "Login" in AdminConfig.disabled_reason()

    # wrong credentials should not grant access
    assert AdminConfig.validate_credentials("wrong", "bad") is False
    assert AdminConfig.has_access() is False

    # correct credentials unlock session
    assert AdminConfig.validate_credentials("plncake", "12345") is True
    assert AdminConfig.has_access() is True
    assert "UI session" in AdminConfig.status_indicator()

    AdminConfig.revoke_session_access()
    assert AdminConfig.has_access() is False
