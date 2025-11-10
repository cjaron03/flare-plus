"""tests for ui server endpoints."""

import pytest
from datetime import datetime
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.ui.server import create_app, _parse_bucket, _throttle_message


@pytest.fixture
def mock_api_url():
    """mock api url."""
    return "http://localhost:5000"


@pytest.fixture
def ui_app(mock_api_url):
    """create ui app for testing."""
    with patch("src.ui.server.get_prediction_service") as mock_service:
        mock_service.return_value = ("api", mock_api_url, None)
        app = create_app(mock_api_url)
        yield app


@pytest.fixture
def client(ui_app):
    """create test client."""
    return TestClient(ui_app)


def test_parse_bucket():
    """test bucket parsing utility."""
    assert _parse_bucket("24h") == 24
    assert _parse_bucket("48h") == 48
    assert _parse_bucket("72h") == 72
    assert _parse_bucket("12-24h") == 12
    assert _parse_bucket("invalid") == 0
    assert _parse_bucket("") == 0


def test_throttle_message_no_last_refresh():
    """test throttle message when no refresh has occurred."""
    assert _throttle_message(None, min_minutes=5) == ""


def test_throttle_message_recent_refresh():
    """test throttle message when refresh was recent."""
    recent_time = datetime.utcnow()
    message = _throttle_message(recent_time, min_minutes=5)
    assert "throttled" in message.lower() or "wait" in message.lower()


def test_throttle_message_old_refresh():
    """test throttle message when refresh was old enough."""
    old_time = datetime.utcnow().replace(year=2020)
    assert _throttle_message(old_time, min_minutes=5) == ""


@patch("src.ui.server.get_latest_data_timestamps")
@patch("src.ui.server.calculate_data_freshness")
@patch("src.ui.server.get_api_model_status")
def test_status_endpoint(mock_status, mock_freshness, mock_timestamps, client):
    """test status endpoint."""
    mock_timestamps.return_value = {
        "flux_latest": datetime.utcnow(),
        "flux_count": 100,
        "regions_latest": datetime.utcnow(),
        "regions_count": 50,
        "flares_latest": datetime.utcnow(),
        "flares_count": 25,
    }
    mock_freshness.return_value = {"hours_ago": 1, "status": "fresh", "color": "green"}
    mock_status.return_value = {
        "classification": True,
        "survival": True,
        "confidence_level": "high",
        "survival_guardrail": False,
    }

    response = client.get("/ui/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "connection" in data
    assert "dataFreshness" in data
    assert "limitation" in data


@patch("src.ui.server.CONFIG")
def test_about_endpoint(mock_config, client):
    """test about endpoint."""
    mock_config.get.return_value = {
        "endpoints": {
            "noaa": "https://example.com",
            "swpc": "https://example.com",
        }
    }

    response = client.get("/ui/api/about")
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert "knownIssues" in data
    assert isinstance(data["knownIssues"], list)


@patch("src.ui.server.AdminConfig")
def test_admin_session_endpoint(mock_admin_config, client):
    """test admin session endpoint."""
    mock_admin_config.has_access.return_value = False
    mock_admin_config.status_indicator.return_value = "locked"
    mock_admin_config.disabled_reason.return_value = "no access"

    response = client.get("/ui/api/admin/session")
    assert response.status_code == 200
    data = response.json()
    assert "hasAccess" in data
    assert "indicator" in data
    assert "disabledReason" in data
    assert data["hasAccess"] is False


@patch("src.ui.server.AdminConfig")
def test_admin_logout_endpoint(mock_admin_config, client):
    """test admin logout endpoint."""
    mock_admin_config.has_access.return_value = False
    mock_admin_config.status_indicator.return_value = "locked"
    mock_admin_config.disabled_reason.return_value = "session cleared"

    response = client.post("/ui/api/admin/logout")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "message" in data
    mock_admin_config.revoke_session_access.assert_called_once()


def test_create_app_with_static_dir(tmp_path):
    """test app creation with static directory."""
    static_dir = tmp_path / "dist"
    static_dir.mkdir()
    (static_dir / "index.html").touch()
    (static_dir / "assets").mkdir()

    with patch("src.ui.server.get_prediction_service") as mock_service:
        mock_service.return_value = ("api", "http://localhost:5000", None)
        app = create_app("http://localhost:5000", static_dir=static_dir)
        assert app is not None


def test_create_app_without_static_dir():
    """test app creation without static directory."""
    with patch("src.ui.server.get_prediction_service") as mock_service:
        mock_service.return_value = ("api", "http://localhost:5000", None)
        app = create_app("http://localhost:5000")
        assert app is not None

