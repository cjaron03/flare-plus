"""API authentication for flare+ model serving."""

import hashlib
import logging
import os
from functools import wraps
from typing import Set

from flask import request, jsonify

logger = logging.getLogger(__name__)


def _get_valid_api_key_hashes() -> Set[str]:
    """
    Load valid API key hashes from environment.

    API_KEYS should be a comma-separated list of keys.
    Keys are hashed with SHA256 for secure comparison.
    """
    api_keys_str = os.getenv("API_KEYS", "")
    if not api_keys_str:
        return set()

    keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
    return {hashlib.sha256(k.encode()).hexdigest() for k in keys}


# Cache valid key hashes at module load (refresh on app restart)
_VALID_KEY_HASHES: Set[str] = set()


def _load_api_keys():
    """Load or reload API keys from environment."""
    global _VALID_KEY_HASHES
    _VALID_KEY_HASHES = _get_valid_api_key_hashes()
    if _VALID_KEY_HASHES:
        logger.info(f"Loaded {len(_VALID_KEY_HASHES)} API key(s)")
    else:
        logger.warning("No API keys configured - authentication will fail")


def require_api_key(f):
    """
    Decorator to require valid API key for endpoint access.

    Expects API key in X-API-Key header.
    Returns 401 if key is missing or invalid.

    Usage:
        @app.route("/protected")
        @require_api_key
        def protected_endpoint():
            return jsonify({"status": "ok"})
    """

    @wraps(f)
    def decorated(*args, **kwargs):
        # Load keys on first request if not already loaded
        if not _VALID_KEY_HASHES:
            _load_api_keys()

        # Check for API key header
        api_key = request.headers.get("X-API-Key")

        if not api_key:
            logger.warning(f"Missing API key for {request.method} {request.path} " f"from {request.remote_addr}")
            return jsonify({"error": "Missing API key", "message": "Provide API key in X-API-Key header"}), 401

        # Hash the provided key and compare
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash not in _VALID_KEY_HASHES:
            logger.warning(f"Invalid API key for {request.method} {request.path} " f"from {request.remote_addr}")
            return jsonify({"error": "Invalid API key", "message": "The provided API key is not valid"}), 401

        # Key is valid - proceed with request
        return f(*args, **kwargs)

    return decorated


def is_auth_enabled() -> bool:
    """Check if API authentication is enabled (keys are configured)."""
    if not _VALID_KEY_HASHES:
        _load_api_keys()
    return len(_VALID_KEY_HASHES) > 0


def reload_api_keys():
    """Reload API keys from environment (useful for testing)."""
    _load_api_keys()
