"""admin utilities for dashboard operations."""

import logging
from typing import Tuple, List, Dict, Any

import requests

from src.config import AdminConfig

logger = logging.getLogger(__name__)


def trigger_validation_via_api(api_url: str, initiated_by: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    trigger the system validation run via api.

    returns:
        (success, response_data, error_message)
    """
    if not AdminConfig.has_access():
        return False, {}, AdminConfig.disabled_reason()

    try:
        response = requests.post(
            f"{api_url}/validate/system",
            json={"initiated_by": initiated_by},
            timeout=900,
        )
        
        # get response data regardless of status code
        data = response.json() if response.content else {}
        returncode = data.get("returncode", 0)
        
        # validation script returns non-zero on failure, but we still want to show the output
        if response.status_code == 400 and returncode != 0:
            # validation failed, but include the output in the response
            stdout = data.get("stdout", "")
            stderr = data.get("stderr", "")
            # create a user-friendly error message
            error_msg = "Validation completed with issues"
            # return the data with the output, error_msg is just for status
            return False, data, error_msg
        
        response.raise_for_status()
        return True, data, ""
    except requests.Timeout:
        logger.error("validation trigger timed out")
        return False, {}, "Validation timed out. Check server logs for details."
    except requests.ConnectionError:
        logger.error("validation trigger connection error")
        return False, {}, "API server unavailable. Ensure the API is running."
    except requests.RequestException as exc:
        logger.error(f"validation trigger request failed: {exc}")
        try:
            data = exc.response.json() if exc.response is not None else {}
            message = data.get("error", str(exc))
            # include stdout/stderr if available
            stdout = data.get("stdout", "")
            stderr = data.get("stderr", "")
            if stdout or stderr:
                if stderr:
                    message += f"\n\nSTDERR:\n{stderr}"
                if stdout:
                    message += f"\n\nSTDOUT:\n{stdout}"
        except Exception:
            message = str(exc)
        return False, {}, f"Validation failed: {message}"


def fetch_validation_logs(api_url: str, limit: int = 5) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    fetch recent validation logs for display in admin panel.

    returns:
        (success, logs, error_message)
    """
    if not AdminConfig.has_access():
        return False, [], AdminConfig.disabled_reason()

    try:
        response = requests.get(
            f"{api_url}/validation/logs",
            params={"limit": limit},
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()
        logs = data.get("logs", [])
        return True, logs, ""
    except requests.Timeout:
        logger.error("validation log request timed out")
        return False, [], "Timeout retrieving validation logs."
    except requests.ConnectionError:
        logger.error("validation log request connection error")
        return False, [], "API server unavailable."
    except requests.RequestException as exc:
        logger.error(f"validation log request failed: {exc}")
        return False, [], "Unable to retrieve validation logs."
