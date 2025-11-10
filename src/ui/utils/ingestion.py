"""helper functions for ui data ingestion: api calls, rate limiting, formatting."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# rate limiting constants
MIN_INGESTION_INTERVAL = 300  # 5 minutes in seconds

# global state for rate limiting and ingestion lock
_ingestion_in_progress = False
_last_ingestion_time: Optional[datetime] = None


def get_ingestion_state() -> Tuple[bool, Optional[datetime]]:
    """get current ingestion state."""
    return _ingestion_in_progress, _last_ingestion_time


def set_ingestion_in_progress(value: bool):
    """set ingestion in progress flag."""
    global _ingestion_in_progress
    _ingestion_in_progress = value


def set_last_ingestion_time(value: Optional[datetime]):
    """set last ingestion time."""
    global _last_ingestion_time
    _last_ingestion_time = value


def run_ingestion_via_api(
    api_url: str, use_cache: bool = True
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """
    call ingestion api endpoint.

    args:
        api_url: base api url
        use_cache: whether to use cached data

    returns:
        tuple: (success, results_dict, error_message, error_type)
            error_type: "transient" (network timeout, connection), "permanent" (api down, auth), or None
    """
    try:
        url = f"{api_url}/ingest"
        payload = {"use_cache": use_cache}

        response = requests.post(url, json=payload, timeout=300)  # 5 minute timeout for ingestion

        response.raise_for_status()
        data = response.json()

        return True, data, None, None

    except requests.Timeout:
        error_msg = "Network error occurred. Retry in 1 minute."
        logger.error(f"ingestion timeout: {error_msg}")
        return False, None, error_msg, "transient"
    except requests.ConnectionError:
        error_msg = "API server unavailable. Contact administrator."
        logger.error(f"ingestion connection error: {error_msg}")
        return False, None, error_msg, "permanent"
    except requests.RequestException as e:
        # check if it's a client error (4xx) vs server error (5xx)
        if hasattr(e, "response") and e.response is not None:
            status_code = e.response.status_code
            if 400 <= status_code < 500:
                error_msg = "API server unavailable. Contact administrator."
                return False, None, error_msg, "permanent"
            else:
                error_msg = "Network error occurred. Retry in 1 minute."
                return False, None, error_msg, "transient"
        else:
            error_msg = "Network error occurred. Retry in 1 minute."
            logger.error(f"ingestion request error: {e}")
            return False, None, error_msg, "transient"
    except ValueError as e:  # json decode error
        error_msg = "API server unavailable. Contact administrator."
        logger.error(f"ingestion json decode error: {e}")
        return False, None, error_msg, "permanent"


def can_trigger_ingestion() -> Tuple[bool, str]:
    """
    check if ingestion can be triggered based on rate limiting.

    returns:
        tuple: (can_trigger, message)
    """
    if _last_ingestion_time is None:
        return True, ""

    time_since_last = (datetime.utcnow() - _last_ingestion_time).total_seconds()

    if time_since_last < MIN_INGESTION_INTERVAL:
        minutes_ago = int(time_since_last / 60)
        minutes_remaining = int((MIN_INGESTION_INTERVAL - time_since_last) / 60)
        message = f"Last refresh: {minutes_ago} minutes ago. Next refresh available in {minutes_remaining} minutes."
        return False, message

    return True, ""


def format_ingestion_summary(response_data: Dict[str, Any]) -> str:
    """
    format ingestion summary with duration and overall status.

    args:
        response_data: api response with overall_status, duration, and results dict

    returns:
        formatted summary string
    """
    overall_status = response_data.get("overall_status", "unknown")
    duration = response_data.get("duration", 0)
    results = response_data.get("results", {})

    lines = ["Ingestion Summary:"]
    lines.append(f"Status: {overall_status.title()}")
    lines.append(f"Duration: {duration:.1f} seconds")
    lines.append("")

    # format each data type result
    for key, result_data in results.items():
        status = result_data.get("status", "failed")
        if status == "success":
            if key == "xray_flux":
                records = result_data.get("records", 0)
                lines.append(f"- Flux: {records:,} records saved")
            elif key == "solar_regions":
                records = result_data.get("records", 0)
                lines.append(f"- Regions: {records} records saved")
            elif key == "magnetogram":
                records = result_data.get("records", 0)
                lines.append(f"- Magnetogram: {records} records saved")
            elif key == "flare_events":
                new_count = result_data.get("new", 0)
                duplicates = result_data.get("duplicates", 0)
                if new_count > 0:
                    lines.append(f"- Flares: {new_count} new, {duplicates} duplicates")
                elif duplicates > 0:
                    lines.append(f"- Flares: {duplicates} detected (all duplicates)")
                else:
                    lines.append("- Flares: none detected")
        else:
            error = result_data.get("error", "unknown error")
            key_name = key.replace("_", " ").title()
            lines.append(f"- {key_name}: failed - {error}")

    return "\n".join(lines)


def get_rate_limit_message(last_time: Optional[datetime]) -> str:
    """
    get formatted rate limit message.

    args:
        last_time: last ingestion timestamp

    returns:
        formatted message string
    """
    if last_time is None:
        return ""

    time_since = (datetime.utcnow() - last_time).total_seconds()
    minutes_ago = int(time_since / 60)
    minutes_remaining = max(0, int((MIN_INGESTION_INTERVAL - time_since) / 60))

    return f"Last refresh: {minutes_ago} minutes ago. Next refresh available in {minutes_remaining} minutes."
