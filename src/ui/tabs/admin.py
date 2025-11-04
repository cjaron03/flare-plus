"""admin system-health tab."""

import logging
from typing import List

import gradio as gr

from src.config import AdminConfig
from src.ui.utils.admin import trigger_validation_via_api, fetch_validation_logs
from src.ui.utils.helpers import get_api_health_status

logger = logging.getLogger(__name__)


def _format_guardrail_status(health: dict) -> str:
    """format guardrail status markdown for admin view."""
    if not health:
        return "Unable to retrieve API health status."

    validation = health.get("validation") or {}
    guardrail_active = health.get("survival_guardrail", False)
    confidence = (health.get("confidence_level") or "unknown").title()
    service_status = (health.get("status") or "unknown").title()
    last_validation = validation.get("run_timestamp", "N/A")
    last_validation_status = (validation.get("status") or "unknown").title()
    guardrail_reason = validation.get("guardrail_reason")

    lines: List[str] = [
        f"**Service Status**: {service_status}",
        f"**Confidence**: {confidence}",
        f"**Last Validation**: {last_validation} ({last_validation_status})",
    ]

    if guardrail_active:
        lines.append("**Guardrail**: Active")
        if guardrail_reason:
            lines.append(f"**Reason**: {guardrail_reason}")
    else:
        lines.append("**Guardrail**: Not triggered")

    return "\n".join(lines)


def build_admin_tab(
    connection_state: gr.State,
    api_url_state: gr.State,
) -> None:
    """render admin system health tab."""
    with gr.Column():
        if not AdminConfig.has_access():
            gr.Markdown(AdminConfig.disabled_reason())
            return

        guardrail_status = gr.Markdown("Loading system health…")
        validation_history = gr.Dataframe(
            headers=["Run Time", "Status", "Guardrail", "Reason", "Initiated By"],
            datatype=["str", "str", "str", "str", "str"],
            interactive=False,
            value=[],
            label="Recent Validation Runs",
        )
        validation_output = gr.Textbox(
            label="Validation Output",
            lines=10,
            interactive=False,
        )

        with gr.Row():
            refresh_button = gr.Button("Refresh Status", variant="secondary")
            run_validation_button = gr.Button("Run System Validation", variant="primary")

        def gather_admin_status():
            """collect guardrail status and validation history."""
            mode = connection_state.value
            api_url = api_url_state.value

            if mode != "api" or not api_url:
                return (
                    "API connection required to retrieve system health. Start the API server and reconnect.",
                    [],
                )

            health = get_api_health_status(api_url)
            guardrail_text = _format_guardrail_status(health)

            success, logs, error = fetch_validation_logs(api_url, limit=10)
            if not success:
                logger.warning(f"failed to fetch validation logs: {error}")
                return guardrail_text + "\n\n⚠️ " + error, []

            rows = []
            for log in logs:
                guardrail_flag = "Yes" if log.get("guardrail_triggered") else "No"
                rows.append(
                    [
                        log.get("run_timestamp", ""),
                        (log.get("status") or "").title(),
                        guardrail_flag,
                        log.get("guardrail_reason") or "",
                        log.get("initiated_by") or "-",
                    ]
                )

            return guardrail_text, rows

        def refresh_admin_panel():
            guardrail_text, rows = gather_admin_status()
            return guardrail_text, rows

        def run_validation():
            mode = connection_state.value
            api_url = api_url_state.value

            if mode != "api" or not api_url:
                return (
                    "API connection required to run validation. Start the API server and reconnect.",
                    [],
                    "Validation not executed.",
                )

            initiated_by = "ui-admin"
            success, data, error = trigger_validation_via_api(api_url, initiated_by)

            output_lines = []
            if success:
                stdout = data.get("stdout", "")
                stderr = data.get("stderr", "")
                return_code = data.get("returncode", 0)
                output_lines.append(f"Return code: {return_code}")
                if stdout:
                    output_lines.append("STDOUT:\n" + stdout)
                if stderr:
                    output_lines.append("STDERR:\n" + stderr)
            else:
                output_lines.append(error or "Validation failed.")

            guardrail_text, rows = gather_admin_status()
            return guardrail_text, rows, "\n\n".join(output_lines)

        refresh_button.click(
            fn=refresh_admin_panel,
            outputs=[guardrail_status, validation_history],
        )

        run_validation_button.click(
            fn=run_validation,
            outputs=[guardrail_status, validation_history, validation_output],
        )

        # populate content on load
        guardrail_text, rows = gather_admin_status()
        guardrail_status.value = guardrail_text
        validation_history.value = rows
