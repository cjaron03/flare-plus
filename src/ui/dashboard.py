"""main dashboard application using gradio."""

import logging
from datetime import datetime
from typing import Optional

import gradio as gr

from src.config import AdminConfig
from src.ui.utils.helpers import get_prediction_service
from src.ui.utils.data_queries import get_latest_data_timestamps, calculate_data_freshness
from src.ui.utils.ingestion import (
    run_ingestion_via_api,
    can_trigger_ingestion,
    get_ingestion_state,
    set_ingestion_in_progress,
    set_last_ingestion_time,
)
from src.ui.tabs.predictions import build_predictions_tab
from src.ui.tabs.timeline import build_timeline_tab
from src.ui.tabs.scenario import build_scenario_tab
from src.ui.tabs.about import build_about_tab
from src.ui.tabs.admin import build_admin_tab
from src.ui.tabs.login import build_login_tab

logger = logging.getLogger(__name__)


def create_dashboard(
    api_url: str = "http://127.0.0.1:5000",
    classification_model_path: Optional[str] = None,
    survival_model_path: Optional[str] = None,
) -> gr.Blocks:
    """
    create main dashboard application.

    args:
        api_url: api server url
        classification_model_path: path to classification model (joblib)
        survival_model_path: path to survival model (joblib)

    returns:
        gradio blocks application
    """
    # initialize connection
    connection_mode, api_url_or_none, loaded_pipelines = get_prediction_service(
        api_url, classification_model_path, survival_model_path
    )

    # create state variables
    connection_state = gr.State(value=connection_mode)
    api_url_state = gr.State(value=api_url_or_none)
    pipelines_state = gr.State(value=loaded_pipelines)
    last_refresh_state = gr.State(value=None)
    # create dashboard
    with gr.Blocks(title="Solar Flare Prediction Dashboard") as dashboard:
        # header
        gr.Markdown(
            """
            # Solar Flare Prediction Dashboard
            **Developed by Jaron Cabral**

            ---
            """
        )

        # limitations warning
        with gr.Row():
            gr.Markdown(
                """
                ### LIMITATION: This prototype has limited training data.
                Predictions may not be accurate until more historical flare events are ingested.
                **Use with caution for operational decisions.**
                """
            )

        # connection status and data freshness sidebar
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Connection Status")

                # build initial connection status text
                def build_connection_status_text():
                    """build connection status text, querying api if available."""
                    from src.ui.utils.helpers import get_api_model_status

                    lines = [f"**Mode**: {connection_mode.upper()}"]

                    if api_url_or_none:
                        lines.append(f"**API URL**: {api_url_or_none}")
                        # query api for model status
                        try:
                            model_status = get_api_model_status(api_url_or_none)
                            loaded_models = []
                            if model_status.get("classification"):
                                loaded_models.append("Classification")
                            if model_status.get("survival"):
                                loaded_models.append("Survival")

                            if loaded_models:
                                lines.append(f"**Models**: {', '.join(loaded_models)} loaded")
                            else:
                                lines.append("**Models**: None loaded")

                            confidence = model_status.get("confidence_level")
                            if confidence:
                                lines.append(f"**Confidence**: {str(confidence).title()}")

                            if model_status.get("survival_guardrail"):
                                reason = model_status.get("guardrail_reason") or "Survival guardrail active"
                                lines.append(f"**Guardrail**: {reason}")

                            last_validation = model_status.get("last_validation")
                            if last_validation:
                                lines.append(f"**Last Validation**: {last_validation}")
                        except Exception as e:
                            logger.warning(f"failed to query api model status: {e}")
                            lines.append("**Models**: Checking...")
                    elif loaded_pipelines:
                        lines.append(f"**Models Loaded**: {', '.join(loaded_pipelines.keys())}")
                    else:
                        lines.append("**Models**: None loaded")

                    return "\n".join(lines)

                connection_status = gr.Markdown(build_connection_status_text())
                admin_indicator = gr.Markdown(f"**Admin Access**: {AdminConfig.status_indicator()}")

                gr.Markdown("### Data Freshness")

                def update_data_status():
                    """update data freshness status."""
                    try:
                        timestamps = get_latest_data_timestamps()

                        status_lines = []

                        # flux data
                        flux_info = calculate_data_freshness(timestamps["flux_latest"])
                        if flux_info["hours_ago"] is not None:
                            status_lines.append(
                                f"**Flux**: {flux_info['hours_ago']:.1f}h ago ({flux_info['status']}) "
                                f"- {timestamps['flux_count']} records"
                            )
                        else:
                            status_lines.append("**Flux**: No data")

                        # regions data
                        regions_info = calculate_data_freshness(timestamps["regions_latest"])
                        if regions_info["hours_ago"] is not None:
                            status_lines.append(
                                f"**Regions**: {regions_info['hours_ago']:.1f}h ago ({regions_info['status']}) "
                                f"- {timestamps['regions_count']} records"
                            )
                        else:
                            status_lines.append("**Regions**: No data")

                        # flares data
                        flares_info = calculate_data_freshness(timestamps["flares_latest"])
                        if flares_info["hours_ago"] is not None:
                            status_lines.append(
                                f"**Flares**: {flares_info['hours_ago']:.1f}h ago - "
                                f"{timestamps['flares_count']} events"
                            )
                        else:
                            status_lines.append("**Flares**: No data")

                        return "\n".join(status_lines)

                    except Exception as e:
                        logger.error(f"error updating data status: {e}")
                        return "Error loading data status"

                data_status = gr.Markdown(update_data_status())

                # ingestion progress/output
                ingestion_progress = gr.Textbox(
                    label="Ingestion Progress",
                    lines=3,
                    interactive=False,
                )
                ingestion_summary = gr.Markdown(label="Ingestion Summary")

                refresh_status_button = gr.Button("Refresh Data & Status", variant="secondary")

                def trigger_ingestion_and_refresh_status():
                    """trigger ingestion and refresh status (day 1: basic functionality)."""
                    # check rate limiting
                    can_trigger, rate_limit_msg = can_trigger_ingestion()
                    if not can_trigger:
                        return (
                            update_data_status(),
                            rate_limit_msg,
                            "",
                        )

                    # check ingestion lock
                    in_progress, _ = get_ingestion_state()
                    if in_progress:
                        return (
                            update_data_status(),
                            "Ingestion already in progress. Please wait for current ingestion to complete.",
                            "",
                        )

                    # get api url from state
                    mode = connection_state.value
                    api = api_url_state.value

                    if mode != "api" or not api:
                        return (
                            update_data_status(),
                            "No API connection available. Please start API server or provide model paths.",
                            "",
                        )

                    # start ingestion
                    set_ingestion_in_progress(True)
                    try:
                        progress_msg = "Starting data ingestion..."
                        success, results, error, error_type = run_ingestion_via_api(api, use_cache=True)

                        if success:
                            duration = results.get("duration", 0)
                            overall_status = results.get("overall_status", "success")

                            # day 4: format summary with overall status and detailed results
                            from src.ui.utils.ingestion import format_ingestion_summary

                            summary = format_ingestion_summary(results)

                            if overall_status == "success":
                                progress_msg = f"Ingestion completed successfully. Duration: {duration:.1f} seconds."
                            elif overall_status == "partial":
                                progress_msg = (
                                    f"Ingestion completed with partial success. Duration: {duration:.1f} seconds."
                                )
                            else:
                                progress_msg = f"Ingestion failed. Duration: {duration:.1f} seconds."

                            set_last_ingestion_time(datetime.now())
                        else:
                            # day 3: error handling with error types
                            if error_type == "transient":
                                progress_msg = f"Ingestion failed (temporary): {error}"
                            elif error_type == "permanent":
                                progress_msg = f"Ingestion failed (permanent): {error}"
                            else:
                                progress_msg = f"Ingestion failed: {error}"
                            summary = f"Ingestion Error:\n{error}"

                        # refresh status after ingestion
                        updated_status = update_data_status()

                        return updated_status, progress_msg, summary

                    except Exception as e:
                        logger.error(f"ingestion error: {e}")
                        return (
                            update_data_status(),
                            f"Ingestion error: {str(e)}",
                            "",
                        )
                    finally:
                        set_ingestion_in_progress(False)

                def update_connection_status():
                    """update connection status by querying api."""
                    from src.ui.utils.helpers import get_api_model_status

                    lines = [f"**Mode**: {connection_mode.upper()}"]

                    if api_url_or_none:
                        lines.append(f"**API URL**: {api_url_or_none}")
                        # query api for model status
                        try:
                            model_status = get_api_model_status(api_url_or_none)
                            loaded_models = []
                            if model_status.get("classification"):
                                loaded_models.append("Classification")
                            if model_status.get("survival"):
                                loaded_models.append("Survival")

                            if loaded_models:
                                lines.append(f"**Models**: {', '.join(loaded_models)} loaded")
                            else:
                                lines.append("**Models**: None loaded")

                            confidence = model_status.get("confidence_level")
                            if confidence:
                                lines.append(f"**Confidence**: {str(confidence).title()}")

                            if model_status.get("survival_guardrail"):
                                reason = model_status.get("guardrail_reason") or "Survival guardrail active"
                                lines.append(f"**Guardrail**: {reason}")

                            last_validation = model_status.get("last_validation")
                            if last_validation:
                                lines.append(f"**Last Validation**: {last_validation}")
                        except Exception as e:
                            logger.warning(f"failed to query api model status: {e}")
                            lines.append("**Models**: Checking...")
                    elif loaded_pipelines:
                        lines.append(f"**Models Loaded**: {', '.join(loaded_pipelines.keys())}")
                    else:
                        lines.append("**Models**: None loaded")

                    return "\n".join(lines)

                def update_connection_status_and_admin():
                    """update connection status and admin indicator."""
                    return update_connection_status(), f"**Admin Access**: {AdminConfig.status_indicator()}"

                refresh_status_button.click(
                    fn=trigger_ingestion_and_refresh_status,
                    outputs=[data_status, ingestion_progress, ingestion_summary],
                )

                # update connection status when refresh button is clicked
                refresh_status_button.click(
                    fn=update_connection_status_and_admin,
                    outputs=[connection_status, admin_indicator],
                )

            with gr.Column(scale=4):
                # main tabs
                with gr.Tabs():
                    with gr.Tab("Predictions"):
                        build_predictions_tab(
                            connection_state,
                            api_url_state,
                            pipelines_state,
                            last_refresh_state,
                        )

                    with gr.Tab("Timeline"):
                        build_timeline_tab(last_refresh_state)

                    with gr.Tab("Scenario"):
                        build_scenario_tab()

                    with gr.Tab("About"):
                        build_about_tab()

                    with gr.Tab("Admin"):
                        admin_components = build_admin_tab(
                            connection_state,
                            api_url_state,
                        )

                    with gr.Tab("Login"):
                        build_login_tab(
                            admin_indicator,
                            admin_components["access_notice"],
                            admin_components["admin_container"],
                            admin_components["guardrail_status"],
                            admin_components["validation_history"],
                            admin_components["refresh_fn"],
                        )

        # footer
        gr.Markdown(
            """
            ---
            **Disclaimer**: This dashboard is a research prototype. Always consult official
            NOAA/SWPC forecasts for operational solar flare predictions.
            """
        )

    return dashboard
