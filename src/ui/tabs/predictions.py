"""predictions tab for ui dashboard."""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

import gradio as gr

from src.config import APIClientConfig
from src.ui.utils.helpers import (
    make_api_request,
    format_classification_prediction,
    format_survival_plain_language,
    format_survival_probability_distribution,
    should_refresh,
    get_api_model_status,
)
from src.ui.utils.charts import (
    create_probability_bar_chart,
    create_survival_curve_chart,
    create_probability_distribution_chart,
)

logger = logging.getLogger(__name__)


def create_classification_prediction(
    timestamp: datetime,
    window: int,
    region_number: Optional[int],
    model_type: str,
    connection_mode: str,
    api_url: Optional[str],
    pipelines: Optional[Dict[str, Any]],
    last_refresh: Optional[datetime],
    force_refresh: bool,
) -> Tuple[str, Any, str]:
    """
    make classification prediction and return formatted output.

    returns:
        tuple: (formatted_text, plotly_figure, status_message)
    """
    try:
        # check throttling (only if not forcing refresh)
        if not force_refresh:
            if not should_refresh(last_refresh, min_interval_minutes=5):
                time_since = datetime.now() - last_refresh if last_refresh else timedelta(0)
                minutes_remaining = max(0, 5 - int(time_since.total_seconds() / 60))
                return (
                    "",
                    None,
                    f"Refresh throttled (wait {minutes_remaining} more minutes or use 'Refresh Now' button)",
                )

        prediction = None

        if connection_mode == "api" and api_url:
            # use api
            payload = {
                "timestamp": timestamp.isoformat(),
                "window": window,
                "model_type": model_type,
            }
            if region_number is not None:
                payload["region_number"] = region_number

            success, data, error = make_api_request(
                api_url,
                "/predict/classification",
                method="POST",
                json_data=payload,
                api_key=APIClientConfig.API_KEY,
            )

            if success and data:
                prediction = data
            else:
                return (
                    "",
                    None,
                    f"API error: {error or 'unknown error'}",
                )
        elif connection_mode == "direct" and pipelines:
            # use direct pipeline
            classification_pipeline = pipelines.get("classification")
            if not classification_pipeline:
                return (
                    "",
                    None,
                    "Classification model not available (only survival model loaded)",
                )

            try:
                prediction = classification_pipeline.predict(
                    timestamp=timestamp,
                    window=window,
                    model_type=model_type,
                    region_number=region_number,
                )
            except Exception as e:
                logger.error(f"classification prediction failed: {e}")
                return (
                    "",
                    None,
                    f"Prediction error: {str(e)}",
                )
        else:
            return (
                "",
                None,
                "No connection available. Please start API server or provide model paths.",
            )

        if not prediction:
            return (
                "",
                None,
                "Unable to generate prediction",
            )

        # format output
        formatted_text = format_classification_prediction(prediction)
        class_probs = prediction.get("class_probabilities", {})

        # create chart
        chart = create_probability_bar_chart(class_probs, title=f"Classification Probabilities ({window}h window)")

        status = "Prediction successful"

        return formatted_text, chart, status

    except Exception as e:
        logger.error(f"error in classification prediction: {e}")
        return (
            "",
            None,
            f"Error: {str(e)}",
        )


def create_survival_prediction(
    timestamp: datetime,
    region_number: Optional[int],
    model_type: str,
    connection_mode: str,
    api_url: Optional[str],
    pipelines: Optional[Dict[str, Any]],
    last_refresh: Optional[datetime],
    force_refresh: bool,
) -> Tuple[str, str, Any, Any, str]:
    """
    make survival prediction and return formatted output.

    returns:
        tuple: (plain_language, distribution_text, survival_curve, prob_dist_chart, status_message)
    """
    try:
        # check throttling (only if not forcing refresh)
        if not force_refresh:
            if not should_refresh(last_refresh, min_interval_minutes=5):
                time_since = datetime.now() - last_refresh if last_refresh else timedelta(0)
                minutes_remaining = max(0, 5 - int(time_since.total_seconds() / 60))
                return (
                    "",
                    "",
                    None,
                    None,
                    f"Refresh throttled (wait {minutes_remaining} more minutes or use 'Refresh Now' button)",
                )

        prediction = None

        if connection_mode == "api" and api_url:
            # use api
            payload = {
                "timestamp": timestamp.isoformat(),
                "model_type": model_type,
            }
            if region_number is not None:
                payload["region_number"] = region_number

            success, data, error = make_api_request(
                api_url,
                "/predict/survival",
                method="POST",
                json_data=payload,
                api_key=APIClientConfig.API_KEY,
            )

            if success and data:
                prediction = data
            else:
                return (
                    "",
                    "",
                    None,
                    None,
                    f"API error: {error or 'unknown error'}",
                )
        elif connection_mode == "direct" and pipelines:
            # use direct pipeline
            survival_pipeline = pipelines.get("survival")
            if not survival_pipeline:
                return (
                    "",
                    "",
                    None,
                    None,
                    "Survival model not available (only classification model loaded)",
                )

            try:
                prediction = survival_pipeline.predict_survival_probabilities(
                    timestamp=timestamp,
                    region_number=region_number,
                    model_type=model_type,
                )
            except Exception as e:
                logger.error(f"survival prediction failed: {e}")
                return (
                    "",
                    "",
                    None,
                    None,
                    f"Prediction error: {str(e)}",
                )
        else:
            return (
                "",
                "",
                None,
                None,
                "No connection available. Please start API server or provide model paths.",
            )

        if not prediction:
            return (
                "",
                "",
                None,
                None,
                "Unable to generate prediction",
            )

        # format outputs
        plain_language = format_survival_plain_language(prediction)
        distribution_text = format_survival_probability_distribution(prediction)

        # create charts
        survival_data = prediction.get("survival_function", {})
        survival_curve = create_survival_curve_chart(survival_data, title="Survival Function")

        prob_dist = prediction.get("probability_distribution", {})
        prob_dist_chart = create_probability_distribution_chart(
            prob_dist, title="Probability Distribution Over Time Buckets"
        )

        status = "Prediction successful"

        return plain_language, distribution_text, survival_curve, prob_dist_chart, status

    except Exception as e:
        logger.error(f"error in survival prediction: {e}")
        return (
            "",
            "",
            None,
            None,
            f"Error: {str(e)}",
        )


def build_predictions_tab(
    connection_mode: gr.State,
    api_url: gr.State,
    pipelines: gr.State,
    last_refresh: gr.State,
) -> gr.Blocks:
    """
    build predictions tab interface.

    args:
        connection_mode: state variable for connection mode
        api_url: state variable for api url
        pipelines: state variable for loaded pipelines
        last_refresh: state variable for last refresh time

    returns:
        gradio blocks for predictions tab
    """
    with gr.Blocks() as tab:
        gr.Markdown("## Current Predictions")
        gr.Markdown(
            "Two complementary prediction models:\n\n"
            "**Classification**: Predicts the maximum flare class (M/X) expected in 24-48h\n"
            "**Survival Analysis**: Predicts when an M-class flare will occur (timing probabilities)\n\n"
            "**Model Performance:** F1: 0.867, Precision: 93%, Recall: 81%\n\n"
            "Note: C-class predictions are currently disabled. These models focus on M/X-class flares only."
        )

        def refresh_confidence_notice():
            mode = connection_mode.value
            api = api_url.value

            if mode == "api" and api:
                status = get_api_model_status(api)
                confidence = status.get("confidence_level")
                guardrail_active = status.get("survival_guardrail", False)

                if confidence:
                    if str(confidence).lower() in ["normal", "high"]:
                        text = (
                            f"**Prediction Confidence:** ✅ **{str(confidence).upper()}** "
                            "— Model operating normally, all validations passing"
                        )
                    elif str(confidence).lower() == "low":
                        text = f"**Prediction Confidence:** ⚠️ **{str(confidence).upper()}**"
                        if guardrail_active:
                            reason = status.get("guardrail_reason")
                            if reason:
                                text += f" — {reason}"
                            else:
                                text += " — Survival model guardrail active"
                        else:
                            text += " — Survival model under review"
                    else:
                        text = f"**Prediction Confidence:** {str(confidence).title()}"
                    return text
                return "Prediction confidence unavailable (API did not report confidence)."

            return "Prediction confidence unavailable. Connect to the API for live guardrail status."

        confidence_notice = gr.Markdown(refresh_confidence_notice())

        with gr.Row():
            with gr.Column(scale=1):
                timestamp_input = gr.DateTime(
                    label="Observation Timestamp",
                )
                region_input = gr.Number(
                    label="Region Number (optional)",
                    value=None,
                    precision=0,
                )

            with gr.Column(scale=1):
                status_output = gr.Textbox(label="Status", interactive=False)

        with gr.Tabs():
            with gr.Tab("Classification (M/X only)"):
                gr.Markdown("### Classification Prediction (24-48h)")
                gr.Markdown(
                    "**Predicts whether an M-class or X-class flare will occur** "
                    "within the specified time window.\n\n"
                    "- **None**: No M or X-class flare in the window\n"
                    "- **M**: At least M-class flare (may include X)\n"
                    "- **X**: X-class flare\n\n"
                    "**Note:** C-class predictions are currently disabled. This model focuses on M/X-class flares only."
                )

                with gr.Row():
                    window_input = gr.Radio(
                        label="Prediction Window",
                        choices=[24, 48],
                        value=24,
                    )
                    model_type_input = gr.Radio(
                        label="Model Type",
                        choices=["logistic", "gradient_boosting"],
                        value="gradient_boosting",
                    )

                with gr.Row():
                    classify_button = gr.Button("Predict Classification", variant="primary")
                    refresh_button_class = gr.Button("Refresh Now", variant="secondary")

                classification_text = gr.Textbox(label="Prediction Results", lines=10, interactive=False)
                classification_chart = gr.Plot(label="Probability Distribution")

                def run_classification(timestamp, window, region_val, model_type, refresh_now):
                    # handle timestamp conversion (gradio DateTime can return float, datetime, or None)
                    if timestamp is None:
                        timestamp = datetime.now()
                    elif isinstance(timestamp, (int, float)):
                        # convert unix timestamp to datetime
                        timestamp = datetime.fromtimestamp(timestamp)
                    elif not isinstance(timestamp, datetime):
                        # try to parse string or convert
                        try:
                            timestamp = datetime.fromisoformat(str(timestamp))
                        except (ValueError, TypeError):
                            timestamp = datetime.now()
                    region = int(region_val) if region_val else None

                    mode = connection_mode.value
                    api = api_url.value
                    pipes = pipelines.value

                    text, chart, status = create_classification_prediction(
                        timestamp,
                        window,
                        region,
                        model_type,
                        mode,
                        api,
                        pipes,
                        last_refresh.value,
                        refresh_now,
                    )

                    if refresh_now:
                        last_refresh.value = datetime.now()

                    return text, chart, status

                classify_button.click(
                    fn=lambda t, w, r, m: run_classification(t, w, r, m, False),
                    inputs=[timestamp_input, window_input, region_input, model_type_input],
                    outputs=[classification_text, classification_chart, status_output],
                )
                classify_button.click(
                    fn=lambda: refresh_confidence_notice(),
                    outputs=[confidence_notice],
                )

                refresh_button_class.click(
                    fn=lambda t, w, r, m: run_classification(t, w, r, m, True),
                    inputs=[timestamp_input, window_input, region_input, model_type_input],
                    outputs=[classification_text, classification_chart, status_output],
                )
                refresh_button_class.click(
                    fn=lambda: refresh_confidence_notice(),
                    outputs=[confidence_notice],
                )

            with gr.Tab("Survival Analysis (M-class)"):
                gr.Markdown("### Survival Analysis Prediction (Time-to-Event)")
                gr.Markdown(
                    "**Predicts WHEN an M-class flare will occur** over different time buckets (0-168 hours).\n\n"
                    "This model focuses specifically on M-class flare timing and provides:\n"
                    "- Probability distribution across time buckets\n"
                    "- Survival curve showing likelihood over time\n"
                    "- Plain-language timing estimates\n\n"
                    "**Model Performance:** F1: 0.867, Precision: 93%, Recall: 81%\n\n"
                    "Note: C-class predictions are currently disabled. This model is trained on M-class flares."
                )

                model_type_survival = gr.Radio(
                    label="Model Type",
                    choices=["cox", "gb"],
                    value="cox",
                )

                with gr.Row():
                    survival_button = gr.Button("Predict Survival", variant="primary")
                    refresh_button_survival = gr.Button("Refresh Now", variant="secondary")

                plain_language_output = gr.Textbox(label="Plain Language Summary", lines=5, interactive=False)
                distribution_text_output = gr.Textbox(
                    label="Probability Distribution Details", lines=10, interactive=False
                )

                with gr.Row():
                    survival_curve_plot = gr.Plot(label="Survival Curve")
                    prob_dist_plot = gr.Plot(label="Probability Distribution")

                def run_survival(timestamp, region_val, model_type, refresh_now):
                    # handle timestamp conversion (gradio DateTime can return float, datetime, or None)
                    if timestamp is None:
                        timestamp = datetime.now()
                    elif isinstance(timestamp, (int, float)):
                        # convert unix timestamp to datetime
                        timestamp = datetime.fromtimestamp(timestamp)
                    elif not isinstance(timestamp, datetime):
                        # try to parse string or convert
                        try:
                            timestamp = datetime.fromisoformat(str(timestamp))
                        except (ValueError, TypeError):
                            timestamp = datetime.now()
                    region = int(region_val) if region_val else None

                    mode = connection_mode.value
                    api = api_url.value
                    pipes = pipelines.value

                    (
                        plain_lang,
                        dist_text,
                        curve,
                        prob_chart,
                        status,
                    ) = create_survival_prediction(
                        timestamp,
                        region,
                        model_type,
                        mode,
                        api,
                        pipes,
                        last_refresh.value,
                        refresh_now,
                    )

                    if refresh_now:
                        last_refresh.value = datetime.now()

                    return plain_lang, dist_text, curve, prob_chart, status

                survival_button.click(
                    fn=lambda t, r, m: run_survival(t, r, m, False),
                    inputs=[timestamp_input, region_input, model_type_survival],
                    outputs=[
                        plain_language_output,
                        distribution_text_output,
                        survival_curve_plot,
                        prob_dist_plot,
                        status_output,
                    ],
                )
                survival_button.click(
                    fn=lambda: refresh_confidence_notice(),
                    outputs=[confidence_notice],
                )

                refresh_button_survival.click(
                    fn=lambda t, r, m: run_survival(t, r, m, True),
                    inputs=[timestamp_input, region_input, model_type_survival],
                    outputs=[
                        plain_language_output,
                        distribution_text_output,
                        survival_curve_plot,
                        prob_dist_plot,
                        status_output,
                    ],
                )
                refresh_button_survival.click(
                    fn=lambda: refresh_confidence_notice(),
                    outputs=[confidence_notice],
                )

    return tab
