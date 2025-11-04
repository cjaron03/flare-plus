"""about tab for ui dashboard with links, author info, and limitations."""

import logging

import gradio as gr

from src.config import CONFIG

logger = logging.getLogger(__name__)


def build_about_tab() -> gr.Blocks:
    """
    build about tab interface with links, author, and limitations.

    returns:
        gradio blocks for about tab
    """
    with gr.Blocks() as tab:
        gr.Markdown("# Solar Flare Prediction Dashboard")
        gr.Markdown("Developed by **Jaron Cabral**")

        gr.Markdown("## Limitations and Disclaimers")

        gr.Markdown(
            """
            **IMPORTANT LIMITATION**: This prototype has limited training data.
            Predictions may not be accurate until more historical flare events are ingested.
            **Use with caution for operational decisions.**

            This dashboard is a research prototype and should not be used as the sole basis
            for operational solar flare forecasting. Always consult official NOAA/SWPC forecasts
            for operational decisions.
            """
        )

        gr.Markdown("## Data Sources")

        # get endpoints from config
        endpoints = CONFIG.get("data_ingestion", {}).get("endpoints", {})

        noaa_links = []
        if endpoints:
            for name, url in endpoints.items():
                if url:
                    noaa_links.append(f"- **{name.replace('_', ' ').title()}**: {url}")

        if noaa_links:
            gr.Markdown("\n".join(noaa_links))
        else:
            gr.Markdown("No endpoint URLs configured in config.yaml")

        # add general noaa links
        gr.Markdown(
            """
            - **NOAA Space Weather Prediction Center**: https://www.swpc.noaa.gov/
            - **GOES XRS Flux Data**: https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json
            - **Solar Region Data**: https://services.swpc.noaa.gov/json/solar_regions.json
            """
        )

        gr.Markdown("## Methodology")

        with gr.Accordion("Classification Models", open=False):
            gr.Markdown(
                """
                Short-term (24-48 hour) flare classification using logistic regression and
                gradient boosting. Predicts probability of flare classes: None, C, M, X.

                Features include:
                - Sunspot complexity metrics (McIntosh, Mount Wilson)
                - X-ray flux trends and rolling statistics
                - Recency-weighted flare history
                - Magnetic field measurements

                Models are calibrated using isotonic or sigmoid calibration methods.
                """
            )

        with gr.Accordion("Survival Analysis Models", open=False):
            gr.Markdown(
                """
                Time-to-event modeling using Cox Proportional Hazards and Gradient Boosting
                Survival models. Predicts probability distribution of flare occurrence over
                time buckets (0h-168h).

                Time-varying covariates capture recent conditions:
                - Recent flux metrics (mean, max, trend)
                - Recent region complexity metrics
                - Recent flare history

                Models use survival functions to estimate time-to-event probabilities.
                """
            )

        gr.Markdown("## Known Issues")

        gr.Markdown(
            """
            - Limited training data may result in inaccurate predictions
            - Some features may be missing for certain time periods
            - Model performance not yet validated against official forecasts
            - Real-time data updates depend on NOAA API availability
            """
        )

        gr.Markdown("## Contact")

        gr.Markdown("For questions or issues, please refer to the project repository.")

    return tab
