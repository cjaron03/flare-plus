"""scenario exploration tab for ui dashboard."""

import logging

import gradio as gr

logger = logging.getLogger(__name__)


def build_scenario_tab() -> gr.Blocks:
    """
    build scenario exploration tab interface.

    returns:
        gradio blocks for scenario tab
    """
    with gr.Blocks() as tab:
        gr.Markdown("## Scenario Exploration")
        gr.Markdown(
            "This feature is planned for future implementation. "
            "Scenario exploration requires full feature engineering recalculation "
            "when adjusting sunspot metrics."
        )

        gr.Markdown("### Planned Features")
        gr.Markdown(
            """
            - Adjust sunspot metrics (McIntosh class, Mount Wilson class, region area)
            - Modify magnetic complexity scores and flux trends
            - Compare baseline vs scenario predictions
            - Visual delta charts showing probability changes
            """
        )

        gr.Markdown("### Current Workaround")
        gr.Markdown(
            "You can explore different scenarios by adjusting the timestamp and region number "
            "in the Predictions tab and comparing results manually."
        )

    return tab
