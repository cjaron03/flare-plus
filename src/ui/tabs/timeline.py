"""historical timeline tab for ui dashboard."""

import logging
from datetime import datetime, timedelta

import gradio as gr

from src.ui.utils.data_queries import query_historical_flares
from src.ui.utils.charts import create_timeline_chart
from src.ui.utils.helpers import should_refresh

logger = logging.getLogger(__name__)


def build_timeline_tab(last_refresh: gr.State) -> gr.Blocks:
    """
    build historical timeline tab interface.

    args:
        last_refresh: state variable for last refresh time

    returns:
        gradio blocks for timeline tab
    """
    with gr.Blocks() as tab:
        gr.Markdown("## Historical Flare Timeline")
        gr.Markdown(
            "View historical flare events from the database. "
            "Filter by date range, minimum flare class, and region number."
        )

        with gr.Row():
            with gr.Column(scale=1):
                start_date = gr.DateTime(
                    label="Start Date",
                )
                end_date = gr.DateTime(
                    label="End Date",
                )

            with gr.Column(scale=1):
                min_class = gr.Dropdown(
                    label="Minimum Flare Class",
                    choices=["B", "C", "M", "X"],
                    value="B",
                )
                region_filter = gr.Number(
                    label="Region Number (optional)",
                    value=None,
                    precision=0,
                )

            with gr.Column(scale=1):
                refresh_button = gr.Button("Refresh Timeline", variant="primary")
                status_output = gr.Textbox(label="Status", interactive=False)

        timeline_plot = gr.Plot(label="Flare Timeline")

        def update_timeline(start, end, min_class_val, region_val, refresh_now):
            """update timeline visualization."""
            try:
                # handle None dates (gradio 4.44.1 DateTime doesn't support value parameter)
                if start is None:
                    start = datetime.now() - timedelta(days=30)
                if end is None:
                    end = datetime.now()

                # check throttling (10 minutes for timeline)
                if not refresh_now and not should_refresh(last_refresh.value, min_interval_minutes=10):
                    return (
                        None,
                        "Refresh throttled (wait 10 minutes or use 'Refresh Now' button)",
                    )

                if start >= end:
                    return (
                        None,
                        "Error: Start date must be before end date",
                    )

                # parse region
                region = int(region_val) if region_val is not None else None

                # query flares
                flares_df = query_historical_flares(
                    start_date=start,
                    end_date=end,
                    min_class=min_class_val,
                    region_number=region,
                )

                if flares_df.empty:
                    # create empty chart with message
                    chart = create_timeline_chart(flares_df)
                    status = f"No flares found in date range ({len(flares_df)} events)"
                else:
                    chart = create_timeline_chart(flares_df)
                    status = f"Found {len(flares_df)} flare events"

                if refresh_now:
                    last_refresh.value = datetime.now()

                return chart, status

            except Exception as e:
                logger.error(f"error updating timeline: {e}")
                return (
                    None,
                    f"Error: {str(e)}",
                )

        refresh_button.click(
            fn=lambda s, e, m, r: update_timeline(s, e, m, r, True),
            inputs=[start_date, end_date, min_class, region_filter],
            outputs=[timeline_plot, status_output],
        )

        # auto-update when filters change
        start_date.change(
            fn=lambda s, e, m, r: update_timeline(s, e, m, r, False),
            inputs=[start_date, end_date, min_class, region_filter],
            outputs=[timeline_plot, status_output],
        )
        end_date.change(
            fn=lambda s, e, m, r: update_timeline(s, e, m, r, False),
            inputs=[start_date, end_date, min_class, region_filter],
            outputs=[timeline_plot, status_output],
        )
        min_class.change(
            fn=lambda s, e, m, r: update_timeline(s, e, m, r, False),
            inputs=[start_date, end_date, min_class, region_filter],
            outputs=[timeline_plot, status_output],
        )
        region_filter.change(
            fn=lambda s, e, m, r: update_timeline(s, e, m, r, False),
            inputs=[start_date, end_date, min_class, region_filter],
            outputs=[timeline_plot, status_output],
        )

        # initial load
        tab.load(
            fn=lambda s, e, m, r: update_timeline(s, e, m, r, False),
            inputs=[start_date, end_date, min_class, region_filter],
            outputs=[timeline_plot, status_output],
        )

    return tab
