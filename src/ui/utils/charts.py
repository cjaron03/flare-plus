"""plotly chart generation utilities for ui dashboard."""

import logging
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

logger = logging.getLogger(__name__)


def create_timeline_chart(flares_df: pd.DataFrame) -> go.Figure:
    """
    create timeline chart for historical flare events.

    args:
        flares_df: dataframe with flare events (columns: peak_time, flare_class, class_category, etc.)

    returns:
        plotly figure
    """
    if flares_df.empty:
        # return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No flares found in selected time range",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Flare Class",
            showlegend=False,
        )
        return fig

    # color mapping for flare classes
    color_map = {
        "B": "#FFD700",  # yellow
        "C": "#FF8C00",  # orange
        "M": "#FF4500",  # red
        "X": "#8B008B",  # purple
    }

    # prepare data
    flares_df = flares_df.copy()
    flares_df["peak_time"] = pd.to_datetime(flares_df["peak_time"])

    # assign colors
    flares_df["color"] = flares_df["class_category"].map(color_map).fillna("#808080")

    # create scatter plot
    fig = go.Figure()

    for class_cat in ["B", "C", "M", "X"]:
        class_data = flares_df[flares_df["class_category"] == class_cat]
        if len(class_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=class_data["peak_time"],
                    y=[class_cat] * len(class_data),
                    mode="markers",
                    name=class_cat,
                    marker=dict(
                        size=10,
                        color=color_map.get(class_cat, "#808080"),
                        line=dict(width=1, color="black"),
                    ),
                    text=[
                        f"Class: {row['flare_class']}<br>"
                        f"Peak: {row['peak_time']}<br>"
                        f"Region: {row.get('active_region', 'N/A')}<br>"
                        f"Location: {row.get('location', 'N/A')}"
                        for _, row in class_data.iterrows()
                    ],
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Historical Flare Events Timeline",
        xaxis_title="Time",
        yaxis_title="Flare Class",
        hovermode="closest",
        showlegend=True,
        height=400,
    )

    return fig


def create_probability_bar_chart(
    probabilities: Dict[str, float],
    title: str = "Class Probabilities",
) -> go.Figure:
    """
    create bar chart for classification probabilities.

    args:
        probabilities: dict mapping class names to probabilities
        title: chart title

    returns:
        plotly figure
    """
    if not probabilities:
        fig = go.Figure()
        fig.add_annotation(
            text="No probability data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(title=title)
        return fig

    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]  # convert to percentage

    fig = go.Figure(
        data=[
            go.Bar(
                x=classes,
                y=probs,
                marker_color="steelblue",
                text=[f"{p:.1f}%" for p in probs],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Flare Class",
        yaxis_title="Probability (%)",
        yaxis_range=[0, max(probs) * 1.2 if probs else 100],
        height=300,
    )

    return fig


def create_survival_curve_chart(
    survival_data: Dict[str, List[float]],
    title: str = "Survival Function",
) -> go.Figure:
    """
    create survival curve line chart.

    args:
        survival_data: dict with 'time_points' and 'probabilities' lists
        title: chart title

    returns:
        plotly figure
    """
    time_points = survival_data.get("time_points", [])
    probabilities = survival_data.get("probabilities", [])

    if not time_points or not probabilities:
        fig = go.Figure()
        fig.add_annotation(
            text="No survival data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(title=title)
        return fig

    fig = go.Figure(
        data=[
            go.Scatter(
                x=time_points,
                y=probabilities,
                mode="lines",
                name="Survival Probability",
                line=dict(color="steelblue", width=2),
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (hours)",
        yaxis_title="Survival Probability",
        yaxis_range=[0, 1.05],
        height=300,
        hovermode="x unified",
    )

    return fig


def create_probability_distribution_chart(
    probability_distribution: Dict[str, float],
    title: str = "Survival Probability Distribution",
) -> go.Figure:
    """
    create bar chart for survival probability distribution over time buckets.

    args:
        probability_distribution: dict mapping bucket ranges (e.g., "0h-6h") to probabilities
        title: chart title

    returns:
        plotly figure
    """
    if not probability_distribution:
        fig = go.Figure()
        fig.add_annotation(
            text="No probability distribution available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(title=title)
        return fig

    # sort buckets by start time
    sorted_buckets = sorted(probability_distribution.items(), key=lambda x: _parse_bucket_range(x[0]))

    bucket_ranges = [b[0] for b in sorted_buckets]
    probs = [b[1] * 100 for b in sorted_buckets]  # convert to percentage

    fig = go.Figure(
        data=[
            go.Bar(
                x=bucket_ranges,
                y=probs,
                marker_color="steelblue",
                text=[f"{p:.1f}%" for p in probs],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time Bucket",
        yaxis_title="Probability (%)",
        yaxis_range=[0, max(probs) * 1.2 if probs else 100],
        height=300,
        xaxis_tickangle=-45,
    )

    return fig


def create_delta_comparison_chart(
    baseline_probs: Dict[str, float],
    scenario_probs: Dict[str, float],
    title: str = "Probability Change (Scenario vs Baseline)",
) -> go.Figure:
    """
    create bar chart showing probability changes between baseline and scenario.

    args:
        baseline_probs: baseline probabilities
        scenario_probs: scenario probabilities
        title: chart title

    returns:
        plotly figure
    """
    # get all keys
    all_keys = set(list(baseline_probs.keys()) + list(scenario_probs.keys()))

    if not all_keys:
        fig = go.Figure()
        fig.add_annotation(
            text="No probability data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        fig.update_layout(title=title)
        return fig

    # calculate deltas
    keys = []
    deltas = []
    colors = []

    for key in sorted(all_keys):
        baseline = baseline_probs.get(key, 0.0)
        scenario = scenario_probs.get(key, 0.0)
        delta = scenario - baseline
        keys.append(key)
        deltas.append(delta * 100)  # convert to percentage

        # color: green for increase, red for decrease, gray for no change
        if abs(delta) < 0.001:
            colors.append("gray")
        elif delta > 0:
            colors.append("green")
        else:
            colors.append("red")

    fig = go.Figure(
        data=[
            go.Bar(
                x=keys,
                y=deltas,
                marker_color=colors,
                text=[f"{d:+.1f}%" for d in deltas],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Flare Class",
        yaxis_title="Probability Change (%)",
        height=300,
        yaxis_range=[
            min(deltas) * 1.2 if deltas else -10,
            max(deltas) * 1.2 if deltas else 10,
        ],
    )

    return fig


def _parse_bucket_range(bucket_str: str) -> int:
    """parse bucket range string to integer for sorting (e.g., '0h-6h' -> 0)."""
    try:
        start = bucket_str.split("-")[0].replace("h", "")
        return int(start)
    except (ValueError, IndexError):
        return 0
