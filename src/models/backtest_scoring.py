"""Scoring a probabilistic forecast against what actually happened.

Kept apart from scripts/backtest_historical.py, which needs the database and
the modelling stack to produce forecasts. Scoring them afterwards is plain
arithmetic over plain dicts, so it lives here where it can be read and
tested on its own.
"""
import re
from datetime import timedelta
from typing import Dict, List

# bucket labels are written as "{start}h-{end}h" by
# survival_labeling.compute_probability_distribution.
BUCKET_LABEL = re.compile(r"^(\d+)h-(\d+)h$")


def reconstruct_survival_pipeline(pipeline_data: Dict) -> SurvivalAnalysisPipeline:
    """reconstruct survival pipeline from saved dict."""
    pipeline = SurvivalAnalysisPipeline(
        target_flare_class=pipeline_data.get("target_flare_class", "X"),
        max_time_hours=pipeline_data.get("max_time_hours", 168),
    )
    pipeline.is_fitted = pipeline_data.get("is_fitted", False)

    # restore model state
    if pipeline_data.get("cox_model"):
        pipeline.cox_model = pipeline_data["cox_model"]
    if pipeline_data.get("gb_model"):
        pipeline.gb_model = pipeline_data["gb_model"]

    return pipeline


def load_survival_model(model_path: Path) -> SurvivalAnalysisPipeline:
    """load survival model from joblib file."""
    logger.info(f"loading model from {model_path}")

    model_data = joblib.load(model_path)

    # reconstruct pipeline if saved as dict
    if isinstance(model_data, dict):
        return reconstruct_survival_pipeline(model_data)

    return model_data


def get_historical_flares(
    db,
    start_date: datetime,
    end_date: datetime,
    min_class: str = "C",
) -> List[Dict]:
    """retrieve historical flare events from database."""
    with db.get_session() as session:
        flares = (
            session.query(FlareEvent)
            .filter(
                FlareEvent.peak_time >= start_date,
                FlareEvent.peak_time <= end_date,
                FlareEvent.flare_class.isnot(None),
            )
            .order_by(FlareEvent.peak_time)
            .all()
        )

        # filter by minimum class and convert to dicts while in session
        class_order = {"B": 0, "C": 1, "M": 2, "X": 3}
        min_class_value = class_order.get(min_class, 0)

        filtered = []
        for flare in flares:
            flare_class = flare.flare_class[0] if flare.flare_class else "B"
            if class_order.get(flare_class, 0) >= min_class_value:
                filtered.append(
                    {
                        "peak_time": flare.peak_time,
                        "flare_class": flare.flare_class,
                        "class_magnitude": flare.class_magnitude,
                    }
                )

    return filtered


def probability_within(prob_dist: Dict[str, float], horizon_hours: int) -> float:
    """probability of at least one event inside the horizon.

    the buckets are disjoint by construction, one per interval, so the
    probability of an event somewhere in [0, horizon] is the sum of the
    buckets that close on or before it. a bucket straddling the horizon is
    not counted, which makes this an underestimate and never an
    overestimate.
    """
    total = 0.0
    for label, p in prob_dist.items():
        m = BUCKET_LABEL.match(label)
        if m and int(m.group(2)) <= horizon_hours:
            total += float(p)
    return min(total, 1.0)


def compute_backtest_metrics(
    predictions: List[Dict],
    actuals: List[Dict],
    horizon_hours: int = 24,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """score the forecasts against what actually happened.

    a forecast is a probability and a probability is not a decision. to
    count hits and false alarms at all, the probability has to be turned
    into a yes or a no by a threshold, and that threshold has to be stated.
    the horizon has to be stated too, because "a flare is coming" means
    nothing without "by when".

    the earlier version of this function did neither. every sampled
    timestamp was counted as a positive forecast no matter what the model
    had said, and the window was the model's full 168h range, so across an
    active fortnight nearly every window contained a flare and the score
    came out high whatever the model predicted. those numbers described the
    sun, not the model. replacing the model with a constant would not have
    moved them.

    so the scores here are always reported next to climatology, which is
    what a forecaster gets for ignoring every input and repeating the base
    rate. a model has to beat that before it has shown anything at all.
    """
    if not predictions:
        return {"error": "no predictions", "n_predictions": 0, "n_actuals": len(actuals)}

    peak_times = sorted(a["peak_time"] for a in actuals)

    rows = []
    for pred in predictions:
        t = pred["timestamp"]
        end = t + timedelta(hours=horizon_hours)
        p_hat = probability_within(pred.get("probability_distribution", {}), horizon_hours)
        rows.append(
            {
                "timestamp": t,
                "p": p_hat,
                "occurred": any(t <= peak <= end for peak in peak_times),
                "forecast": p_hat >= threshold,
            }
        )

    tp = sum(1 for r in rows if r["forecast"] and r["occurred"])
    fp = sum(1 for r in rows if r["forecast"] and not r["occurred"])
    fn = sum(1 for r in rows if not r["forecast"] and r["occurred"])
    tn = sum(1 for r in rows if not r["forecast"] and not r["occurred"])

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # climatology. the rate at which a window contained a flare at all is
    # the whole of what a forecaster needs to know to score this well, so
    # it is the bar, not the floor.
    n = len(rows)
    n_windows_with_event = sum(1 for r in rows if r["occurred"])
    base_rate = n_windows_with_event / n

    truth = [1.0 if r["occurred"] else 0.0 for r in rows]
    brier = sum((r["p"] - y) ** 2 for r, y in zip(rows, truth)) / n
    brier_climatology = sum((base_rate - y) ** 2 for y in truth) / n
    # the fraction of climatology's error the model removed. zero means the
    # model is worth exactly as much as knowing the base rate and nothing
    # more. below zero means it is worth less than that.
    bss = (1 - brier / brier_climatology) if brier_climatology > 0 else 0.0

    # the two forecasters that read no inputs, scored the same way, because
    # a number is only good or bad next to what they get for free.
    always_yes_precision = base_rate
    always_yes_recall = 1.0 if n_windows_with_event else 0.0
    always_yes_f1 = (
        2 * always_yes_precision * always_yes_recall / (always_yes_precision + always_yes_recall)
        if (always_yes_precision + always_yes_recall)
        else 0.0
    )

    # one threshold is one arbitrary choice, and a model that is genuinely
    # informative separates the two classes at more than one of them.
    sweep = []
    for i in range(21):
        thr = i / 20
        s_tp = sum(1 for r in rows if r["p"] >= thr and r["occurred"])
        s_fp = sum(1 for r in rows if r["p"] >= thr and not r["occurred"])
        s_fn = sum(1 for r in rows if r["p"] < thr and r["occurred"])
        s_pr = s_tp / (s_tp + s_fp) if (s_tp + s_fp) else 0.0
        s_rc = s_tp / (s_tp + s_fn) if (s_tp + s_fn) else 0.0
        sweep.append(
            {
                "threshold": thr,
                "n_positive": s_tp + s_fp,
                "precision": s_pr,
                "recall": s_rc,
                "f1": 2 * s_pr * s_rc / (s_pr + s_rc) if (s_pr + s_rc) else 0.0,
            }
        )

    return {
        "horizon_hours": horizon_hours,
        "threshold": threshold,
        "n_predictions": n,
        "n_actuals": len(actuals),
        "n_windows_with_event": n_windows_with_event,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "brier_score": brier,
        "base_rate": base_rate,
        "brier_climatology": brier_climatology,
        "brier_skill_score": bss,
        "always_yes_precision": always_yes_precision,
        "always_yes_recall": always_yes_recall,
        "always_yes_f1": always_yes_f1,
        "always_no_accuracy": 1 - base_rate,
        "threshold_sweep": sweep,
    }
