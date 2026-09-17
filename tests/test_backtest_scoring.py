"""Tests for backtest scoring.

The case these exist for: a scoring function that ignores the model's output
will happily report a high score, and the high score is the thing nobody
questions. The tests below pin the three behaviours that stop that, so the
old bug cannot come back quietly.
"""
from datetime import datetime, timedelta

import pytest

from src.models.backtest_scoring import compute_backtest_metrics, probability_within

T0 = datetime(2025, 10, 28)
BUCKETS = [(0, 6), (6, 12), (12, 24), (24, 48), (48, 72), (72, 96), (96, 120), (120, 168)]

# nine flares across thirteen days, the density of an active fortnight.
FLARE_DAYS = [0, 1, 3, 4, 6, 7, 9, 10, 12]
ACTUALS = [{"peak_time": T0 + timedelta(days=d, hours=6)} for d in FLARE_DAYS]


def flat_forecasts(p_per_bucket, n=14):
    """A model that has learned nothing: the same answer every day."""
    return [
        {
            "timestamp": T0 + timedelta(days=i),
            "probability_distribution": {
                f"{a}h-{b}h": p_per_bucket for a, b in BUCKETS
            },
        }
        for i in range(n)
    ]


def informed_forecasts(horizon_hours=24, n=14):
    """A model that does know: high before a flare, low otherwise."""
    peaks = [a["peak_time"] for a in ACTUALS]
    out = []
    for i in range(n):
        t = T0 + timedelta(days=i)
        hot = any(t <= p <= t + timedelta(hours=horizon_hours) for p in peaks)
        per = (0.9 if hot else 0.02) / 3
        out.append(
            {
                "timestamp": t,
                "probability_distribution": {
                    f"{a}h-{b}h": (per if b <= horizon_hours else 0.001)
                    for a, b in BUCKETS
                },
            }
        )
    return out


class TestProbabilityWithin:
    def test_sums_only_buckets_that_close_inside_the_horizon(self):
        dist = {"0h-6h": 0.1, "6h-12h": 0.2, "12h-24h": 0.3, "24h-48h": 0.4}
        assert probability_within(dist, 24) == pytest.approx(0.6)
        assert probability_within(dist, 12) == pytest.approx(0.3)
        assert probability_within(dist, 48) == pytest.approx(1.0)

    def test_straddling_bucket_is_excluded_not_prorated(self):
        # 24h-48h does not close by 36h, so it does not count. an
        # underestimate is acceptable here, an overestimate is not.
        assert probability_within({"0h-6h": 0.1, "24h-48h": 0.9}, 36) == pytest.approx(0.1)

    def test_never_exceeds_one(self):
        assert probability_within({f"{a}h-{b}h": 0.9 for a, b in BUCKETS}, 168) == 1.0

    def test_ignores_labels_it_does_not_recognise(self):
        assert probability_within({"0h-6h": 0.5, "lifetime": 0.4}, 24) == pytest.approx(0.5)


class TestConstantModelEarnsNoSkill:
    """The regression test for the bug this module was written to fix.

    A model emitting the same number every day knows nothing. It must not
    be able to score well, however common flares happen to be in the
    window, and the report must say so.
    """

    def test_confident_constant_matches_the_base_rate_and_scores_no_skill(self):
        m = compute_backtest_metrics(
            flat_forecasts(0.2), ACTUALS, horizon_hours=48, threshold=0.5
        )
        # the precision a constant earns is exactly the base rate, which is
        # the whole point: it is a property of the sun, not of the model.
        assert m["precision"] == pytest.approx(m["base_rate"])
        assert m["precision"] > 0.9
        # and the skill score refuses to be impressed by it.
        assert m["brier_skill_score"] <= 0

    def test_f1_alone_would_have_passed_it(self):
        m = compute_backtest_metrics(
            flat_forecasts(0.2), ACTUALS, horizon_hours=48, threshold=0.5
        )
        assert m["f1_score"] > 0.9                      # looks excellent
        assert m["f1_score"] <= m["always_yes_f1"]      # and is worth nothing

    def test_silent_constant_is_not_rewarded_either(self):
        m = compute_backtest_metrics(
            flat_forecasts(0.004), ACTUALS, horizon_hours=24, threshold=0.5
        )
        assert m["true_positives"] == 0
        assert m["brier_skill_score"] < 0


class TestInformedModelIsRewarded:
    def test_separating_the_classes_scores_real_skill(self):
        m = compute_backtest_metrics(
            informed_forecasts(24), ACTUALS, horizon_hours=24, threshold=0.5
        )
        assert m["brier_skill_score"] > 0.5
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)


class TestConfusionMatrix:
    def test_every_forecast_lands_in_exactly_one_cell(self):
        m = compute_backtest_metrics(flat_forecasts(0.2), ACTUALS, horizon_hours=24)
        total = (m["true_positives"] + m["false_positives"]
                 + m["false_negatives"] + m["true_negatives"])
        assert total == m["n_predictions"]

    def test_false_negatives_count_windows_not_flares(self):
        # the old code took actuals[0] of each window, so extra flares
        # inside one window were miscounted as misses. a silent model over
        # this period should record one false negative per event window and
        # no more.
        m = compute_backtest_metrics(
            flat_forecasts(0.0), ACTUALS, horizon_hours=24, threshold=0.5
        )
        assert m["false_negatives"] == m["n_windows_with_event"]
        assert m["false_negatives"] <= m["n_predictions"]

    def test_horizon_changes_the_base_rate(self):
        # the same fortnight is a different problem at 24h and at 168h, and
        # a score quoted without its horizon means nothing.
        narrow = compute_backtest_metrics(flat_forecasts(0.2), ACTUALS, horizon_hours=24)
        wide = compute_backtest_metrics(flat_forecasts(0.2), ACTUALS, horizon_hours=168)
        assert wide["base_rate"] > narrow["base_rate"]


class TestEdges:
    def test_no_predictions_returns_an_error_not_a_score(self):
        m = compute_backtest_metrics([], ACTUALS, horizon_hours=24)
        assert "error" in m
        assert "precision" not in m

    def test_no_flares_gives_a_zero_base_rate_and_no_division_error(self):
        m = compute_backtest_metrics(flat_forecasts(0.2), [], horizon_hours=24)
        assert m["base_rate"] == 0
        assert m["recall"] == 0
        assert m["brier_skill_score"] == 0

    def test_sweep_covers_the_unit_interval(self):
        m = compute_backtest_metrics(flat_forecasts(0.2), ACTUALS, horizon_hours=24)
        thresholds = [r["threshold"] for r in m["threshold_sweep"]]
        assert thresholds[0] == 0.0 and thresholds[-1] == 1.0
        # issuing fewer forecasts as the bar rises is the one monotonic
        # property the sweep must have.
        counts = [r["n_positive"] for r in m["threshold_sweep"]]
        assert counts == sorted(counts, reverse=True)
