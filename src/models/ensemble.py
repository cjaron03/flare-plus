"""Simple probability-averaging ensemble for pre-trained classifiers."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np


class ProbabilityAveragingEnsemble:
    """Average class probabilities from multiple fitted classifiers."""

    def __init__(
        self,
        models: List[Any],
        model_names: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
    ):
        if len(models) < 2:
            raise ValueError("ensemble requires at least two models")
        self.models = list(models)
        self.model_names = list(model_names) if model_names else [f"model_{idx}" for idx in range(len(models))]
        if len(self.model_names) != len(self.models):
            raise ValueError("model_names length must match number of models")

        if weights is None:
            weights_arr = np.ones(len(self.models), dtype=float)
        else:
            if len(weights) != len(self.models):
                raise ValueError("weights length must match number of models")
            weights_arr = np.asarray(weights, dtype=float)
            if np.any(weights_arr < 0):
                raise ValueError("weights must be non-negative")

        total = float(weights_arr.sum())
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = (weights_arr / total).tolist()

        # keep classes_ when available for compatibility with downstream tools.
        self.classes_ = getattr(self.models[0], "classes_", None)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return weighted average probability distribution."""
        weighted_sum: Optional[np.ndarray] = None
        expected_shape = None

        for weight, model in zip(self.weights, self.models):
            probs = np.asarray(model.predict_proba(X), dtype=float)
            if expected_shape is None:
                expected_shape = probs.shape
            elif probs.shape != expected_shape:
                raise ValueError(f"ensemble probability shape mismatch: expected {expected_shape}, got {probs.shape}")

            contrib = probs * float(weight)
            weighted_sum = contrib if weighted_sum is None else weighted_sum + contrib

        if weighted_sum is None:
            raise ValueError("ensemble produced no probabilities")

        # Normalize rows defensively in case component models are slightly off.
        row_sums = weighted_sum.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return weighted_sum / row_sums

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return argmax class index."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
