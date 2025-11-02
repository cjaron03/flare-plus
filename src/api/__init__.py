"""api service for model predictions."""

from src.api.service import PredictionService
from src.api.app import create_app

__all__ = ["PredictionService", "create_app"]
