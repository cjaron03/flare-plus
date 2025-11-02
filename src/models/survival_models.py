# fmt: off
"""survival analysis models - cox proportional hazards and gradient boosting survival."""

import logging
from typing import Optional, Dict, Any, List, Tuple
import warnings

import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)


class CoxProportionalHazards:
    """cox proportional hazards model for survival analysis."""

    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0):
        """
        initialize cox ph model.

        args:
            penalizer: l2 penalty coefficient
            l1_ratio: l1/l2 ratio (0.0 = pure l2, 1.0 = pure l1)
        """
        self.model = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self.is_fitted = False
        self.feature_cols_ = None  # store feature columns used during training

    def fit(self, df: pd.DataFrame, duration_col: str = "duration", event_col: str = "event"):
        """
        fit cox ph model.

        args:
            df: dataframe with features, duration, and event columns
            duration_col: name of duration column
            event_col: name of event column
        """
        try:
            # prepare dataframe for lifelines
            # lifelines expects duration and event columns, plus feature columns
            feature_cols = [c for c in df.columns if c not in [duration_col, event_col, "timestamp", "region_number", "event_time"]]

            # ensure no NaN values in duration or event
            df_clean = df[[duration_col, event_col] + feature_cols].copy()
            df_clean = df_clean.dropna(subset=[duration_col, event_col])

            if len(df_clean) == 0:
                raise ValueError("no valid data after cleaning")

            # remove constant features (zero variance) to avoid divide-by-zero in normalization
            constant_features = []
            for col in feature_cols:
                if df_clean[col].std() == 0 or pd.isna(df_clean[col].std()):
                    constant_features.append(col)

            if constant_features:
                logger.warning(f"removing {len(constant_features)} constant features: {constant_features[:10]}")
                feature_cols = [c for c in feature_cols if c not in constant_features]
                df_clean = df_clean[[duration_col, event_col] + feature_cols].copy()

            if len(feature_cols) == 0:
                raise ValueError("no features remaining after removing constant features")

            # fit model
            self.model.fit(df_clean, duration_col=duration_col, event_col=event_col)
            self.is_fitted = True
            self.feature_cols_ = feature_cols  # store for prediction

            logger.info(f"cox ph model fitted with {len(df_clean)} samples and {len(feature_cols)} features")

        except Exception as e:
            logger.error(f"error fitting cox ph model: {e}")
            raise

    def predict_survival_function(
        self,
        df: pd.DataFrame,
        duration_col: Optional[str] = None,
        time_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        predict survival function for given samples.

        args:
            df: dataframe with features
            duration_col: optional duration column (ignored for prediction)
            time_points: optional time points for survival function (default: 0 to max duration)

        returns:
            array of survival probabilities (shape: [n_samples, n_time_points])
        """
        if not self.is_fitted:
            raise ValueError("model must be fitted before prediction")

        # prepare feature columns - use same features as training
        if self.feature_cols_ is not None:
            # use features from training
            feature_cols = [c for c in self.feature_cols_ if c in df.columns]
        else:
            # fallback: exclude non-feature columns
            feature_cols = [c for c in df.columns if c not in ["duration", "event", "timestamp", "region_number", "event_time"]]

        if len(feature_cols) == 0:
            raise ValueError("no feature columns found for prediction")

        df_features = df[feature_cols].copy()

        # predict survival function
        # lifelines returns DataFrame with time index and columns for each sample
        survival_df = self.model.predict_survival_function(df_features)

        # lifelines returns DataFrame with time as index (in hours from training data)
        lifelines_time_points = survival_df.index.values
        lifelines_survival = survival_df.values.T  # [n_samples, n_time_points_from_lifelines]
        
        # debug: log what lifelines actually returns
        logger.debug(f"lifelines returned {len(lifelines_time_points)} time points: min={lifelines_time_points.min():.2f}h, max={lifelines_time_points.max():.2f}h")
        logger.debug(f"lifelines survival range: min={lifelines_survival.min():.6f}, max={lifelines_survival.max():.6f}")
        if len(lifelines_time_points) > 0 and len(lifelines_survival) > 0:
            logger.debug(f"lifelines survival at boundaries: start={lifelines_survival[0][0]:.6f}, end={lifelines_survival[0][-1]:.6f}")

        # if specific time_points requested, interpolate to those points
        if time_points is not None:
            # interpolate survival function to requested time points
            n_samples = lifelines_survival.shape[0]
            n_requested = len(time_points)
            survival_interpolated = np.zeros((n_samples, n_requested))
            
            for i in range(n_samples):
                # interpolate each sample's survival function
                # get the last survival value for extrapolation (survival shouldn't jump to 0, should decay gradually)
                last_survival = lifelines_survival[i][-1] if len(lifelines_survival[i]) > 0 else 1.0
                
                survival_interpolated[i] = np.interp(
                    time_points,
                    lifelines_time_points,
                    lifelines_survival[i],
                    left=1.0,  # before first time point, survival = 1.0 (no events yet)
                    right=last_survival  # after last observed time, use last survival value (extrapolate constant)
                )
            
            return survival_interpolated, time_points
        else:
            # use time index from survival DataFrame
            return lifelines_survival, lifelines_time_points

    def predict_partial_hazard(self, df: pd.DataFrame) -> np.ndarray:
        """
        predict partial hazard (relative risk) for given samples.

        args:
            df: dataframe with features

        returns:
            array of partial hazards
        """
        if not self.is_fitted:
            raise ValueError("model must be fitted before prediction")

        # use same features as training
        if self.feature_cols_ is not None:
            feature_cols = [c for c in self.feature_cols_ if c in df.columns]
            # ensure we have all required features
            if len(feature_cols) < len(self.feature_cols_):
                missing = set(self.feature_cols_) - set(feature_cols)
                raise ValueError(
                    f"missing {len(missing)} required features for prediction: {list(missing)[:10]}"
                )
        else:
            feature_cols = [c for c in df.columns if c not in ["duration", "event", "timestamp", "region_number", "event_time"]]

        if len(feature_cols) == 0:
            raise ValueError("no feature columns found for prediction")

        df_features = df[feature_cols].copy()

        partial_hazards = self.model.predict_partial_hazard(df_features)
        return partial_hazards.values

    def compute_concordance_index(self, df: pd.DataFrame) -> float:
        """
        compute concordance index (c-index) on given data.

        args:
            df: dataframe with features, duration, and event columns

        returns:
            c-index score
        """
        if not self.is_fitted:
            raise ValueError("model must be fitted before computing c-index")

        # use same features as training
        if self.feature_cols_ is not None:
            feature_cols = [c for c in self.feature_cols_ if c in df.columns]
        else:
            feature_cols = [c for c in df.columns if c not in ["duration", "event", "timestamp", "region_number", "event_time"]]

        if len(feature_cols) == 0:
            return 0.0

        df_clean = df[["duration", "event"] + feature_cols].copy()
        df_clean = df_clean.dropna()

        if len(df_clean) == 0:
            return 0.0

        # predict partial hazards
        partial_hazards = self.predict_partial_hazard(df_clean)

        # higher hazard = shorter survival time (inverse relationship)
        # so we negate for concordance
        c_index = concordance_index(df_clean["duration"], -partial_hazards, df_clean["event"])

        return c_index

    def get_summary(self) -> pd.DataFrame:
        """
        get model summary (coefficients, p-values, etc.).

        returns:
            dataframe with model summary
        """
        if not self.is_fitted:
            raise ValueError("model must be fitted before getting summary")

        return self.model.summary


class GradientBoostingSurvival(BaseEstimator, RegressorMixin):
    """gradient boosting survival model (using regression to predict hazard)."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ):
        """
        initialize gradient boosting survival model.

        args:
            n_estimators: number of boosting stages
            learning_rate: learning rate
            max_depth: maximum depth of trees
            min_samples_split: minimum samples to split
            random_state: random seed
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
        )

        self.is_fitted = False
        self.feature_cols_ = None

    def _prepare_target(self, df: pd.DataFrame, duration_col: str = "duration", event_col: str = "event") -> np.ndarray:
        """
        prepare target for survival regression.

        we use a transformed target: -log(survival_time) for events, or max_duration for censored.

        args:
            df: dataframe with duration and event columns
            duration_col: name of duration column
            event_col: name of event column

        returns:
            transformed target array
        """
        durations = df[duration_col].values
        events = df[event_col].values

        # for events: use negative log of duration (higher hazard for shorter times)
        # for censored: use max duration (assume they survived at least that long)
        max_duration = durations.max()

        target = np.where(
            events == 1,
            -np.log(durations + 1e-6),  # add small epsilon to avoid log(0)
            -np.log(max_duration + 1e-6),  # censored observations
        )

        return target

    def fit(self, df: pd.DataFrame, duration_col: str = "duration", event_col: str = "event"):
        """
        fit gradient boosting survival model.

        args:
            df: dataframe with features, duration, and event columns
            duration_col: name of duration column
            event_col: name of event column
        """
        try:
            # prepare features
            feature_cols = [c for c in df.columns if c not in [duration_col, event_col, "timestamp", "region_number", "event_time"]]

            df_clean = df[[duration_col, event_col] + feature_cols].copy()
            df_clean = df_clean.dropna()

            if len(df_clean) == 0:
                raise ValueError("no valid data after cleaning")

            # remove constant features (zero variance) to avoid numerical issues
            constant_features = []
            for col in feature_cols:
                if df_clean[col].std() == 0 or pd.isna(df_clean[col].std()):
                    constant_features.append(col)

            if constant_features:
                logger.warning(f"removing {len(constant_features)} constant features: {constant_features[:10]}")
                feature_cols = [c for c in feature_cols if c not in constant_features]
                df_clean = df_clean[[duration_col, event_col] + feature_cols].copy()

            if len(feature_cols) == 0:
                raise ValueError("no features remaining after removing constant features")

            self.feature_cols_ = feature_cols

            X = df_clean[feature_cols].values
            y = self._prepare_target(df_clean, duration_col, event_col)

            # fit model
            self.model.fit(X, y)
            self.is_fitted = True

            logger.info(f"gradient boosting survival model fitted with {len(df_clean)} samples and {len(feature_cols)} features")

        except Exception as e:
            logger.error(f"error fitting gradient boosting survival model: {e}")
            raise

    def predict_hazard(self, df: pd.DataFrame) -> np.ndarray:
        """
        predict hazard scores (higher = shorter survival time).

        args:
            df: dataframe with features

        returns:
            array of hazard scores
        """
        if not self.is_fitted:
            raise ValueError("model must be fitted before prediction")

        if self.feature_cols_ is None:
            raise ValueError("model must be fitted before prediction")

        X = df[self.feature_cols_].values
        hazards = self.model.predict(X)

        return hazards

    def predict_survival_function(
        self,
        df: pd.DataFrame,
        time_points: Optional[np.ndarray] = None,
        baseline_hazard: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        predict survival function using exponential distribution assumption.

        s(t) = exp(-hazard * t)

        args:
            df: dataframe with features
            time_points: time points for survival function
            baseline_hazard: optional baseline hazard function

        returns:
            tuple of (survival_array, time_points)
        """
        if not self.is_fitted:
            raise ValueError("model must be fitted before prediction")

        hazards = self.predict_hazard(df)

        if time_points is None:
            # default: 0 to 168 hours (7 days) in 1-hour increments
            time_points = np.arange(0, 169, 1)

        # convert hazard to survival function
        # higher hazard = lower survival probability
        # s(t) = exp(-hazard * t / scale_factor)
        # we normalize hazard to reasonable scale
        hazard_scale = np.abs(hazards).max()
        if hazard_scale > 0:
            hazards_normalized = hazards / hazard_scale
        else:
            hazards_normalized = hazards

        # compute survival probabilities
        survival_array = np.exp(-np.outer(hazards_normalized, time_points))

        # clip to [0, 1]
        survival_array = np.clip(survival_array, 0.0, 1.0)

        return survival_array, time_points

    def compute_concordance_index(self, df: pd.DataFrame, duration_col: str = "duration", event_col: str = "event") -> float:
        """
        compute concordance index (c-index) on given data.

        args:
            df: dataframe with features, duration, and event columns
            duration_col: name of duration column
            event_col: name of event column

        returns:
            c-index score
        """
        if not self.is_fitted:
            raise ValueError("model must be fitted before computing c-index")

        if self.feature_cols_ is None:
            raise ValueError("model must be fitted before computing c-index")

        df_clean = df[[duration_col, event_col] + self.feature_cols_].copy()
        df_clean = df_clean.dropna()

        if len(df_clean) == 0:
            return 0.0

        # predict hazards (higher hazard = shorter survival)
        hazards = self.predict_hazard(df_clean)

        # compute c-index
        c_index = concordance_index(df_clean[duration_col].values, -hazards, df_clean[event_col].values)

        return c_index

    def get_feature_importance(self) -> pd.DataFrame:
        """
        get feature importance from gradient boosting model.

        returns:
            dataframe with feature importance
        """
        if not self.is_fitted:
            raise ValueError("model must be fitted before getting feature importance")

        if self.feature_cols_ is None:
            raise ValueError("model must be fitted before getting feature importance")

        importance = self.model.feature_importances_

        return pd.DataFrame(
            {
                "feature": self.feature_cols_,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)
# fmt: on

