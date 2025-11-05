# fmt: off
"""end-to-end pipeline for time-to-event survival analysis."""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.config import CONFIG
from src.models.survival_labeling import SurvivalLabeler
from src.models.time_varying_covariates import TimeVaryingCovariateEngineer
from src.models.survival_models import CoxProportionalHazards, GradientBoostingSurvival
from src.ml.experiment_tracking import MLflowTracker

logger = logging.getLogger(__name__)

# survival config
SURVIVAL_CONFIG = CONFIG.get("survival", {})


class SurvivalAnalysisPipeline:
    """end-to-end pipeline for survival analysis of time-to-event flare prediction."""

    def __init__(
        self,
        target_flare_class: str = "X",
        max_time_hours: int = 168,
        cox_penalizer: float = 0.1,
        gb_n_estimators: int = 100,
        gb_learning_rate: float = 0.1,
        random_state: Optional[int] = None,
        use_mlflow: bool = True,
    ):
        """
        initialize survival analysis pipeline.

        args:
            target_flare_class: flare class to predict (X, M, C, etc.)
            max_time_hours: maximum observation time (censoring window)
            cox_penalizer: l2 penalty for cox model
            gb_n_estimators: number of trees for gradient boosting
            use_mlflow: whether to use mlflow tracking
            gb_learning_rate: learning rate for gradient boosting
            random_state: random seed
        """
        self.labeler = SurvivalLabeler(target_flare_class, max_time_hours)
        self.covariate_engineer = TimeVaryingCovariateEngineer()
        self.cox_model = CoxProportionalHazards(penalizer=cox_penalizer)
        self.gb_model = GradientBoostingSurvival(
            n_estimators=gb_n_estimators,
            learning_rate=gb_learning_rate,
            random_state=random_state,
        )

        self.target_flare_class = target_flare_class
        self.max_time_hours = max_time_hours
        self.use_mlflow = use_mlflow
        self.mlflow_tracker = MLflowTracker() if use_mlflow else None
        self.is_fitted = False

    def prepare_dataset(
        self,
        timestamps: List[datetime],
        region_number: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        prepare dataset with time-varying covariates and survival labels.

        args:
            timestamps: list of observation timestamps
            region_number: optional region number

        returns:
            dataframe with covariates and survival labels
        """
        logger.info(f"preparing dataset for {len(timestamps)} timestamps")

        # compute time-varying covariates (with progress bar)
        covariates_df = self.covariate_engineer.compute_time_varying_covariates_batch(
            timestamps,
            region_number,
            show_progress=True,
        )

        if len(covariates_df) == 0:
            logger.warning("no covariates computed")
            return pd.DataFrame()

        # create survival labels (with progress bar)
        labels_df = self.labeler.create_survival_labels(timestamps, region_number, show_progress=True)

        if len(labels_df) == 0:
            logger.warning("no survival labels created")
            return pd.DataFrame()

        # merge covariates and labels
        dataset = covariates_df.merge(labels_df, on="timestamp", how="inner")

        if len(dataset) == 0:
            logger.warning("no data after merging covariates and labels")
            return pd.DataFrame()

        logger.info(f"dataset prepared with {len(dataset)} samples")
        logger.info(f"event rate: {dataset['event'].mean():.2%}")

        return dataset

    def train_and_evaluate(
        self,
        dataset: pd.DataFrame,
        test_size: float = 0.2,
        models: Optional[List[str]] = None,
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        train and evaluate survival models.

        args:
            dataset: dataframe with covariates and survival labels
            test_size: fraction of data for testing
            models: list of models to train (["cox", "gb"] or None for both)
            run_name: optional mlflow run name

        returns:
            dict with training results and evaluation metrics
        """
        if models is None:
            models = ["cox", "gb"]

        # start mlflow run if enabled
        mlflow_run = None
        if self.use_mlflow and self.mlflow_tracker:
            run_name_default = (
                f"survival_{self.target_flare_class}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )
            mlflow_run = self.mlflow_tracker.start_run(
                run_name=run_name or run_name_default,
                tags={
                    "model_type": "survival",
                    "pipeline": "survival",
                    "target_class": self.target_flare_class,
                },
            )

            # log parameters
            self.mlflow_tracker.log_params({
                "dataset_size": len(dataset),
                "test_size": test_size,
                "target_flare_class": self.target_flare_class,
                "max_time_hours": self.max_time_hours,
                "cox_penalizer": self.cox_model.penalizer,
                "gb_n_estimators": self.gb_model.n_estimators,
                "gb_learning_rate": self.gb_model.learning_rate,
                "models": ",".join(models),
            })

        results = {
            "train_size": 0,
            "test_size": 0,
            "cox_trained": False,
            "gb_trained": False,
            "cox_c_index_train": None,
            "cox_c_index_test": None,
            "gb_c_index_train": None,
            "gb_c_index_test": None,
        }

        # split dataset
        n_test = int(len(dataset) * test_size)
        test_df = dataset.tail(n_test).copy()
        train_df = dataset.head(len(dataset) - n_test).copy()

        results["train_size"] = len(train_df)
        results["test_size"] = len(test_df)

        logger.info(f"train size: {len(train_df)}, test size: {len(test_df)}")

        # train cox ph model
        if "cox" in models:
            try:
                logger.info("training cox proportional hazards model")
                self.cox_model.fit(train_df)

                # evaluate
                cox_c_train = self.cox_model.compute_concordance_index(train_df)
                cox_c_test = self.cox_model.compute_concordance_index(test_df)

                results["cox_trained"] = True
                results["cox_c_index_train"] = float(cox_c_train)
                results["cox_c_index_test"] = float(cox_c_test)

                logger.info(f"cox ph c-index: train={cox_c_train:.4f}, test={cox_c_test:.4f}")

                # log to mlflow
                if self.use_mlflow and self.mlflow_tracker:
                    try:
                        self.mlflow_tracker.log_metrics({
                            "cox_c_index_train": float(cox_c_train),
                            "cox_c_index_test": float(cox_c_test),
                        })
                        # log model (cox models are custom, so we'll save as joblib artifact)
                        import tempfile
                        import os
                        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                            joblib.dump(self.cox_model, tmp.name)
                            self.mlflow_tracker.log_dataset(tmp.name, "cox_model")
                            os.unlink(tmp.name)
                    except Exception as e:
                        logger.warning(f"failed to log cox model to mlflow: {e}")

            except Exception as e:
                logger.error(f"error training cox model: {e}")
                results["cox_trained"] = False

        # train gradient boosting survival model
        if "gb" in models:
            try:
                logger.info("training gradient boosting survival model")
                self.gb_model.fit(train_df)

                # evaluate
                gb_c_train = self.gb_model.compute_concordance_index(train_df)
                gb_c_test = self.gb_model.compute_concordance_index(test_df)

                results["gb_trained"] = True
                results["gb_c_index_train"] = float(gb_c_train)
                results["gb_c_index_test"] = float(gb_c_test)

                logger.info(f"gb survival c-index: train={gb_c_train:.4f}, test={gb_c_test:.4f}")

                # log to mlflow
                if self.use_mlflow and self.mlflow_tracker:
                    try:
                        self.mlflow_tracker.log_metrics({
                            "gb_c_index_train": float(gb_c_train),
                            "gb_c_index_test": float(gb_c_test),
                        })
                        # log model (gb models are custom, so we'll save as joblib artifact)
                        import tempfile
                        import os
                        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
                            joblib.dump(self.gb_model, tmp.name)
                            self.mlflow_tracker.log_dataset(tmp.name, "gb_model")
                            os.unlink(tmp.name)
                    except Exception as e:
                        logger.warning(f"failed to log gb model to mlflow: {e}")

            except Exception as e:
                logger.error(f"error training gb model: {e}")
                results["gb_trained"] = False

        # only mark as fitted if at least one model trained successfully
        if results.get("cox_trained") or results.get("gb_trained"):
            self.is_fitted = True
        else:
            logger.error("no models trained successfully - cannot mark pipeline as fitted")
            self.is_fitted = False

        # end mlflow run
        if self.use_mlflow and self.mlflow_tracker and mlflow_run:
            try:
                self.mlflow_tracker.end_run(status="FINISHED")
            except Exception as e:
                logger.warning(f"failed to end mlflow run: {e}")

        return results

    def predict_survival_probabilities(
        self,
        timestamp: datetime,
        region_number: Optional[int] = None,
        model_type: str = "cox",
        time_buckets: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        predict survival probabilities (probability of no flare in time buckets).

        args:
            timestamp: observation timestamp
            region_number: optional region number
            model_type: "cox" or "gb"
            time_buckets: list of bucket boundaries in hours

        returns:
            dict with survival probabilities over time buckets
        """
        if not self.is_fitted:
            raise ValueError("model must be trained before prediction")

        # compute covariates
        covariates_df = self.covariate_engineer.compute_time_varying_covariates(
            timestamp,
            region_number,
        )

        if len(covariates_df) == 0:
            raise ValueError("could not compute covariates")

        # predict survival function
        if model_type == "cox":
            # ensure we only use features that were used during training
            if self.cox_model.is_fitted and self.cox_model.feature_cols_ is not None:
                # filter to training features only
                available_features = [f for f in self.cox_model.feature_cols_ if f in covariates_df.columns]
                if len(available_features) < len(self.cox_model.feature_cols_):
                    missing = set(self.cox_model.feature_cols_) - set(available_features)
                    logger.warning(f"missing {len(missing)} features for prediction: {list(missing)[:5]}")
                    # add missing features with default values (0.0)
                    for feat in missing:
                        covariates_df[feat] = 0.0
                    # now get all required features
                    available_features = self.cox_model.feature_cols_

                # ensure columns are in the same order as training
                covariates_df = covariates_df[available_features]

            # ensure time_points cover full bucket range (0 to max bucket)
            if time_buckets is None:
                time_buckets = self.labeler.time_buckets
            max_time = max(time_buckets) if time_buckets else 168
            # request time_points up to max bucket to avoid extrapolation issues
            requested_time_points = np.linspace(0, max_time, max(int(max_time) + 1, 169))  # at least hourly

            survival_array, time_points = self.cox_model.predict_survival_function(
                covariates_df, time_points=requested_time_points
            )
            survival_probs = survival_array[0]  # single sample

            # debug logging
            logger.info(
                f"cox survival function: min={survival_probs.min():.6f}, "
                f"max={survival_probs.max():.6f}"
            )
            logger.info(
                f"cox time_points: min={time_points.min():.2f}h, "
                f"max={time_points.max():.2f}h, count={len(time_points)}"
            )
            survival_24h = survival_probs[min(24, len(survival_probs) - 1)]
            logger.info(
                f"cox survival at key points: 0h={survival_probs[0]:.6f}, "
                f"24h={survival_24h:.6f}, 168h={survival_probs[-1]:.6f}"
            )

            # check if survival is flat
            survival_range = survival_probs.max() - survival_probs.min()
            if survival_range < 0.001:
                logger.warning(
                    f"cox survival function is nearly flat (range={survival_range:.6f}). "
                    f"probabilities will be near zero."
                )
        elif model_type == "gb":
            # ensure we only use features that were used during training
            if self.gb_model.is_fitted and self.gb_model.feature_cols_ is not None:
                # filter to training features only
                available_features = [f for f in self.gb_model.feature_cols_ if f in covariates_df.columns]
                if len(available_features) < len(self.gb_model.feature_cols_):
                    missing = set(self.gb_model.feature_cols_) - set(available_features)
                    logger.warning(f"missing {len(missing)} features for prediction: {list(missing)[:5]}")
                    # add missing features with default values (0.0)
                    for feat in missing:
                        covariates_df[feat] = 0.0
                    # now get all required features
                    available_features = self.gb_model.feature_cols_

                # ensure columns are in the same order as training
                covariates_df = covariates_df[available_features]

            # ensure time_points cover full bucket range (0 to max bucket)
            if time_buckets is None:
                time_buckets = self.labeler.time_buckets
            max_time = max(time_buckets) if time_buckets else 168
            # request time_points up to max bucket
            requested_time_points = np.linspace(0, max_time, max(int(max_time) + 1, 169))  # at least hourly

            survival_array, time_points = self.gb_model.predict_survival_function(
                covariates_df,
                time_points=requested_time_points
            )
            survival_probs = survival_array[0]  # single sample
        else:
            raise ValueError(f"unknown model type: {model_type}")

        # compute probability distribution over time buckets
        if time_buckets is None:
            time_buckets = self.labeler.time_buckets

        prob_dist = self.labeler.compute_probability_distribution(
            survival_probs,
            time_points,
            time_buckets,
        )

        # debug: check if all probabilities are zero
        total_prob = sum(prob_dist.values())
        logger.info(f"total probability across all buckets: {total_prob:.6f}")

        if total_prob < 0.001:
            survival_min = survival_probs.min()
            survival_max = survival_probs.max()
            time_min = time_points.min()
            time_max = time_points.max()
            logger.warning(
                f"total probability across all buckets is very low ({total_prob:.6f}). "
                f"survival function may be too flat. "
                f"survival range: [{survival_min:.6f}, {survival_max:.6f}], "
                f"time_points range: [{time_min:.2f}h, {time_max:.2f}h]"
            )
            # log sample bucket probabilities for debugging
            sample_buckets = list(prob_dist.items())[:3]
            logger.info(f"sample bucket probabilities: {sample_buckets}")

            # log actual survival values at bucket boundaries for debugging
            if len(time_buckets) > 0:
                bucket_indices = [int(t) for t in time_buckets if t <= time_points.max()]
                if bucket_indices:
                    bucket_data = [
                        (t, survival_probs[min(int(t), len(survival_probs) - 1)])
                        for t in time_buckets[:4]
                    ]
                    logger.info(f"survival at bucket boundaries: {bucket_data}")

        # also compute hazard (risk) score
        # reuse the filtered covariates_df from above
        if model_type == "cox":
            hazard = self.cox_model.predict_partial_hazard(covariates_df)[0]
        else:
            hazard = self.gb_model.predict_hazard(covariates_df)[0]

        return {
            "timestamp": timestamp,
            "region_number": region_number,
            "model_type": model_type,
            "hazard_score": float(hazard),
            "time_buckets": time_buckets,
            "probability_distribution": prob_dist,
            "survival_function": {
                "time_points": time_points.tolist(),
                "probabilities": survival_probs.tolist(),
            },
        }

    def compare_models(
        self,
        dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        compare cox and gb models side-by-side.

        args:
            dataset: dataframe with covariates and survival labels

        returns:
            dataframe comparing model performance
        """
        results = []

        for model_name, model in [("cox", self.cox_model), ("gb", self.gb_model)]:
            if not model.is_fitted:  # type: ignore[attr-defined]
                continue

            try:
                if model_name == "cox":
                    c_index = model.compute_concordance_index(dataset)  # type: ignore[attr-defined]
                else:
                    c_index = model.compute_concordance_index(dataset)  # type: ignore[attr-defined]

                results.append(
                    {
                        "model": model_name,
                        "c_index": float(c_index),
                        "n_samples": len(dataset),
                    }
                )
            except Exception as e:
                logger.error(f"error evaluating {model_name}: {e}")

        return pd.DataFrame(results)

    def save_model(self, filepath: str):
        """
        save trained pipeline to file.

        args:
            filepath: path to save model
        """
        if not self.is_fitted:
            raise ValueError("model must be trained before saving")

        model_data = {
            "target_flare_class": self.target_flare_class,
            "max_time_hours": self.max_time_hours,
            "cox_model": self.cox_model if self.cox_model.is_fitted else None,
            "gb_model": self.gb_model if self.gb_model.is_fitted else None,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> "SurvivalAnalysisPipeline":
        """
        load trained pipeline from file.

        args:
            filepath: path to saved model

        returns:
            loaded pipeline
        """
        model_data = joblib.load(filepath)

        pipeline = cls(
            target_flare_class=model_data["target_flare_class"],
            max_time_hours=model_data["max_time_hours"],
        )

        if model_data.get("cox_model"):
            pipeline.cox_model = model_data["cox_model"]

        if model_data.get("gb_model"):
            pipeline.gb_model = model_data["gb_model"]

        pipeline.is_fitted = model_data["is_fitted"]

        logger.info(f"model loaded from {filepath}")
        return pipeline
# fmt: on
