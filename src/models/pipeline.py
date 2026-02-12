# fmt: off
"""main pipeline for short-term classification."""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import sys

try:
    from tqdm import tqdm
    HAS_TQDM = True
    # configure tqdm for docker/non-interactive terminals
    TQDM_KWARGS = {
        "file": sys.stderr,  # use stderr to avoid buffering
        "mininterval": 1.0,  # update at least every second
        "miniters": 1,  # update after each iteration
        "disable": False,  # explicitly enable
    }
except ImportError:
    HAS_TQDM = False
    TQDM_KWARGS = {}

    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.config import CONFIG
from src.features.pipeline import FeatureEngineer
from src.models.labeling import FlareLabeler
from src.models.training import ModelTrainer
from src.models.evaluation import ModelEvaluator
from src.models.ensemble import ProbabilityAveragingEnsemble
from src.ml.experiment_tracking import MLflowTracker, mlflow_enabled

logger = logging.getLogger(__name__)

# model config
MODEL_CONFIG = CONFIG.get("model", {})
TARGET_WINDOWS = MODEL_CONFIG.get("target_windows", [24, 48])


class ClassificationPipeline:
    """end-to-end pipeline for short-term flare classification."""

    def __init__(
        self,
        use_smote: bool = True,
        cv_folds: int = 5,
        calibrate: bool = True,
        random_state: int = 42,
        use_mlflow: Optional[bool] = None,
    ):
        """
        initialize classification pipeline.

        args:
            use_smote: whether to use smote for oversampling
            cv_folds: number of cross-validation folds
            calibrate: whether to calibrate probabilities
            random_state: random seed
            use_mlflow: whether to use mlflow tracking (None -> auto from config/env)
        """
        self.feature_engineer = FeatureEngineer()
        self.labeler = FlareLabeler()
        self.trainer = ModelTrainer(
            use_smote=use_smote, cv_folds=cv_folds, random_state=random_state
        )
        self.evaluator = ModelEvaluator()
        self.use_smote = use_smote
        self.cv_folds = cv_folds
        self.calibrate = calibrate
        self.random_state = random_state
        resolved_use_mlflow = mlflow_enabled() if use_mlflow is None else use_mlflow
        self.use_mlflow = resolved_use_mlflow
        self.mlflow_tracker = MLflowTracker() if resolved_use_mlflow else None
        self.models: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Any] = {}

    @staticmethod
    def _chronological_train_test_split(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """split arrays in time order so train data always precedes test data."""
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")

        n_samples = len(X)
        if n_samples < 2:
            raise ValueError("at least 2 samples are required for train/test split")

        split_idx = int(np.floor(n_samples * (1 - test_size)))
        split_idx = max(1, min(split_idx, n_samples - 1))

        return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]

    @staticmethod
    def _extract_selection_metrics(
        evaluation_results: Dict[str, Any],
    ) -> Dict[str, Optional[float]]:
        """Extract comparable metrics used for selecting the best deployed model."""
        report = evaluation_results.get("classification_report", {})
        weighted = report.get("weighted avg", {})
        event_metrics = evaluation_results.get("event_metrics", {}) or {}
        calibrated_event_metrics = evaluation_results.get("calibrated_event_metrics", {}) or {}

        brier = evaluation_results.get("brier_score", {}).get("macro_avg")
        calibrated_brier = evaluation_results.get("calibrated_brier_score", {}).get("macro_avg")

        test_event_f1 = None
        test_event_recall = None
        test_event_precision = None
        if calibrated_event_metrics.get("available"):
            test_event_f1 = calibrated_event_metrics.get("f1")
            test_event_recall = calibrated_event_metrics.get("recall")
            test_event_precision = calibrated_event_metrics.get("precision")
        elif event_metrics.get("available"):
            test_event_f1 = event_metrics.get("f1")
            test_event_recall = event_metrics.get("recall")
            test_event_precision = event_metrics.get("precision")

        return {
            "test_accuracy": report.get("accuracy"),
            "test_weighted_f1": weighted.get("f1-score"),
            "test_event_f1": test_event_f1,
            "test_event_recall": test_event_recall,
            "test_event_precision": test_event_precision,
            "test_brier_macro": brier,
            "test_brier_macro_calibrated": calibrated_brier,
        }

    @staticmethod
    def _build_feature_fill_values(
        X_train_eval: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        """Build per-feature inference defaults from training medians."""
        fill_values: Dict[str, float] = {}
        for idx, name in enumerate(feature_names):
            col = np.asarray(X_train_eval[:, idx], dtype=float)
            finite = col[np.isfinite(col)]
            if len(finite) == 0:
                fill_values[name] = 0.0
            else:
                fill_values[name] = float(np.median(finite))
        return fill_values

    @staticmethod
    def _align_feature_matrix(
        X: np.ndarray,
        source_feature_names: List[str],
        target_feature_names: List[str],
    ) -> np.ndarray:
        """Project feature matrix into target feature order."""
        if source_feature_names == target_feature_names:
            return X
        indices = [source_feature_names.index(name) for name in target_feature_names]
        return X[:, indices]

    def _build_logistic_tree_ensemble(
        self,
        window_results: Dict[str, Any],
        feature_cols: List[str],
        X_test: np.ndarray,
        X_train: np.ndarray,
        y_test: np.ndarray,
        classes: List[str],
        label_encoder: Any,
        plot_reliability: bool,
        reliability_dir: Optional[str],
        window: int,
    ) -> Optional[Dict[str, Any]]:
        """Build/evaluate logistic + best-tree probability averaging ensemble."""
        if "logistic" not in window_results:
            return None

        tree_keys = [key for key in ["gradient_boosting", "lightgbm", "random_forest"] if key in window_results]
        if not tree_keys:
            return None

        best_tree_key = max(
            tree_keys,
            key=lambda key: ModelTrainer._model_selection_sort_key(window_results[key]["training_info"]),
        )

        logistic_entry = window_results["logistic"]
        tree_entry = window_results[best_tree_key]
        logistic_features = logistic_entry.get("feature_names", feature_cols)
        tree_features = tree_entry.get("feature_names", feature_cols)
        if logistic_features != tree_features:
            logger.warning(
                "skipping ensemble for %sh: logistic/tree feature schemas differ (%d vs %d)",
                window,
                len(logistic_features),
                len(tree_features),
            )
            return None

        ensemble_features = list(logistic_features)
        X_test_eval = self._align_feature_matrix(X_test, feature_cols, ensemble_features)
        X_train_eval = self._align_feature_matrix(X_train, feature_cols, ensemble_features)

        ensemble_model = ProbabilityAveragingEnsemble(
            models=[logistic_entry["model"], tree_entry["model"]],
            model_names=["logistic", best_tree_key],
            weights=[0.5, 0.5],
        )

        reliability_filepath = None
        if plot_reliability and reliability_dir:
            import os

            os.makedirs(reliability_dir, exist_ok=True)
            reliability_filepath = os.path.join(
                reliability_dir, f"reliability_{window}h_ensemble_voting.png"
            )

        evaluation_results = self.evaluator.evaluate_model(
            ensemble_model,
            X_test_eval,
            y_test,
            classes=classes,
            calibrate=False,
            X_calibration=X_train_eval,
            y_calibration=None,
            plot_reliability=plot_reliability,
            reliability_filepath=reliability_filepath,
            return_calibrated_model=False,
        )

        selection_metrics = self._extract_selection_metrics(evaluation_results)
        selection_metrics["calibration_method"] = None
        selection_metrics["calibration_deployed"] = False

        cv_values = [
            window_results["logistic"]["training_info"].get("cv_mean"),
            window_results[best_tree_key]["training_info"].get("cv_mean"),
        ]
        cv_values = [float(v) for v in cv_values if v is not None]
        cv_stds = [
            window_results["logistic"]["training_info"].get("cv_std"),
            window_results[best_tree_key]["training_info"].get("cv_std"),
        ]
        cv_stds = [float(v) for v in cv_stds if v is not None]

        training_info = {
            "model_type": "ensemble_voting",
            "ensemble_components": ["logistic", best_tree_key],
            "ensemble_weights": [0.5, 0.5],
            "cv_mean": float(np.mean(cv_values)) if cv_values else None,
            "cv_std": float(np.mean(cv_stds)) if cv_stds else None,
        }
        training_info.update(selection_metrics)
        training_info["feature_fill_values"] = self._build_feature_fill_values(X_train_eval, ensemble_features)

        logger.info(
            "ensemble_voting (%s + %s): test accuracy=%.4f, event_f1=%s",
            "logistic",
            best_tree_key,
            evaluation_results["classification_report"]["accuracy"],
            selection_metrics.get("test_event_f1"),
        )

        return {
            "model": ensemble_model,
            "training_info": training_info,
            "evaluation_results": evaluation_results,
            "label_encoder": label_encoder,
            "feature_names": ensemble_features,
            "feature_fill_values": training_info["feature_fill_values"],
        }

    def prepare_dataset(
        self,
        start_date: datetime,
        end_date: datetime,
        sample_interval_hours: int = 1,
        region_number: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        prepare dataset with features and labels.

        args:
            start_date: start date for dataset
            end_date: end date for dataset
            sample_interval_hours: hours between samples
            region_number: optional region number to filter by

        returns:
            dataframe with features and labels
        """
        logger.info(f"preparing dataset from {start_date} to {end_date}")

        # generate timestamps
        timestamps = []
        current = start_date
        while current <= end_date:
            timestamps.append(current)
            current += timedelta(hours=sample_interval_hours)

        logger.info(f"generating features for {len(timestamps)} timestamps")

        # compute features
        features_df = self.feature_engineer.compute_features_batch(
            timestamps,
            region_number=region_number,
            normalize=False,
            standardize=False,
            handle_missing=True,
        )

        if len(features_df) == 0:
            logger.warning("no features generated")
            return pd.DataFrame()

        logger.info(f"created features for {len(features_df)} timestamps")

        # create labels
        logger.info("creating labels")
        labeled_df = self.labeler.create_labels_from_features(
            features_df, windows=TARGET_WINDOWS
        )

        # filter out rows without labels
        for window in TARGET_WINDOWS:
            label_col = f"label_{window}h"
            if label_col in labeled_df.columns:
                initial_count = len(labeled_df)
                labeled_df = labeled_df[labeled_df[label_col].notna()]
                filtered_count = len(labeled_df)
                logger.info(
                    f"filtered {initial_count - filtered_count} rows without {label_col} labels"
                )

        logger.info(f"final dataset size: {len(labeled_df)} samples")

        return labeled_df

    def train_and_evaluate(
        self,
        dataset: pd.DataFrame,
        test_size: float = 0.2,
        models: Optional[List[str]] = None,
        plot_reliability: bool = False,
        reliability_dir: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        train and evaluate models on dataset.

        args:
            dataset: dataframe with features and labels
            test_size: fraction of data to use for testing
            models: list of model types to train
            plot_reliability: whether to plot reliability diagrams
            reliability_dir: directory to save reliability diagrams
            run_name: optional mlflow run name

        returns:
            dict with training and evaluation results
        """
        if models is None:
            models = ["logistic", "gradient_boosting", "lightgbm", "random_forest"]

        # start mlflow run if enabled
        mlflow_run = None
        if self.use_mlflow and self.mlflow_tracker:
            mlflow_run = self.mlflow_tracker.start_run(
                run_name=run_name or f"classification_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                tags={"model_type": "classification", "pipeline": "classification"},
            )

            # log dataset info
            self.mlflow_tracker.log_params({
                "dataset_size": len(dataset),
                "test_size": test_size,
                "num_features": len([c for c in dataset.columns if not c.startswith("label_")]),
                "use_smote": self.use_smote,
                "cv_folds": self.cv_folds,
                "calibrate": self.calibrate,
                "random_state": self.random_state,
                "models": ",".join(models),
            })

        results = {}

        # train each window with progress logging
        for window_idx, window in enumerate(TARGET_WINDOWS, 1):
            logger.info(f"[window {window_idx}/{len(TARGET_WINDOWS)}] training {window}h prediction window...")
            label_col = f"label_{window}h"
            if label_col not in dataset.columns:
                logger.warning(f"label column {label_col} not found, skipping")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"training models for {window}h prediction window")
            logger.info(f"{'='*60}")

            # prepare features and labels
            # exclude label columns and historical features (max_magnitude, flare_classes)
            exclude_cols = ["timestamp", "region_number"] + [
                col
                for col in dataset.columns
                if (
                    col.startswith("label_")
                    or col.startswith("num_flares_")
                    or col.startswith("max_magnitude_")
                    or col.startswith("flare_classes_")
                )
            ]

            # only include numeric columns (exclude string/object columns)
            feature_cols = [
                col for col in dataset.columns
                if col not in exclude_cols
                and pd.api.types.is_numeric_dtype(dataset[col])
            ]

            if len(feature_cols) == 0:
                raise ValueError("no numeric feature columns found in dataset")

            # check for any remaining non-numeric columns
            non_numeric = [
                col for col in feature_cols
                if not pd.api.types.is_numeric_dtype(dataset[col])
            ]
            if non_numeric:
                logger.warning(f"excluding non-numeric columns from features: {non_numeric}")
                feature_cols = [col for col in feature_cols if col not in non_numeric]

            if "timestamp" in dataset.columns:
                dataset_for_window = dataset.sort_values("timestamp").reset_index(drop=True)
            else:
                logger.warning(
                    "dataset has no timestamp column; falling back to input order for train/test split"
                )
                dataset_for_window = dataset.reset_index(drop=True)

            X = dataset_for_window[feature_cols].values
            y = dataset_for_window[label_col].values

            # encode labels
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            classes = label_encoder.classes_.tolist()

            # handle missing values - ensure X is numeric
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # ensure X is float type (not object/string)
            if X.dtype == object:
                logger.error(f"feature matrix contains non-numeric data. columns: {feature_cols}")
                logger.error(f"data types: {dataset[feature_cols].dtypes.to_dict()}")
                raise ValueError("feature matrix must be numeric but contains object/string data")

            # split train/test in chronological order (no shuffling)
            X_train, X_test, y_train, y_test = self._chronological_train_test_split(
                X,
                y_encoded,
                test_size=test_size,
            )

            logger.info(f"train size: {len(X_train)}, test size: {len(X_test)}")
            # fix: correctly map classes to counts
            unique_labels, counts = np.unique(y_train, return_counts=True)
            class_dist = dict(zip([classes[i] for i in unique_labels], counts))
            logger.info(f"class distribution (train): {class_dist}")

            # train models with progress logging
            window_results = {}
            for model_idx, model_type in enumerate(models, 1):
                logger.info(f"[model {model_idx}/{len(models)}] training {model_type} model for {window}h window...")

                try:
                    # prepare temporary dataframe for training
                    train_df = pd.DataFrame(X_train, columns=feature_cols)
                    train_df[label_col] = label_encoder.inverse_transform(y_train)

                    # train model
                    trained_models = self.trainer.train_baseline_models(
                        train_df, label_col, models=[model_type]
                    )

                    if model_type not in trained_models:
                        logger.warning(f"failed to train {model_type}")
                        continue

                    model, training_info = trained_models[model_type]

                    # align test set to selected features when feature selection is active
                    selected_feature_names = training_info.get("feature_names", feature_cols)
                    X_test_eval = self._align_feature_matrix(X_test, feature_cols, selected_feature_names)
                    X_train_eval = self._align_feature_matrix(X_train, feature_cols, selected_feature_names)
                    feature_fill_values = self._build_feature_fill_values(X_train_eval, selected_feature_names)

                    # evaluate on test set
                    logger.info(f"evaluating {model_type} model...")

                    # update label encoder for evaluator
                    self.evaluator.label_encoder = label_encoder

                    # determine reliability filepath
                    reliability_filepath = None
                    if plot_reliability and reliability_dir:
                        import os
                        os.makedirs(reliability_dir, exist_ok=True)
                        reliability_filepath = os.path.join(
                            reliability_dir, f"reliability_{window}h_{model_type}.png"
                        )

                    # fix: pass training data for calibration, test data for evaluation
                    evaluation_results = self.evaluator.evaluate_model(
                        model,
                        X_test_eval,
                        y_test,
                        classes=classes,
                        calibrate=self.calibrate,
                        X_calibration=X_train_eval,  # use training data for calibration
                        y_calibration=y_train,  # use training labels for calibration
                        plot_reliability=plot_reliability,
                        reliability_filepath=reliability_filepath,
                        return_calibrated_model=self.calibrate,
                    )

                    selected_model = model
                    calibration_deployed = False
                    if self.calibrate and "calibrated_model" in evaluation_results:
                        baseline_brier = (
                            evaluation_results.get("brier_score", {}).get("macro_avg")
                        )
                        calibrated_brier = (
                            evaluation_results.get("calibrated_brier_score", {}).get("macro_avg")
                        )
                        if (
                            calibrated_brier is not None
                            and (baseline_brier is None or calibrated_brier <= baseline_brier + 1e-9)
                        ):
                            selected_model = evaluation_results["calibrated_model"]
                            calibration_deployed = True
                        # keep model artifacts lightweight/serializable in results payload
                        evaluation_results.pop("calibrated_model", None)

                    selection_metrics = self._extract_selection_metrics(evaluation_results)
                    selection_metrics["calibration_method"] = evaluation_results.get(
                        "selected_calibration_method"
                    )
                    selection_metrics["calibration_deployed"] = calibration_deployed

                    training_info_for_selection = dict(training_info)
                    training_info_for_selection.update(selection_metrics)
                    training_info_for_selection["feature_fill_values"] = feature_fill_values

                    window_results[model_type] = {
                        "model": selected_model,
                        "training_info": training_info_for_selection,
                        "evaluation_results": evaluation_results,
                        "label_encoder": label_encoder,
                        "feature_names": selected_feature_names,
                        "feature_fill_values": feature_fill_values,
                    }

                    # log key metrics
                    logger.info(f"\n{model_type} results:")
                    logger.info(
                        f"  cv accuracy: {training_info['cv_mean']:.4f} (+/- {training_info['cv_std']*2:.4f})"
                    )
                    logger.info(
                        f"  test accuracy: {evaluation_results['classification_report']['accuracy']:.4f}"
                    )
                    logger.info(
                        f"  macro avg brier score: {evaluation_results['brier_score']['macro_avg']:.4f}"
                    )
                    logger.info(
                        f"  macro avg roc-auc: {evaluation_results['roc_auc']['macro_avg']:.4f}"
                    )

                    # log to mlflow
                    if self.use_mlflow and self.mlflow_tracker:
                        try:
                            # log metrics
                            metrics = {
                                f"{window}h_{model_type}_cv_accuracy": training_info_for_selection['cv_mean'],
                                f"{window}h_{model_type}_cv_std": training_info_for_selection['cv_std'],
                                f"{window}h_{model_type}_test_accuracy": (
                                    evaluation_results['classification_report']['accuracy']
                                ),
                                f"{window}h_{model_type}_brier_macro": evaluation_results['brier_score']['macro_avg'],
                                f"{window}h_{model_type}_roc_auc_macro": evaluation_results['roc_auc']['macro_avg'],
                            }
                            if selection_metrics.get("test_event_f1") is not None:
                                metrics[f"{window}h_{model_type}_event_f1"] = selection_metrics["test_event_f1"]
                            if selection_metrics.get("test_event_recall") is not None:
                                metrics[f"{window}h_{model_type}_event_recall"] = selection_metrics["test_event_recall"]
                            if selection_metrics.get("test_brier_macro_calibrated") is not None:
                                metrics[f"{window}h_{model_type}_brier_macro_calibrated"] = (
                                    selection_metrics["test_brier_macro_calibrated"]
                                )
                            self.mlflow_tracker.log_metrics(metrics)

                            # log model
                            model_artifact_path = f"models/{window}h_{model_type}"
                            self.mlflow_tracker.log_model(
                                selected_model,
                                artifact_path=model_artifact_path,
                                model_type="sklearn",
                            )
                        except Exception as e:
                            logger.warning(f"failed to log to mlflow: {e}", exc_info=True)

                except Exception as e:
                    logger.error(f"error training/evaluating {model_type}: {e}")
                    continue

            # optional logistic + tree ensemble candidate
            try:
                ensemble_entry = self._build_logistic_tree_ensemble(
                    window_results=window_results,
                    feature_cols=feature_cols,
                    X_test=X_test,
                    X_train=X_train,
                    y_test=y_test,
                    classes=classes,
                    label_encoder=label_encoder,
                    plot_reliability=plot_reliability,
                    reliability_dir=reliability_dir,
                    window=window,
                )
                if ensemble_entry is not None:
                    window_results["ensemble_voting"] = ensemble_entry

                    if self.use_mlflow and self.mlflow_tracker:
                        ensemble_eval = ensemble_entry["evaluation_results"]
                        ensemble_sel = self._extract_selection_metrics(ensemble_eval)
                        metrics = {
                            f"{window}h_ensemble_voting_test_accuracy": (
                                ensemble_eval["classification_report"]["accuracy"]
                            ),
                            f"{window}h_ensemble_voting_brier_macro": (
                                ensemble_eval["brier_score"]["macro_avg"]
                            ),
                            f"{window}h_ensemble_voting_roc_auc_macro": (
                                ensemble_eval["roc_auc"]["macro_avg"]
                            ),
                        }
                        if ensemble_sel.get("test_event_f1") is not None:
                            metrics[f"{window}h_ensemble_voting_event_f1"] = ensemble_sel["test_event_f1"]
                        if ensemble_sel.get("test_event_recall") is not None:
                            metrics[f"{window}h_ensemble_voting_event_recall"] = ensemble_sel["test_event_recall"]
                        self.mlflow_tracker.log_metrics(metrics)
            except Exception as e:
                logger.warning(f"failed to build ensemble_voting candidate: {e}")

            # auto-select best model for this window
            if window_results:
                from src.models.training import ModelTrainer as _MT
                best_candidates = {k: (v["model"], v["training_info"]) for k, v in window_results.items()}
                best_result = _MT.select_best_model(best_candidates)
                if best_result is not None:
                    best_model, best_info = best_result
                    selected_from = best_info.get("selected_from", "unknown")
                    logger.info(
                        "auto-selected best model for %sh: %s (event_f1=%s, accuracy=%s)",
                        window,
                        selected_from,
                        best_info.get("test_event_f1", best_info.get("cv_event_f1_mean")),
                        best_info.get("test_accuracy", best_info.get("cv_mean")),
                    )
                    window_results["best"] = {
                        "model": best_model,
                        "training_info": best_info,
                        "evaluation_results": window_results.get(selected_from, {}).get("evaluation_results", {}),
                        "label_encoder": label_encoder,
                        "feature_names": window_results.get(selected_from, {}).get("feature_names", feature_cols),
                        "feature_fill_values": window_results.get(selected_from, {}).get("feature_fill_values", {}),
                    }

            results[f"{window}h"] = window_results

        self.models = results
        self.evaluation_results = results

        # end mlflow run
        if self.use_mlflow and self.mlflow_tracker and mlflow_run:
            try:
                self.mlflow_tracker.end_run(status="FINISHED")
            except Exception as e:
                logger.warning(f"failed to end mlflow run: {e}")

        return results

    def predict(
        self,
        timestamp: datetime,
        window: int,
        model_type: str = "best",
        region_number: Optional[int] = None,
        include_explanation: bool = False,
    ) -> Dict[str, Any]:
        """
        make prediction for a given timestamp.

        args:
            timestamp: timestamp to predict for
            window: prediction window in hours (24 or 48)
            model_type: model type to use ('best', 'logistic', 'gradient_boosting', etc.)
            region_number: optional region number to filter by
            include_explanation: whether to include SHAP explanation

        returns:
            dict with prediction results
        """
        window_key = f"{window}h"
        if window_key not in self.models:
            raise ValueError(f"no models trained for {window}h window")

        # fallback from "best" to "gradient_boosting" if best not available
        if model_type == "best" and "best" not in self.models[window_key]:
            model_type = "gradient_boosting"

        if model_type not in self.models[window_key]:
            raise ValueError(f"model type {model_type} not found for {window}h window")

        model_info = self.models[window_key][model_type]
        model = model_info["model"]
        label_encoder = model_info["label_encoder"]
        feature_names = model_info["feature_names"]
        feature_fill_values = model_info.get("feature_fill_values", {}) or {}

        # compute features
        features_df = self.feature_engineer.compute_features(
            timestamp,
            region_number=region_number,
            normalize=False,
            standardize=False,
            handle_missing=True,
        )

        if len(features_df) == 0:
            raise ValueError("could not compute features")

        # extract features in correct order
        # handle missing features (e.g., max_magnitude_* which are historical labels)
        missing_features = [f for f in feature_names if f not in features_df.columns]
        if missing_features:
            logger.warning(
                "missing features in computed data (using training fill values): %s",
                missing_features,
            )
            for feat in missing_features:
                features_df[feat] = float(feature_fill_values.get(feat, 0.0))

        X_df = features_df[feature_names].copy()
        for feat in feature_names:
            fallback = float(feature_fill_values.get(feat, 0.0))
            X_df[feat] = pd.to_numeric(X_df[feat], errors="coerce").fillna(fallback)

        X = X_df.values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # predict
        y_pred = model.predict(X)[0]
        y_prob = model.predict_proba(X)[0]

        # decode prediction
        predicted_class = label_encoder.inverse_transform([y_pred])[0]

        # convert numpy types to native Python types for JSON serialization
        class_probs = {}
        for class_name, prob in zip(label_encoder.classes_, y_prob):
            # convert numpy types to native Python types
            if hasattr(prob, 'item'):
                class_probs[str(class_name)] = float(prob.item())
            else:
                class_probs[str(class_name)] = float(prob)

        result = {
            "timestamp": timestamp,
            "window_hours": window,
            "predicted_class": str(predicted_class),
            "class_probabilities": class_probs,
            "model_type": model_type,
        }

        # add SHAP explanation if requested
        if include_explanation:
            try:
                from src.api.explainer import get_explainer

                explainer = get_explainer()
                explanation = explainer.explain_classification(
                    model=model,
                    X=X,
                    feature_names=feature_names,
                    label_encoder=label_encoder,
                    model_type=model_type,
                    window=window,
                )
                result["explanation"] = explanation
            except Exception as e:
                logger.warning(f"failed to compute SHAP explanation: {e}")
                result["explanation"] = {"error": str(e)}

        return result

# fmt: on
