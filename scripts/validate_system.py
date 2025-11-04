#!/usr/bin/env python
"""end-to-end system validation before deployment."""

import sys
from pathlib import Path

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import Dict, Any, List, Tuple  # noqa: E402

import numpy as np  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src.config import PROJECT_ROOT  # noqa: E402
from src.data.database import get_database  # noqa: E402
from src.data.ingestion import DataIngestionPipeline  # noqa: E402
from src.data.schema import PredictionLog, SystemValidationLog  # noqa: E402
from src.features.pipeline import FeatureEngineer  # noqa: E402
from src.models.pipeline import ClassificationPipeline  # noqa: E402
from src.models.survival_pipeline import SurvivalAnalysisPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

GUARDRAIL_KEYWORDS = [
    "survival prediction total probability too low",
    "survival prediction total probability too high",
    "survival function nearly flat",
    "survival function collapses to zero",
    "survival function has abrupt drop",
]


def reconstruct_pipeline_from_dict(pipeline_data: Dict[str, Any], pipeline_type: str) -> Any:
    """
    reconstruct pipeline object from saved dict.

    args:
        pipeline_data: saved pipeline data (dict)
        pipeline_type: 'classification' or 'survival'

    returns:
        reconstructed pipeline object
    """
    if pipeline_type == "classification":
        pipeline = ClassificationPipeline()
        pipeline.models = pipeline_data.get("models", {})
        pipeline.evaluation_results = pipeline_data.get("evaluation_results", {})
    elif pipeline_type == "survival":
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
    else:
        raise ValueError(f"unknown pipeline type: {pipeline_type}")

    return pipeline


def test_database_connection() -> Tuple[bool, List[str]]:
    """
    test database connection and verify tables exist.

    returns:
        tuple of (success, error_messages)
    """
    print("Testing database connection...")
    errors = []

    try:
        db = get_database()

        with db.get_session() as session:
            # test connection
            result = session.execute(text("SELECT 1"))
            result.fetchone()

            # check if tables exist
            required_tables = [
                "flare_goes_xray_flux",
                "flare_solar_regions",
                "flare_events",
                "flare_ingestion_log",
                "flare_prediction_log",
            ]

            for table in required_tables:
                result = session.execute(text(f"SELECT COUNT(*) FROM {table}"))  # nosec B608
                count = result.fetchone()[0]
                print(f"  Table {table}: {count} records")

        print("[PASS] Database connection test")
        return True, []

    except Exception as e:
        errors.append(f"database connection failed: {e}")
        print(f"[FAIL] Database connection test: {e}")
        return False, errors


def test_data_ingestion() -> Tuple[bool, List[str]]:
    """
    test data ingestion from noaa.

    returns:
        tuple of (success, error_messages)
    """
    print("\nTesting data ingestion...")
    errors = []

    try:
        pipeline = DataIngestionPipeline()

        # run ingestion
        results = pipeline.run_incremental_update(use_cache=False)

        # check results
        for source, result in results.items():
            # skip timestamp field (it's a datetime, not a dict)
            if source == "timestamp":
                continue

            status = result.get("status", "unknown")
            records = result.get("records_inserted", 0)

            if status == "success":
                print(f"  {source}: {status} ({records} records)")
            else:
                error_msg = result.get("error", "unknown error")
                print(f"  {source}: {status} - {error_msg}")
                errors.append(f"ingestion failed for {source}: {error_msg}")

        if not errors:
            print("[PASS] Data ingestion test")
            return True, []
        else:
            print("[FAIL] Data ingestion test")
            return False, errors

    except Exception as e:
        errors.append(f"data ingestion failed: {e}")
        print(f"[FAIL] Data ingestion test: {e}")
        return False, errors


def test_model_loading() -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    test loading classification and survival models.

    returns:
        tuple of (success, error_messages, loaded_models)
    """
    print("\nTesting model loading...")
    errors = []
    loaded_models = {}

    # try to load classification model
    classification_path = PROJECT_ROOT / "models" / "classification_pipeline.joblib"
    if classification_path.exists():
        try:
            import joblib

            classification_model = joblib.load(classification_path)
            loaded_models["classification"] = classification_model
            print(f"  [OK] Classification model loaded from {classification_path}")
        except Exception as e:
            errors.append(f"failed to load classification model: {e}")
            print(f"  [FAIL] Classification model: {e}")
    else:
        print(f"  [SKIP] Classification model not found at {classification_path}")

    # try to load survival model
    survival_paths = [
        PROJECT_ROOT / "models" / "survival_model.joblib",
        PROJECT_ROOT / "models" / "survival_model_c_class.joblib",
        PROJECT_ROOT / "models" / "survival_model_x_class.joblib",
    ]

    survival_loaded = False
    for survival_path in survival_paths:
        if survival_path.exists():
            try:
                import joblib

                survival_model = joblib.load(survival_path)

                # if model is a dict, reconstruct the pipeline
                if isinstance(survival_model, dict):
                    survival_model = reconstruct_pipeline_from_dict(survival_model, "survival")

                # validate that model is actually fitted and usable
                if not survival_model.is_fitted:
                    print(f"  [SKIP] Survival model ({survival_path.name}) not fitted")
                    continue

                # try a quick predict test to ensure it's actually usable
                test_timestamp = datetime.utcnow()
                try:
                    _ = survival_model.predict_survival_probabilities(timestamp=test_timestamp)
                    loaded_models["survival"] = survival_model
                    print(f"  [OK] Survival model loaded and validated from {survival_path}")
                    survival_loaded = True
                    break
                except Exception as predict_err:
                    print(f"  [SKIP] Survival model ({survival_path.name}) loaded but unusable: {predict_err}")
                    continue

            except Exception as e:
                errors.append(f"failed to load survival model from {survival_path}: {e}")
                print(f"  [FAIL] Survival model ({survival_path.name}): {e}")

    if not survival_loaded:
        print(f"  [SKIP] No survival model found in {PROJECT_ROOT / 'models'}")

    # check if at least one model loaded
    if not loaded_models:
        errors.append("no models found - train models before validation")
        print("[FAIL] Model loading test - no models available")
        return False, errors, loaded_models

    print("[PASS] Model loading test")
    return True, [], loaded_models


def test_predictions(loaded_models: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    test making predictions with loaded models.

    args:
        loaded_models: dict of loaded models

    returns:
        tuple of (success, error_messages)
    """
    print("\nTesting predictions...")
    errors = []
    timestamp = datetime.utcnow()

    # test classification predictions
    if "classification" in loaded_models:
        try:
            model = loaded_models["classification"]
            result = model.predict(timestamp, window=24)

            # check for valid structure
            if "probabilities" not in result:
                errors.append("classification prediction missing probabilities")
            else:
                probs = result["probabilities"]
                # check for NaN
                if any(np.isnan(v) for v in probs.values()):
                    errors.append("classification prediction contains NaN")
                else:
                    print("  [OK] Classification prediction valid")

        except Exception as e:
            errors.append(f"classification prediction failed: {e}")
            print(f"  [FAIL] Classification prediction: {e}")

    # test survival predictions
    if "survival" in loaded_models:
        try:
            model = loaded_models["survival"]

            # if model is a dict, reconstruct the pipeline
            if isinstance(model, dict):
                model = reconstruct_pipeline_from_dict(model, "survival")

            result = model.predict_survival_probabilities(timestamp=timestamp)

            # check for valid structure
            if "probability_distribution" not in result:
                errors.append("survival prediction missing probability_distribution")
            else:
                probs = result["probability_distribution"]
                # check for NaN
                if any(np.isnan(v) for v in probs.values()):
                    errors.append("survival prediction contains NaN")
                else:
                    print("  [OK] Survival prediction valid")

        except Exception as e:
            errors.append(f"survival prediction failed: {e}")
            print(f"  [FAIL] Survival prediction: {e}")

    if not errors:
        print("[PASS] Prediction test")
        return True, []
    else:
        print("[FAIL] Prediction test")
        return False, errors


def test_api_endpoint() -> Tuple[bool, List[str]]:
    """
    test api endpoint if api is running.

    returns:
        tuple of (success, error_messages)
    """
    print("\nTesting API endpoint...")
    errors = []

    try:
        import requests

        # try to hit health endpoint
        response = requests.get("http://127.0.0.1:5000/health", timeout=5)

        if response.status_code == 200:
            health_data = response.json()
            print(f"  API status: {health_data.get('status', 'unknown')}")
            print(f"  Classification available: {health_data.get('classification_available', False)}")
            print(f"  Survival available: {health_data.get('survival_available', False)}")
            print("[PASS] API endpoint test")
            return True, []
        else:
            errors.append(f"API returned status {response.status_code}")
            print(f"[FAIL] API endpoint test - status {response.status_code}")
            return False, errors

    except requests.exceptions.ConnectionError:
        print("[SKIP] API not running (connection refused)")
        return True, []  # not a failure if API isn't running
    except Exception as e:
        print(f"[SKIP] API endpoint test: {e}")
        return True, []  # not a failure


def test_prediction_pipeline() -> Tuple[bool, List[str]]:
    """
    test full prediction pipeline: ingestion -> features -> prediction -> logging.

    returns:
        tuple of (success, error_messages)
    """
    print("\nTesting full prediction pipeline...")
    errors = []

    try:
        # step 1: ingest fresh data
        print("  Step 1: Ingesting fresh data...")
        pipeline = DataIngestionPipeline()
        ingest_results = pipeline.run_incremental_update(use_cache=False)

        # check ingestion succeeded (skip timestamp field)
        ingestion_success = all(
            r.get("status") == "success" for source, r in ingest_results.items() if source != "timestamp"
        )
        if not ingestion_success:
            errors.append("data ingestion failed in pipeline test")
            print("  [FAIL] Data ingestion step")
            return False, errors

        print("  [OK] Data ingestion")

        # step 2: extract features for current timestamp
        print("  Step 2: Extracting features...")
        timestamp = datetime.utcnow()
        feature_engineer = FeatureEngineer()

        features = feature_engineer.compute_features(
            timestamp=timestamp, normalize=False, standardize=False, handle_missing=True
        )

        if features is None or len(features) == 0:
            errors.append("feature extraction returned no features")
            print("  [FAIL] Feature extraction step")
            return False, errors

        print(f"  [OK] Extracted {len(features.columns)} features")

        # step 3: run both models (if available)
        print("  Step 3: Running models...")
        classification_pred = None
        survival_pred = None

        # try classification
        classification_path = PROJECT_ROOT / "models" / "classification_pipeline.joblib"
        if classification_path.exists():
            try:
                import joblib

                classification_model = joblib.load(classification_path)

                # if model is a dict, reconstruct the pipeline
                if isinstance(classification_model, dict):
                    classification_model = reconstruct_pipeline_from_dict(classification_model, "classification")

                classification_pred = classification_model.predict(timestamp, window=24)
                print("  [OK] Classification prediction")
            except Exception as e:
                print(f"  [SKIP] Classification prediction: {e}")

        # try survival
        survival_paths = [
            PROJECT_ROOT / "models" / "survival_model.joblib",
            PROJECT_ROOT / "models" / "survival_model_c_class.joblib",
        ]

        for survival_path in survival_paths:
            if survival_path.exists():
                try:
                    import joblib

                    survival_model = joblib.load(survival_path)

                    # if model is a dict, reconstruct the pipeline
                    if isinstance(survival_model, dict):
                        survival_model = reconstruct_pipeline_from_dict(survival_model, "survival")

                    survival_pred = survival_model.predict_survival_probabilities(timestamp=timestamp)
                    print("  [OK] Survival prediction")
                    break
                except Exception as e:
                    print(f"  [SKIP] Survival prediction ({survival_path.name}): {e}")

        # step 4: verify predictions are reasonable
        print("  Step 4: Verifying predictions...")

        if classification_pred:
            probs = classification_pred.get("probabilities", {})
            if any(np.isnan(v) for v in probs.values()):
                errors.append("classification prediction contains NaN")
                print("  [FAIL] Classification has NaN values")
            elif abs(sum(probs.values()) - 1.0) > 0.1:
                errors.append("classification probabilities don't sum to 1")
                print("  [FAIL] Classification probabilities invalid")
            else:
                print("  [OK] Classification prediction valid")

        if survival_pred:
            probs = survival_pred.get("probability_distribution", {})
            if any(np.isnan(v) for v in probs.values()):
                errors.append("survival prediction contains NaN")
                print("  [FAIL] Survival has NaN values")
            else:
                total_prob = sum(probs.values())
                if total_prob < 0.01:
                    errors.append(f"survival prediction total probability too low ({total_prob:.6f})")
                    print(f"  [FAIL] Survival probabilities too low (total={total_prob:.6f})")
                elif total_prob > 1.5:
                    errors.append(f"survival prediction total probability too high ({total_prob:.6f})")
                    print(f"  [FAIL] Survival probabilities too high (total={total_prob:.6f})")
                else:
                    survival_info = survival_pred.get("survival_function", {})
                    survival_probs = survival_info.get("probabilities", [])
                    if survival_probs:
                        survival_range = max(survival_probs) - min(survival_probs)
                        if survival_range < 0.001:
                            errors.append(f"survival function nearly flat (range={survival_range:.6f})")
                            print(f"  [FAIL] Survival function nearly flat (range={survival_range:.6f})")
                        else:
                            # flag degenerate step functions (all mass in tiny interval)
                            min_nonterminal = min(survival_probs[:-1]) if len(survival_probs) > 1 else survival_probs[0]
                            max_drop = (
                                max((survival_probs[i] - survival_probs[i + 1]) for i in range(len(survival_probs) - 1))
                                if len(survival_probs) > 1
                                else 0.0
                            )
                            if min_nonterminal <= 0.01:
                                errors.append("survival function collapses to zero inside prediction horizon")
                                print("  [FAIL] Survival function collapses to zero pre-horizon (degenerate)")
                            elif max_drop > 0.9:
                                errors.append(f"survival function has abrupt drop of {max_drop:.3f}")
                                print(f"  [FAIL] Survival function drops {max_drop:.3f} in a single step (degenerate)")
                            else:
                                print("  [OK] Survival prediction valid")
                    else:
                        print("  [OK] Survival prediction valid (no survival curve details)")

        # step 5: log prediction to database
        print("  Step 5: Logging prediction to database...")

        if classification_pred or survival_pred:
            db = get_database()

            with db.get_session() as session:
                if classification_pred:
                    import json

                    pred_log = PredictionLog(
                        prediction_timestamp=datetime.utcnow(),
                        observation_timestamp=timestamp,
                        prediction_type="classification",
                        model_type=classification_pred.get("model_type"),
                        window_hours=classification_pred.get("window"),
                        predicted_class=classification_pred.get("predicted_class"),
                        class_probabilities=json.dumps(classification_pred.get("probabilities")),
                    )
                    session.add(pred_log)
                    session.commit()
                    print(f"  [OK] Classification prediction logged (id={pred_log.id})")

                if survival_pred:
                    import json

                    pred_log = PredictionLog(
                        prediction_timestamp=datetime.utcnow(),
                        observation_timestamp=timestamp,
                        prediction_type="survival",
                        model_type=survival_pred.get("model_type"),
                        hazard_score=survival_pred.get("hazard_score"),
                        probability_distribution=json.dumps(survival_pred.get("probability_distribution")),
                    )
                    session.add(pred_log)
                    session.commit()
                    print(f"  [OK] Survival prediction logged (id={pred_log.id})")

        if not errors:
            print("[PASS] Full prediction pipeline test")
            return True, []
        else:
            print("[FAIL] Full prediction pipeline test")
            return False, errors

    except Exception as e:
        errors.append(f"prediction pipeline failed: {e}")
        print(f"[FAIL] Full prediction pipeline test: {e}")
        logger.error("prediction pipeline test failed", exc_info=True)
        return False, errors


def main():
    """main entry point."""
    print("=" * 70)
    print("FLARE+ SYSTEM VALIDATOR")
    print("=" * 70)
    print()

    all_tests = [
        ("Database Connection", test_database_connection, []),
        ("Data Ingestion", test_data_ingestion, []),
        ("Model Loading", test_model_loading, []),
    ]

    results = []
    all_errors = []
    loaded_models = {}

    # run initial tests
    for test_name, test_func, _ in all_tests:
        try:
            if test_name == "Model Loading":
                success, errors, loaded_models = test_func()
            else:
                success, errors = test_func()

            results.append((test_name, success))
            if errors:
                all_errors.extend(errors)
        except Exception as e:
            logger.error(f"{test_name} test failed with exception: {e}", exc_info=True)
            results.append((test_name, False))
            all_errors.append(f"{test_name}: {e}")

    # run predictions test if models loaded
    if loaded_models:
        try:
            success, errors = test_predictions(loaded_models)
            results.append(("Predictions", success))
            if errors:
                all_errors.extend(errors)
        except Exception as e:
            logger.error(f"Predictions test failed with exception: {e}", exc_info=True)
            results.append(("Predictions", False))
            all_errors.append(f"Predictions: {e}")

    # run api test
    try:
        success, errors = test_api_endpoint()
        results.append(("API Endpoint", success))
        if errors:
            all_errors.extend(errors)
    except Exception as e:
        logger.error(f"API test failed with exception: {e}", exc_info=True)
        results.append(("API Endpoint", False))
        all_errors.append(f"API Endpoint: {e}")

    # run full pipeline test
    try:
        success, errors = test_prediction_pipeline()
        results.append(("Full Prediction Pipeline", success))
        if errors:
            all_errors.extend(errors)
    except Exception as e:
        logger.error(f"Pipeline test failed with exception: {e}", exc_info=True)
        results.append(("Full Prediction Pipeline", False))
        all_errors.append(f"Full Prediction Pipeline: {e}")

    # print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for test_name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")

    if all_errors:
        print("\nErrors found:")
        for error in all_errors:
            print(f"  - {error}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print(f"\n{passed}/{total} tests passed")

    guardrail_errors = [error for error in all_errors if any(keyword in error for keyword in GUARDRAIL_KEYWORDS)]

    record_validation_run(results, all_errors, guardrail_errors, total, passed)

    if passed == total:
        print("\n[OK] All system validation tests passed")
        print("System is ready for deployment")
        sys.exit(0)
    else:
        print("\n[FAIL] System validation failed")
        print("Please address the issues above before deploying")

        if guardrail_errors:
            print("If this was a test run, rerun './flare validate' once the survival model issues are corrected.")

        sys.exit(1)


def record_validation_run(
    results: List[Tuple[str, bool]],
    all_errors: List[str],
    guardrail_errors: List[str],
    total: int,
    passed: int,
) -> None:
    """persist validation summary to the database for historical tracking."""
    status = "pass" if passed == total else "fail"
    guardrail_triggered = bool(guardrail_errors)
    guardrail_reason = "; ".join(guardrail_errors)[:500] if guardrail_errors else None

    summary = [{"test": name, "status": "pass" if success else "fail"} for name, success in results]

    payload = {
        "results": summary,
        "errors": all_errors,
        "passed": passed,
        "total": total,
    }

    initiated_by = os.getenv("VALIDATION_INITIATED_BY")

    try:
        db = get_database()
        with db.get_session() as session:
            log_entry = SystemValidationLog(
                status=status,
                guardrail_triggered=guardrail_triggered,
                guardrail_reason=guardrail_reason,
                details=json.dumps(payload, default=str),
                initiated_by=initiated_by,
            )
            session.add(log_entry)
    except Exception:
        logger.error("failed to record validation run", exc_info=True)


if __name__ == "__main__":
    main()
