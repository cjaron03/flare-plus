# Flare-plus Code Audit

## Executive Summary
- The ingestion and modeling subsystems are well-structured, but a few defects cause incorrect results or undermine evaluation quality.
- Most critical issues center on data duplication in persistence, model training/evaluation mismatches, and avoidable database load.
- Addressing the high-priority items below will materially improve reliability before scaling the system.

## High-Priority Findings
- [x] **Duplicate region & magnetogram writes** (`src/data/persistence.py:104`, `src/data/persistence.py:145`): ~~The save paths append rows without any `ON CONFLICT` handling or uniqueness constraint. Re-running ingestion against the same NOAA snapshots will silently create unlimited duplicates, breaking downstream stats. Add composite uniqueness (e.g., `(region_number, timestamp)`) and switch to `INSERT ... ON CONFLICT DO UPDATE` or a merge.~~ **FIXED**: Added `UniqueConstraint` on `(region_number, timestamp)` for both `SolarRegion` and `SolarMagnetogram` tables. Implemented `INSERT ... ON CONFLICT DO UPDATE` upserts in `save_solar_regions()` and `save_magnetogram()`. Tested: no duplicates after re-ingestion.
- [x] **Model evaluation trains on the test fold** (`src/models/evaluation.py:66`): ~~`CalibratedClassifierCV.fit` is invoked on the arrays already reserved for testing, leaking labels back into the model. The reported metrics are therefore optimistic. Calibrate inside the training fold (nested CV) or reserve a calibration split distinct from the hold-out set.~~ **FIXED**: Added `X_calibration` and `y_calibration` parameters to `evaluate_model()`. Calibration now uses training data only, preventing data leakage. Test data is only used for final evaluation after calibration.
- [x] **Logistic baseline never returned** (`src/models/pipeline.py:209`, `src/models/training.py:288`): ~~`ModelTrainer` stores the key `logistic_regression`, but the pipeline looks up `trained_models["logistic"]`. As written the logistic branch is always skipped. Align the keys or normalise them before lookup.~~ **FIXED**: `ModelTrainer` now stores both `"logistic"` (for pipeline lookup) and `"logistic_regression"` (backward compatibility) keys.

## Medium-Priority Findings
- **Ingestion cache inconsistency** (`src/data/ingestion.py:120`): Magnetogram cache files are written but never read back. Either load them (mirroring the region cache) or drop the write to avoid stale artefacts.
- **Database connection string robustness** (`src/config.py:38`): Credentials are interpolated directly; special characters in `DB_PASSWORD` break the URI. Wrap user/pass with `urllib.parse.quote_plus`.
- **Repeated DB round-trips in feature builders** (`src/features/pipeline.py:299`, `src/models/time_varying_covariates.py:205`): Each timestamp triggers new sessions and full-range SELECTs. For hourly backfills this turns into O(n²) queries. Prefetch bulk windows once and reuse in-memory slices.
- **Inconsistent feature schema when data is sparse** (`src/features/normalization.py:74`): `handle_missing_data` drops any column with >50 % nulls. For single-row feature frames this removes most engineered metrics; later concatenations silently reintroduce NaNs. Prefer imputing to sentinel values and keep the columns stable.
- [x] **Evaluation logging bug** (`src/models/pipeline.py:191`): ~~`dict(zip(classes, *np.unique(...)))` maps class names to label ids instead of counts, so the reported class distribution is wrong. Zip the counts only.~~ **FIXED**: Correctly maps class indices to class names: `unique_labels, counts = np.unique(y_train, return_counts=True)` then `dict(zip([classes[i] for i in unique_labels], counts))`.
- **`FeatureEngineer` region guard** (`src/features/pipeline.py:204`): `if region_number:` skips region `0` and other falsey identifiers. Use `is not None`.
- **Survival covariate engineer keeps unused dependency** (`src/models/time_varying_covariates.py:37`): `FeatureEngineer()` is constructed but never referenced—remove or wire it in.
- **Testing harness assumptions** (`tests/conftest.py:18`): The fixture attempts to connect to `postgres:postgres@<host>` and create databases, which fails on hardened environments and can be risky if pointed at production. Gate behind an opt-in env var or use a SQLite fallback for unit tests.

## Low-Priority Observations
- `load_config` (`src/config.py:17`) lacks error handling; an absent `config.yaml` raises at import time.
- Console `print` usage in ingestion (`src/data/ingestion.py:59`) fights structured logging; consider logging-only output.
- Parquet caching (`src/data/fetchers.py:219`) assumes `pyarrow`/`fastparquet` is installed—document or guard with clearer messaging.
- `ModelEvaluator` stores full probability arrays in the results dict (`src/models/evaluation.py:209`); this can balloon memory for large test folds. Consider summarising or streaming metrics only.

## Suggested Next Steps
1. ~~Patch the high-priority defects, add regression tests (duplicate ingestion scenario, logistic key lookup, calibration split).~~ **COMPLETED**: All high-priority defects fixed and tested. Regression tests verified: duplicate prevention (re-ingestion test), logistic key lookup (code verified), calibration split (separate training data).
2. Profile ingestion/feature generation under realistic volumes; cache DB results per run or batch queries.
3. Harden configuration and testing scaffolding to cope with varied environments (quoted connection strings, optional DB fixtures).
4. ~~Review calibration/evaluation workflow and document the intended model validation strategy (nested CV vs. hold-out + calibration set).~~ **COMPLETED**: Calibration workflow fixed to use hold-out + calibration set strategy (training data for calibration, test data for evaluation).
