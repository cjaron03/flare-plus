# Solar Flare Prediction Roadmap

## Data Ingestion
- [x] Inventory NOAA/SWPC endpoints for GOES XRS flux, sunspot classifications, magnetograms; document auth + cadence.
  - [OK] GOES XRS flux endpoints documented (`goes_xrs_7day`, `goes_xrs_6hour`)
  - [OK] Solar regions endpoint documented (`solar_regions`) - fixed column mapping to match NOAA API format
  - [OK] Magnetograms endpoint implemented (`MagnetogramFetcher` extracts magnetic field data from solar regions)
  - [OK] Auth documented (public, no API keys required)
  - [OK] Cadence documented (GOES XRS: real-time ~5min, solar regions: daily, recommended ingestion: 60min)
- [x] Build data fetcher with caching (24-48h window, plus historical backfill) and persistence (e.g., parquet or postgres).
  - [OK] Fetchers implemented (`GOESXRayFetcher`, `SolarRegionFetcher`, `MagnetogramFetcher`)
  - [OK] Caching layer with configurable 24-48h window
  - [OK] Persistence to PostgreSQL (`DataPersister`)
  - [OK] Data quality handling (filters invalid records, handles NaN values)
  - [NOTE] Historical backfill placeholder exists (future enhancement - requires manual NOAA archive processing)
- [x] Schedule incremental updates and verify schema covers features needed for both models.
  - [OK] Incremental update script (`scripts/run_ingestion.py`)
  - [OK] Schema covers flux, regions, magnetogram, flares tables
  - [OK] All data sources tested and working (10k+ xray flux, 384+ solar regions, 384+ magnetogram records)
  - [NOTE] Schema verification against model feature requirements pending (models not yet built)

## Feature Engineering
- [x] Derive sunspot complexity metrics (McIntosh/Mount Wilson) and flux trend features from magnetograms.
  - [OK] McIntosh complexity metrics implemented (`compute_mcintosh_complexity`)
  - [OK] Mount Wilson complexity metrics implemented (`compute_mount_wilson_complexity`)
  - [OK] Magnetic complexity score implemented (`compute_magnetic_complexity_score`)
  - [OK] Flux trend features implemented (`compute_flux_trends`, `compute_flux_rate_of_change`)
- [x] Aggregate rolling statistics (last 6/12/24h) and recency-weighted flare counts by class.
  - [OK] Rolling statistics implemented (`compute_rolling_statistics`) with configurable windows
  - [OK] Recency-weighted flare counts implemented (`compute_recency_weighted_flare_counts`) with exponential decay
  - [OK] Supports multiple time windows (6/12/24h) and flare classes (B/C/M/X)
- [x] Normalize/standardize features; flag missing data paths and imputation strategy.
  - [OK] Feature normalization implemented (`normalize_features`) with min-max and robust scaling
  - [OK] Feature standardization implemented (`standardize_features`) with standard and robust scaling
  - [OK] Missing data handling implemented (`handle_missing_data`) with multiple strategies
  - [OK] Missing data flagging implemented (`flag_missing_data_paths`) for monitoring
- [x] Feature engineering pipeline (`FeatureEngineer`) for end-to-end feature computation

## Short-Term Classification (24-48h)
- [x] Frame supervised labels for next-24h and next-48h flare classes {None, C, M, X}.
  - [OK] Label creation module implemented (`FlareLabeler` class)
  - [OK] Supports multiple prediction windows (24h, 48h)
  - [OK] Handles class hierarchy (X > M > C > None)
  - [OK] Integrates with feature engineering pipeline
- [x] Train baseline models (logistic, gradient boosting) with cross-validation and class-balancing.
  - [OK] Logistic regression with class weights and SMOTE support
  - [OK] Gradient boosting classifier with cross-validation
  - [OK] Stratified K-fold cross-validation implemented
  - [OK] Class balancing via class weights and SMOTE oversampling
  - [OK] Model training pipeline (`ModelTrainer` class)
- [x] Calibrate probabilities; evaluate with Brier score, ROC-AUC per class, and reliability diagrams.
  - [OK] Probability calibration implemented (isotonic and sigmoid methods)
  - [OK] Brier score computation per class and macro average
  - [OK] ROC-AUC per class (one-vs-rest) with macro average
  - [OK] Reliability diagrams (calibration curves) implemented
  - [OK] Comprehensive evaluation metrics (`ModelEvaluator` class)
  - [OK] End-to-end pipeline (`ClassificationPipeline`) for training and evaluation

## Time-to-Event Modeling
- [x] Define target windows for next X-class flare; experiment with survival analysis or hazard models.
  - [OK] Survival labeling implemented (`SurvivalLabeler`) with configurable target flare class (default: X, supports M/C)
  - [OK] Configurable observation window (default: 168 hours / 7 days)
  - [OK] Handles both event observations and censored observations
  - [OK] Time-to-event labels computed: duration (hours to event) and event indicator (1=occurred, 0=censored)
  - [OK] Automatic flare detection from X-ray flux data (`FlareDetector`) for historical data
- [x] Engineer time-varying covariates from recent conditions; compare Cox PH vs gradient boosting survival.
  - [OK] Time-varying covariates implemented (`TimeVaryingCovariateEngineer`)
  - [OK] Recent flux metrics (mean, max, trend) over multiple lookback windows
  - [OK] Recent region complexity metrics (McIntosh, Mount Wilson) over lookback windows
  - [OK] Recent flare history (counts, max class, hours since last flare)
  - [OK] Cox Proportional Hazards model implemented (`CoxProportionalHazards`)
  - [OK] Gradient Boosting Survival model implemented (`GradientBoostingSurvival`)
  - [OK] Model comparison and evaluation framework
  - [OK] Progress bars for batch processing with tqdm
- [x] Produce probability distribution over time buckets; validate with concordance index.
  - [OK] Probability distribution over time buckets implemented (6h, 12h, 24h, 48h, 72h, 96h, 120h, 168h)
  - [OK] Survival function prediction from both models
  - [OK] Concordance index (C-index) validation for both models
  - [OK] End-to-end pipeline (`SurvivalAnalysisPipeline`) for training, evaluation, and prediction
  - [OK] Command-line script for training and prediction (`scripts/train_and_predict_survival.py`)
  - [OK] Model persistence (save/load trained models)
  - [OK] Flexible flare class targeting (X/M/C) with automatic data availability checking

## Model Serving
- [x] Wrap models behind a single prediction service exposing both probability endpoints.
  - [OK] Prediction methods implemented (`ClassificationPipeline.predict()` and `SurvivalAnalysisPipeline.predict_survival_probabilities()`)
  - [OK] Command-line scripts for predictions (`scripts/train_and_predict_survival.py`)
  - [OK] HTTP API service (Flask) implemented (`src/api/app.py`) with endpoints: `/health`, `/predict/classification`, `/predict/survival`, `/predict/all`
  - [OK] Unified service interface (`PredictionService`) combining classification and survival predictions
  - [OK] Server script for running API (`scripts/run_api_server.py`) with model loading support
- [x] Implement monitoring hooks for input drift and outcome logging.
  - [OK] Input drift detection implemented (`InputDriftDetector`) with statistical tests (Kolmogorov-Smirnov, Mann-Whitney U)
  - [OK] Outcome logging infrastructure implemented (`OutcomeLogger`) for storing predictions and actuals
  - [OK] Model performance tracking over time (`OutcomeLogger.get_performance_metrics()`)
  - [OK] Database schema for prediction logs (`PredictionLog` table)
  - [OK] Comprehensive test suite (19 tests covering all endpoints and monitoring)

## UI Prototype
- [x] Build lightweight dashboard (Streamlit/Gradio) showing current probabilities and historical flare timelines.
  - [OK] Gradio-based dashboard implemented with modular structure
  - [OK] Predictions tab: classification and survival analysis with plain language summaries
  - [OK] Timeline tab: historical flare event visualization with Plotly charts
  - [OK] About tab: NOAA source links, author attribution (Jaron Cabral), limitations disclaimer
  - [OK] Hybrid connection: API-first with fallback to direct model loading
  - [OK] Performance throttling for refresh operations
  - [OK] Data freshness indicators in sidebar
  - [NOTE] Scenario exploration tab placeholder (requires full feature recalculation - future enhancement)
- [x] Add controls for scenario exploration (e.g., tweak sunspot metrics) and display NOAA source links.
  - [OK] NOAA source links displayed in About tab (from config.yaml)
  - [OK] Author attribution (Jaron Cabral) prominently displayed
  - [OK] Limitations clearly marked throughout UI
  - [NOTE] Scenario controls simplified (full implementation requires feature engineering recalculation)

## Validation & Ops
- [ ] Backtest against recent solar cycles; compare against NOAA forecasts.
- [ ] Document deployment plan (schedule retraining cadence, model registry) and publish README.

## Infrastructure & Automation
- [ ] Stand up CI workflow that runs unit tests, linting, and scheduled data/feature drift checks.
- [ ] Containerize data + model services with reproducible environment specs (Docker/Conda).
- [ ] Provision experiment tracking (MLflow/W&B) and artifact versioning for datasets + models.

## Interpretability & Communication
- [ ] Implement SHAP or permutation importance to explain key drivers per prediction.
- [ ] Produce analyst-friendly daily report summarizing probabilities, confidence, and notable drivers.
- [ ] Add alert thresholds and messaging templates for ops handoff (e.g., Slack/email).

## Resilience & Data Governance
- [ ] Define fallback strategies when NOAA endpoints lag (e.g., cached feeds, alternate providers).
- [ ] Audit licensing/usage terms for NOAA and third-party datasets; note attribution requirements.
- [ ] Establish data retention, quality checks, and anomaly alerting policies.

## Extended Data Sources
- [ ] Explore integrating SDO/AIA imagery, SOHO data, or helioseismology signals for richer context.
- [ ] Evaluate solar wind and CME catalogs as auxiliary predictors for time-to-event modeling.

## Future Research Directions
- [ ] Investigate hybrid physics-informed ML approaches or transfer learning from NASA datasets.
- [ ] Prototype transformer-based sequence models for multi-day flare forecasting.
- [ ] Assess feasibility of probabilistic forecasts beyond 48h (e.g., 7-day outlook) with uncertainty quantification.
