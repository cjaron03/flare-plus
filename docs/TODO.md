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
- [ ] Frame supervised labels for next-24h and next-48h flare classes {None, C, M, X}.
- [ ] Train baseline models (logistic, gradient boosting) with cross-validation and class-balancing.
- [ ] Calibrate probabilities; evaluate with Brier score, ROC-AUC per class, and reliability diagrams.

## Time-to-Event Modeling
- [ ] Define target windows for next X-class flare; experiment with survival analysis or hazard models.
- [ ] Engineer time-varying covariates from recent conditions; compare Cox PH vs gradient boosting survival.
- [ ] Produce probability distribution over time buckets; validate with concordance index.

## Model Serving
- [ ] Wrap models behind a single prediction service exposing both probability endpoints.
- [ ] Implement monitoring hooks for input drift and outcome logging.

## UI Prototype
- [ ] Build lightweight dashboard (Streamlit/Gradio) showing current probabilities and historical flare timelines.
- [ ] Add controls for scenario exploration (e.g., tweak sunspot metrics) and display NOAA source links.

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
