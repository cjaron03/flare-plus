# flare+ project summary (final)

last updated: **February 12, 2026**

## 1) scope and objective

`flare+` is a machine learning system for short-horizon solar flare forecasting.

primary goals:
- ingest operational space-weather signals continuously
- predict flare risk (classification + probability) over 24-48h windows
- benchmark against NOAA/SWPC public probability feed where available
- expose predictions and system status via API/UI

## 2) architecture

high-level flow:
1. **data ingestion**
   - NOAA/SWPC operational feeds (xray, regions, magnetogram)
   - NASA DONKI flare catalog (historical flare events)
   - NOAA NCEI GOES archive support for deeper historical xray backfill
2. **feature engineering**
   - flux trend and rolling-window statistics
   - magnetic/sunspot complexity features
   - recency-weighted flare history features
3. **modeling**
   - multi-class classification (`None`, `C`, `M`, `X`)
   - probability outputs used for thresholded event decisions
4. **serving + ops**
   - Flask prediction API
   - FastAPI/Svelte UI backend + dashboard
   - realtime NOAA-vs-flare monitoring daemon + watchdog + staleness checks

## 3) data pipeline status

implemented:
- incremental operational ingestion via `./flare ingest`
- DONKI ingestion via `./flare import-donki`
- NCEI historical XRS ingestion via `./flare ingest-ncei`
- realtime benchmark logging table `flare_noaa_realtime_log`

operational logging (currently running):
- daily predictions for flare+ and NOAA
- resolved actual outcomes
- timestamped status/report artifacts in `scripts/runtime/`

## 4) performance snapshot

### historical benchmark run (latest)

run date: **February 12, 2026**  
target: **>= M-class in next 1 day**

sample windows from `data/noaa_benchmark_details.csv`:
- **31-day sample** (`2026-01-13` to `2026-02-12`, deployed thresholds `flare=0.38`, `NOAA=0.56`)
  - flare+ accuracy: **67.74%**
  - NOAA accuracy: **70.97%**
  - delta: **-3.23 points**
- **14-day holdout** (`2026-01-30` to `2026-02-12`, deployed thresholds `flare=0.38`, `NOAA=0.56`)
  - flare+ accuracy: **71.43%**
  - NOAA accuracy: **71.43%**
  - delta: **0.00 points**

### realtime tracker snapshot (live)

as of **February 12, 2026**:
- resolved realtime rows: **n=1** (`2026-02-11`)
- flare+ accuracy: **100.00%**
- NOAA accuracy: **0.00%**

note: realtime sample is currently too small for strong inference.

## 5) key limitation (decision-driving)

the dominant constraint is **historical NOAA probability feed availability**.

while the project can backfill historical features/events (NCEI + DONKI), the exact NOAA/SWPC probability stream used in operational comparison is not available as a long, structured historical dataset in this repo workflow. that limits robust long-timescale apples-to-apples validation versus NOAA probabilities.

practical implication:
- model improvements can be developed
- but confidence in “consistently beats NOAA over long horizons” remains constrained until sufficient realtime accumulation (or a separate archived NOAA forecast reconstruction pipeline) exists

## 6) project closeout position

engineering conclusion:
- system is functional end-to-end
- benchmarking against NOAA is implemented and operational in realtime
- limitations are explicit and documented

this is an acceptable completion point for portfolio/interview use:
- demonstrates full-stack ML/system execution
- includes honest treatment of data constraints and validation boundaries
- shows sound scope control and product judgment
