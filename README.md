# flare+ solar flare prediction system

a machine learning system for predicting solar flares using noaa/swpc data.

## overview

flare+ implements short-term (24-48h) classification and time-to-event modeling for solar flare prediction. the system ingests real-time data from noaa goes satellites and solar region observations to predict flare probability and timing.

## features

- **data ingestion**: automated fetching from noaa/swpc endpoints with caching and persistence
  - goes x-ray flux data (real-time, ~5min cadence)
  - solar region observations (daily updates)
  - magnetogram data extraction
  - automatic flare detection from flux data
- **feature engineering**: comprehensive feature pipeline
  - sunspot complexity metrics (mcinosh, mount wilson, magnetic complexity)
  - flux trend analysis (mean, max, trend, rate of change, acceleration)
  - rolling statistics over multiple time windows (6h, 12h, 24h)
  - recency-weighted flare counts with exponential decay
  - normalization and standardization with missing data handling
- **24-48h classification**: predict flare class probability (none, c, m, x)
  - logistic regression and gradient boosting models
  - probability calibration (isotonic, sigmoid)
  - comprehensive evaluation metrics (brier score, roc-auc, reliability diagrams)
- **time-to-event modeling**: survival analysis for flare timing prediction
  - cox proportional hazards and gradient boosting survival models
  - configurable target flare classes (x, m, or c)
  - probability distributions over time buckets (6h-168h)
  - concordance index (c-index) validation
  - time-varying covariates from recent conditions
- **model persistence**: save/load trained models for production use

## project structure

```
flare+/
├── src/
│   ├── data/           # data ingestion, persistence, and flare detection
│   ├── features/       # feature engineering (complexity, trends, rolling stats)
│   ├── models/         # ml models (classification and survival analysis)
│   │   ├── labeling.py          # flare label creation for classification
│   │   ├── training.py          # model training with cross-validation
│   │   ├── evaluation.py        # model evaluation and calibration
│   │   ├── pipeline.py         # classification pipeline
│   │   ├── survival_labeling.py         # survival analysis labels
│   │   ├── time_varying_covariates.py   # dynamic features for survival
│   │   ├── survival_models.py          # cox ph and gb survival models
│   │   └── survival_pipeline.py       # end-to-end survival pipeline
│   ├── api/            # flask api (todo)
│   └── ui/             # streamlit dashboard (todo)
├── scripts/
│   ├── run_ingestion.py              # data ingestion script
│   ├── init_db.py                   # database initialization
│   └── train_and_predict_survival.py # survival model training/prediction
├── docs/               # documentation and roadmap
├── config.yaml         # configuration
├── requirements.txt    # python dependencies
└── README.md
```

## setup

### prerequisites

- python 3.9+ (3.9, 3.10, or 3.11)
- postgresql 12+
- docker and docker-compose (recommended)

### installation

1. clone the repository:
```bash
cd "flare-plus"
```

2. create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # on windows: venv\Scripts\activate
```

3. install dependencies:
```bash
pip install -r requirements.txt

# for development (includes linting/formatting tools)
pip install -r requirements-dev.txt
```

4. configure environment:
```bash
cp .env.example .env
# edit .env with your database credentials
```

5. initialize database:
```bash
# using docker-compose (recommended)
docker-compose up -d
docker-compose exec app python scripts/init_db.py

# or manually
createdb flare_prediction
python scripts/init_db.py
```

### docker setup (recommended)

the project includes docker compose configuration for easy setup:

```bash
# build and start containers
docker-compose up -d

# run ingestion
docker-compose exec app python scripts/run_ingestion.py

# run model training
docker-compose exec app python scripts/train_and_predict_survival.py --train --predict --target-class C

# view logs
docker-compose logs -f app
```

## usage

### data ingestion

run incremental data update:
```bash
docker-compose exec app python scripts/run_ingestion.py
```

or directly:
```bash
python scripts/run_ingestion.py
```

this will:
- fetch last 7 days of goes x-ray flux data
- fetch current active solar regions
- extract magnetogram data from regions
- automatically detect flare events from flux data
- store all data in postgresql with caching

### training time-to-event survival models

train a survival model for predicting the next flare (c-class example):
```bash
docker-compose exec app python scripts/train_and_predict_survival.py \
  --train \
  --predict \
  --target-class C \
  --detect-flares \
  --save-model /app/models/survival_model_c_class.joblib
```

options:
- `--train`: train a new model
- `--predict`: make a prediction after training
- `--target-class`: target flare class (X, M, or C)
- `--detect-flares`: detect flares from historical flux data first
- `--load-model`: load a previously saved model
- `--start-date` / `--end-date`: date range for training data
- `--model`: which model to use for prediction (cox or gb)

example output:
```
============================================================
SOLAR FLARE TIME-TO-EVENT PREDICTION
============================================================
timestamp: 2025-10-31 00:07:27
model: COX
hazard score: 2.3456

probability distribution (flare in time bucket):
------------------------------------------------------------
  0h-6h          12.50%
  6h-12h         15.30%
  12h-24h        18.20%
  24h-48h        22.10%
  ...
```

### training classification models

use the classification pipeline to train short-term (24-48h) flare prediction models:
```python
from src.models.pipeline import ClassificationPipeline
from datetime import datetime, timedelta

pipeline = ClassificationPipeline()
dataset = pipeline.prepare_dataset(
    start_date=datetime(2024, 1, 1),
    end_date=datetime.utcnow(),
    sample_interval_hours=1,
)

results = pipeline.train_and_evaluate(dataset, test_size=0.2)
```

### scheduled updates

configure scheduled updates using cron or system scheduler:
```bash
# example cron job - run every hour
0 * * * * cd /path/to/flare+ && docker-compose exec app python scripts/run_ingestion.py
```

## data sources

all data comes from noaa space weather prediction center (swpc):

- **goes xrs flux**: https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json
  - update cadence: real-time (updates every ~5 minutes)
  - data window: last 7 days via `goes_xrs_7day`, last 6 hours via `goes_xrs_6hour`
- **solar regions**: https://services.swpc.noaa.gov/json/solar_regions.json
  - update cadence: daily (typically updated once per day around midnight UTC)
  - data window: current active regions with historical tracking
- **historical archive**: https://www.ncei.noaa.gov/data/goes-space-environment-monitor/
  - requires manual download and processing

no api keys required - data is publicly accessible.

### recommended update frequency

- **production**: run ingestion every 60 minutes (default `UPDATE_INTERVAL_MINUTES=60`)
  - ensures fresh data without excessive API calls
  - cache window (48 hours) prevents redundant fetches
- **development**: run manually as needed (`python scripts/run_ingestion.py`)

## development status

### completed

- [x] project structure and configuration
- [x] database schema for flux, regions, magnetograms, and flares
- [x] noaa data fetchers with retry logic and caching
- [x] automatic flare detection from x-ray flux data
- [x] feature engineering pipeline
  - sunspot complexity metrics (mcinosh, mount wilson)
  - flux trend analysis and rolling statistics
  - recency-weighted flare counts
  - normalization and standardization
- [x] 24-48h classification models
  - logistic regression and gradient boosting
  - probability calibration and evaluation metrics
  - end-to-end training and prediction pipeline
- [x] time-to-event survival analysis
  - cox proportional hazards model
  - gradient boosting survival model
  - time-varying covariates
  - probability distributions over time buckets
  - command-line training and prediction script

### in progress / planned

- [ ] model serving: flask api for probability endpoints
- [ ] ui dashboard: streamlit interface for visualization
- [ ] model monitoring: input drift detection and outcome logging
- [ ] backtesting: validation against historical solar cycles
- [ ] experiment tracking: mlflow/weights & biases integration

see `docs/TODO.md` for the complete roadmap.

## development setup

### pre-commit hooks

install pre-commit hooks for automatic code formatting:

```bash
pip install pre-commit
pre-commit install
```

this will automatically format code with black before each commit, preventing formatting issues in ci.

## contributing

this is a personal project for learning and experimentation. the roadmap is documented in `docs/TODO.md`.

## license

mit

## acknowledgments

data provided by noaa space weather prediction center.
