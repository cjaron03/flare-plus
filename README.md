# flare+ solar flare prediction system

a machine learning system for predicting solar flares using noaa/swpc data.

## overview

flare+ implements short-term (24-48h) classification and time-to-event modeling for solar flare prediction. the system ingests real-time data from noaa goes satellites and solar region observations to predict flare probability and timing.

## features

- **data ingestion**: automated fetching from noaa/swpc endpoints
  - goes x-ray flux data (real-time, 5min cadence)
  - solar region observations (daily updates)
  - magnetogram data extraction
  - automatic flare detection from flux data
  - caching and persistence to postgresql

- **feature engineering**: comprehensive pipeline
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

- **model serving**: flask api with monitoring
  - rest endpoints for classification and survival predictions
  - health monitoring with database and disk space checks
  - input drift detection
  - outcome logging to database

- **interactive ui**: gradio-based dashboard
  - real-time predictions (classification and survival)
  - historical flare event timeline with filters
  - system health monitoring

## quick start

### prerequisites

- docker and docker-compose
- 2gb+ free disk space

### setup and run

1. clone repository and start services:
```bash
cd flare-plus
./flare up
```

2. initialize database:
```bash
./flare init-db
```

3. ingest data:
```bash
./flare ingest
```

4. start api server:
```bash
./flare api-bg
```

5. start ui dashboard:
```bash
./flare ui-bg
```

6. access dashboard at http://127.0.0.1:7860

### validate system

before deployment, run full system validation:
```bash
./flare validate
```

## project structure

```
flare-plus/
├── src/
│   ├── data/           # data ingestion, persistence, flare detection
│   ├── features/       # feature engineering (complexity, trends, rolling stats)
│   ├── models/         # ml models (classification and survival analysis)
│   ├── api/            # flask api with monitoring and drift detection
│   └── ui/             # gradio dashboard
├── scripts/
│   ├── run_ingestion.py              # data ingestion script
│   ├── run_api_server.py             # api server
│   ├── run_ui.py                     # ui dashboard
│   ├── init_db.py                    # database initialization
│   ├── validate_system.py            # end-to-end validation
│   ├── validate_models.py            # model validation
│   ├── check_config.py               # configuration validation
│   └── train_and_predict_*.py        # model training scripts
├── models/             # trained model artifacts (gitignored)
├── data/cache/         # ingestion cache (gitignored)
├── docs/               # documentation and roadmap
├── config.yaml         # configuration
├── requirements.txt    # python dependencies
├── docker-compose.yml  # docker services
├── Makefile            # convenience targets
├── flare               # main command wrapper
└── README.md
```

## usage

all commands use the `./flare` wrapper script for consistency.

### docker services

```bash
./flare up         # start all docker services (postgres, app)
./flare down       # stop all docker services
./flare logs       # view logs from all services
./flare shell      # open interactive shell in app container
./flare db-shell   # open psql shell in database
./flare clean      # remove containers and volumes
```

### data ingestion

```bash
./flare ingest           # run data ingestion from noaa sources
./flare ingest-api       # trigger ingestion via api endpoint
```

ingestion fetches:
- last 7 days of goes x-ray flux data
- current active solar regions
- magnetogram data from regions
- automatically detects flare events from flux data

### model serving (api)

```bash
./flare api              # start api server (foreground)
./flare api-bg           # start api server in background
./flare api-stop         # stop api server
./flare api-logs         # view api server logs
```

api available at http://127.0.0.1:5001

endpoints:
- `GET /health` - health check with system status
- `POST /predict/classification` - 24-48h flare class prediction
- `POST /predict/survival` - time-to-event prediction
- `POST /predict/all` - combined predictions
- `POST /ingest` - trigger data ingestion

### ui dashboard

```bash
./flare ui               # start ui dashboard (foreground)
./flare ui-bg            # start ui dashboard in background
./flare ui-stop          # stop ui dashboard
./flare ui-logs          # view ui dashboard logs
```

dashboard available at http://127.0.0.1:7860

features:
- real-time predictions (classification and survival)
- historical flare event timeline with filters
- system health monitoring
- data source information and limitations

### development

```bash
./flare test             # run test suite
./flare lint             # run linters (flake8, black check)
./flare format           # format code with black
```

### makefile shortcuts

all `./flare` commands have equivalent `make` targets:

```bash
make up              # ./flare up
make api-bg          # ./flare api-bg
make ui-bg           # ./flare ui-bg
make validate        # ./flare validate
```

## validation and testing

flare+ includes comprehensive validation tools to ensure system reliability.

### system validation

run full end-to-end validation:

```bash
./flare validate
```

validates:
- database connection and table integrity (5 required tables)
- data ingestion from all noaa sources
- model loading and reconstruction from saved format
- prediction generation with valid outputs (no nan values)
- api endpoint availability and health
- full pipeline: ingestion → features → prediction → database logging

output example:
```
======================================================================
FLARE+ SYSTEM VALIDATOR
======================================================================

Testing database connection...
  Table flare_goes_xray_flux: 11319 records
  Table flare_solar_regions: 378 records
  Table flare_events: 54 records
  Table flare_ingestion_log: 72 records
  Table flare_prediction_log: 2 records
[PASS] Database connection test

Testing data ingestion...
  xray_flux: success (10075 records)
  solar_regions: success (365 records)
  magnetogram: success (365 records)
  flare_events: success (0 records)
[PASS] Data ingestion test

...

======================================================================
VALIDATION SUMMARY
======================================================================
[PASS] Database Connection
[PASS] Data Ingestion
[PASS] Model Loading
[PASS] Predictions
[PASS] API Endpoint
[PASS] Full Prediction Pipeline

6/6 tests passed

[OK] All system validation tests passed
System is ready for deployment
```

### model validation

validate a trained model:

```bash
./flare validate-model /app/models/survival_model.joblib
```

checks:
- model loads without errors
- required methods and attributes exist
- predictions contain no nan values
- probabilities sum to approximately 1.0
- performance metrics meet thresholds (c-index > 0.5)
- comparison with previous model version

### configuration validation

verify environment setup:

```bash
./flare check-config
```

validates:
- .env file with required database credentials
- config.yaml structure and required sections
- database connection
- required directories (models/, data/cache/)
- disk space availability

### prediction logging

predictions are automatically logged to database for monitoring:

```python
from src.api.monitoring import OutcomeLogger

# logger persists to flare_prediction_log table
logger = OutcomeLogger(persist_to_db=True)

# retrieve logged predictions
predictions = logger.get_predictions_from_db(
    prediction_type="classification",
    start_date=datetime(2024, 1, 1),
    limit=100
)
```

### health monitoring

check system health:

```bash
curl http://127.0.0.1:5001/health
```

response includes:
- model availability status (classification/survival)
- database connection status
- last ingestion timestamp
- total predictions logged
- disk space information
- drift detection status

## training models

### survival models

train time-to-event survival model:

```bash
docker-compose exec app python scripts/train_and_predict_survival.py \
  --train \
  --predict \
  --target-class C \
  --detect-flares \
  --save-model /app/models/survival_model_c_class.joblib
```

options:
- `--train` - train new model
- `--predict` - make prediction after training
- `--target-class` - target flare class (X, M, or C)
- `--detect-flares` - detect flares from historical flux first
- `--load-model` - load previously saved model
- `--start-date` / `--end-date` - training date range
- `--model` - model type for prediction (cox or gb)

output example:
```
============================================================
SOLAR FLARE TIME-TO-EVENT PREDICTION
============================================================
timestamp: 2025-11-04 18:05:11
model: COX
hazard score: 1.234

probability distribution (flare in time bucket):
------------------------------------------------------------
  0h-6h          8.5%
  6h-12h         12.3%
  12h-24h        18.7%
  24h-48h        24.2%
  48h-72h        15.8%
  72h-168h       20.5%
```

### classification models

train 24-48h classification model:

```python
from src.models.pipeline import ClassificationPipeline
from datetime import datetime

pipeline = ClassificationPipeline()
dataset = pipeline.prepare_dataset(
    start_date=datetime(2024, 1, 1),
    end_date=datetime.utcnow(),
    sample_interval_hours=1,
)

results = pipeline.train_and_evaluate(dataset, test_size=0.2)
```

## data sources

all data from noaa space weather prediction center (swpc):

- **goes xrs flux**: https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json
  - real-time updates (every 5 minutes)
  - last 7 days of data

- **solar regions**: https://services.swpc.noaa.gov/json/solar_regions.json
  - daily updates (around midnight utc)
  - current active regions with tracking

- **historical archive**: https://www.ncei.noaa.gov/data/goes-space-environment-monitor/
  - manual download and processing required

no api keys required - all data publicly accessible.

### update frequency

- **production**: run ingestion every 60 minutes
  - fresh data without excessive api calls
  - 48-hour cache prevents redundant fetches
- **development**: run manually as needed

## development setup

### local development (without docker)

```bash
# install dependencies
pip install uv
uv pip install -r requirements-dev.txt

# run tests
pytest tests/ -v

# lint and format
flake8 src/ tests/
black src/ tests/ scripts/
```

### code formatting

this project uses black for consistent formatting.

install pre-commit hooks for automatic formatting:
```bash
pip install pre-commit
pre-commit install
```

manual formatting:
```bash
./flare format              # format all python files
make format                 # alternative using make
black src/ tests/ scripts/  # direct invocation
```

check formatting:
```bash
black --check src/ tests/ scripts/
```

all code must pass black formatting before merging.

### environment configuration

create `.env` file with database credentials:
```bash
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password
DB_NAME=flare_prediction
```

configure system in `config.yaml`:
```yaml
data_ingestion:
  cache_expiry_hours: 48
  update_interval_minutes: 60

model_training:
  test_size: 0.2
  random_state: 42
```

## scheduled updates

configure cron for automatic updates:

```bash
# run ingestion every hour
0 * * * * cd /path/to/flare-plus && ./flare ingest-api

# daily model retraining (optional)
0 2 * * * cd /path/to/flare-plus && docker-compose exec -T app python scripts/train_and_predict_survival.py --train --target-class C
```

## contributing

this is a personal project for learning and experimentation. see `docs/TODO.md` for the roadmap.

## license

mit

## acknowledgments

data provided by noaa space weather prediction center.
