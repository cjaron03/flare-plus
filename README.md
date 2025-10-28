# flare+ solar flare prediction system

a machine learning system for predicting solar flares using noaa/swpc data.

## overview

flare+ implements short-term (24-48h) classification and time-to-event modeling for solar flare prediction. the system ingests real-time data from noaa goes satellites and solar region observations to predict flare probability and timing.

## features

- **data ingestion**: automated fetching from noaa/swpc endpoints with caching and persistence
- **feature engineering**: sunspot complexity metrics, flux trends, and rolling statistics
- **24-48h classification**: predict flare class probability (none, c, m, x)
- **time-to-event modeling**: estimate timing distribution for next x-class flare
- **model serving**: flask api for probability endpoints
- **ui dashboard**: streamlit interface for visualization and exploration

## project structure

```
flare+/
├── src/
│   ├── data/           # data ingestion and persistence
│   ├── features/       # feature engineering (todo)
│   ├── models/         # ml models (todo)
│   ├── api/            # flask api (todo)
│   └── ui/             # streamlit dashboard (todo)
├── docs/               # documentation
├── config.yaml         # configuration
├── requirements.txt    # python dependencies
└── README.md
```

## setup

### prerequisites

- python 3.9+
- postgresql 12+

### installation

1. clone the repository:
```bash
cd "Flare+"
```

2. create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # on windows: venv\Scripts\activate
```

3. install dependencies:
```bash
pip install -r requirements.txt
```

4. configure environment:
```bash
cp .env.example .env
# edit .env with your database credentials
```

5. initialize database:
```bash
# create postgresql database
createdb flare_prediction

# initialize tables
python -m src.data.ingestion
```

## usage

### data ingestion

run incremental data update:
```bash
python -m src.data.ingestion
```

this will:
- fetch last 7 days of goes x-ray flux data
- fetch current active solar regions
- store data in postgresql with caching

### scheduled updates

configure scheduled updates using cron or system scheduler:
```bash
# example cron job - run every hour
0 * * * * cd /path/to/flare+ && /path/to/venv/bin/python -m src.data.ingestion
```

## data sources

all data comes from noaa space weather prediction center (swpc):

- **goes xrs flux**: https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json
- **solar regions**: https://services.swpc.noaa.gov/json/solar_regions.json
- **historical archive**: https://www.ncei.noaa.gov/data/goes-space-environment-monitor/

no api keys required - data is publicly accessible.

## development status

current implementation:
- [x] project structure and configuration
- [x] database schema for flux, regions, flares
- [x] noaa data fetchers with retry logic
- [x] caching layer for efficiency
- [x] postgresql persistence
- [x] ingestion logging and monitoring

next steps:
- [ ] feature engineering module
- [ ] 24-48h classification model
- [ ] model training pipeline
- [ ] flask api for predictions
- [ ] streamlit dashboard
- [ ] time-to-event modeling

## contributing

this is a personal project for learning and experimentation. the roadmap is documented in `docs/TODO.md`.

## license

mit

## acknowledgments

data provided by noaa space weather prediction center.

