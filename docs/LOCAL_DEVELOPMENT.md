# local development guide

this guide covers running flare-plus locally using docker compose or native python.

## prerequisites

- docker and docker-compose installed (for docker method)
- python 3.9+ installed (for native method)
- make (optional, but recommended)

## quick start with docker

### option 1: using make (recommended)

```bash
# start all services
make up

# view logs
make logs

# initialize database
make init-db

# run data ingestion
make ingest

# open shell in app container
make shell

# run tests
make test

# stop services
make down
```

### option 2: using docker-compose directly

```bash
# start services
docker-compose up -d

# view logs
docker-compose logs -f

# initialize database
docker-compose exec app python scripts/init_db.py

# run data ingestion
docker-compose exec app python scripts/run_ingestion.py

# stop services
docker-compose down
```

## native development (without docker)

### setup with uv (recommended for speed)

```bash
# install uv
pip install uv

# install runtime dependencies only
uv pip install -r requirements.txt

# or install dev dependencies for testing
uv pip install -r requirements-dev.txt

# set up local postgres (macos)
brew install postgresql@14
brew services start postgresql@14

# create database
createdb flare_prediction

# set environment variables
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=flare_prediction
export DB_USER=your_username
export DB_PASSWORD=

# initialize database
python scripts/init_db.py

# run data ingestion
python scripts/run_ingestion.py

# run tests
pytest tests/ -v
```

### traditional pip setup

```bash
# create virtual environment
python3.9 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements-dev.txt

# follow same setup steps as above
```

## dependency management

### requirements files

- `requirements.txt` - runtime dependencies only (used in docker image)
- `requirements-dev.txt` - includes runtime + development/testing tools

### updating dependencies

```bash
# with uv (faster)
uv pip install package_name
uv pip freeze > requirements.txt

# traditional pip
pip install package_name
pip freeze > requirements.txt
```

## services

### docker-compose setup includes:

- **postgres** - postgresql 14 database
  - accessible at `localhost:5432`
  - default credentials: `postgres/postgres`
  - database: `flare_prediction`

- **app** - flare-plus application container
  - includes all python dependencies
  - mounts source code for live development
  - connects to postgres automatically

## environment variables

create a `.env` file in the project root to customize settings:

```bash
# database configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=flare_prediction
DB_USER=postgres
DB_PASSWORD=postgres

# data ingestion settings
DATA_CACHE_HOURS=48
UPDATE_INTERVAL_MINUTES=60
BACKFILL_START_DATE=2023-01-01
```

## development workflow

### 1. start services
```bash
make up
```

### 2. initialize database
```bash
make init-db
```

### 3. run data ingestion
```bash
make ingest
```

### 4. develop with live reload

the `src/` directory is mounted as a volume, so changes are reflected immediately:

```bash
# edit files locally
vim src/data/fetchers.py

# run tests to verify
make test
```

### 5. database access

```bash
# open psql shell
make db-shell

# or manually
docker-compose exec postgres psql -U postgres -d flare_prediction
```

### 6. run python commands

```bash
# open app shell
make shell

# then run any python command
python -c "from src.data.fetchers import XRayFluxFetcher; print('works!')"
```

## testing

### with docker

```bash
# run all tests
make test

# run specific test file
docker-compose exec app pytest tests/test_database.py -v

# run with coverage
docker-compose exec app pytest tests/ -v --cov=src --cov-report=term-missing
```

### native (faster for development)

```bash
# set up test database
createdb flare_prediction_test

# export test environment
export DB_NAME=flare_prediction_test

# run tests
pytest tests/ -v

# with coverage
pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80
```

## linting and formatting

```bash
# with docker
make lint
make format

# native
flake8 src/
black src/
mypy src/ --ignore-missing-imports
```

## cleaning up

```bash
# stop services (keep volumes)
make down

# stop services and remove volumes (clean slate)
make clean
```

## docker optimization

the dockerfile uses several optimizations:

- **uv** for 10-100x faster dependency installation
- **multi-stage build** to keep final image small
- **virtualenv** at `/opt/venv` for clean python environment
- **runtime-only dependencies** in the image (no dev tools)
- **non-root user** for security

## troubleshooting

### database connection issues

```bash
# check if postgres is healthy
docker-compose ps

# check postgres logs
docker-compose logs postgres

# restart services
make down && make up
```

### python dependency issues

```bash
# rebuild containers
make build

# or
docker-compose build --no-cache
```

### port conflicts

if port 5432 is already in use, update `.env`:

```bash
DB_PORT=5433
```

then restart:

```bash
make down && make up
```

### slow dependency installs

switch to uv for much faster installs:

```bash
# install uv
pip install uv

# use uv for installs
uv pip install -r requirements-dev.txt
```

## ci/cd integration

the optimized ci/cd workflows use:

- **uv** for faster dependency installation in ci
- **pip caching** via actions/setup-python
- **concurrency cancellation** to stop outdated workflow runs
- **split test matrix** - lint runs once on 3.9, tests run on 3.9/3.10/3.11
- **docker caching** via github actions cache
- **pr builds** validate but don't push images
- **main/tag pushes** build and publish to ghcr.io

## available make commands

run `make help` to see all available commands:

```
make help       - show all commands
make build      - build docker images
make up         - start all services
make down       - stop all services
make logs       - view logs
make shell      - open shell in app container
make db-shell   - open psql shell
make test       - run test suite
make lint       - run linters
make format     - format code
make clean      - remove containers and volumes
make init-db    - initialize database
make ingest     - run data ingestion
```
