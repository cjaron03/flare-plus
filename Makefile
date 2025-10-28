# flare+ makefile for common development tasks

.PHONY: help build up down logs shell db-shell test lint format clean

help:
	@echo "flare+ development commands:"
	@echo "  make build      - build docker images"
	@echo "  make up         - start all services"
	@echo "  make down       - stop all services"
	@echo "  make logs       - view logs (ctrl+c to exit)"
	@echo "  make shell      - open shell in app container"
	@echo "  make db-shell   - open psql shell in database"
	@echo "  make test       - run test suite"
	@echo "  make lint       - run linters"
	@echo "  make format     - format code with black"
	@echo "  make clean      - remove containers and volumes"
	@echo "  make init-db    - initialize database schema"
	@echo "  make ingest     - run data ingestion"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "services started!"
	@echo "streamlit ui: http://localhost:8501"
	@echo "flask api: http://localhost:5000"

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker-compose exec app /bin/bash

db-shell:
	docker-compose exec postgres psql -U postgres -d flare_prediction

test:
	docker-compose exec app pytest tests/ -v

lint:
	docker-compose exec app flake8 src/ tests/
	docker-compose exec app black --check src/ tests/

format:
	docker-compose exec app black src/ tests/

clean:
	docker-compose down -v
	docker system prune -f

init-db:
	docker-compose exec app python scripts/init_db.py

ingest:
	docker-compose exec app python scripts/run_ingestion.py

# local development (without docker)
local-install:
	pip install -r requirements.txt

local-test:
	pytest tests/ -v

local-lint:
	flake8 src/ tests/
	black --check src/ tests/

local-format:
	black src/ tests/

