# flare+ makefile for common development tasks
# note: ./flare script is the preferred/default way to run commands
# this makefile provides make targets that call ./flare for convenience

.PHONY: help build up down logs shell db-shell test lint format format-check clean init-db ingest api api-bg api-stop api-logs ui ui-bg ui-stop ui-logs ingest-api validate validate-model check-config backtest

help:
	@echo "flare+ development commands"
	@echo ""
	@echo "preferred usage: ./flare <command>"
	@echo "  (see ./flare help for full command list)"
	@echo ""
	@echo "makefile targets (wrappers around ./flare):"
	@echo ""
	@echo "docker services:"
	@echo "  make build        - build docker images"
	@echo "  make up           - start core docker services (postgres, app)"
	@echo "  make down         - stop all docker services"
	@echo "  make logs         - view logs from all services (ctrl+c to exit)"
	@echo "  make shell        - open interactive shell in app container"
	@echo "  make db-shell     - open psql shell in database container"
	@echo "  make clean        - remove containers and volumes (cleanup)"
	@echo ""
	@echo "database:"
	@echo "  make init-db      - initialize database schema"
	@echo "  make ingest       - run data ingestion via dedicated ingestion service"
	@echo ""
	@echo "services (api & ui):"
	@echo "  make api          - start flask api service (foreground, default http://127.0.0.1:5001)"
	@echo "  make api-bg       - start flask api service in background"
	@echo "  make api-stop     - stop api service"
	@echo "  make api-logs     - stream api service logs"
	@echo "  make ui           - start gradio ui dashboard (foreground, default http://127.0.0.1:7860)"
	@echo "  make ui-bg        - start gradio ui dashboard in background"
	@echo "  make ui-stop      - stop ui dashboard"
	@echo "  make ui-logs      - stream ui dashboard logs"
	@echo "  make ingest-api   - trigger data ingestion via api endpoint (requires api-bg)"
	@echo ""
	@echo "development:"
	@echo "  make test         - run test suite"
	@echo "  make lint         - run linters (flake8, black check)"
	@echo "  make format       - format code with black"
	@echo "  make format-check - check formatting without changing files"
	@echo ""
	@echo "validation:"
	@echo "  make validate           - run full system validation"
	@echo "  make validate-model     - validate specific model (MODEL_PATH required)"
	@echo "  make check-config       - check environment configuration"
	@echo ""
	@echo "examples:"
	@echo "  make up           # start docker services"
	@echo "  make ingest       # populate database via ingestion service"
	@echo "  make api          # start api service"
	@echo "  make ui           # start ui dashboard"
	@echo "  make ingest-api   # trigger ingestion via api"
	@echo "  make validate     # run system validation"
	@echo ""
	@echo "or use ./flare directly:"
	@echo "  ./flare up        # start postgres + toolbox container"
	@echo "  ./flare ingest    # populate database via ingestion service"
	@echo "  ./flare api-bg    # start api service in background"
	@echo "  ./flare ui-bg     # start ui dashboard in background"
	@echo "  ./flare validate  # run system validation"

build:
	./flare build

up:
	./flare up

down:
	./flare down

logs:
	./flare logs

shell:
	./flare shell

db-shell:
	./flare db-shell

test:
	./flare test

lint:
	./flare lint

format:
	./flare format

format-check:
	docker-compose exec app black --check src/ tests/ scripts/

clean:
	./flare clean

init-db:
	./flare init-db

ingest:
	./flare ingest

api:
	./flare api

api-bg:
	./flare api-bg

api-stop:
	./flare api-stop

api-logs:
	./flare api-logs

ui:
	./flare ui

ui-bg:
	./flare ui-bg

ui-stop:
	./flare ui-stop

ui-logs:
	./flare ui-logs

ingest-api:
	./flare ingest-api

validate:
	./flare validate

validate-model:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH required"; \
		echo "Usage: make validate-model MODEL_PATH=/app/models/survival_model.joblib"; \
		exit 1; \
	fi
	./flare validate-model "$(MODEL_PATH)"

check-config:
	./flare check-config

backtest:
	./flare backtest $(BACKTEST_ARGS)

# local development (without docker)
.PHONY: local-install local-test local-lint local-format

local-install:
	pip install uv
	uv pip install -r requirements-dev.txt

local-test:
	pytest tests/ -v

local-lint:
	flake8 src/ tests/
	black --check src/ tests/ scripts/

local-format:
	black src/ tests/ scripts/
