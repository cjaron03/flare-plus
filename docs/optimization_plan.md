# Optimization Plan for CI/CD and Container Builds

## Constraints & Context
- Runtime must support Python 3.9 (future work can add 3.10+ testing but deployment sticks to 3.9).
- Container will eventually handle both inference and periodic training tasks.
- Team is open to adopting a faster dependency manager such as `uv` (preferred) or Poetry.

## CI Pipeline Improvements
- Collapse redundant caches: rely on `actions/setup-python` built-in pip cache and drop extra `actions/cache` steps in CI, PR checks, and nightly workflows.
- Separate concerns in the matrix: keep pytest on Python 3.9–3.11 but run lint/format/mypy once on 3.9 to trim job time.
- External API stability: stub NOAA calls in fast CI workflows; keep live checks only in nightly schedule with retries + timeout logging.
- Reusable steps: factor lint/security/tooling install into a composite action or workflow call to remove duplicated setup blocks.
- PR hygiene: add concurrency cancellation (`concurrency.group`) to drop superseded runs and enable required checks gating.
- Docker build workflow: only push images on main/tags; PR builds should use `docker build` with `push: false` and leverage build cache for faster validation.

## Nightly / Scheduled Jobs
- Harden ingestion: wrap `run_ingestion.py` in retry logic and emit structured logs for NOAA failures.
- Persist drift metrics: store outputs as workflow artifacts or push to monitoring to align with future ops alerting.
- Extend coverage selectively: schedule 3.10/3.11 test jobs weekly rather than nightly to keep runtime manageable.

## Docker Optimization
- Stay on `python:3.9-slim` base image; upgrade to point releases as available.
- Switch builder stage to `uv` (`pip install uv && uv pip install`) using cache mounts to accelerate rebuilds.
- Split dependencies: create `requirements.txt` (runtime) and `requirements-dev.txt` (dev/test). CI installs both; image only installs runtime list.
- Use virtualenv at `/opt/venv` built in the first stage, copy the venv into runtime, and adjust `PATH`/`VIRTUAL_ENV` instead of relying on `/root/.local`.
- Combine COPY instructions and ensure `.dockerignore` doesn’t omit needed files (remove `docs/` exclusion if docs are required).
- Add multi-platform build targets (linux/amd64, linux/arm64) once base image optimizations are stable.
- Introduce image scanning gate: keep Trivy scan but consider failing build only on high/critical to avoid noise.

## Follow-Up Tasks for Claude
- Draft updated CI workflow(s) implementing cache cleanup, matrix split, and concurrency guardrails.
- Author composite action (or reusable workflow) for shared lint/security setup.
- Refactor Dockerfile to incorporate `uv`, virtualenv copy, runtime/dev dependency split, and improved caching.
- Review `.dockerignore` and adjust to align with new Docker build strategy.
- Document local development workflow changes (e.g., using `uv` for installs) in README.
