# multi-stage docker build for flare-plus
# optimized with uv for faster dependency installation

# stage 1: builder
FROM python:3.9-slim AS builder

WORKDIR /build

# install build dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# create virtualenv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy and install dependencies
# for production, use requirements.txt only
# for local dev with docker-compose, we install dev deps for testing
ARG INSTALL_DEV=true
COPY requirements.txt requirements-dev.txt ./
RUN if [ "$INSTALL_DEV" = "true" ]; then \
      uv pip install --no-cache -r requirements-dev.txt; \
    else \
      uv pip install --no-cache -r requirements.txt; \
    fi

# stage 2: runtime
FROM python:3.9-slim

# metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="flare-plus team" \
      org.opencontainers.image.url="https://github.com/cjaron03/flare-plus" \
      org.opencontainers.image.source="https://github.com/cjaron03/flare-plus" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="flare-plus" \
      org.opencontainers.image.description="solar flare prediction system"

WORKDIR /app

# install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

# copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config.yaml ./
COPY pyproject.toml ./

# create data directories
RUN mkdir -p /app/data/cache /app/models

# create non-root user and set ownership
RUN useradd -m -u 1000 flareuser && \
    chown -R flareuser:flareuser /app

# make scripts executable
RUN chmod +x scripts/*.py

USER flareuser

# set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    VIRTUAL_ENV="/opt/venv" \
    PYTHONUNBUFFERED="1"

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# default command
CMD ["python", "-m", "src"]
