# multi-stage docker build for flare-plus
# optimized with buildkit cache mounts and uv for speed

# ===== BUILDER STAGE =====
FROM python:3.10-slim AS builder

WORKDIR /build

# install build dependencies with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade "pip>=24.3.1" \
    && pip install --no-cache-dir uv

# create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# ensure pip is up to date before installing dependencies
# pin to pip 24.3.1+ to fix CVE-2025-8869 and CVE-2023-5752
RUN pip install --upgrade "pip>=24.3.1"

# copy requirements
COPY requirements.txt requirements-dev.txt ./

# install dependencies with uv cache mount for speed
# ensure numpy installs correctly before other packages that depend on it
ARG INSTALL_DEV=true
ENV CUDA_VISIBLE_DEVICES="" \
    FORCE_CUDA=0
RUN --mount=type=cache,target=/root/.cache/uv \
    # install numpy first to ensure C extensions are built correctly \
    uv pip install numpy==1.26.4 && \
    # verify numpy works \
    python -c "import numpy; print(f'numpy {numpy.__version__} OK')" && \
    if [ "$INSTALL_DEV" = "true" ]; then \
      uv pip install -r requirements-dev.txt; \
    else \
      uv pip install -r requirements.txt; \
    fi && \
    # verify numpy still works after installing other packages \
    python -c "import numpy; import pandas; print('numpy/pandas OK')"

# ===== RUNTIME STAGE =====
FROM python:3.10-slim

# metadata
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION=main

LABEL org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.authors="flare-plus team" \
      org.opencontainers.image.url="https://github.com/cjaron03/flare-plus" \
      org.opencontainers.image.source="https://github.com/cjaron03/flare-plus" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.title="flare-plus" \
      org.opencontainers.image.description="solar flare prediction system"

WORKDIR /app

# install runtime dependencies with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    VIRTUAL_ENV="/opt/venv" \
    PYTHONUNBUFFERED="1"

# copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pyproject.toml ./
# copy config.yaml if it exists (optional - code has defaults via load_config)
# note: docker-compose mounts this as volume, but for standalone builds it should exist
COPY config.yaml* ./

# copy tests directory (for local dev/testing)
ARG INSTALL_DEV=true
COPY tests/ ./tests/

# create data directories, non-root user, and set permissions in one layer
RUN mkdir -p /app/data/cache /app/models && \
    useradd -m -u 1000 flareuser && \
    chown -R flareuser:flareuser /app && \
    find scripts -type f \( -name "*.py" -o -name "*.sh" \) -exec chmod +x {} \;

USER flareuser

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# default command
CMD ["python", "-m", "src"]
