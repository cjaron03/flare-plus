# multi-stage docker build for flare-plus
# optimized with buildkit cache mounts and uv for speed

# ===== BUILDER STAGE =====
FROM python:3.9-slim AS builder

WORKDIR /build

# install build dependencies with cache mount (gcc/g++/make needed for xgboost cpu-only build)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    cmake \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# copy requirements
COPY requirements.txt requirements-dev.txt ./

# install dependencies with uv cache mount for speed
# build xgboost from source with cpu-only to avoid 500mb cuda wheels
ARG INSTALL_DEV=true
ENV CMAKE_ARGS="-DUSE_CUDA=OFF" \
    CUDA_VISIBLE_DEVICES="" \
    FORCE_CUDA=0
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_DEV" = "true" ]; then \
      # install numpy/scipy first (xgboost dependencies) \
      uv pip install --python-platform linux numpy==1.26.4 scipy==1.13.1; \
      # install xgboost from source without its declared dependencies \
      uv pip install --no-deps --no-binary xgboost --python-platform linux xgboost==2.1.4; \
      # install remaining requirements \
      uv pip install --python-platform linux -r requirements-dev.txt; \
    else \
      # install numpy/scipy first (xgboost dependencies) \
      uv pip install --python-platform linux numpy==1.26.4 scipy==1.13.1; \
      # install xgboost from source without its declared dependencies \
      uv pip install --no-deps --no-binary xgboost --python-platform linux xgboost==2.1.4; \
      # install remaining requirements \
      uv pip install --python-platform linux -r requirements.txt; \
    fi

# ===== RUNTIME STAGE =====
FROM python:3.9-slim

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

# install runtime dependencies with cache mount (libgomp1 needed for compiled xgboost)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    libstdc++6 \
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
COPY config.yaml pyproject.toml ./

# copy tests directory (for local dev/testing)
ARG INSTALL_DEV=true
COPY tests/ ./tests/

# create data directories, non-root user, and set permissions in one layer
RUN mkdir -p /app/data/cache /app/models && \
    useradd -m -u 1000 flareuser && \
    chown -R flareuser:flareuser /app && \
    chmod +x scripts/*.py

USER flareuser

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# default command
CMD ["python", "-m", "src"]
