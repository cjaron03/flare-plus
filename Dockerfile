# multi-stage docker build for flare-plus
# optimized with buildkit cache mounts and uv for speed

# ===== BUILDER STAGE =====
FROM python:3.12-slim AS builder

WORKDIR /build

# update system packages and install uv for faster dependency resolution
# upgrade system pip to fix cve-2025-8869 (Trivy scans system pip, not just venv)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get upgrade -y -o Dpkg::Options::="--force-confold" && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --no-cache-dir --upgrade "pip>=25.3" && \
    python -m pip install --no-cache-dir uv

# create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# upgrade pip in venv to fix cve-2023-5752 and cve-2025-8869
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade "pip>=25.3"

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
FROM python:3.12-slim

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

# create non-root user first for proper ownership
RUN useradd -m -u 1000 flareuser

# install runtime libraries and apply security updates
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get upgrade -y -o Dpkg::Options::="--force-confold" && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# set environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    VIRTUAL_ENV="/opt/venv" \
    PYTHONUNBUFFERED="1" \
    PYTHONDONTWRITEBYTECODE="1"

# copy application code with proper ownership
COPY --chown=flareuser:flareuser src/ ./src/
COPY --chown=flareuser:flareuser scripts/ ./scripts/
COPY --chown=flareuser:flareuser pyproject.toml ./
# copy config.yaml if it exists (optional - code has defaults via load_config)
# note: docker-compose mounts this as volume, but for standalone builds it should exist
COPY --chown=flareuser:flareuser config.yaml* ./

# create data directories, fix venv ownership, and set permissions
RUN mkdir -p /app/data/cache /app/models && \
    chown -R flareuser:flareuser /app /opt/venv && \
    find scripts -type f \( -name "*.py" -o -name "*.sh" \) -exec chmod +x {} \;

USER flareuser

# default command (serve API)
CMD ["python", "scripts/run_api_server.py", "--host", "0.0.0.0", "--port", "5000", "--workers", "2"]
