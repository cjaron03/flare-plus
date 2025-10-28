# multi-stage docker build for flare-plus

# stage 1: builder
FROM python:3.10-slim as builder

WORKDIR /build

# install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# stage 2: runtime
FROM python:3.10-slim

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

# copy python packages from builder
COPY --from=builder /root/.local /root/.local

# copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config.yaml ./
COPY pyproject.toml ./

# make sure scripts are executable
RUN chmod +x scripts/*.py

# create non-root user
RUN useradd -m -u 1000 flareuser && \
    chown -R flareuser:flareuser /app

USER flareuser

# set python path
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# default command
CMD ["python", "-m", "src"]

