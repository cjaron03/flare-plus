# Deployment Guide

This guide covers local deployment, environment configuration, and model performance metrics for the Flare+ solar flare prediction system.

## Quick Start (Local Deployment)

### Prerequisites

- Docker and Docker Compose installed
- 2GB+ free disk space
- Ports 5001 (API) and 7860 (UI) available

### Step-by-Step Setup

1. **Clone and start services:**
   ```bash
   git clone <repository-url>
   cd flare-plus
   ./flare up
   ```

2. **Initialize database:**
   ```bash
   ./flare init-db
   ```

3. **Ingest initial data:**
   ```bash
   ./flare ingest
   ```

4. **Start API service:**
   ```bash
   ./flare api-bg
   ```
   API will be available at `http://127.0.0.1:5001`

5. **Start UI dashboard:**
   ```bash
   ./flare ui-bg
   ```
   Dashboard will be available at `http://127.0.0.1:7860`

6. **Validate system:**
   ```bash
   ./flare validate
   ```

## Environment Variables

### Required Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_password_here
DB_NAME=flare_prediction

# API Configuration
API_KEYS=your-api-key-here  # Required! Comma-separated API keys for authentication
API_HOST_PORT=5001          # Host port for API (default: 5001)
API_PORT=5000               # Container port for API (default: 5000)

# UI Configuration (optional)
UI_HOST_PORT=7860           # Host port for UI (default: 7860)
UI_PORT=7860                # Container port for UI (default: 7860)
UI_API_URL=http://127.0.0.1:5001  # API URL for frontend (default: auto-detected)

# Model Paths (optional)
CLASSIFICATION_MODEL_PATH=/app/models/classification_model.joblib
SURVIVAL_MODEL_PATH=/app/models/survival_model_m_class.joblib

# Admin UI (optional - change these for production!)
ADMIN_UI_LOGIN_ENABLED=true
ADMIN_UI_USERNAME=your_admin_username
ADMIN_UI_PASSWORD=your_secure_password
ADMIN_UI_MAX_ATTEMPTS=5
ADMIN_UI_ATTEMPT_WINDOW=300  # seconds
ADMIN_UI_LOCKOUT_SECONDS=600

# NASA DONKI API (optional, for historical flare import)
NASA_API_KEY=DEMO_KEY        # Get free key at https://api.nasa.gov/ (required for DONKI ingestion)

# Error Monitoring (optional)
SENTRY_DSN=                  # Sentry DSN for error tracking (get from https://sentry.io)
ENVIRONMENT=development      # Environment name (development, staging, production)
```

### Docker Compose Environment Variables

The following can be set in your shell environment before running `./flare` commands:

- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` - Database connection
- `API_HOST_PORT`, `API_PORT` - API service ports
- `UI_HOST_PORT`, `UI_PORT` - UI service ports
- `CLASSIFICATION_MODEL_PATH`, `SURVIVAL_MODEL_PATH` - Model artifact paths

### Example `.env` File

```bash
# Minimal configuration
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=changeme
DB_NAME=flare_prediction

# Custom ports
API_HOST_PORT=5001
UI_HOST_PORT=7860

# NASA API key for historical data import
NASA_API_KEY=your_nasa_api_key_here
```

## Model Performance Metrics

### M-Class Survival Model (Current Production Model)

**Performance Metrics:**
- **F1 Score:** 0.867
- **Precision:** 93%
- **Recall:** 81%
- **Brier Score:** 0.071 (lower is better)
- **C-Index:** ~0.75 (survival model concordance)

**Training Data:**
- Model trained on M-class flares from October 28 - November 10, 2025
- Training window: 13 days
- M-class flares in training set: 8 events
- Total flares in training window: 125 C-class, 8 M-class, 1 X-class

**Backtesting Results:**
- Test period: October 28 - November 10, 2025
- Total predictions: 14
- Actual flares: 9
- Correct predictions: 13
- Missed predictions: 1
- False negatives: 3

**Model Characteristics:**
- **Model Type:** Cox Proportional Hazards (survival analysis)
- **Target Class:** M-class flares only
- **Time Buckets:** 0-6h, 6-12h, 12-24h, 24-48h, 48-72h, 72-96h, 96-120h, 120-168h
- **Features:** Flux trends, solar region complexity, magnetic field measurements, recency-weighted flare counts

**Limitations:**
- C-class predictions are currently disabled (model focuses on M/X-class only)
- Training data is limited to recent window (Oct 28 - Nov 10, 2025)
- Model performance may degrade during different solar cycle phases
- Predictions are for research/advisory purposes only

### Model File Location

Production model is stored at:
```
/app/models/survival_model_m_class.joblib
```

Or in the host filesystem:
```
./models/survival_model_m_class.joblib
```

## Service Management

### Starting Services

```bash
# Start all core services (database + app container)
./flare up

# Start API in background
./flare api-bg

# Start UI in background
./flare ui-bg
```

### Stopping Services

```bash
# Stop API
./flare api-stop

# Stop UI
./flare ui-stop

# Stop all services
./flare down
```

### Viewing Logs

```bash
# All services
./flare logs

# API only
./flare api-logs

# UI only
./flare ui-logs
```

## Health Checks

### API Health Endpoint

```bash
curl http://127.0.0.1:5001/health
```

Response includes:
- Model availability status
- Database connection status
- Last ingestion timestamp
- Total predictions logged
- Disk space information
- Drift detection status

### System Validation

Run full system validation:

```bash
./flare validate
```

Validates:
- Database connection and table integrity
- Data ingestion from NOAA sources
- Model loading and reconstruction
- Prediction generation
- API endpoint availability
- Full pipeline integration

## Data Ingestion

### Manual Ingestion

```bash
# Run one-time ingestion
./flare ingest

# Trigger via API
./flare ingest-api
```

### Historical Data Import

Import historical flares from NASA DONKI:

```bash
# Import last 2 years (default)
./flare import-donki

# Import specific date range
./flare import-donki 2023-01-01 2024-12-31

# With custom API key (or set NASA_API_KEY env var)
./flare import-donki 2023-01-01 2024-12-31 your_nasa_api_key

# Or set environment variable (recommended)
export NASA_API_KEY=your_nasa_api_key_here
./flare import-donki 2023-01-01 2024-12-31
```

## Troubleshooting

### Database Connection Issues

```bash
# Check database is running
docker compose ps postgres

# Test connection
./flare db-shell

# Reset database (WARNING: deletes all data)
docker compose down -v
./flare up
./flare init-db
```

### Model Loading Issues

```bash
# Check model file exists
docker compose exec app ls -lh /app/models/

# Validate model
./flare validate-model /app/models/survival_model_m_class.joblib

# Check model path in config
docker compose exec app cat config.yaml | grep -A 5 model
```

### Port Conflicts

If ports 5001 or 7860 are already in use:

```bash
# Set custom ports
export API_HOST_PORT=5002
export UI_HOST_PORT=7861
./flare api-bg
./flare ui-bg
```

### API Not Responding

```bash
# Check API is running
./flare api-logs

# Restart API
./flare api-stop
./flare api-bg

# Check health endpoint
curl http://127.0.0.1:5001/health
```

## Production Deployment Checklist

Before deploying to production:

- [ ] All environment variables configured in `.env`
- [ ] Database credentials secured (not committed to git)
- [ ] Model files present in `models/` directory
- [ ] System validation passes (`./flare validate`)
- [ ] Health checks responding correctly
- [ ] Logs are being monitored
- [ ] Backup strategy in place for database
- [ ] Model performance metrics documented
- [ ] Rate limiting configured (if applicable)
- [ ] Monitoring/alerting set up

## Support

For issues or questions:
- Check logs: `./flare logs`
- Run validation: `./flare validate`
- Review documentation: `README.md`
- Check configuration: `./flare check-config`

