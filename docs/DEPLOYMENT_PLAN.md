# Flare+ Deployment Plan

**Version:** 1.0
**Last Updated:** 2025-11-04
**Status:** Active

---

## Overview

This document outlines the operational deployment procedures for the Flare+ solar flare prediction system, including model retraining cadence, validation requirements, and rollback procedures.

## Model Retraining Schedule

### Automatic Retraining Triggers

Retraining should be initiated when:

1. **Time-based (Monthly)**
   - Schedule: First day of each month at 02:00 UTC
   - Minimum 6 months of historical data required
   - Command: `./flare train-survival --target-class C --save-model`

2. **Performance-based**
   - Brier score increases >10% from baseline
   - Precision drops below 0.5
   - Recall drops below 0.5
   - Detection: Monitor via `/metrics/performance` endpoint

3. **Data-based**
   - Major solar event occurred (X-class flare)
   - Significant change in solar cycle phase
   - New data sources integrated

### Manual Retraining

For ad-hoc retraining:

```bash
# Train survival model for C-class flares
./flare up
./flare ingest  # Ensure latest data
docker-compose exec app python scripts/train_and_predict_survival.py \
  --train \
  --target-class C \
  --detect-flares \
  --save-model /app/models/survival_model_c_class_$(date +%Y%m%d).joblib

# Validate new model
./flare validate-model /app/models/survival_model_c_class_$(date +%Y%m%d).joblib
```

---

## Model Registry

### Current State (Phase 1)

**Storage:** Local joblib files in `models/` directory

**Naming Convention:**
- Survival models: `survival_model_{class}_{date}.joblib`
  - Example: `survival_model_c_class_20251104.joblib`
- Classification models: `classification_pipeline_{date}.joblib`
  - Example: `classification_pipeline_20251104.joblib`

**Versioning Strategy:**
- Keep last 3 versions of each model type
- Archive older versions to `models/archive/`
- Delete versions older than 90 days from archive

### Future State (Phase 2)

**Tool:** MLflow Model Registry (see `docs/PHASE_2.md` item #3)

**Benefits:**
- Centralized model tracking
- Metadata storage (training metrics, hyperparameters)
- Model versioning and promotion workflow
- A/B testing capability
- Automated rollback

---

## Pre-Deployment Validation

Before deploying any new model, complete this checklist:

### 1. System Validation

```bash
./flare validate
```

**Requirements:**
- All 6 tests must pass:
  - [x] Database connection
  - [x] Data ingestion
  - [x] Model loading
  - [x] Predictions
  - [x] API endpoint
  - [x] Full pipeline

**If validation fails:**
- Check logs: `./flare logs`
- Review validation history: Query `flare_system_validation_log` table
- Fix issues before proceeding

### 2. Configuration Check

```bash
./flare check-config
```

**Requirements:**
- [x] .env file exists with required variables
- [x] config.yaml valid
- [x] Database connection works
- [x] Required directories exist
- [x] Sufficient disk space (>1GB free)

### 3. Model Validation

```bash
./flare validate-model /app/models/survival_model_c_class_new.joblib
```

**Requirements:**
- Model loads without errors
- Predictions contain no NaN values
- Probabilities sum to ~1.0
- C-index > 0.5 (for survival models)
- Comparison with previous version shows improvement

### 4. Backtesting

```bash
./flare backtest --model models/survival_model_c_class_new.joblib
```

**Requirements:**
- F1 score > 0.4 (moderate performance)
- Precision > 0.3 (acceptable false alarm rate)
- Brier score < 0.3 (reasonable calibration)

**Review output:**
- Check `data/backtest_report.txt`
- Compare metrics to previous model version
- Verify performance is stable or improving

### 5. Integration Test

```bash
# Start services
./flare up
./flare api-bg

# Test prediction endpoint
curl -X POST http://127.0.0.1:5001/predict/survival \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "2024-11-04T12:00:00"}'

# Verify response structure
```

**Requirements:**
- API responds within 10 seconds
- Response contains probability_distribution
- No error messages in logs

---

## Deployment Procedure

### Step-by-Step Deployment

1. **Backup Current Model**
   ```bash
   # Create backup directory
   mkdir -p models/archive/$(date +%Y%m)

   # Backup current production model
   cp models/survival_model_c_class.joblib \
      models/archive/$(date +%Y%m)/survival_model_c_class_$(date +%Y%m%d).joblib
   ```

2. **Deploy New Model**
   ```bash
   # Copy new model to production location
   cp models/survival_model_c_class_new.joblib \
      models/survival_model_c_class.joblib

   # Restart API to load new model
   ./flare api-stop
   ./flare api-bg
   ```

3. **Verify Deployment**
   ```bash
   # Check API health
   curl http://127.0.0.1:5001/health

   # Make test prediction
   curl -X POST http://127.0.0.1:5001/predict/survival \
     -H "Content-Type: application/json" \
     -d '{"timestamp": "2024-11-04T12:00:00"}'

   # Check logs for errors
   ./flare api-logs
   ```

4. **Monitor for 48 Hours**
   - Watch prediction logs: `SELECT * FROM flare_prediction_log ORDER BY prediction_timestamp DESC LIMIT 100`
   - Check for NaN values or anomalies
   - Monitor API response times
   - Review any error logs

5. **Update Documentation**
   ```bash
   # Log deployment in validation history
   docker-compose exec app python -c "
   from src.data.database import get_database
   from src.data.schema import SystemValidationLog
   from datetime import datetime

   db = get_database()
   with db.get_session() as session:
       log = SystemValidationLog(
           run_timestamp=datetime.utcnow(),
           validation_type='model_deployment',
           status='success',
           details={'model': 'survival_model_c_class', 'version': '$(date +%Y%m%d)'},
           initiated_by='manual'
       )
       session.add(log)
       session.commit()
   "
   ```

---

## Rollback Procedure

If new model shows poor performance or errors:

### Quick Rollback

```bash
# Stop API
./flare api-stop

# Restore previous model
cp models/archive/$(date +%Y%m)/survival_model_c_class_YYYYMMDD.joblib \
   models/survival_model_c_class.joblib

# Restart API
./flare api-bg

# Verify
curl http://127.0.0.1:5001/health
```

### Post-Rollback Actions

1. **Document the Issue**
   - Record why rollback was needed
   - Save error logs to `logs/rollback_$(date +%Y%m%d).log`
   - Add entry to validation history

2. **Investigate Root Cause**
   - Check training data quality
   - Review feature engineering changes
   - Analyze validation metrics
   - Check for data drift

3. **Fix and Retry**
   - Address root cause
   - Retrain model with fixes
   - Run full validation suite
   - Deploy with extra monitoring

---

## Model Performance Monitoring

### Daily Checks

**Manual (until automated monitoring):**

```sql
-- Check prediction volume
SELECT
    DATE(prediction_timestamp) as date,
    COUNT(*) as predictions,
    COUNT(DISTINCT model_version) as model_versions
FROM flare_prediction_log
WHERE prediction_timestamp > NOW() - INTERVAL '7 days'
GROUP BY DATE(prediction_timestamp)
ORDER BY date DESC;

-- Check for errors
SELECT * FROM flare_prediction_log
WHERE prediction_timestamp > NOW() - INTERVAL '24 hours'
  AND (predicted_class IS NULL OR class_probabilities IS NULL)
ORDER BY prediction_timestamp DESC;

-- Review recent flares and predictions
SELECT
    p.prediction_timestamp,
    p.predicted_class,
    p.actual_flare_class,
    p.class_probabilities
FROM flare_prediction_log p
WHERE p.prediction_timestamp > NOW() - INTERVAL '7 days'
  AND p.actual_flare_class IS NOT NULL
ORDER BY p.prediction_timestamp DESC;
```

### Weekly Performance Review

1. **Match Predictions to Actuals**
   ```bash
   docker-compose exec app python scripts/match_predictions_to_actuals.py
   ```
   (Future enhancement - see PHASE_2.md #9)

2. **Calculate Metrics**
   - Precision: Correct predictions / Total predictions
   - Recall: Correct predictions / Total actual flares
   - Brier score: Average squared error of probabilities

3. **Compare to Baseline**
   - Track metrics in spreadsheet or dashboard
   - Alert if metrics degrade >10%

### Monthly Model Review

1. Run backtesting on past month
2. Review prediction quality trends
3. Decide if retraining is needed
4. Document findings

---

## Configuration Management

### Environment Variables

Required in `.env`:

```bash
DB_HOST=postgres
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=your_secure_password
DB_NAME=flare_prediction

# Optional: Model serving
MODEL_PATH=/app/models/survival_model_c_class.joblib
API_PORT=5001
UI_PORT=7860
```

### Application Configuration

Edit `config.yaml` for:

- Data ingestion intervals
- Feature engineering parameters
- Model hyperparameters
- Validation thresholds

**After config changes:**
```bash
./flare check-config  # Verify config is valid
./flare down && ./flare up  # Restart services
```

---

## Incident Response

### API Service Down

**Symptoms:** `/health` endpoint unreachable or returning errors

**Actions:**
1. Check service status: `./flare logs`
2. Restart API: `./flare api-stop && ./flare api-bg`
3. If still failing, check database: `./flare db-shell`
4. Review recent deployments
5. Rollback model if recently deployed

### Predictions Failing

**Symptoms:** High error rate in prediction logs

**Actions:**
1. Check validation status: `./flare validate`
2. Review feature computation: Check for NaN values in features
3. Verify data freshness: `./flare ingest`
4. Check model file integrity: `./flare validate-model`
5. Restore backup if corrupted

### Database Issues

**Symptoms:** Connection errors, timeouts, data corruption

**Actions:**
1. Check database status: `docker-compose ps postgres`
2. Verify connectivity: `./flare db-shell`
3. Check disk space: `df -h`
4. Review logs: `docker-compose logs postgres`
5. Restore from backup if needed (see DISASTER_RECOVERY.md - future doc)

### Data Ingestion Failures

**Symptoms:** No new data, stale predictions

**Actions:**
1. Check NOAA API status manually: https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json
2. Review ingestion logs: Query `flare_ingestion_log` table
3. Retry ingestion: `./flare ingest`
4. Check cache expiry settings in config.yaml
5. Use cached data if NOAA unavailable

---

## Contact & Escalation

### Operational Owner
- **Name:** [Your Name]
- **Email:** [your.email@example.com]
- **On-call:** [Schedule if applicable]

### Escalation Path

**Level 1:** Self-service
- Check this deployment plan
- Review logs and validation output
- Attempt standard remediation

**Level 2:** Manual intervention
- Rollback model
- Restart services
- Check data sources

**Level 3:** Development team
- Code fixes required
- Infrastructure changes needed
- Contact: [dev-team@example.com]

---

## Future Enhancements

Items planned for Phase 2 (see `docs/PHASE_2.md`):

1. **MLflow Model Registry** (P0 #3)
   - Automated model versioning
   - Promotion workflow (staging → production)
   - Rollback with one command

2. **Automated Performance Monitoring** (P1 #9)
   - Daily performance metrics calculation
   - Automated alerting on degradation
   - Prediction-to-actual matching

3. **Automated Retraining** (P1 #17)
   - Scheduled monthly retraining
   - Performance-triggered retraining
   - Automatic validation and promotion

4. **Centralized Logging** (P0 #4)
   - ELK stack or CloudWatch
   - Request correlation IDs
   - Search and analysis tools

5. **Alerting System** (P0 #6)
   - Slack/PagerDuty integration
   - Automatic incident creation
   - Health check monitoring

---

**Document Owner:** Engineering Team
**Review Frequency:** Monthly
**Next Review:** 2025-12-01
