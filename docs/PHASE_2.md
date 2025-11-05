# Phase 2: Operational Readiness Roadmap

**Document Version:** 1.0
**Date:** 2025-11-04
**Status:** Planning
**Target Completion:** Q2 2025

---

## Executive Summary

### Overall Assessment: ADVANCED PROTOTYPE - NOT READY FOR OPERATIONAL DEPLOYMENT

Flare-plus demonstrates sophisticated ML architecture and excellent engineering hygiene for a research project, with approximately 60% test coverage, comprehensive validation infrastructure, and thoughtful monitoring design. However, **critical gaps in operational resilience, security, scalability, and interpretability** prevent immediate operational deployment.

### Verdict

**NOT READY** for operational deployment without 3-6 months of hardening work focused on:
- Security (authentication, rate limiting, TLS)
- Reliability (backups, alerting, monitoring)
- Interpretability (model explainability, performance tracking)
- Scalability (async processing, horizontal scaling)

### Key Metrics

- **Codebase Size:** ~9,889 lines of Python source code
- **Test Coverage:** ~60-70% (1,497 lines of tests)
- **Architecture Quality:** Excellent (clean separation of concerns)
- **Operational Maturity:** Prototype-level (critical gaps in production patterns)

### Timeline to Production

| Scenario | Duration | Prerequisites |
|----------|----------|---------------|
| Optimistic (dedicated team) | 3 months | Full-time ML Eng, Backend Eng, DevOps Eng |
| Realistic (part-time team) | 6 months | Part-time dedicated resources |
| Conservative (limited resources) | 9-12 months | Ad-hoc availability |

### Resource Requirements

**Team Composition:**
- 1x ML Engineer (model improvements, explainability)
- 1x Backend Engineer (API hardening, async processing)
- 1x DevOps Engineer (infrastructure, monitoring, security)
- 0.5x QA Engineer (testing, validation)
- 0.25x Compliance Specialist (documentation, audit prep)

**Infrastructure Costs (monthly estimate):**
- Cloud hosting (AWS/GCP/Azure): $500-1000
- Monitoring tools (Datadog/New Relic): $200-500
- Log aggregation (ELK/CloudWatch): $100-300
- Model registry (MLflow): $50-100 (self-hosted)
- Backup storage (S3/GCS): $50-100
- **Total: $900-2000/month**

---

## Critical Blockers (P0) - 18 Engineering Days

These items **MUST** be resolved before any operational deployment. Deploying without addressing these creates unacceptable security, reliability, and audit risks.

### 1. API Authentication & Authorization (3 days)

**Current State:**
- All API endpoints completely open with no authentication
- Anyone can call `/predict/*`, `/ingest`, `/validate/system`

**Risk:**
- Data exfiltration (query prediction logs)
- Resource exhaustion (spam prediction requests)
- Sabotage (trigger expensive validation runs)

**Location:** `src/api/app.py` - all routes (lines 132-173, 187-228, 242-288, 302-357, 371-409)

**Current Code:**
```python
# src/api/app.py line 132 - NO authentication!
@app.route("/predict/classification", methods=["POST"])
def predict_classification():
    data = request.get_json()  # No API key check, no auth
    # ... prediction logic
```

**Implementation Plan:**

1. **Add API Key Authentication (Day 1)**
   ```python
   # src/api/auth.py (NEW FILE)
   from functools import wraps
   from flask import request, jsonify
   import hashlib
   import secrets

   # Store in database or environment variable
   VALID_API_KEYS = {
       hashlib.sha256(os.environ.get("API_KEY_1", "").encode()).hexdigest(),
       hashlib.sha256(os.environ.get("API_KEY_2", "").encode()).hexdigest(),
   }

   def require_api_key(f):
       @wraps(f)
       def decorated(*args, **kwargs):
           api_key = request.headers.get("X-API-Key")
           if not api_key:
               return jsonify({"error": "missing API key"}), 401

           key_hash = hashlib.sha256(api_key.encode()).hexdigest()
           if key_hash not in VALID_API_KEYS:
               return jsonify({"error": "invalid API key"}), 401

           return f(*args, **kwargs)
       return decorated

   def require_admin_key(f):
       @wraps(f)
       def decorated(*args, **kwargs):
           api_key = request.headers.get("X-Admin-Key")
           if not api_key:
               return jsonify({"error": "admin access required"}), 403

           admin_key_hash = hashlib.sha256(os.environ["ADMIN_API_KEY"].encode()).hexdigest()
           key_hash = hashlib.sha256(api_key.encode()).hexdigest()

           if key_hash != admin_key_hash:
               return jsonify({"error": "invalid admin key"}), 403

           return f(*args, **kwargs)
       return decorated
   ```

2. **Apply Decorators to Endpoints (Day 2)**
   ```python
   # src/api/app.py
   from src.api.auth import require_api_key, require_admin_key

   @app.route("/predict/classification", methods=["POST"])
   @require_api_key
   def predict_classification():
       ...

   @app.route("/validate/system", methods=["POST"])
   @require_admin_key  # Admin-only endpoint
   def validate_system():
       ...

   # Leave /health unauthenticated for monitoring
   @app.route("/health", methods=["GET"])
   def health_check():
       ...
   ```

3. **Add Key Management (Day 3)**
   - Create `scripts/generate_api_key.py` to generate secure keys
   - Document key rotation procedure in `docs/SECURITY.md`
   - Add API key usage tracking to database

**Testing:**
- Test all endpoints return 401 without valid key
- Test admin endpoints return 403 with regular key
- Test /health remains accessible

**Documentation:**
- Update README with authentication instructions
- Add key management section to SECURITY.md

### 2. Rate Limiting (1 day)

**Current State:**
- No protection against abuse or DoS attacks
- Single user can exhaust resources with rapid requests

**Risk:**
- Resource exhaustion (CPU, database connections, memory)
- Denial of service for legitimate users
- Cost explosion on cloud infrastructure

**Implementation Plan:**

1. **Install Flask-Limiter**
   ```bash
   # requirements.txt
   Flask-Limiter==3.5.0
   ```

2. **Configure Rate Limits**
   ```python
   # src/api/app.py
   from flask_limiter import Limiter
   from flask_limiter.util import get_remote_address

   limiter = Limiter(
       app=app,
       key_func=get_remote_address,
       default_limits=["100 per hour"],
       storage_uri="redis://localhost:6379",  # Or memory:// for development
   )

   @app.route("/predict/classification", methods=["POST"])
   @limiter.limit("10 per minute")  # Stricter limit for compute-heavy endpoints
   @require_api_key
   def predict_classification():
       ...

   @app.route("/predict/all", methods=["POST"])
   @limiter.limit("5 per minute")  # Very strict for expensive all-predictions
   @require_api_key
   def predict_all():
       ...

   @app.route("/health", methods=["GET"])
   @limiter.exempt  # No rate limit for health checks
   def health_check():
       ...
   ```

3. **Add Custom Rate Limit Headers**
   ```python
   @app.after_request
   def add_rate_limit_headers(response):
       # Flask-Limiter adds X-RateLimit-* headers automatically
       return response
   ```

4. **Configure Redis for Rate Limit Storage** (production)
   ```yaml
   # docker-compose.yml
   services:
     redis:
       image: redis:7-alpine
       ports:
         - "6379:6379"
       volumes:
         - redis_data:/data
       healthcheck:
         test: ["CMD", "redis-cli", "ping"]
         interval: 5s
         timeout: 3s
         retries: 5

   volumes:
     redis_data:
   ```

**Testing:**
- Test rate limit headers present in responses
- Test exceeding limit returns 429 Too Many Requests
- Test Retry-After header present on 429 responses

### 3. Model Versioning & Registry (5 days)

**Current State:**
- Models saved as `models/survival_model.joblib` (no version in filename)
- No metadata tracking (training date, hyperparameters, metrics)
- No rollback capability
- Overwrites existing model on save

**Risk:**
- Cannot audit which model version made which prediction
- Cannot rollback bad model deployments
- Cannot A/B test new models
- Loss of model lineage and reproducibility

**Location:** `scripts/train_and_predict_survival.py` line 340, `src/models/pipeline.py`

**Current Code:**
```python
# scripts/train_and_predict_survival.py line 340 - OVERWRITES!
save_path = args.save_model or PROJECT_ROOT / "models" / "survival_model.joblib"
joblib.dump(pipeline_data, save_path)  # No versioning, no metadata
```

**Implementation Plan:**

1. **Set Up MLflow (Day 1)**
   ```bash
   # requirements.txt
   mlflow==2.10.0
   ```

   ```python
   # src/models/registry.py (NEW FILE)
   import mlflow
   from mlflow.models.signature import infer_signature
   import joblib
   from datetime import datetime
   from pathlib import Path

   class ModelRegistry:
       def __init__(self, tracking_uri="sqlite:///mlflow.db"):
           mlflow.set_tracking_uri(tracking_uri)

       def save_model(
           self,
           model,
           model_type: str,  # "classification" or "survival"
           metrics: dict,
           params: dict,
           X_sample,  # For signature inference
           y_sample=None,
       ):
           with mlflow.start_run():
               # Log parameters
               mlflow.log_params(params)

               # Log metrics
               mlflow.log_metrics(metrics)

               # Log model with signature
               signature = infer_signature(X_sample, y_sample)
               mlflow.sklearn.log_model(
                   model,
                   artifact_path=model_type,
                   signature=signature,
                   registered_model_name=f"flare_{model_type}",
               )

               # Log additional metadata
               mlflow.set_tag("model_type", model_type)
               mlflow.set_tag("training_date", datetime.utcnow().isoformat())

               run_id = mlflow.active_run().info.run_id
               return run_id

       def load_model(self, model_name: str, version: str = "latest"):
           if version == "latest":
               model_uri = f"models:/{model_name}/Production"
           else:
               model_uri = f"models:/{model_name}/{version}"

           return mlflow.sklearn.load_model(model_uri)

       def promote_to_production(self, model_name: str, version: int):
           client = mlflow.tracking.MlflowClient()
           client.transition_model_version_stage(
               name=model_name,
               version=version,
               stage="Production",
           )
   ```

2. **Update Training Scripts (Day 2-3)**
   ```python
   # scripts/train_and_predict_survival.py
   from src.models.registry import ModelRegistry

   def train_model(args):
       # ... existing training code ...

       # After training
       registry = ModelRegistry()

       metrics = {
           "c_index": c_index,
           "brier_score": brier_score,
       }

       params = {
           "target_class": args.target_class,
           "max_time_hours": 168,
           "model_type": "cox",  # or "gb"
       }

       run_id = registry.save_model(
           model=pipeline,
           model_type="survival",
           metrics=metrics,
           params=params,
           X_sample=X_train[:10],
       )

       logger.info(f"Model saved with run_id: {run_id}")

       # Optionally promote to production
       if args.promote_to_production:
           registry.promote_to_production("flare_survival", version=1)
   ```

3. **Update Model Loading in API (Day 4)**
   ```python
   # src/api/service.py
   from src.models.registry import ModelRegistry

   class PredictionService:
       def __init__(self, model_version: str = "latest"):
           self.registry = ModelRegistry()
           self.classification_pipeline = self.registry.load_model(
               "flare_classification", version=model_version
           )
           self.survival_pipeline = self.registry.load_model(
               "flare_survival", version=model_version
           )
   ```

4. **Add Version Tracking to Predictions (Day 5)**
   ```python
   # src/data/schema.py - Update PredictionLog
   class PredictionLog(Base):
       # ... existing columns ...
       model_version = Column(String(50), nullable=True)  # MLflow run_id
       model_stage = Column(String(20), nullable=True)    # "Production", "Staging"

   # src/api/monitoring.py - Update logging
   def log_prediction(self, prediction, prediction_type, timestamp, model_version=None):
       # ... existing code ...
       pred_log = PredictionLog(
           # ... existing fields ...
           model_version=model_version,
       )
   ```

**Testing:**
- Test model save creates MLflow run
- Test model load from registry works
- Test version tracking in predictions
- Test rollback to previous version

**Documentation:**
- Add model versioning guide to docs/
- Document promotion workflow

### 4. Centralized Logging (3 days)

**Current State:**
- Logs only in Docker container stdout
- No log aggregation or search capability
- Cannot trace requests across components
- No request correlation IDs

**Risk:**
- Cannot debug production issues
- No audit trail for security events
- Cannot track request flow through system

**Implementation Plan:**

1. **Set Up Logging Infrastructure (Day 1)**

   **Option A: ELK Stack (Self-hosted)**
   ```yaml
   # docker-compose.yml
   services:
     elasticsearch:
       image: elasticsearch:8.11.0
       environment:
         - discovery.type=single-node
         - xpack.security.enabled=false
       ports:
         - "9200:9200"
       volumes:
         - elasticsearch_data:/usr/share/elasticsearch/data

     logstash:
       image: logstash:8.11.0
       volumes:
         - ./logstash/pipeline:/usr/share/logstash/pipeline
       ports:
         - "5000:5000"
       depends_on:
         - elasticsearch

     kibana:
       image: kibana:8.11.0
       ports:
         - "5601:5601"
       depends_on:
         - elasticsearch
   ```

   **Option B: CloudWatch (AWS)**
   ```python
   # requirements.txt
   watchtower==3.0.1
   ```

2. **Add Structured Logging (Day 2)**
   ```python
   # src/logging_config.py (NEW FILE)
   import logging
   import sys
   from pythonjsonlogger import jsonlogger
   import watchtower

   def setup_logging(service_name: str, log_level: str = "INFO"):
       logger = logging.getLogger()
       logger.setLevel(log_level)

       # JSON formatter for structured logging
       formatter = jsonlogger.JsonFormatter(
           "%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s"
       )

       # Console handler
       console_handler = logging.StreamHandler(sys.stdout)
       console_handler.setFormatter(formatter)
       logger.addHandler(console_handler)

       # CloudWatch handler (if AWS)
       if os.environ.get("USE_CLOUDWATCH"):
           cloudwatch_handler = watchtower.CloudWatchLogHandler(
               log_group=f"/flare-plus/{service_name}",
               stream_name="{strftime}-%Y-%m-%d",
           )
           cloudwatch_handler.setFormatter(formatter)
           logger.addHandler(cloudwatch_handler)

       return logger
   ```

3. **Add Request Correlation IDs (Day 3)**
   ```python
   # src/api/middleware.py (NEW FILE)
   import uuid
   from flask import request, g

   def add_request_id():
       """Middleware to add request ID to all logs"""
       request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
       g.request_id = request_id
       return None

   # src/api/app.py
   from src.api.middleware import add_request_id

   app.before_request(add_request_id)

   @app.after_request
   def add_request_id_header(response):
       if hasattr(g, "request_id"):
           response.headers["X-Request-ID"] = g.request_id
       return response

   # Update all log calls to include request_id
   # Example:
   logger.info(
       "prediction request",
       extra={"request_id": g.get("request_id", "unknown")}
   )
   ```

**Testing:**
- Test logs appear in centralized system
- Test request IDs propagate through logs
- Test log search and filtering

### 5. Automated Backups (2 days)

**Current State:**
- PostgreSQL data in Docker volume with no backups
- Model files in Docker volume with no backups
- No recovery procedures documented

**Risk:**
- Complete data loss if volume corrupted
- No disaster recovery capability
- Cannot restore to point in time

**Implementation Plan:**

1. **Database Backups (Day 1)**
   ```bash
   # scripts/backup_database.sh (NEW FILE)
   #!/bin/bash
   set -e

   BACKUP_DIR="/backups/postgres"
   TIMESTAMP=$(date +%Y%m%d_%H%M%S)
   BACKUP_FILE="$BACKUP_DIR/flare_db_$TIMESTAMP.sql.gz"

   # Create backup directory
   mkdir -p $BACKUP_DIR

   # Run pg_dump
   docker-compose exec -T postgres pg_dump -U $DB_USER $DB_NAME | gzip > $BACKUP_FILE

   # Upload to S3 (if configured)
   if [ -n "$S3_BACKUP_BUCKET" ]; then
       aws s3 cp $BACKUP_FILE s3://$S3_BACKUP_BUCKET/postgres/
   fi

   # Keep only last 7 days locally
   find $BACKUP_DIR -name "flare_db_*.sql.gz" -mtime +7 -delete

   echo "Backup completed: $BACKUP_FILE"
   ```

2. **Model Backups (Day 1)**
   ```bash
   # scripts/backup_models.sh (NEW FILE)
   #!/bin/bash
   set -e

   BACKUP_DIR="/backups/models"
   TIMESTAMP=$(date +%Y%m%d_%H%M%S)
   BACKUP_FILE="$BACKUP_DIR/models_$TIMESTAMP.tar.gz"

   mkdir -p $BACKUP_DIR

   # Backup models directory
   tar -czf $BACKUP_FILE -C /app models/

   # Upload to S3
   if [ -n "$S3_BACKUP_BUCKET" ]; then
       aws s3 cp $BACKUP_FILE s3://$S3_BACKUP_BUCKET/models/
   fi

   # Keep only last 30 days locally
   find $BACKUP_DIR -name "models_*.tar.gz" -mtime +30 -delete

   echo "Model backup completed: $BACKUP_FILE"
   ```

3. **Automated Scheduling (Day 2)**
   ```yaml
   # docker-compose.yml - Add backup service
   services:
     backup:
       build: .
       volumes:
         - postgres_data:/var/lib/postgresql/data:ro
         - app_models:/app/models:ro
         - ./backups:/backups
       environment:
         - S3_BACKUP_BUCKET=${S3_BACKUP_BUCKET}
       command: |
         sh -c "
         while true; do
           /app/scripts/backup_database.sh
           /app/scripts/backup_models.sh
           sleep 86400  # Run daily
         done
         "
       depends_on:
         - postgres
   ```

4. **Restore Procedures**
   ```bash
   # scripts/restore_database.sh (NEW FILE)
   #!/bin/bash
   set -e

   BACKUP_FILE=$1

   if [ -z "$BACKUP_FILE" ]; then
       echo "Usage: $0 <backup_file.sql.gz>"
       exit 1
   fi

   echo "Restoring database from $BACKUP_FILE..."

   # Download from S3 if needed
   if [[ $BACKUP_FILE == s3://* ]]; then
       aws s3 cp $BACKUP_FILE /tmp/restore.sql.gz
       BACKUP_FILE="/tmp/restore.sql.gz"
   fi

   # Restore
   gunzip -c $BACKUP_FILE | docker-compose exec -T postgres psql -U $DB_USER $DB_NAME

   echo "Restore completed"
   ```

**Testing:**
- Test backup scripts run successfully
- Test restore from backup
- Test S3 upload (if configured)

**Documentation:**
- Add backup/restore procedures to docs/DISASTER_RECOVERY.md

### 6. Alerting System (2 days)

**Current State:**
- No notifications on system failures
- Validation failures, API errors, data ingestion issues go unnoticed

**Risk:**
- Extended downtime without awareness
- Silent failures (predictions not updating)
- Data quality issues undetected

**Implementation Plan:**

1. **Set Up Alert Infrastructure (Day 1)**

   **Option A: Slack Webhooks**
   ```python
   # src/alerting/slack.py (NEW FILE)
   import requests
   import os

   class SlackAlerter:
       def __init__(self, webhook_url: str = None):
           self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

       def send_alert(
           self,
           message: str,
           severity: str = "warning",  # "critical", "warning", "info"
           details: dict = None,
       ):
           color = {
               "critical": "#FF0000",
               "warning": "#FFA500",
               "info": "#0000FF",
           }[severity]

           payload = {
               "attachments": [
                   {
                       "color": color,
                       "title": f"[{severity.upper()}] Flare-plus Alert",
                       "text": message,
                       "fields": [
                           {"title": k, "value": str(v), "short": True}
                           for k, v in (details or {}).items()
                       ],
                       "ts": int(time.time()),
                   }
               ]
           }

           response = requests.post(self.webhook_url, json=payload)
           response.raise_for_status()
   ```

   **Option B: PagerDuty**
   ```python
   # src/alerting/pagerduty.py (NEW FILE)
   from pdpyras import EventsAPISession

   class PagerDutyAlerter:
       def __init__(self, routing_key: str = None):
           self.session = EventsAPISession(
               routing_key or os.environ["PAGERDUTY_ROUTING_KEY"]
           )

       def trigger_incident(self, summary: str, severity: str, details: dict):
           self.session.trigger(
               summary=summary,
               severity=severity,
               source="flare-plus",
               custom_details=details,
           )
   ```

2. **Integrate Alerting (Day 2)**
   ```python
   # src/api/app.py - Add error handler
   from src.alerting.slack import SlackAlerter

   alerter = SlackAlerter()

   @app.errorhandler(500)
   def internal_error(error):
       logger.error(f"Internal server error: {error}", exc_info=True)

       alerter.send_alert(
           message="API internal server error",
           severity="critical",
           details={
               "error": str(error),
               "endpoint": request.path,
               "method": request.method,
           },
       )

       return jsonify({"error": "internal server error"}), 500

   # src/data/ingestion.py - Alert on ingestion failures
   def run_incremental_update(self):
       try:
           # ... existing code ...
       except Exception as e:
           alerter.send_alert(
               message="Data ingestion failed",
               severity="warning",
               details={"source": source_name, "error": str(e)},
           )
           raise

   # scripts/validate_system.py - Alert on validation failures
   if not all_passed:
       alerter.send_alert(
           message="System validation failed",
           severity="critical",
           details={"failed_tests": failed_test_names},
       )
   ```

3. **Add Health Check Monitoring**
   ```python
   # scripts/monitor_health.py (NEW FILE)
   import requests
   import time
   from src.alerting.slack import SlackAlerter

   def check_health():
       alerter = SlackAlerter()
       failures = 0

       while True:
           try:
               response = requests.get("http://127.0.0.1:5001/health", timeout=10)

               if response.status_code != 200:
                   failures += 1
                   if failures >= 3:
                       alerter.send_alert(
                           message="API health check failing",
                           severity="critical",
                           details={"status_code": response.status_code},
                       )
               else:
                   failures = 0

           except requests.RequestException as e:
               failures += 1
               if failures >= 3:
                   alerter.send_alert(
                       message="API is unreachable",
                       severity="critical",
                       details={"error": str(e)},
                   )

           time.sleep(60)  # Check every minute

   if __name__ == "__main__":
       check_health()
   ```

**Testing:**
- Test alerts trigger on errors
- Test alert formatting in Slack/PagerDuty
- Test health check monitoring

### 7. HTTPS/TLS Security (2 days)

**Current State:**
- API served over HTTP only
- Data transmitted in plaintext
- No SSL/TLS certificates

**Risk:**
- API keys intercepted
- Prediction data visible to network sniffers
- Man-in-the-middle attacks

**Implementation Plan:**

1. **Set Up Nginx Reverse Proxy (Day 1)**
   ```nginx
   # nginx/nginx.conf (NEW FILE)
   upstream flare_api {
       server app:5001;
   }

   upstream flare_ui {
       server app:7860;
   }

   server {
       listen 80;
       server_name flare-plus.example.com;

       # Redirect HTTP to HTTPS
       return 301 https://$server_name$request_uri;
   }

   server {
       listen 443 ssl http2;
       server_name flare-plus.example.com;

       # SSL certificates (use Let's Encrypt)
       ssl_certificate /etc/nginx/ssl/cert.pem;
       ssl_certificate_key /etc/nginx/ssl/key.pem;

       # Modern SSL configuration
       ssl_protocols TLSv1.2 TLSv1.3;
       ssl_ciphers HIGH:!aNULL:!MD5;
       ssl_prefer_server_ciphers on;

       # Security headers
       add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
       add_header X-Frame-Options "DENY" always;
       add_header X-Content-Type-Options "nosniff" always;
       add_header X-XSS-Protection "1; mode=block" always;

       # API endpoints
       location /api/ {
           proxy_pass http://flare_api/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;

           # Timeouts for long-running predictions
           proxy_read_timeout 30s;
           proxy_connect_timeout 10s;
       }

       # UI
       location / {
           proxy_pass http://flare_ui/;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

2. **Update Docker Compose (Day 2)**
   ```yaml
   # docker-compose.yml
   services:
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
         - ./nginx/ssl:/etc/nginx/ssl:ro
       depends_on:
         - app

     app:
       # ... existing config ...
       # Remove port mapping (nginx handles it)
       expose:
         - "5001"
         - "7860"
   ```

3. **Set Up Let's Encrypt (Production)**
   ```bash
   # scripts/setup_ssl.sh (NEW FILE)
   #!/bin/bash

   # Install certbot
   apt-get update && apt-get install -y certbot python3-certbot-nginx

   # Get certificate
   certbot --nginx -d flare-plus.example.com --non-interactive --agree-tos -m admin@example.com

   # Auto-renewal
   echo "0 0,12 * * * certbot renew --quiet" | crontab -
   ```

**Testing:**
- Test HTTP redirects to HTTPS
- Test SSL certificate valid
- Test security headers present

---

## High Priority (P1) - 44 Engineering Days

These items are essential for operational stability and should be completed before full production deployment.

### 8. Model Explainability - SHAP/LIME (5 days)

**Current State:**
- Zero interpretability features
- Cannot explain why predictions were made
- No feature importance insights

**Impact:**
- Operators cannot trust or understand model decisions
- Cannot debug incorrect predictions
- Regulatory/compliance risk (lack of transparency)

**Implementation Plan:**

1. **Add SHAP Library (Day 1)**
   ```bash
   # requirements.txt
   shap==0.43.0
   ```

2. **Implement Explainer for Survival Models (Day 2-3)**
   ```python
   # src/models/explainability.py (NEW FILE)
   import shap
   import numpy as np
   import pandas as pd
   from typing import Dict, List, Tuple

   class ModelExplainer:
       def __init__(self, model, model_type: str):
           self.model = model
           self.model_type = model_type
           self.explainer = None

       def fit_explainer(self, X_background: pd.DataFrame):
           """Fit SHAP explainer on background data"""
           if self.model_type == "survival":
               # For Cox PH model
               self.explainer = shap.LinearExplainer(
                   self.model.cox_model.model,
                   X_background,
               )
           elif self.model_type == "classification":
               # For tree-based models
               self.explainer = shap.TreeExplainer(self.model)

       def explain_prediction(
           self,
           features: pd.DataFrame,
           top_k: int = 5,
       ) -> Dict[str, any]:
           """Generate explanation for a single prediction"""
           shap_values = self.explainer.shap_values(features)

           if isinstance(shap_values, list):
               shap_values = shap_values[1]  # Positive class for binary

           # Get top K features
           feature_importance = pd.DataFrame({
               "feature": features.columns,
               "importance": np.abs(shap_values[0]),
               "value": features.iloc[0].values,
               "shap_value": shap_values[0],
           })
           feature_importance = feature_importance.sort_values(
               "importance", ascending=False
           ).head(top_k)

           return {
               "base_value": self.explainer.expected_value,
               "prediction_value": self.explainer.expected_value + shap_values[0].sum(),
               "top_features": feature_importance.to_dict(orient="records"),
           }
   ```

3. **Integrate into API (Day 4)**
   ```python
   # src/api/service.py
   from src.models.explainability import ModelExplainer

   class PredictionService:
       def __init__(self, enable_explanations: bool = True):
           # ... existing init ...

           if enable_explanations:
               # Fit explainers on background data
               X_background = self._get_background_data()
               self.survival_explainer = ModelExplainer(
                   self.survival_pipeline, "survival"
               )
               self.survival_explainer.fit_explainer(X_background)

       def predict_survival(self, timestamp, explain: bool = True):
           # ... existing prediction code ...

           if explain and self.survival_explainer:
               explanation = self.survival_explainer.explain_prediction(features_df)
               result["explanation"] = explanation

           return result
   ```

4. **Add LIME for Local Explanations (Day 5)**
   ```python
   # src/models/explainability.py
   from lime import lime_tabular

   class ModelExplainer:
       def explain_with_lime(
           self,
           features: pd.DataFrame,
           num_features: int = 5,
       ) -> Dict:
           """Generate LIME explanation"""
           lime_explainer = lime_tabular.LimeTabularExplainer(
               training_data=self.X_background.values,
               feature_names=self.X_background.columns.tolist(),
               mode='regression' if self.model_type == 'survival' else 'classification',
           )

           explanation = lime_explainer.explain_instance(
               features.iloc[0].values,
               self.model.predict,
               num_features=num_features,
           )

           return {
               "lime_explanation": explanation.as_list(),
               "lime_score": explanation.score,
           }
   ```

**Testing:**
- Test SHAP values sum to prediction
- Test top features make physical sense
- Test explanation endpoint performance (<2s)

**Documentation:**
- Add interpretation guide for operators
- Document feature importance thresholds

### 9. Online Performance Monitoring (5 days)

**Current State:**
- Models deployed with no ongoing validation
- No tracking of prediction quality over time
- No automated retraining triggers

**Implementation Plan:**

1. **Create Performance Tracking System (Day 1-2)**
   ```python
   # src/models/performance_tracker.py (NEW FILE)
   from datetime import datetime, timedelta
   from typing import Dict, List
   import pandas as pd
   from sqlalchemy import func

   class PerformanceTracker:
       def __init__(self, db):
           self.db = db

       def calculate_metrics(
           self,
           start_date: datetime,
           end_date: datetime,
           prediction_type: str = "classification",
       ) -> Dict:
           """Calculate performance metrics for a time period"""
           with self.db.get_session() as session:
               predictions = (
                   session.query(PredictionLog)
                   .filter(
                       PredictionLog.prediction_type == prediction_type,
                       PredictionLog.prediction_timestamp >= start_date,
                       PredictionLog.prediction_timestamp <= end_date,
                       PredictionLog.actual_flare_class.isnot(None),
                   )
                   .all()
               )

               if not predictions:
                   return {"status": "insufficient_data"}

               # Calculate Brier score
               brier_scores = []
               for pred in predictions:
                   true_class = pred.actual_flare_class
                   probs = json.loads(pred.class_probabilities)

                   # Brier score for predicted class
                   predicted_prob = probs.get(true_class, 0)
                   brier_score = (1 - predicted_prob) ** 2
                   brier_scores.append(brier_score)

               avg_brier = np.mean(brier_scores)

               # Calculate accuracy
               correct = sum(
                   1 for p in predictions
                   if p.predicted_class == p.actual_flare_class
               )
               accuracy = correct / len(predictions)

               return {
                   "period_start": start_date.isoformat(),
                   "period_end": end_date.isoformat(),
                   "n_predictions": len(predictions),
                   "accuracy": accuracy,
                   "brier_score": avg_brier,
               }

       def check_degradation(self, baseline_metrics: Dict) -> bool:
           """Check if recent performance has degraded significantly"""
           recent_metrics = self.calculate_metrics(
               start_date=datetime.utcnow() - timedelta(days=7),
               end_date=datetime.utcnow(),
           )

           if recent_metrics.get("status") == "insufficient_data":
               return False

           # Alert if Brier score increased >10%
           brier_degradation = (
               recent_metrics["brier_score"] - baseline_metrics["brier_score"]
           ) / baseline_metrics["brier_score"]

           if brier_degradation > 0.10:
               return True

           # Alert if accuracy dropped >5%
           accuracy_drop = baseline_metrics["accuracy"] - recent_metrics["accuracy"]
           if accuracy_drop > 0.05:
               return True

           return False
   ```

2. **Automated Outcome Matching (Day 3)**
   ```python
   # scripts/match_predictions_to_actuals.py (NEW FILE)
   from datetime import datetime, timedelta
   from src.data.database import get_database
   from src.data.schema import PredictionLog, FlareEvent
   from sqlalchemy import and_

   def match_predictions():
       """Match predictions to actual flares (run daily with 48h delay)"""
       db = get_database()
       cutoff = datetime.utcnow() - timedelta(hours=48)

       with db.get_session() as session:
           # Get unmatched predictions older than 48h
           predictions = (
               session.query(PredictionLog)
               .filter(
                   PredictionLog.prediction_timestamp < cutoff,
                   PredictionLog.actual_flare_class.is_(None),
               )
               .all()
           )

           for pred in predictions:
               # Find flares in prediction window
               window_start = pred.observation_timestamp
               window_end = window_start + timedelta(hours=pred.window_hours or 48)

               actual_flares = (
                   session.query(FlareEvent)
                   .filter(
                       FlareEvent.peak_time >= window_start,
                       FlareEvent.peak_time <= window_end,
                   )
                   .order_by(FlareEvent.class_magnitude.desc())
                   .all()
               )

               if actual_flares:
                   # Record highest-magnitude flare
                   pred.actual_flare_class = actual_flares[0].flare_class
                   pred.actual_flare_time = actual_flares[0].peak_time
               else:
                   # No flare occurred
                   pred.actual_flare_class = "None"

               pred.outcome_recorded_at = datetime.utcnow()
               session.commit()

   if __name__ == "__main__":
       match_predictions()
   ```

3. **Performance Monitoring Dashboard Endpoint (Day 4)**
   ```python
   # src/api/app.py
   from src.models.performance_tracker import PerformanceTracker

   @app.route("/metrics/performance", methods=["GET"])
   @require_api_key
   def get_performance_metrics():
       """Get model performance metrics over time"""
       days = int(request.args.get("days", 30))
       prediction_type = request.args.get("type", "classification")

       tracker = PerformanceTracker(get_database())

       metrics = tracker.calculate_metrics(
           start_date=datetime.utcnow() - timedelta(days=days),
           end_date=datetime.utcnow(),
           prediction_type=prediction_type,
       )

       return jsonify(metrics)
   ```

4. **Scheduled Performance Checks (Day 5)**
   ```python
   # scripts/monitor_performance.py (NEW FILE)
   import time
   from src.models.performance_tracker import PerformanceTracker
   from src.alerting.slack import SlackAlerter
   from src.data.database import get_database

   def monitor_performance():
       tracker = PerformanceTracker(get_database())
       alerter = SlackAlerter()

       # Baseline from past 30 days
       baseline = tracker.calculate_metrics(
           start_date=datetime.utcnow() - timedelta(days=30),
           end_date=datetime.utcnow() - timedelta(days=7),
       )

       while True:
           if tracker.check_degradation(baseline):
               alerter.send_alert(
                   message="Model performance degradation detected",
                   severity="warning",
                   details=baseline,
               )

           time.sleep(86400)  # Check daily

   if __name__ == "__main__":
       monitor_performance()
   ```

**Testing:**
- Test outcome matching works correctly
- Test performance calculation accuracy
- Test degradation detection triggers

### 10. Data Retention Policy (3 days)

**Current State:**
- Database grows indefinitely
- No archival or deletion procedures
- Compliance risk (GDPR requires data minimization)

**Implementation Plan:**

1. **Define Retention Periods (Day 1)**
   ```python
   # src/config.py
   class DataRetentionConfig:
       # Retention periods in days
       PREDICTION_LOGS = 730  # 2 years
       INGESTION_LOGS = 365   # 1 year
       VALIDATION_LOGS = 180  # 6 months
       RAW_FLUX_DATA = 730    # 2 years
       FLARE_EVENTS = 3650    # 10 years (permanent record)
   ```

2. **Create Archival Script (Day 2)**
   ```python
   # scripts/archive_old_data.py (NEW FILE)
   from datetime import datetime, timedelta
   import pandas as pd
   from src.data.database import get_database
   from src.data.schema import PredictionLog, DataIngestionLog, GOESXRayFlux
   from src.config import DataRetentionConfig

   def archive_prediction_logs():
       """Archive prediction logs older than retention period"""
       db = get_database()
       cutoff = datetime.utcnow() - timedelta(days=DataRetentionConfig.PREDICTION_LOGS)

       with db.get_session() as session:
           # Export to parquet before deletion
           old_predictions = (
               session.query(PredictionLog)
               .filter(PredictionLog.prediction_timestamp < cutoff)
               .all()
           )

           if old_predictions:
               # Convert to DataFrame and save
               df = pd.DataFrame([{
                   "id": p.id,
                   "prediction_timestamp": p.prediction_timestamp,
                   "predicted_class": p.predicted_class,
                   "actual_flare_class": p.actual_flare_class,
                   # ... other fields ...
               } for p in old_predictions])

               archive_path = f"data/archive/predictions_{cutoff.strftime('%Y%m%d')}.parquet"
               df.to_parquet(archive_path)

               # Delete from database
               session.query(PredictionLog).filter(
                   PredictionLog.prediction_timestamp < cutoff
               ).delete()
               session.commit()

               logger.info(f"Archived {len(old_predictions)} prediction logs to {archive_path}")

   def cleanup_ingestion_logs():
       """Delete ingestion logs older than retention period"""
       db = get_database()
       cutoff = datetime.utcnow() - timedelta(days=DataRetentionConfig.INGESTION_LOGS)

       with db.get_session() as session:
           deleted = session.query(DataIngestionLog).filter(
               DataIngestionLog.run_timestamp < cutoff
           ).delete()
           session.commit()

           logger.info(f"Deleted {deleted} old ingestion logs")

   def archive_flux_data():
       """Archive old flux data to cold storage"""
       db = get_database()
       cutoff = datetime.utcnow() - timedelta(days=DataRetentionConfig.RAW_FLUX_DATA)

       # Similar to prediction logs, export to parquet then delete
       # ...
   ```

3. **Schedule Archival (Day 3)**
   ```bash
   # Add to crontab
   # Run archival weekly
   0 2 * * 0 cd /app && python scripts/archive_old_data.py
   ```

**Testing:**
- Test archival exports data correctly
- Test archived data can be restored
- Test deletion doesn't affect recent data

**Documentation:**
- Add retention policy to docs/DATA_RETENTION.md
- Document restoration procedures

### 11. Feature Schema Validation (3 days)

**Current State:**
- No validation that features match model expectations
- Risk of silent failures if feature engineering changes

**Implementation Plan:**

1. **Create Feature Schema Registry (Day 1)**
   ```python
   # src/features/schema.py (NEW FILE)
   from dataclasses import dataclass
   from typing import List, Dict
   import json
   from pathlib import Path

   @dataclass
   class FeatureSchema:
       version: str
       features: List[str]
       dtypes: Dict[str, str]
       valid_ranges: Dict[str, tuple]  # (min, max) for each feature

       def validate(self, df) -> tuple[bool, List[str]]:
           """Validate DataFrame against schema"""
           errors = []

           # Check all required features present
           missing = set(self.features) - set(df.columns)
           if missing:
               errors.append(f"Missing features: {missing}")

           # Check dtypes
           for col, expected_dtype in self.dtypes.items():
               if col in df.columns:
                   actual_dtype = str(df[col].dtype)
                   if actual_dtype != expected_dtype:
                       errors.append(f"Feature {col} has dtype {actual_dtype}, expected {expected_dtype}")

           # Check ranges
           for col, (min_val, max_val) in self.valid_ranges.items():
               if col in df.columns:
                   if df[col].min() < min_val or df[col].max() > max_val:
                       errors.append(f"Feature {col} outside valid range [{min_val}, {max_val}]")

           return len(errors) == 0, errors

       def save(self, path: Path):
           """Save schema to JSON"""
           with open(path, 'w') as f:
               json.dump({
                   'version': self.version,
                   'features': self.features,
                   'dtypes': self.dtypes,
                   'valid_ranges': self.valid_ranges,
               }, f, indent=2)

       @classmethod
       def load(cls, path: Path):
           """Load schema from JSON"""
           with open(path) as f:
               data = json.load(f)
           return cls(**data)
   ```

2. **Generate Schemas from Training (Day 2)**
   ```python
   # scripts/generate_feature_schema.py (NEW FILE)
   from src.features.schema import FeatureSchema
   from src.features.pipeline import FeatureEngineer
   import numpy as np

   def generate_schema_from_training_data(X_train, version: str):
       """Generate feature schema from training data"""

       # Get feature names and dtypes
       features = X_train.columns.tolist()
       dtypes = {col: str(X_train[col].dtype) for col in features}

       # Calculate valid ranges (5th to 95th percentile)
       valid_ranges = {}
       for col in features:
           if X_train[col].dtype in [np.float64, np.int64]:
               min_val = X_train[col].quantile(0.05)
               max_val = X_train[col].quantile(0.95)
               valid_ranges[col] = (min_val, max_val)

       schema = FeatureSchema(
           version=version,
           features=features,
           dtypes=dtypes,
           valid_ranges=valid_ranges,
       )

       # Save schema
       schema_path = PROJECT_ROOT / "models" / f"feature_schema_{version}.json"
       schema.save(schema_path)

       logger.info(f"Generated feature schema version {version}")
       return schema
   ```

3. **Integrate Validation (Day 3)**
   ```python
   # src/api/service.py
   from src.features.schema import FeatureSchema

   class PredictionService:
       def __init__(self, feature_schema_path: str = None):
           # Load feature schema
           schema_path = feature_schema_path or PROJECT_ROOT / "models" / "feature_schema_v1.json"
           self.feature_schema = FeatureSchema.load(schema_path)

       def predict_classification(self, timestamp):
           # Compute features
           features_df = self.feature_engineer.compute_features(timestamp)

           # Validate features
           is_valid, errors = self.feature_schema.validate(features_df)
           if not is_valid:
               logger.error(f"Feature validation failed: {errors}")
               return {
                   "error": "feature validation failed",
                   "details": errors,
               }

           # Proceed with prediction
           return self.classification_pipeline.predict(timestamp)
   ```

**Testing:**
- Test schema generation from training data
- Test validation catches missing features
- Test validation catches out-of-range values

### 12-17. Additional P1 Items

Due to length constraints, additional P1 items are summarized:

**12. Async Task Queue (Celery) - 7 days**
- Move long-running predictions to background tasks
- Implement task status tracking
- Add result caching with Redis

**13. Integration & Load Tests - 5 days**
- E2E tests for ingestion → prediction flow
- Load testing with Locust (100 concurrent users)
- Performance benchmarking

**14. Circuit Breakers - 2 days**
- Implement circuit breaker for NOAA API
- Add fallback to cached data
- Configure failure thresholds

**15. OpenAPI Documentation - 2 days**
- Generate Swagger spec from Flask routes
- Add interactive API docs UI
- Include example requests/responses

**16. Horizontal Scalability - 7 days**
- Containerize with Kubernetes manifests
- Add load balancer configuration
- Implement stateless API design

**17. Automated Model Retraining - 5 days**
- Create retraining pipeline
- Schedule monthly retraining
- Add performance-based triggers

---

## Medium Priority (P2) - 34 Engineering Days

### 18-27. Medium Priority Items Summary

**18. LIME Explanations in UI (3 days)**
- Add explanation visualizations to dashboard
- Interactive feature importance plots

**19. Fairness Metrics (3 days)**
- Analyze performance across region types
- Check for bias by latitude/longitude

**20. Feature Importance Tracking (3 days)**
- Monitor which features models rely on
- Alert on sudden importance shifts

**21. Distributed Caching (Redis) (3 days)**
- Replace file-based cache with Redis
- Share cache across API instances

**22. Model Bias Detection (5 days)**
- Implement fairness analysis
- Regular bias audits

**23. Compliance Documentation (5 days)**
- Create model cards
- Document data lineage
- Privacy impact assessment

**24. Chaos Engineering Tests (3 days)**
- Inject database failures
- Test network partitions
- Verify graceful degradation

**25. Config Validation (Pydantic) (2 days)**
- Validate config.yaml at startup
- Prevent misconfiguration errors

**26. Prediction Quality Dashboard (5 days)**
- Grafana dashboard for metrics
- Real-time accuracy tracking

**27. Incident Response Documentation (2 days)**
- Create runbook for common issues
- Define escalation procedures

---

## Architecture Deep Dive

### System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         External Layer                          │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ NOAA SWPC  │    │   Operators  │    │  Monitoring      │   │
│  │ Data APIs  │    │   (Web UI)   │    │  (Grafana/Logs)  │   │
│  └─────┬──────┘    └──────┬───────┘    └────────┬─────────┘   │
└────────┼──────────────────┼─────────────────────┼──────────────┘
         │                  │                     │
┌────────┼──────────────────┼─────────────────────┼──────────────┐
│        │       API Gateway / Load Balancer     │              │
│        │       (Nginx with TLS/Rate Limiting)  │              │
└────────┼──────────────────┼─────────────────────┼──────────────┘
         │                  │                     │
┌────────┼──────────────────┼─────────────────────┼──────────────┐
│        │          Application Layer            │              │
│  ┌─────▼──────┐    ┌─────▼──────┐    ┌────────▼─────────┐    │
│  │ Ingestion  │    │   Flask    │    │   Monitoring     │    │
│  │  Service   │    │    API     │    │    Service       │    │
│  │ (Celery)   │    │ (Gunicorn) │    │ (Prometheus)     │    │
│  └─────┬──────┘    └─────┬──────┘    └──────────────────┘    │
└────────┼──────────────────┼────────────────────────────────────┘
         │                  │
┌────────┼──────────────────┼────────────────────────────────────┐
│        │        Business Logic Layer          │              │
│  ┌─────▼──────┐    ┌─────▼──────────┐    ┌──────────────┐    │
│  │   Data     │    │  Prediction    │    │   Feature    │    │
│  │ Ingestion  │───▶│   Service      │◀──▶│ Engineering  │    │
│  │  Pipeline  │    │ (Models+Drift) │    │   Pipeline   │    │
│  └─────┬──────┘    └─────┬──────────┘    └───────┬──────┘    │
└────────┼──────────────────┼────────────────────────┼───────────┘
         │                  │                        │
┌────────┼──────────────────┼────────────────────────┼───────────┐
│        │          Data Layer                       │           │
│  ┌─────▼──────┐    ┌─────▼──────────┐    ┌────────▼──────┐   │
│  │ PostgreSQL │    │   MLflow       │    │   Redis       │   │
│  │ (Time-     │    │   (Model       │    │   (Cache/     │   │
│  │  Series)   │    │   Registry)    │    │   Sessions)   │   │
│  └────────────┘    └────────────────┘    └───────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

### Current Strengths

1. **Clean Separation of Concerns**
   - Data layer (ingestion, persistence) isolated from business logic
   - Feature engineering decoupled from models
   - API layer cleanly wraps services

2. **Database Schema Design**
   - Proper normalization with composite unique constraints
   - Audit trails on all tables (created_at, ingested_at)
   - Time-series optimizations (indexes on timestamps)

3. **Error Handling Patterns**
   - Transaction rollback on database errors
   - Graceful degradation in feature computation
   - Try-except blocks throughout

### Current Weaknesses

1. **Tight Coupling to NOAA**
   - Hardcoded endpoint URLs
   - No circuit breaker or fallback
   - Single point of failure

2. **Synchronous API**
   - Blocking feature computation (5-10s)
   - Cannot handle concurrent load
   - No async task queue

3. **Global State**
   - Database singleton prevents horizontal scaling
   - File-based cache not shared across instances

---

## Data Integrity Assessment

### Database Schema Strengths

**Well-Designed Tables:**
```sql
-- Example: Composite unique constraints prevent duplicates
CREATE UNIQUE INDEX idx_flux_timestamp
  ON flare_goes_xray_flux(timestamp);

CREATE UNIQUE INDEX idx_region_time
  ON flare_solar_regions(region_number, timestamp);

-- Audit trail columns on all tables
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
```

### Critical Data Gaps

1. **No Data Retention Implementation**
   - Projection: ~100K flux records/year
   - Database will grow to ~1M records after 10 years
   - No archival or cleanup procedures

2. **Weak Input Validation**
   ```python
   # Current code - no bounds checking
   "flux_short": row.get("flux_short"),  # Accepts any value
   "flux_long": row.get("flux_long"),    # No NaN detection

   # Should be:
   flux_short = row.get("flux_short")
   if flux_short is not None:
       if not (1e-9 <= flux_short <= 1e-2):  # Physical bounds
           raise ValueError(f"Invalid flux value: {flux_short}")
   ```

3. **No Data Lineage**
   - Cannot trace which raw data contributed to predictions
   - Missing foreign keys from predictions to data sources

4. **No Anomaly Detection**
   - No automated detection of data gaps or corruption
   - Example: Stale data from NOAA goes undetected

---

## Model Governance Gaps

### Critical Missing Components

1. **No Model Versioning**
   ```python
   # Current: Overwrites existing model
   joblib.dump(pipeline_data, "models/survival_model.joblib")

   # Should be: Versioned with metadata
   mlflow.sklearn.log_model(
       pipeline,
       artifact_path="survival_model",
       registered_model_name="flare_survival",
   )
   ```

2. **No Explainability**
   - Cannot explain predictions to operators
   - No SHAP/LIME implementations
   - Regulatory/compliance risk

3. **No Online Monitoring**
   - Models deployed with no ongoing validation
   - Cannot detect performance degradation
   - No automated retraining triggers

---

## Go/No-Go Decision Framework

### DO NOT DEPLOY if:
- ❌ Any P0 (Critical) item remains unaddressed
- ❌ No API authentication implemented
- ❌ No backup/recovery plan in place
- ❌ No alerting system configured
- ❌ No model versioning capability

### PROCEED TO LIMITED PILOT if:
- ✅ All P0 items resolved
- ✅ >70% of P1 items resolved
- ✅ Comprehensive incident response plan documented
- ✅ 24/7 on-call support available
- ✅ Rollback procedures tested

### FULL OPERATIONAL DEPLOYMENT when:
- ✅ All P0 and P1 items resolved
- ✅ >50% of P2 items resolved
- ✅ 3 months of successful pilot operation
- ✅ Load testing confirms scalability (100+ concurrent users)
- ✅ External security audit passed
- ✅ Model performance monitoring shows <5% degradation

---

## Success Metrics

### Reliability Targets

| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| API Uptime | 99.9% | Unknown | Need monitoring |
| Prediction Latency (p95) | <3s | ~5-10s | Need async tasks |
| Data Loss Events | 0 | High risk | Need backups |
| MTTR (Mean Time to Recovery) | <30 min | Unknown | Need alerting |

### Model Performance Targets

| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| Brier Score (24h) | <0.15 | Unknown | Need monitoring |
| C-Index (Survival) | >0.65 | 0.72 (training) | Need validation |
| False Positive Rate | <20% | Unknown | Need monitoring |
| Monthly Degradation | <5% | Unknown | Need tracking |

### Operational Excellence Targets

| Metric | Target | Current | Gap |
|--------|--------|---------|-----|
| Incidents with RCA | 100% | 0% | Need process |
| Model Retraining Success | >95% | Manual only | Need automation |
| Validation Pass Rate | >90% | 100% (dev) | Need production |
| Critical Vulnerabilities | 0 | Unknown | Need scanning |

---

## Timeline & Milestones

### Phase 2A: Security & Reliability (Weeks 1-4)

**Milestone: Production-Ready Security**

- Week 1: API Authentication + Rate Limiting + TLS
- Week 2: Centralized Logging + Alerting
- Week 3: Model Versioning (MLflow setup)
- Week 4: Automated Backups + Testing

**Exit Criteria:**
- ✅ All API endpoints authenticated
- ✅ HTTPS enforced
- ✅ Alerts firing on test failures
- ✅ Successful backup restore test

### Phase 2B: Interpretability & Monitoring (Weeks 5-8)

**Milestone: Observable System**

- Week 5-6: Model Explainability (SHAP/LIME)
- Week 7: Online Performance Monitoring
- Week 8: Data Retention + Feature Validation

**Exit Criteria:**
- ✅ Predictions include top 5 feature explanations
- ✅ Performance metrics tracked daily
- ✅ Archival process running

### Phase 2C: Scalability & Testing (Weeks 9-12)

**Milestone: Production-Grade Performance**

- Week 9-10: Async Task Queue (Celery)
- Week 11: Integration & Load Testing
- Week 12: Horizontal Scalability (K8s)

**Exit Criteria:**
- ✅ API handles 100 concurrent users
- ✅ p95 latency <3s
- ✅ Load balancer deployed

### Phase 2D: Pilot Deployment (Months 4-6)

**Milestone: Limited Production Use**

- Month 4: Deploy to staging environment
- Month 5: Limited pilot with select users
- Month 6: Monitor, iterate, fix issues

**Exit Criteria:**
- ✅ 99% uptime for 30 days
- ✅ Model performance within targets
- ✅ No critical incidents

---

## Resource Planning

### Engineering Team Requirements

| Role | Allocation | Responsibilities |
|------|------------|------------------|
| ML Engineer | 1 FTE | Model improvements, explainability, monitoring |
| Backend Engineer | 1 FTE | API hardening, async processing, integration |
| DevOps Engineer | 1 FTE | Infrastructure, monitoring, security, CI/CD |
| QA Engineer | 0.5 FTE | Testing, validation, load testing |
| Compliance | 0.25 FTE | Documentation, audit prep, policies |

### Infrastructure Budget

| Component | Monthly Cost | Annual Cost |
|-----------|--------------|-------------|
| Cloud Compute (AWS ECS/EKS) | $500-800 | $6,000-9,600 |
| Database (RDS PostgreSQL) | $200-300 | $2,400-3,600 |
| Monitoring (Datadog/New Relic) | $200-500 | $2,400-6,000 |
| Log Aggregation (CloudWatch) | $100-300 | $1,200-3,600 |
| Model Registry (MLflow S3) | $50-100 | $600-1,200 |
| Backup Storage (S3 Glacier) | $50-100 | $600-1,200 |
| **Total** | **$1,100-2,100** | **$13,200-25,200** |

---

## Risk Assessment

### Critical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Data breach (no auth) | High | Critical | P0 item #1 - API authentication |
| Data loss (no backups) | Medium | Critical | P0 item #5 - Automated backups |
| Model degradation undetected | High | High | P1 item #9 - Online monitoring |
| NOAA API outage | Medium | High | P1 item #13 - Circuit breakers |
| Scaling bottleneck | High | Medium | P1 item #12 - Async tasks |

### Acceptance Criteria

Before declaring "Production Ready":

1. **Security:**
   - ✅ External penetration test passed
   - ✅ Zero critical vulnerabilities in dependencies
   - ✅ HTTPS enforced, TLS 1.2+ only
   - ✅ API keys rotated successfully

2. **Reliability:**
   - ✅ 99% uptime for 90 consecutive days
   - ✅ Successful disaster recovery drill
   - ✅ Mean time to detection (MTTD) <5 minutes

3. **Performance:**
   - ✅ Load test: 100 concurrent users, <3s p95 latency
   - ✅ Model performance: Brier <0.15, C-index >0.65
   - ✅ <5% performance degradation per month

4. **Governance:**
   - ✅ Model cards for all production models
   - ✅ Data lineage documented
   - ✅ Incident response runbook tested

---

## Conclusion

Flare-plus is a **well-engineered ML prototype** that requires 3-6 months of hardening before operational deployment. The codebase quality is high, with clean architecture and comprehensive testing, but operational maturity is prototype-level.

### Immediate Next Steps (This Week)

1. **Security First:** Implement API authentication (P0 #1)
2. **Risk Mitigation:** Set up automated backups (P0 #5)
3. **Visibility:** Configure centralized logging (P0 #4)

### Success Path

**Month 1:** Complete all P0 items
**Month 2:** Complete 50% of P1 items
**Month 3:** Complete remaining P1 items
**Month 4:** Deploy to staging, begin pilot
**Month 5-6:** Monitor pilot, iterate
**Month 7:** Full operational deployment

With disciplined execution of this roadmap, Flare-plus can transition from advanced prototype to production-grade operational system within 6 months.

---

**Document Maintained By:** Engineering Team
**Last Updated:** 2025-11-04
**Next Review:** 2025-12-01
