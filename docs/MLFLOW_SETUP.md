# MLflow Setup and Usage

MLflow is integrated into flare+ for experiment tracking, model versioning, and artifact management.

## Overview

MLflow tracks:
- **Parameters**: hyperparameters, model configuration, dataset info
- **Metrics**: training accuracy, test accuracy, brier scores, ROC-AUC, C-index
- **Models**: trained model artifacts (sklearn models and custom survival models)
- **Datasets**: dataset metadata and versions

## Setup

### Initial Setup

MLflow is automatically configured when you train models. The default setup uses a local SQLite backend:

```bash
# setup mlflow (creates default experiment)
python scripts/setup_mlflow.py
```

This creates:
- SQLite database at `mlruns.db` (or path specified in `config.yaml`)
- Default experiment named "flare-plus"

### Configuration

MLflow settings are in `config.yaml`:

```yaml
mlflow:
  tracking_uri: "sqlite:///mlruns.db"  # local sqlite backend
  experiment_name: "flare-plus"  # default experiment name
  model_registry_name: "flare-plus-models"  # model registry name
```

To use a different backend (e.g., PostgreSQL), update `tracking_uri`:

```yaml
mlflow:
  tracking_uri: "postgresql://user:pass@localhost:5432/mlflow_db"
```

## Using MLflow

### Automatic Tracking

MLflow tracking is enabled by default when training models. Both classification and survival pipelines automatically:

1. Start a new MLflow run
2. Log training parameters (dataset size, hyperparameters, model config)
3. Log evaluation metrics (accuracy, brier score, ROC-AUC, C-index)
4. Save model artifacts
5. Tag runs with metadata (model type, target class, etc.)

### Training with MLflow

Classification training:

```bash
python scripts/train_and_predict_classification.py \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --save-path models/classification_model.joblib
```

Survival training:

```bash
python scripts/train_and_predict_survival.py \
  --train \
  --start-date 2023-01-01 \
  --end-date 2023-12-31 \
  --target-class X
```

Both scripts automatically log to MLflow.

### Disabling MLflow

To disable MLflow tracking (e.g., for quick tests):

```python
from src.models.pipeline import ClassificationPipeline

pipeline = ClassificationPipeline(use_mlflow=False)
# ... training code ...
```

## Viewing Experiments

### MLflow UI (Local)

Start the MLflow UI to view experiments:

```bash
# using default sqlite backend
mlflow ui --backend-store-uri sqlite:///mlruns.db

# or if using postgresql
mlflow ui --backend-store-uri postgresql://user:pass@localhost:5432/mlflow_db
```

Then open http://localhost:5000 in your browser.

The UI shows:
- **Experiments**: list of all experiments
- **Runs**: individual training runs with parameters, metrics, and artifacts
- **Model Registry**: registered models with versions and stages

### MLflow UI (Docker)

To run MLflow UI in Docker (optional, commented out in `docker-compose.yml`):

1. Uncomment the `mlflow` service in `docker-compose.yml`
2. Start the service:

```bash
docker-compose up mlflow
```

Access at http://localhost:5002 (or port specified by `MLFLOW_PORT`).

## Model Registry

### Registering Models

Models are automatically logged during training. To register a model in the model registry:

```python
from src.ml.experiment_tracking import MLflowTracker

tracker = MLflowTracker()

# get model uri from a run
model_uri = "runs:/<run_id>/models/24h_gradient_boosting"

# register model
version = tracker.register_model(
    model_uri=model_uri,
    model_name="flare-classification-24h",
    tags={"stage": "production", "model_type": "gradient_boosting"}
)
```

### Loading Registered Models

Load a model from the registry:

```python
from src.ml.experiment_tracking import MLflowTracker

tracker = MLflowTracker()

# load latest version
model = tracker.load_model("models:/flare-classification-24h/latest")

# or specific version
model = tracker.load_model("models:/flare-classification-24h/1")
```

### Model Versions

Get latest model version:

```python
latest_version = tracker.get_latest_model_version("flare-classification-24h")
```

## Programmatic Access

### Query Experiments

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# list experiments
experiments = client.list_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")

# get runs for an experiment
runs = client.search_runs(
    experiment_ids=["0"],
    filter_string="metrics.test_accuracy > 0.8",
    max_results=10
)

for run in runs:
    print(f"Run {run.info.run_id}: accuracy={run.data.metrics.get('test_accuracy')}")
```

### Compare Runs

```python
import mlflow

# compare runs
runs = mlflow.search_runs(
    experiment_ids=["0"],
    order_by=["metrics.test_accuracy DESC"],
    max_results=5
)

print(runs[["run_id", "metrics.test_accuracy", "params.model_type"]])
```

## Best Practices

1. **Meaningful Run Names**: Use descriptive run names when training:
   ```python
   pipeline.train_and_evaluate(dataset, run_name="baseline_20240101")
   ```

2. **Tag Important Runs**: Use tags to mark production models:
   ```python
   tracker.start_run(run_name="prod_model", tags={"stage": "production"})
   ```

3. **Version Models**: Register models in the model registry for production use

4. **Track Datasets**: Log dataset metadata for reproducibility:
   ```python
   tracker.log_dataset("data/train.csv", "training_data", {"rows": 10000, "features": 50})
   ```

5. **Regular Cleanup**: Periodically archive old experiments to keep the database manageable

## Troubleshooting

### MLflow Not Logging

- Check that MLflow is enabled: `pipeline.use_mlflow` should be `True`
- Verify tracking URI is accessible
- Check logs for MLflow errors (warnings are logged but don't fail training)

### Database Locked (SQLite)

If using SQLite and getting "database is locked" errors:
- Ensure only one process writes to the database at a time
- Consider switching to PostgreSQL for production use

### Model Artifacts Not Found

- Verify model artifacts were saved (check MLflow UI)
- Ensure artifact paths are correct when loading models
- Check file permissions if using shared storage

## Migration to Production

For production deployments:

1. **Use PostgreSQL Backend**: Update `config.yaml`:
   ```yaml
   mlflow:
     tracking_uri: "postgresql://user:pass@db-host:5432/mlflow_db"
   ```

2. **Use Model Registry**: Register production models in the registry with proper versioning

3. **Set Up MLflow Server**: Deploy MLflow server for centralized tracking:
   ```bash
   mlflow server \
     --backend-store-uri postgresql://user:pass@db-host:5432/mlflow_db \
     --default-artifact-root s3://mlflow-artifacts/ \
     --host 0.0.0.0 \
     --port 5000
   ```

4. **Monitor Model Performance**: Use MLflow to track model drift and performance degradation over time

