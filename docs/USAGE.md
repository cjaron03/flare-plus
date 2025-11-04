# flare-plus usage guide

## overview

flare-plus provides two types of predictions for solar flare events:

1. **classification prediction**: predicts the probability of different flare classes (None, C, M, X) occurring within a 24-48 hour window
2. **survival analysis**: predicts the probability distribution of when a flare might occur over different time buckets (0-168 hours)

## starting the services

### 1. start docker services

```bash
./flare up
```

this starts postgres and the app container.

### 2. initialize database

```bash
./flare init-db
```

### 3. ingest data

```bash
./flare ingest
```

this fetches data from noaa sources and populates the database.

### 4. train models (if needed)

**classification model:**
```bash
docker-compose exec app python scripts/train_and_predict_classification.py \
  --train \
  --start-date 2024-01-01 \
  --end-date 2024-11-01 \
  --save-model /app/models/classification_model.joblib
```

**survival model:**
```bash
docker-compose exec app python scripts/train_and_predict_survival.py \
  --train \
  --start-date 2024-01-01 \
  --end-date 2024-11-01 \
  --save-model /app/models/survival_model.joblib
```

### 5. start api server

```bash
./flare api-bg
```

this starts the flask api server in the background. it will automatically detect and load models from `/app/models/`.

the api is available at:
- inside container: `http://127.0.0.1:5000`
- from host: `http://127.0.0.1:5001`

### 6. start ui dashboard

```bash
./flare ui-bg
```

this starts the gradio ui dashboard in the background.

the dashboard is available at:
- `http://127.0.0.1:7860`

## using the ui dashboard

### predictions tab

the predictions tab allows you to make real-time predictions for solar flare events.

#### classification prediction

1. **observation timestamp**: select the timestamp for which you want to make a prediction (defaults to current time)
2. **region number** (optional): specify a specific solar region number to focus on
3. **prediction window**: choose 24h or 48h window
4. **model type**: choose between logistic regression or gradient boosting
5. click **"predict classification"** to generate predictions

**output:**
- predicted flare class (None, C, M, or X)
- probability distribution for each class
- bar chart showing class probabilities

#### survival analysis prediction

1. **observation timestamp**: select the timestamp for prediction
2. **region number** (optional): specify a specific solar region
3. **model type**: choose between cox proportional hazards or gradient boosting survival
4. click **"predict survival"** to generate predictions

**output:**
- plain language summary (e.g., "there is a 42% chance of an x-class flare within 24-48h")
- detailed probability distribution over time buckets
- survival curve chart showing probability over time
- probability distribution chart showing likelihood in each time bucket

### timeline tab

view historical flare events and data:

1. **start date**: beginning of the date range
2. **end date**: end of the date range
3. click **"update timeline"** to view flares in the selected range

the timeline shows:
- flare events with class, magnitude, and timestamp
- links to noaa source data
- interactive timeline visualization

### scenario tab (placeholder)

future feature for exploring "what-if" scenarios by adjusting feature values.

### about tab

information about:
- noaa data sources
- model limitations
- author attribution

## using the api directly

### health check

```bash
curl http://127.0.0.1:5001/health
```

returns:
```json
{
  "status": "healthy",
  "classification_available": true,
  "survival_available": true,
  ...
}
```

### classification prediction

```bash
curl -X POST http://127.0.0.1:5001/predict/classification \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-11-03T12:00:00",
    "window": 24,
    "model_type": "gradient_boosting",
    "region_number": 1234
  }'
```

response:
```json
{
  "timestamp": "2024-11-03T12:00:00",
  "window_hours": 24,
  "predicted_class": "None",
  "class_probabilities": {
    "None": 0.95,
    "C": 0.04,
    "M": 0.01,
    "X": 0.00
  },
  "model_type": "gradient_boosting",
  "drift_detection": {...}
}
```

### survival prediction

```bash
curl -X POST http://127.0.0.1:5001/predict/survival \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-11-03T12:00:00",
    "model_type": "cox",
    "region_number": 1234
  }'
```

response:
```json
{
  "timestamp": "2024-11-03T12:00:00",
  "target_flare_class": "X",
  "hazard_score": 0.42,
  "probability_distribution": {
    "0-6h": 0.01,
    "6-12h": 0.02,
    "12-24h": 0.15,
    "24-48h": 0.20,
    ...
  },
  "survival_function": {...},
  "model_type": "cox"
}
```

### combined predictions

```bash
curl -X POST http://127.0.0.1:5001/predict/all \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-11-03T12:00:00",
    "region_number": 1234
  }'
```

returns both classification and survival predictions in a single request.

### data ingestion

```bash
curl -X POST http://127.0.0.1:5001/ingest \
  -H "Content-Type: application/json" \
  -d '{"use_cache": true}'
```

triggers data ingestion from noaa sources. returns summary of records ingested.

## troubleshooting

### models not loading

1. check that model files exist:
   ```bash
   docker-compose exec app ls -lh /app/models/*.joblib
   ```

2. check api logs:
   ```bash
   ./flare api-logs
   ```

3. verify models are detected:
   ```bash
   ./flare api-bg
   ```
   look for "found classification model" and "found survival model" messages.

### api returns 500 error

1. check api logs for detailed error:
   ```bash
   ./flare api-logs
   ```

2. verify database has data:
   ```bash
   ./flare ingest
   ```

3. check if models are trained:
   ```bash
   docker-compose exec app python -c "import joblib; data = joblib.load('/app/models/classification_model.joblib'); print('Models:', list(data.get('models', {}).keys()))"
   ```

### ui shows "models: none loaded"

1. ensure api server is running:
   ```bash
   ./flare api-bg
   ```

2. check api health:
   ```bash
   docker-compose exec app python -c "import requests; r = requests.get('http://127.0.0.1:5000/health'); print(r.json())"
   ```

3. refresh the ui page (the connection status updates on page load).

### predictions return "none" or low probabilities

this is normal when solar activity is low. the model is predicting based on current conditions:

- if no active regions are present, probabilities will be low
- if recent flux activity is low, probabilities will be low
- check the timeline tab to see recent flare activity

### feature mismatch errors

if you see errors like "max_magnitude_24h not in index", this means the model was trained with features that aren't available during prediction. the system automatically handles this by defaulting missing historical features to 0.

to fix properly, retrain the model after ensuring feature extraction excludes historical label features.

## model interpretation

### classification probabilities

- **none**: probability of no significant flare (class C or above) in the window
- **c**: probability of a c-class flare (low intensity)
- **m**: probability of an m-class flare (moderate intensity)
- **x**: probability of an x-class flare (high intensity)

higher probabilities indicate greater likelihood based on current solar conditions.

### survival analysis

the survival analysis provides:

1. **hazard score**: overall risk level (higher = more risk)
2. **probability distribution**: likelihood of flare in each time bucket
3. **survival curve**: probability of no flare over time (decreasing curve)

use the plain language summary for quick interpretation, and the detailed distribution for precise timing estimates.

## best practices

1. **data freshness**: refresh data regularly using the "refresh data & status" button in the ui, or run `./flare ingest` periodically

2. **model retraining**: retrain models periodically (monthly or quarterly) as new data becomes available

3. **interpretation**: combine classification and survival predictions for a complete picture:
   - classification: "will there be a flare?" (yes/no probability)
   - survival: "when might it occur?" (timing probability)

4. **region-specific predictions**: specify a region number when focusing on a particular active region

5. **timestamp selection**: use recent timestamps (within last 24h) for current predictions; older timestamps for historical analysis

## limitations

- models are trained on historical data and may not capture rare or novel solar activity patterns
- predictions are probabilistic, not deterministic
- feature computation requires sufficient historical data (at least 24-48 hours of flux data)
- region-specific predictions require that the specified region exists in the database
- model performance depends on data quality and completeness from noaa sources

