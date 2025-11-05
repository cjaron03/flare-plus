# Historical Data Collection Guide

## Current Status

You currently have:
- **37 C-class flares** (need: 200+)
- **1 M-class flare** (need: 50+)
- **0 X-class flares** (need: 10+)
- **9 days** of data coverage

**This is insufficient for reliable ML model training.**

## The Problem

NOAA SWPC's public API only provides the **last 7 days** of data. To get 2020-2024 historical data, you need alternative approaches.

## Solution Options

### Option 1: Continuous Collection (Recommended for Learning)

**Set up automated ingestion to collect data going forward:**

1. **Start Collection Now:**
   ```bash
   # Run ingestion hourly via cron
   0 * * * * cd /path/to/flare-plus && docker-compose exec app python scripts/run_ingestion_with_retry.py
   ```

2. **Timeline:**
   - After 3 months: ~100-200 C-class, ~10-20 M-class (marginal)
   - After 6 months: ~200-400 C-class, ~20-40 M-class (good for C/M-class)
   - After 12 months: ~500+ C-class, ~50+ M-class (reliable training)

3. **Pros:**
   - Free
   - Automated
   - Gets you experience with operational systems

4. **Cons:**
   - Takes months
   - Won't have historical context

### Option 2: Contact NOAA for Bulk Data (Best for Real Research)

**Request historical data directly from NOAA:**

1. **NOAA National Centers for Environmental Information (NCEI):**
   - Website: https://www.ncei.noaa.gov/
   - Email: ncei.orders@noaa.gov
   - Request: GOES X-ray Sensor (XRS) Science Quality data, 2020-2024

2. **What to Request:**
   - GOES-16 XRS data (primary satellite 2020-2024)
   - 1-minute averaged flux data
   - CSV or NetCDF format
   - Time period: 2020-01-01 to 2024-11-01

3. **Expected Response Time:**
   - 1-2 weeks for data request processing
   - May require filling out research use form

4. **Pros:**
   - Gets you full 4.9 years immediately
   - Science-quality data
   - Proper for research/publication

5. **Cons:**
   - Requires formal request
   - Possible waiting period

### Option 3: Use NASA's DONKI Database (Intermediate)

**NASA's Database Of Notifications, Knowledge, Information:**

1. **Access DONKI API:**
   - Website: https://kauai.ccmc.gsfc.nasa.gov/DONKI/
   - API: https://api.nasa.gov/ (get API key)
   - Provides flare event catalog

2. **What You Get:**
   - Pre-detected flare events (2011-present)
   - Flare class, time, active region
   - No need to run flare detection yourself

3. **Implementation:**
   ```python
   # Add to your fetchers
   import requests

   def fetch_donki_flares(start_date, end_date, api_key):
       url = f"https://api.nasa.gov/DONKI/FLR"
       params = {
           'startDate': start_date.strftime('%Y-%m-%d'),
           'endDate': end_date.strftime('%Y-%m-%d'),
           'api_key': api_key
       }
       response = requests.get(url, params=params)
       return response.json()
   ```

4. **Pros:**
   - Free API access
   - Historical flare catalog
   - Well-maintained

5. **Cons:**
   - Still need GOES flux data for features
   - API rate limits

### Option 4: Demo Mode (Use What You Have)

**Train with current 9 days for demonstration only:**

1. **Accept Limitations:**
   - Model will overfit
   - Predictions won't generalize
   - OK for learning/portfolio

2. **Add Warnings:**
   ```python
   if c_class_count < 200:
       print("WARNING: Insufficient training data")
       print("Model predictions are unreliable")
       print("For demonstration purposes only")
   ```

3. **Focus on:**
   - Pipeline architecture
   - Feature engineering
   - Model comparison methodology
   - Visualization and dashboards

## Recommended Approach

### Short Term (This Week)

1. **Set up continuous collection:**
   ```bash
   docker-compose exec app python scripts/run_ingestion_with_retry.py
   ```
   Schedule this hourly to start accumulating data.

2. **Train demo model with warnings:**
   - Use current 37 events
   - Add prominent "DEMO ONLY" warnings
   - Document limitations

3. **Build infrastructure:**
   - Complete model serving setup
   - Create API endpoints
   - Build dashboards

### Medium Term (Next Month)

1. **Submit NOAA NCEI data request**
2. **Continue collecting real-time data**
3. **Explore NASA DONKI integration**

### Long Term (3-6 Months)

1. **Re-train with full historical dataset**
2. **Validate on held-out test set**
3. **Compare to NOAA operational forecasts**
4. **Publish results**

## Next Steps for Training

Since you want to train now, I'll:

1. Add data sufficiency warnings to training script
2. Document that this is a demo
3. Proceed with training on current data
4. Show you how predictions look (with caveats)

Then you can decide:
- Keep running continuous collection for better data
- Request historical data from NOAA
- Use for portfolio/learning as-is
