# Performance Profiling Guide

## Overview

This guide describes how to profile the flare-plus system to identify performance bottlenecks, particularly for large-scale data ingestion and feature generation.

## Quick Start

### Profile Data Ingestion

```bash
# using cprofile
python -m cProfile -o ingestion_profile.stats -s cumulative scripts/run_ingestion.py

# analyze results
python -m pstats ingestion_profile.stats
```

### Profile Feature Generation

```python
# example profiling script
import cProfile
import pstats
from datetime import datetime, timedelta
from src.features.pipeline import FeatureEngineer

# generate timestamps for profiling
timestamps = [datetime.utcnow() - timedelta(hours=i) for i in range(100)]

# profile
profiler = cProfile.Profile()
profiler.enable()

engineer = FeatureEngineer()
features = engineer.compute_features_batch(timestamps)

profiler.disable()
profiler.dump_stats('feature_profile.stats')

# analyze
stats = pstats.Stats('feature_profile.stats')
stats.sort_stats('cumulative')
stats.print_stats(20)  # top 20 functions
```

## Key Areas to Profile

### 1. Database Query Performance

Monitor database round-trips:
- Check that bulk loading is used for batch operations
- Verify that preloaded_data is passed through feature computation chains
- Profile individual query execution times

```python
import logging
import time
from contextlib import contextmanager

@contextmanager
def profile_db_query(query_name):
    start = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start
        logger.info(f"query '{query_name}' took {duration:.3f}s")
```

### 2. Feature Engineering Bottlenecks

Profile:
- `FeatureEngineer.compute_features()` - single timestamp computation
- `FeatureEngineer.compute_features_batch()` - batch computation
- `TimeVaryingCovariateEngineer.compute_time_varying_covariates_batch()` - survival covariates

Look for:
- Excessive database queries (should be bulk loaded)
- Slow pandas operations on large dataframes
- Unnecessary data copying

### 3. Memory Usage

Monitor memory consumption for large feature batches:

```python
import tracemalloc

tracemalloc.start()
# ... run operation ...
current, peak = tracemalloc.get_traced_memory()
print(f"peak memory: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

### 4. Data Ingestion Performance

Profile:
- Network requests to NOAA endpoints
- Database writes (batch vs individual inserts)
- Cache read/write operations

## Profiling Tools

### cProfile

Built-in Python profiler, good for overall performance:

```bash
python -m cProfile -o profile.stats script.py
python -m pstats profile.stats
```

### line_profiler

For line-by-line profiling:

```bash
pip install line_profiler
kernprof -l -v script.py
```

### memory_profiler

For memory usage:

```bash
pip install memory-profiler
python -m memory_profiler script.py
```

### py-spy

For runtime profiling without code changes:

```bash
pip install py-spy
py-spy record -o profile.svg -- python script.py
```

## Recommended Profiling Scenarios

### Scenario 1: Hourly Backfill (24 hours)

```python
from datetime import datetime, timedelta
from src.features.pipeline import FeatureEngineer

timestamps = [datetime.utcnow() - timedelta(hours=i) for i in range(24)]
engineer = FeatureEngineer()

# should complete in < 30 seconds with bulk loading
features = engineer.compute_features_batch(timestamps)
```

Expected: Single bulk database query, < 30s execution time

### Scenario 2: Large Batch Feature Generation (1000 timestamps)

```python
timestamps = [datetime.utcnow() - timedelta(hours=i*0.1) for i in range(1000)]
features = engineer.compute_features_batch(timestamps)
```

Expected: Efficient bulk loading, memory usage < 500MB

### Scenario 3: Full Data Ingestion Run

```bash
time python scripts/run_ingestion.py
```

Expected: < 5 minutes for typical incremental update

## Interpreting Results

### High Database Query Time

If database queries dominate:
- Check that `preloaded_data` is being used
- Verify bulk loading functions are called
- Consider database indexing on timestamp columns

### High Memory Usage

If memory usage is excessive:
- Check for large probability arrays in evaluation results (should use summary stats)
- Verify dataframes are released after use
- Consider processing in smaller batches

### Slow Feature Computation

If feature computation is slow:
- Profile individual feature computation functions
- Check for unnecessary pandas operations
- Consider caching computed features

## Continuous Monitoring

For production systems, consider:

1. **Logging execution times**: Add timing logs to critical operations
2. **Database query logging**: Enable SQLAlchemy query logging
3. **Resource monitoring**: Track CPU, memory, and I/O usage over time

## Example: Automated Profiling Script

```python
#!/usr/bin/env python3
"""profile feature generation performance."""

import cProfile
import pstats
import io
from datetime import datetime, timedelta
from src.features.pipeline import FeatureEngineer

def profile_feature_generation(n_timestamps=100):
    """profile batch feature generation."""
    timestamps = [datetime.utcnow() - timedelta(hours=i) for i in range(n_timestamps)]
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    engineer = FeatureEngineer()
    features = engineer.compute_features_batch(timestamps)
    
    profiler.disable()
    
    # generate report
    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    
    print(f"Generated {len(features)} feature rows")
    print("\nTop 30 functions by cumulative time:")
    print(s.getvalue())

if __name__ == "__main__":
    profile_feature_generation(n_timestamps=100)
```

