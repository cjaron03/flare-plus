<script>
  import { onMount, onDestroy } from "svelte";
  import { fetchStatus, triggerIngestion, predictClassification, predictSurvival, fetchTimeline } from "../lib/api";
  import StatusPill from "../components/StatusPill.svelte";
  import LoadingSpinner from "../components/LoadingSpinner.svelte";
  import PlotlyChart from "../components/PlotlyChart.svelte";

  let loading = true;
  let status;
  let error = "";
  let ingestMessage = "";
  let ingestSummary = "";
  let ingestLoading = false;
  
  let predictionLoading = true;
  let predictionError = "";
  let predictionLevel = null;
  let predictionConfidence = null; // Dynamic confidence based on model probabilities
  let predictionTimestamp = null; // When the prediction was made
  let nextUpdateTime = null; // When the next update will happen
  let showScaleInfo = false;
  let showBacktestReport = false;
  let predictionUpdateInterval = null;
  let testMode = false; // Set to true to test dynamic updates - cycles through C, M, X
  let testCycleIndex = 0;
  let testInterval = null;
  
  // Unified prediction card state
  let autoUpdateMode = true; // Auto/Manual toggle
  let manualPredictionType = "classification"; // "classification" or "survival" for manual mode
  let showViewDetails = false; // View Details modal
  let recentFlares = []; // Recent flares for View Details
  let lastMClassDate = null; // Last M-class flare date
  let lastXClassDate = null; // Last X-class flare date
  let modelMetadata = null; // Model training metadata
  let riskLevel = null; // Calculated risk level (EXTREME/HIGH/MEDIUM/LOW)
  let classProbabilities = {}; // Full probability breakdown
  let flareChartData = null; // Chart data for recent flares visualization
  let flareRefreshInterval = null; // Interval for auto-refreshing flares
  let showTimelineChart = false; // Collapse chart by default
  let filterClasses = { C: true, M: true, X: true }; // Class filter toggles

  const loadStatus = async () => {
    loading = true;
    error = "";
    try {
      status = await fetchStatus();
    } catch (err) {
      error = err.message;
    } finally {
      loading = false;
    }
  };

  const loadPrediction = async (useTestData = false, predictionType = null) => {
    predictionLoading = true;
    predictionError = "";
    try {
      // Determine which prediction type to use
      const predType = predictionType || (autoUpdateMode ? "classification" : manualPredictionType);
      
      let result;
      
      if (useTestData || testMode) {
        // Test mode: cycle through C moderate, M strong, X extreme
        const testCases = [
          {
            success: true,
            result: {
              predictedClass: "C",
              probabilities: { "None": 0.3, "C": 0.55, "M": 0.1, "X": 0.05 }
            }
          },
          {
            success: true,
            result: {
              predictedClass: "M",
              probabilities: { "None": 0.1, "C": 0.2, "M": 0.65, "X": 0.05 }
            }
          },
          {
            success: true,
            result: {
              predictedClass: "X",
              probabilities: { "None": 0.05, "C": 0.1, "M": 0.2, "X": 0.65 }
            }
          }
        ];
        
        result = testCases[testCycleIndex % testCases.length];
        testCycleIndex++;
        console.log('Test mode: Using test data', result.result);
      } else {
        const now = new Date();
        
        if (predType === "survival") {
          // Use survival prediction for time-to-event analysis
          result = await predictSurvival({
            timestamp: now.toISOString(),
            modelType: "cox",
            forceRefresh: false
          });
          
          // Process survival prediction result
          if (result && result.result) {
            // Survival prediction returns probability distribution over time buckets
            const probDist = result.result.probabilityDistribution || {};
            const targetClass = result.result.targetClass || "M";
            
            // Calculate 24-hour probability from survival buckets (0-24h)
            let prob24h = 0;
            for (const [bucket, prob] of Object.entries(probDist)) {
              if (bucket.includes("0-") || bucket.includes("24")) {
                const bucketStart = parseInt(bucket.split("-")[0].replace("h", "")) || 0;
                if (bucketStart < 24) {
                  prob24h += prob;
                }
              }
            }
            
            // Convert survival prediction to classification-like format for display
            const classProbs = {
              "None": Math.max(0, 1 - prob24h),
              "M": prob24h,
              "C": 0,
              "X": 0
            };
            
            const predictedClass = prob24h > 0.5 ? "M" : "None";
            
            // Process as classification prediction for display
            processClassificationResult(classProbs, predictedClass);
            return;
          }
        } else {
          // Use classification prediction to get probabilities for all flare types (None, C, M, X)
          result = await predictClassification({
            timestamp: now.toISOString(),
            window: 24,
            modelType: "gradient_boosting"
          });
        }
      }
      
      if (result && result.result) {
        // Classification prediction returns probabilities for None, C, M, X classes
        // API returns 'probabilities' field with decimal values (0-1)
        const classProbs = result.result.probabilities || {};
        const predictedClass = result.result.predictedClass || "None";
        
        processClassificationResult(classProbs, predictedClass);
      }
    } catch (err) {
      predictionError = err.message;
      console.error('Prediction error:', err);
    } finally {
      predictionLoading = false;
    }
  };
  
  // Helper function to process classification results
  const processClassificationResult = (classProbs, predictedClass) => {
        
    console.log('Classification prediction:', {
      predictedClass,
      classProbabilities: classProbs
    });
    
    // Find the highest probability flare class
    // For NOAA-style display, show the highest flare class probability if >5%
    // Otherwise show "None"
    let maxProb = 0;
    let maxClass = "None";
    let maxProbValue = classProbs["None"] || 0;
    const noneProb = classProbs["None"] || 0;
    
    // Check C, M, X classes (in order of severity)
    // Find the highest probability flare class
    const flareClasses = ["C", "M", "X"];
    for (const flareClass of flareClasses) {
      const prob = classProbs[flareClass] || 0;
      if (prob > maxProb) {
        maxProb = prob;
        maxClass = flareClass;
        maxProbValue = prob;
      }
    }
    
    // Only use "None" if no flare class has significant probability (>5%)
    // This ensures we show flare predictions even if "None" is higher
    if (maxProb < 0.05) {
      maxClass = "None";
      maxProbValue = noneProb;
    }
    
    console.log('Selected flare class:', maxClass, 'with probability:', maxProbValue);
    
    // Calculate confidence based on prediction probabilities
    // Higher max probability = higher confidence
    // Also consider how much higher it is than the next highest probability
    const sortedProbs = Object.values(classProbs).sort((a, b) => b - a);
    const topProb = sortedProbs[0] || 0;
    const secondProb = sortedProbs[1] || 0;
    const probGap = topProb - secondProb;
    
    // Confidence calculation:
    // - High confidence: top prob > 0.7 and gap > 0.3
    // - Normal confidence: top prob > 0.5 or gap > 0.2
    // - Low confidence: otherwise
    let confidenceLevel = "low";
    let confidencePercent = 0;
    
    if (topProb >= 0.7 && probGap >= 0.3) {
      confidenceLevel = "high";
      confidencePercent = Math.min(100, (topProb * 100) + (probGap * 50)); // Boost for high certainty
    } else if (topProb >= 0.5 || probGap >= 0.2) {
      confidenceLevel = "normal";
      confidencePercent = Math.min(90, (topProb * 100) + (probGap * 30));
    } else {
      confidenceLevel = "low";
      confidencePercent = Math.max(20, topProb * 100);
    }
    
    // Clamp to reasonable range
    confidencePercent = Math.min(100, Math.max(10, confidencePercent));
    
    predictionConfidence = {
      level: confidenceLevel,
      percent: confidencePercent,
      maxProb: topProb,
      probGap: probGap
    };
    
    console.log('Prediction confidence:', predictionConfidence);
    
    // Store prediction timestamp
    predictionTimestamp = new Date();
    
    // Calculate next update time (6 hours for auto, null for manual)
    if (autoUpdateMode) {
      nextUpdateTime = new Date(predictionTimestamp.getTime() + 6 * 60 * 60 * 1000);
    } else {
      nextUpdateTime = null;
    }
    
    // Store full probability breakdown for View Details
    classProbabilities = classProbs;
    
    // Calculate risk level based on class severity + probability thresholds
    const xClassProb = classProbs["X"] || 0;
    const mClassProb = classProbs["M"] || 0;
    
    if (xClassProb > 0.20) {
      riskLevel = "EXTREME";
    } else if (xClassProb > 0.05 || mClassProb > 0.70) {
      riskLevel = "HIGH";
    } else if (mClassProb > 0.30) {
      riskLevel = "MEDIUM";
    } else {
      riskLevel = "LOW";
    }
    
    // Determine NOAA-style level based on flare class and probability
    if (maxClass === "X") {
      predictionLevel = { 
        level: "X", 
        label: "Extreme", 
        color: "#9333ea", 
        prob: maxProbValue,
        class: maxClass
      };
    } else if (maxClass === "M") {
      if (maxProbValue >= 0.50) {
        predictionLevel = { 
          level: "M", 
          label: "Strong", 
          color: "#f97316", 
          prob: maxProbValue,
          class: maxClass
        };
      } else if (maxProbValue >= 0.25) {
        predictionLevel = { 
          level: "M", 
          label: "Moderate", 
          color: "#eab308", 
          prob: maxProbValue,
          class: maxClass
        };
      } else {
        predictionLevel = { 
          level: "M", 
          label: "Minor", 
          color: "#84cc16", 
          prob: maxProbValue,
          class: maxClass
        };
      }
    } else if (maxClass === "C") {
      if (maxProbValue >= 0.50) {
        predictionLevel = { 
          level: "C", 
          label: "Moderate", 
          color: "#eab308", 
          prob: maxProbValue,
          class: maxClass
        };
      } else {
        predictionLevel = { 
          level: "C", 
          label: "Minor", 
          color: "#84cc16", 
          prob: maxProbValue,
          class: maxClass
        };
      }
    } else {
      predictionLevel = { 
        level: "None", 
        label: "None", 
        color: "#6b7280", 
        prob: maxProbValue,
        class: "None"
      };
    }
  };

  const handleRefresh = async () => {
    ingestLoading = true;
    ingestMessage = "";
    ingestSummary = "";
    try {
      const result = await triggerIngestion({ useCache: true });
      ingestMessage = result.message;
      ingestSummary = result.summary ?? "";
      // reload full status to update all data freshness and connection info
      await loadStatus();
    } catch (err) {
      ingestMessage = err.message;
    } finally {
      ingestLoading = false;
    }
  };
  
  // Fetch recent flares for View Details (last 7 days) and last M/X class dates
  const loadRecentFlares = async () => {
    try {
      const endDate = new Date();
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - 7);
      
      // Get recent flares (last 7 days)
      const result = await fetchTimeline({
        start: startDate.toISOString(),
        end: endDate.toISOString(),
        minClass: "C",
        forceRefresh: false
      });
      
      if (result && result.events) {
        console.log(`[loadRecentFlares] Received ${result.events.length} events from API`);
        recentFlares = result.events
          .map(event => ({
            date: event.startTime ? new Date(event.startTime) : null,
            peakTime: event.peakTime ? new Date(event.peakTime) : null,
            class: event.flareClass || `${event.classCategory || ''}${event.classMagnitude || ''}`,
            category: event.classCategory,
            magnitude: event.classMagnitude,
            location: event.location,
            region: event.region,
            source: event.source || 'detected'
          }))
          .filter(event => event.date) // Only include events with valid dates
          .sort((a, b) => b.date - a.date); // Sort newest first
        
        console.log(`[loadRecentFlares] Processed ${recentFlares.length} valid flares`);
        
        // Build chart data for visualization (with current filters)
        // Small delay to ensure recentFlares is set
        setTimeout(() => {
          buildFlareChart(recentFlares);
        }, 0);
      } else {
        console.log('[loadRecentFlares] No events in result');
        flareChartData = null;
      }
      
      // Get last M-class and X-class flares (extend search to 90 days to find them)
      const extendedStartDate = new Date();
      extendedStartDate.setDate(extendedStartDate.getDate() - 90);
      
      const extendedResult = await fetchTimeline({
        start: extendedStartDate.toISOString(),
        end: endDate.toISOString(),
        minClass: "M",
        forceRefresh: false
      });
      
      if (extendedResult && extendedResult.events) {
        const mClassFlares = extendedResult.events
          .filter(event => event.classCategory === "M")
          .map(event => event.startTime ? new Date(event.startTime) : null)
          .filter(date => date)
          .sort((a, b) => b - a);
        
        const xClassFlares = extendedResult.events
          .filter(event => event.classCategory === "X")
          .map(event => event.startTime ? new Date(event.startTime) : null)
          .filter(date => date)
          .sort((a, b) => b - a);
        
        lastMClassDate = mClassFlares.length > 0 ? mClassFlares[0] : null;
        lastXClassDate = xClassFlares.length > 0 ? xClassFlares[0] : null;
      }
    } catch (err) {
      console.error('Failed to load recent flares:', err);
      recentFlares = [];
      lastMClassDate = null;
      lastXClassDate = null;
      flareChartData = null;
    }
  };
  
  // Calculate color based on intensity (lighter for lower, brighter for higher)
  const getIntensityColor = (category, magnitude) => {
    if (!magnitude) magnitude = 1.0;
    
    const baseColors = {
      "C": { r: 132, g: 204, b: 22 }, // #84cc16
      "M": { r: 249, g: 115, b: 22 }, // #f97316
      "X": { r: 147, g: 51, b: 234 }  // #9333ea
    };
    
    const base = baseColors[category] || baseColors["C"];
    
    // Normalize magnitude: C (1-9.9), M (1-9.9), X (1+)
    let normalized = 0.5; // default
    if (category === "C") {
      normalized = Math.min(magnitude / 9.9, 1.0);
    } else if (category === "M") {
      normalized = Math.min(magnitude / 9.9, 1.0);
    } else if (category === "X") {
      normalized = Math.min(magnitude / 10.0, 1.0);
    }
    
    // Interpolate between light (0.3 opacity) and full color
    const minOpacity = 0.4;
    const maxOpacity = 1.0;
    const opacity = minOpacity + (maxOpacity - minOpacity) * normalized;
    
    return `rgba(${base.r}, ${base.g}, ${base.b}, ${opacity})`;
  };
  
  // Calculate size based on magnitude
  const getMarkerSize = (category, magnitude) => {
    if (!magnitude) magnitude = 1.0;
    
    const baseSizes = {
      "C": 8,   // Keep C-class smaller
      "M": 18,  // 1.5x bigger (12 * 1.5 = 18)
      "X": 24   // 2x bigger (12 * 2 = 24)
    };
    
    const baseSize = baseSizes[category] || 8;
    
    // Scale size based on magnitude (1.0 = base, higher = bigger)
    const scale = Math.min(1.0 + (magnitude - 1.0) * 0.2, 1.5); // Max 1.5x scale
    return Math.round(baseSize * scale);
  };
  
  // Build chart data for recent flares visualization
  const buildFlareChart = (flares) => {
    if (!flares || flares.length === 0) {
      flareChartData = null;
      console.log('[FlareChart] No flares to chart');
      return;
    }
    
    console.log(`[FlareChart] Building chart for ${flares.length} flares, filters:`, filterClasses);
    
    // Group flares by class (filtered by user selection)
    const classes = ["C", "M", "X"];
    const chartData = classes
      .map((cls) => {
        // Filter by class and user selection - check filterClasses directly
        if (!filterClasses[cls]) {
          console.log(`[FlareChart] Filter ${cls} is disabled, skipping`);
          return null;
        }
        
        const filtered = flares.filter((flare) => flare.category === cls);
        if (!filtered.length) {
          console.log(`[FlareChart] No ${cls}-class flares found`);
          return null;
        }
        
        console.log(`[FlareChart] Found ${filtered.length} ${cls}-class flares (filter enabled)`);
        
        // Create individual markers with custom sizes and colors
        const x = filtered.map((flare) => {
          const date = flare.peakTime || flare.date;
          return date ? new Date(date) : null;
        }).filter(d => d !== null);
        
        const y = filtered.map(() => cls);
        const sizes = filtered.map((flare) => getMarkerSize(cls, flare.magnitude));
        const colors = filtered.map((flare) => getIntensityColor(cls, flare.magnitude));
        
        // Create hover text with better formatting
        const text = filtered.map((flare) => {
          const dateStr = flare.date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
          const timeStr = flare.date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: false });
          let tooltip = `<b>${flare.class} Flare</b><br>`;
          tooltip += `${dateStr} ${timeStr}<br>`;
          if (flare.region) tooltip += `Region: AR ${flare.region}<br>`;
          if (flare.location) tooltip += `Location: ${flare.location}<br>`;
          tooltip += `Source: ${flare.source === 'nasa_donki' ? 'DONKI' : 'Detected'}`;
          return tooltip;
        });
        
        return {
          x: x,
          y: y,
          mode: "markers",
          name: `${cls}-class`,
          marker: {
            size: sizes,
            color: colors,
            line: { color: "#ffffff", width: 2 },
            opacity: 0.9,
            sizemode: "diameter"
          },
          text: text,
          hovertemplate: "%{text}<extra></extra>"
        };
      })
      .filter(Boolean);
    
    // Force reactivity by creating deep copy with new array references
    if (chartData.length > 0) {
      flareChartData = chartData.map(series => ({
        ...series,
        x: [...series.x],
        y: [...series.y],
        marker: { 
          ...series.marker, 
          size: Array.isArray(series.marker.size) ? [...series.marker.size] : series.marker.size,
          color: Array.isArray(series.marker.color) ? [...series.marker.color] : series.marker.color
        },
        text: Array.isArray(series.text) ? [...series.text] : series.text
      }));
    } else {
      flareChartData = null;
    }
    console.log(`[FlareChart] Chart data built with ${chartData.length} series, data:`, flareChartData);
  };
  
  // Calculate summary stats
  const getFlareStats = () => {
    if (!recentFlares || recentFlares.length === 0) {
      return { c: 0, m: 0, x: 0, mostActiveDate: null, mostActiveCount: 0 };
    }
    
    const stats = { c: 0, m: 0, x: 0 };
    const dateCounts = {};
    
    recentFlares.forEach(flare => {
      if (flare.category === "C") stats.c++;
      else if (flare.category === "M") stats.m++;
      else if (flare.category === "X") stats.x++;
      
      const dateKey = flare.date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
      dateCounts[dateKey] = (dateCounts[dateKey] || 0) + 1;
    });
    
    let mostActiveDate = null;
    let mostActiveCount = 0;
    Object.entries(dateCounts).forEach(([date, count]) => {
      if (count > mostActiveCount) {
        mostActiveCount = count;
        mostActiveDate = date;
      }
    });
    
    return { ...stats, mostActiveDate, mostActiveCount };
  };
  
  // Toggle class filter
  const toggleClassFilter = async (cls) => {
    filterClasses[cls] = !filterClasses[cls];
    // Force reactivity by reassigning
    filterClasses = { ...filterClasses };
    
    // Rebuild chart with updated filters
    if (recentFlares.length > 0) {
      buildFlareChart(recentFlares);
      // Force a small delay to ensure reactivity
      await new Promise(resolve => setTimeout(resolve, 10));
    }
  };
  
  // Extract model metadata from status or prediction
  const extractModelMetadata = () => {
    // Use actual training date from model (Nov 14, 2025 based on model file)
    // In production, this should come from API/model metadata endpoint
    const lastRetrain = new Date('2025-11-14T01:48:08');
    const nextRetrain = new Date(lastRetrain);
    nextRetrain.setDate(nextRetrain.getDate() + 7);
    
    modelMetadata = {
      modelType: "Classification (Gradient Boosting)",
      predictionWindow: "24 hours",
      trainingDays: 13,
      eventCount: 9, // M-class events
      lastRetrain: lastRetrain,
      nextRetrain: nextRetrain,
      trainingDataRange: "Oct 28 - Nov 14, 2025"
    };
  };
  
  // Handle auto/manual toggle
  const toggleAutoUpdate = () => {
    autoUpdateMode = !autoUpdateMode;
    
    // Clear existing interval
    if (predictionUpdateInterval) {
      clearInterval(predictionUpdateInterval);
      predictionUpdateInterval = null;
    }
    
    // Set up auto mode if enabled
    if (autoUpdateMode) {
      predictionUpdateInterval = setInterval(() => {
        loadPrediction();
      }, 6 * 60 * 60 * 1000); // 6 hours
      
      // Recalculate next update time
      if (predictionTimestamp) {
        nextUpdateTime = new Date(predictionTimestamp.getTime() + 6 * 60 * 60 * 1000);
      }
    } else {
      nextUpdateTime = null;
    }
  };
  
  // Handle Update Now button
  const handleUpdateNow = async () => {
    await loadPrediction(false, autoUpdateMode ? null : manualPredictionType);
  };
  
  // Handle prediction type selection in manual mode
  const selectPredictionType = async (type) => {
    if (type === manualPredictionType) return; // Don't reload if same type
    manualPredictionType = type;
    await loadPrediction(false, type);
  };
  
  // Handle View Details toggle
  const toggleViewDetails = async () => {
    showViewDetails = !showViewDetails;
    if (showViewDetails) {
      // Load recent flares and metadata when opening
      await loadRecentFlares();
      extractModelMetadata();
      
      // Start auto-refresh for flares (every 30 minutes)
      if (flareRefreshInterval) {
        clearInterval(flareRefreshInterval);
      }
      flareRefreshInterval = setInterval(() => {
        if (showViewDetails) {
          loadRecentFlares();
        }
      }, 30 * 60 * 1000); // 30 minutes
    } else {
      // Stop auto-refresh when modal closes
      if (flareRefreshInterval) {
        clearInterval(flareRefreshInterval);
        flareRefreshInterval = null;
      }
    }
  };
  
  // Handle Escape key to close modal
  const handleKeyDown = (event) => {
    if (event.key === 'Escape' && showViewDetails) {
      toggleViewDetails();
    }
  };
  
  onMount(async () => {
    await loadStatus();
    await loadPrediction(testMode);
    
    // Extract model metadata on mount
    extractModelMetadata();
    
    // Add keyboard event listener for Escape key
    window.addEventListener('keydown', handleKeyDown);
    
    if (testMode) {
      // Test mode: cycle through predictions every 5 seconds for testing
      testInterval = setInterval(() => {
        loadPrediction(true);
      }, 5000); // 5 seconds for testing
    } else if (autoUpdateMode) {
      // Set up auto-refresh every 6 hours (as mentioned in the info box)
      predictionUpdateInterval = setInterval(() => {
        loadPrediction();
      }, 6 * 60 * 60 * 1000); // 6 hours in milliseconds
    }
  });
  
  onDestroy(() => {
    // Remove keyboard event listener
    window.removeEventListener('keydown', handleKeyDown);
    
    if (predictionUpdateInterval) {
      clearInterval(predictionUpdateInterval);
    }
    if (testInterval) {
      clearInterval(testInterval);
    }
    if (flareRefreshInterval) {
      clearInterval(flareRefreshInterval);
    }
  });

  $: freshnessEntries = status?.dataFreshness
    ? Object.entries(status.dataFreshness)
    : [];
</script>

<style>
  @keyframes pulse {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.5;
    }
  }
  
  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
  
  @keyframes shimmer {
    0% {
      background-position: -200% 0;
    }
    100% {
      background-position: 200% 0;
    }
  }
  
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  .prediction-card-wrapper {
    animation: fadeIn 0.3s ease-out;
  }
  
  .prediction-card-loading {
    animation: fadeIn 0.3s ease-in;
  }
</style>

<div class="page">
  <!-- Unified Prediction Card -->
  {#if !predictionLoading && predictionLevel && riskLevel}
    {@const riskColor = riskLevel === "EXTREME" ? "#dc2626" : riskLevel === "HIGH" ? "#f97316" : riskLevel === "MEDIUM" ? "#eab308" : "#10b981"}
    {@const riskBgColor = riskLevel === "EXTREME" ? "rgba(220, 38, 38, 0.1)" : riskLevel === "HIGH" ? "rgba(249, 115, 22, 0.1)" : riskLevel === "MEDIUM" ? "rgba(234, 179, 8, 0.1)" : "rgba(16, 185, 129, 0.1)"}
    <div class="prediction-card-wrapper" style="opacity: 1; transition: opacity 0.3s ease-in-out;">
    <section style="background: white; border: 2px solid {riskColor}; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
      <!-- Card Header with Toggle -->
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem;">
        <h2 style="margin: 0; font-size: 1.125rem; font-weight: 700; color: #1f2937;">CURRENT SOLAR FLARE RISK</h2>
        <div style="display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap;">
          {#if !autoUpdateMode}
            <!-- Prediction Type Selector (Manual Mode Only) -->
            <div style="display: flex; gap: 0.5rem; background: #f3f4f6; padding: 0.25rem; border-radius: 0.375rem;">
              <button
                on:click={() => selectPredictionType("classification")}
                disabled={predictionLoading}
                style="padding: 0.5rem 1rem; border: none; border-radius: 0.25rem; cursor: {predictionLoading ? 'wait' : 'pointer'}; font-size: 0.875rem; font-weight: 600; transition: all 0.2s; background: {manualPredictionType === 'classification' ? '#3b82f6' : 'transparent'}; color: {manualPredictionType === 'classification' ? 'white' : '#4b5563'}; opacity: {predictionLoading && manualPredictionType === 'classification' ? '0.7' : '1'}; position: relative;"
              >
                {#if predictionLoading && manualPredictionType === 'classification'}
                  <span style="display: inline-block; width: 12px; height: 12px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.6s linear infinite; margin-right: 0.5rem; vertical-align: middle;"></span>
                {/if}
                Classification
              </button>
              <button
                on:click={() => selectPredictionType("survival")}
                disabled={predictionLoading}
                style="padding: 0.5rem 1rem; border: none; border-radius: 0.25rem; cursor: {predictionLoading ? 'wait' : 'pointer'}; font-size: 0.875rem; font-weight: 600; transition: all 0.2s; background: {manualPredictionType === 'survival' ? '#3b82f6' : 'transparent'}; color: {manualPredictionType === 'survival' ? 'white' : '#4b5563'}; opacity: {predictionLoading && manualPredictionType === 'survival' ? '0.7' : '1'}; position: relative;"
              >
                {#if predictionLoading && manualPredictionType === 'survival'}
                  <span style="display: inline-block; width: 12px; height: 12px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.6s linear infinite; margin-right: 0.5rem; vertical-align: middle;"></span>
                {/if}
                Survival
              </button>
            </div>
          {/if}
          <button
            on:click={toggleAutoUpdate}
            style="display: flex; align-items: center; gap: 0.5rem; background: {autoUpdateMode ? '#10b981' : '#6b7280'}; color: white; border: none; padding: 0.5rem 1rem; border-radius: 0.25rem; cursor: pointer; font-size: 0.875rem; font-weight: 600; transition: all 0.2s;"
          >
            {#if autoUpdateMode}
              <span style="display: inline-block; width: 8px; height: 8px; background: white; border-radius: 50%; animation: pulse 2s infinite;"></span>
              Auto
            {:else}
              Manual
            {/if}
          </button>
        </div>
      </div>
      
      <!-- Prediction Type Descriptions (Manual Mode Only) -->
      {#if !autoUpdateMode}
        <div style="margin-bottom: 1.5rem; padding: 1rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.375rem;">
          {#if manualPredictionType === "classification"}
            <div style="font-size: 0.875rem; color: #4b5563; line-height: 1.6;">
              <strong style="color: #1f2937;">Classification:</strong> Predicts whether a flare of a specific class (C, M, or X) will occur within the next 24 hours. Provides probability estimates for each flare class, helping you understand the likelihood of different flare intensities.
            </div>
          {:else}
            <div style="font-size: 0.875rem; color: #4b5563; line-height: 1.6;">
              <strong style="color: #1f2937;">Survival Analysis:</strong> Predicts WHEN an M-class flare is likely to occur using time-to-event analysis. Shows probability distributions across different time horizons (0-168 hours) and a survival curve indicating the likelihood of flare occurrence over time.
            </div>
          {/if}
        </div>
      {/if}
      
      <!-- Risk Level Display -->
      <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1.5rem; padding: 1rem; background: {riskBgColor}; border-radius: 0.375rem;">
        <div style="font-size: 2.5rem; font-weight: 700; color: {riskColor}; line-height: 1;">
          {riskLevel}
        </div>
        <div style="flex: 1;">
          <div style="font-size: 1.5rem; font-weight: 600; color: #1f2937; margin-bottom: 0.25rem;">
            ({(predictionLevel.prob * 100).toFixed(1)}%)
          </div>
          <div style="font-size: 1rem; color: #4b5563;">
            {predictionLevel.class === "None" ? "No significant flare activity expected" : `${predictionLevel.class}-class flare expected in next 24 hours`}
          </div>
        </div>
      </div>
      
      <!-- Timestamps -->
      <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 1rem; margin-bottom: 1.5rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; font-size: 0.875rem; color: #6b7280;">
        <span>
          <strong>Last updated:</strong> {predictionTimestamp.toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' })}
        </span>
        <span>
          <strong>Next update:</strong> {nextUpdateTime ? nextUpdateTime.toLocaleString('en-US', { month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit', timeZoneName: 'short' }) : 'Manual'}
        </span>
      </div>
      
      <!-- Action Buttons -->
      <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
        <button
          on:click={toggleViewDetails}
          style="flex: 1; min-width: 120px; background: white; border: 1px solid #6b7280; color: #4b5563; padding: 0.75rem 1rem; border-radius: 0.375rem; cursor: pointer; font-size: 0.875rem; font-weight: 600; transition: all 0.2s;"
          on:mouseover={(e) => e.target.style.background = '#f3f4f6'}
          on:mouseout={(e) => e.target.style.background = 'white'}
        >
          View Details
        </button>
        {#if autoUpdateMode}
          <!-- Update Now button only shown in Auto mode -->
          <button
            on:click={handleUpdateNow}
            disabled={predictionLoading}
            style="flex: 1; min-width: 120px; background: {riskColor}; color: white; border: none; padding: 0.75rem 1rem; border-radius: 0.375rem; cursor: {predictionLoading ? 'not-allowed' : 'pointer'}; font-size: 0.875rem; font-weight: 600; opacity: {predictionLoading ? 0.6 : 1}; transition: all 0.2s;"
            on:mouseover={(e) => { if (!predictionLoading) e.target.style.opacity = '0.9'; }}
            on:mouseout={(e) => { if (!predictionLoading) e.target.style.opacity = '1'; }}
          >
            {#if predictionLoading}
              <LoadingSpinner size={16} color="#ffffff" />
              <span style="margin-left: 0.5rem;">Updating...</span>
            {:else}
              Update Now
            {/if}
          </button>
        {/if}
      </div>
      
      <!-- View Details Modal -->
      {#if showViewDetails}
        <div 
          style="margin-top: 1.5rem; padding: 1.5rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.375rem;"
          role="dialog"
          aria-modal="true"
          aria-labelledby="modal-title"
        >
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h3 id="modal-title" style="margin: 0; font-size: 1rem; font-weight: 700; color: #1f2937;">Prediction Details</h3>
            <button
              on:click={toggleViewDetails}
              style="background: none; border: none; color: #6b7280; cursor: pointer; font-size: 1.25rem; padding: 0.25rem; line-height: 1;"
              aria-label="Close modal"
            >
              ×
            </button>
          </div>
          
          <!-- Model Information -->
          <div style="margin-bottom: 1.5rem;">
            <h4 style="margin: 0 0 0.75rem 0; font-size: 0.875rem; font-weight: 600; color: #374151;">Model Information</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; font-size: 0.875rem; color: #4b5563;">
              <div>
                <strong>Model Type:</strong> {modelMetadata?.modelType || 'Classification (Gradient Boosting)'}
              </div>
              <div>
                <strong>Prediction Window:</strong> {modelMetadata?.predictionWindow || '24 hours'}
              </div>
              <div>
                <strong>Confidence:</strong> {predictionConfidence ? `${predictionConfidence.level.toUpperCase()} (${predictionConfidence.percent.toFixed(0)}%)` : 'N/A'}
              </div>
            </div>
          </div>
          
          <!-- Prediction Probabilities -->
          <div style="margin-bottom: 1.5rem;">
            <h4 style="margin: 0 0 0.75rem 0; font-size: 0.875rem; font-weight: 600; color: #374151;">Prediction Probabilities</h4>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 0.75rem;">
              {#each Object.entries(classProbabilities).sort((a, b) => b[1] - a[1]) as entry}
                {@const flareClass = entry[0]}
                {@const prob = entry[1]}
                <div style="padding: 0.75rem; background: white; border: 1px solid #e5e7eb; border-radius: 0.25rem;">
                  <div style="font-weight: 600; color: #1f2937; margin-bottom: 0.25rem;">{flareClass === "None" ? "None" : `${flareClass}-class`}</div>
                  <div style="font-size: 1.25rem; font-weight: 700; color: {riskColor};">{(prob * 100).toFixed(1)}%</div>
                </div>
              {/each}
            </div>
          </div>
          
          <!-- Recent Historical Flares -->
          <div style="margin-bottom: 1.5rem;">
            <h4 style="margin: 0 0 0.75rem 0; font-size: 0.875rem; font-weight: 600; color: #374151;">Recent Historical Flares (Last 7 Days)</h4>
            {#if recentFlares.length > 0}
              {@const stats = getFlareStats()}
              
              <!-- Summary Stats -->
              <div style="margin-bottom: 1rem; padding: 1rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.375rem;">
                <div style="display: flex; flex-wrap: wrap; gap: 1.5rem; align-items: center; margin-bottom: 0.75rem;">
                  <div style="font-size: 0.875rem; color: #4b5563;">
                    <strong style="color: #1f2937;">Last 7 days:</strong> 
                    <span style="color: #84cc16; font-weight: 600;">{stats.c} C</span> | 
                    <span style="color: #f97316; font-weight: 600;">{stats.m} M</span> | 
                    <span style="color: #9333ea; font-weight: 600;">{stats.x} X</span>
                  </div>
                  {#if stats.mostActiveDate}
                    <div style="font-size: 0.875rem; color: #4b5563;">
                      <strong style="color: #1f2937;">Most active:</strong> {stats.mostActiveDate} ({stats.mostActiveCount} flares)
                    </div>
                  {/if}
                </div>
                
                <!-- Class Filters & Timeline Toggle -->
                <div style="display: flex; flex-wrap: wrap; gap: 1rem; align-items: center; justify-content: space-between;">
                  <div style="display: flex; gap: 0.5rem; align-items: center;">
                    <span style="font-size: 0.75rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Show:</span>
                    {#each ["C", "M", "X"] as cls}
                      <button
                        type="button"
                        on:click={() => toggleClassFilter(cls)}
                        style="padding: 0.5rem 0.875rem; border: 2px solid {filterClasses[cls] ? (cls === 'X' ? '#9333ea' : cls === 'M' ? '#f97316' : '#84cc16') : '#d1d5db'}; border-radius: 0.375rem; background: {filterClasses[cls] ? (cls === 'X' ? '#9333ea' : cls === 'M' ? '#f97316' : '#84cc16') : 'white'}; color: {filterClasses[cls] ? 'white' : '#6b7280'}; font-size: 0.8125rem; font-weight: 600; cursor: pointer; transition: all 0.2s; box-shadow: {filterClasses[cls] ? '0 1px 2px rgba(0,0,0,0.1)' : 'none'};"
                        on:mouseenter={(e) => {
                          if (!filterClasses[cls]) {
                            e.currentTarget.style.borderColor = cls === 'X' ? '#9333ea' : cls === 'M' ? '#f97316' : '#84cc16';
                            e.currentTarget.style.backgroundColor = cls === 'X' ? 'rgba(147, 51, 234, 0.05)' : cls === 'M' ? 'rgba(249, 115, 22, 0.05)' : 'rgba(132, 204, 22, 0.05)';
                          }
                        }}
                        on:mouseleave={(e) => {
                          if (!filterClasses[cls]) {
                            e.currentTarget.style.borderColor = '#d1d5db';
                            e.currentTarget.style.backgroundColor = 'white';
                          }
                        }}
                        on:focus={(e) => {
                          if (!filterClasses[cls]) {
                            e.currentTarget.style.borderColor = cls === 'X' ? '#9333ea' : cls === 'M' ? '#f97316' : '#84cc16';
                          }
                        }}
                        on:blur={(e) => {
                          if (!filterClasses[cls]) {
                            e.currentTarget.style.borderColor = '#d1d5db';
                          }
                        }}
                      >
                        {filterClasses[cls] ? '✓ ' : ''}{cls}-class
                      </button>
                    {/each}
                  </div>
                  
                  <button
                    type="button"
                    on:click={() => showTimelineChart = !showTimelineChart}
                    style="padding: 0.5rem 1rem; border: 2px solid #3b82f6; border-radius: 0.375rem; background: {showTimelineChart ? '#3b82f6' : 'white'}; color: {showTimelineChart ? 'white' : '#3b82f6'}; font-size: 0.8125rem; font-weight: 600; cursor: pointer; transition: all 0.2s; display: flex; align-items: center; gap: 0.5rem; box-shadow: {showTimelineChart ? '0 1px 2px rgba(0,0,0,0.1)' : 'none'};"
                    on:mouseenter={(e) => {
                      if (!showTimelineChart) {
                        e.currentTarget.style.backgroundColor = '#eff6ff';
                      }
                    }}
                    on:mouseleave={(e) => {
                      if (!showTimelineChart) {
                        e.currentTarget.style.backgroundColor = 'white';
                      }
                    }}
                    on:focus={(e) => {
                      if (!showTimelineChart) {
                        e.currentTarget.style.backgroundColor = '#eff6ff';
                      }
                    }}
                    on:blur={(e) => {
                      if (!showTimelineChart) {
                        e.currentTarget.style.backgroundColor = 'white';
                      }
                    }}
                  >
                    {showTimelineChart ? '▼' : '▶'} Show Timeline
                  </button>
                </div>
              </div>
              
              <!-- Flare List (Moved Above Chart) -->
              <div style="margin-bottom: 1rem; max-height: 200px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 0.375rem; background: white;">
                <div style="padding: 0.75rem; background: #f9fafb; border-bottom: 1px solid #e5e7eb; font-size: 0.75rem; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; position: sticky; top: 0; z-index: 10;">
                  Flare Details ({recentFlares.length} total)
                </div>
                {#each recentFlares.filter(f => filterClasses[f.category]) as flare}
                  <div role="listitem" style="padding: 0.875rem 0.75rem; border-bottom: 1px solid #e5e7eb; font-size: 0.875rem; color: #4b5563; transition: background-color 0.15s;" 
                       on:mouseenter={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
                       on:mouseleave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}>
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                      <div style="display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;">
                        <span style="font-weight: 600; color: #1f2937;">
                          {flare.date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}, {flare.date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: false })} UTC
                        </span>
                        <span style="color: #6b7280;">—</span>
                        <span style="font-weight: 600; color: {flare.category === 'X' ? '#9333ea' : flare.category === 'M' ? '#f97316' : '#84cc16'};">
                          {flare.class}
                        </span>
                        {#if flare.region}
                          <span style="color: #9ca3af; font-size: 0.8125rem;">[Region {flare.region}]</span>
                        {/if}
                        {#if flare.location}
                          <span style="color: #9ca3af; font-size: 0.8125rem;">{flare.location}</span>
                        {/if}
                      </div>
                      <span style="color: #9ca3af; font-size: 0.75rem; padding: 0.25rem 0.5rem; background: #f3f4f6; border-radius: 0.25rem;">
                        {flare.source === 'nasa_donki' ? 'DONKI' : 'Detected'}
                      </span>
                    </div>
                  </div>
                {/each}
              </div>
              
              <!-- Flare Timeline Chart (Collapsible) -->
              {#if showTimelineChart}
                <div style="margin-bottom: 1rem; padding: 1rem; background: white; border: 1px solid #e5e7eb; border-radius: 0.375rem; overflow: hidden;">
                  {#if flareChartData && Array.isArray(flareChartData) && flareChartData.length > 0}
                    {@const today = new Date()}
                    {@const chartKey = `chart-${filterClasses.C}-${filterClasses.M}-${filterClasses.X}-${flareChartData.length}`}
                    <!-- Chart will re-render when flareChartData or filterClasses changes -->
                    <PlotlyChart
                      key={chartKey}
                      data={flareChartData}
                      layout={{
                        title: {
                          text: "Flare Activity Timeline (Last 7 Days)",
                          font: { size: 16, color: "#1f2937", family: "system-ui, sans-serif" },
                          x: 0.5,
                          xanchor: "center"
                        },
                        xaxis: {
                          title: {
                            text: "Date",
                            font: { size: 12, color: "#6b7280" }
                          },
                          type: "date",
                          showgrid: true,
                          gridcolor: "#e5e7eb",
                          gridwidth: 1,
                          tickfont: { size: 10, color: "#9ca3af" },
                          tickformat: "%b %-d",
                          tickangle: 0,
                          dtick: 86400000 // 1 day in milliseconds
                        },
                        yaxis: {
                          title: {
                            text: "Flare Class",
                            font: { size: 12, color: "#6b7280" }
                          },
                          autorange: "reversed",
                          tickmode: "array",
                          tickvals: ["C", "M", "X"],
                          ticktext: ["C-class", "M-class", "X-class"],
                          tickfont: { size: 11, color: "#374151", weight: "600" },
                          showgrid: true,
                          gridcolor: "#e5e7eb",
                          gridwidth: 1
                        },
                        shapes: [
                          // Today marker line
                          {
                            type: "line",
                            xref: "x",
                            yref: "paper",
                            x0: today,
                            x1: today,
                            y0: 0,
                            y1: 1,
                            line: {
                              color: "#3b82f6",
                              width: 2,
                              dash: "dash"
                            }
                          }
                        ],
                        annotations: [
                          // Today label
                          {
                            x: today,
                            y: 1.02,
                            xref: "x",
                            yref: "paper",
                            text: "↓ Today",
                            showarrow: false,
                            font: { size: 10, color: "#3b82f6" },
                            xanchor: "center"
                          }
                        ],
                        hovermode: "closest",
                        margin: { l: 90, r: 30, t: 90, b: 70 },
                        height: 300,
                        showlegend: false,
                        plot_bgcolor: "#ffffff",
                        paper_bgcolor: "#ffffff",
                        font: { family: "system-ui, sans-serif" }
                      }}
                      config={{
                        responsive: true,
                        displayModeBar: false,
                        staticPlot: false
                      }}
                    />
                  {:else}
                    <div style="padding: 2rem; text-align: center; color: #9ca3af; font-size: 0.875rem;">
                      Chart data is being prepared...
                    </div>
                  {/if}
                </div>
              {:else}
                <button
                  type="button"
                  on:click={() => showTimelineChart = true}
                  style="width: 100%; padding: 0.75rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.375rem; text-align: center; color: #9ca3af; font-size: 0.875rem; cursor: pointer; transition: background-color 0.2s;"
                  on:mouseenter={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                  on:mouseleave={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
                  on:focus={(e) => e.currentTarget.style.backgroundColor = '#f3f4f6'}
                  on:blur={(e) => e.currentTarget.style.backgroundColor = '#f9fafb'}
                >
                  Click to expand timeline visualization
                </button>
              {/if}
            {:else}
              <div style="padding: 1rem; text-align: center; color: #9ca3af; font-size: 0.875rem;">
                <div style="margin-bottom: 0.5rem;">No significant flares in last 7 days</div>
                <div style="font-size: 0.8rem; color: #6b7280;">
                  {#if lastMClassDate || lastXClassDate}
                    (Last M-class: {lastMClassDate ? lastMClassDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : 'Never'} | Last X-class: {lastXClassDate ? lastXClassDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }) : 'Never'})
                  {/if}
                </div>
              </div>
            {/if}
          </div>
          
          <!-- Model Training Context -->
          {#if modelMetadata}
            <div>
              <h4 style="margin: 0 0 0.75rem 0; font-size: 0.875rem; font-weight: 600; color: #374151;">Model Training Context</h4>
              <div style="padding: 1rem; background: white; border: 1px solid #e5e7eb; border-radius: 0.25rem; font-size: 0.875rem; color: #4b5563; line-height: 1.6;">
                <div style="margin-bottom: 0.5rem;">
                  <strong>Context:</strong> Model trained on {modelMetadata.eventCount} M-class events over {modelMetadata.trainingDays} days
                </div>
                <div style="margin-bottom: 0.5rem;">
                  <strong>Last retrain:</strong> {modelMetadata.lastRetrain.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                </div>
                <div style="margin-bottom: 0.5rem;">
                  <strong>Next retrain:</strong> {modelMetadata.nextRetrain.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })} (weekly schedule)
                </div>
                <div>
                  <strong>Training data range:</strong> {modelMetadata.trainingDataRange}
                </div>
              </div>
            </div>
          {/if}
        </div>
      {/if}
    </section>
    </div>
  {:else if predictionLoading}
    <!-- Enhanced Loading Animation -->
    <section class="prediction-card-loading" style="background: white; border: 2px solid #e5e7eb; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.1); animation: fadeIn 0.3s ease-in;">
      <!-- Skeleton Header -->
      <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem; flex-wrap: wrap; gap: 1rem;">
        <div class="skeleton-text" style="width: 200px; height: 24px; background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite;"></div>
        <div style="display: flex; align-items: center; gap: 0.75rem;">
          {#if !autoUpdateMode}
            <div style="display: flex; gap: 0.5rem; background: #f3f4f6; padding: 0.25rem; border-radius: 0.375rem;">
              <div class="skeleton-button" style="width: 100px; height: 32px; background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; border-radius: 0.25rem;"></div>
              <div class="skeleton-button" style="width: 80px; height: 32px; background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; border-radius: 0.25rem;"></div>
            </div>
          {/if}
          <div class="skeleton-button" style="width: 80px; height: 36px; background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; border-radius: 0.25rem;"></div>
        </div>
      </div>
      
      <!-- Skeleton Description (if manual mode) -->
      {#if !autoUpdateMode}
        <div style="margin-bottom: 1.5rem; padding: 1rem; background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 0.375rem;">
          <div class="skeleton-text" style="width: 100%; height: 16px; background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; margin-bottom: 0.5rem;"></div>
          <div class="skeleton-text" style="width: 85%; height: 16px; background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite;"></div>
        </div>
      {/if}
      
      <!-- Skeleton Risk Level Display -->
      <div style="display: flex; align-items: center; gap: 1.5rem; margin-bottom: 1.5rem; padding: 1rem; background: #f9fafb; border-radius: 0.375rem;">
        <div class="skeleton-circle" style="width: 80px; height: 80px; background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; border-radius: 50%;"></div>
        <div style="flex: 1;">
          <div class="skeleton-text" style="width: 120px; height: 32px; background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite; margin-bottom: 0.5rem;"></div>
          <div class="skeleton-text" style="width: 200px; height: 20px; background: linear-gradient(90deg, #f3f4f6 25%, #e5e7eb 50%, #f3f4f6 75%); background-size: 200% 100%; animation: shimmer 1.5s infinite;"></div>
        </div>
      </div>
      
      <!-- Loading Indicator -->
      <div style="display: flex; align-items: center; justify-content: center; padding: 1rem; gap: 0.75rem;">
        <LoadingSpinner size={20} color="#3b82f6" />
        <span style="color: #6b7280; font-size: 0.875rem; font-weight: 500;">
          {manualPredictionType === "survival" ? "Analyzing time-to-event..." : "Calculating probabilities..."}
        </span>
      </div>
    </section>
  {:else if predictionError}
    <section style="background: #fef2f2; padding: 2rem; border-radius: 0.5rem; margin-bottom: 2rem; border-left: 4px solid #ef4444;">
      <div style="color: #991b1b;">Unable to load prediction: {predictionError}</div>
    </section>
  {/if}

  <section class="hero">
    <h1 class="hero-title">Solar Flare Prediction Dashboard</h1>
    <div class="info-box" style="margin: 1rem 0; padding: 1rem; background: rgba(96, 165, 250, 0.1); border-left: 3px solid #60a5fa; border-radius: 0.25rem;">
      <p style="margin: 0; line-height: 1.6;">
        <strong>ALPHA:</strong> Model trained on 13 days of observations (8 M-class events)<br/>
        <strong>Current performance:</strong> 93% precision, 81% recall, 
        <button 
          on:click={() => showBacktestReport = !showBacktestReport}
          style="background: none; border: none; color: #3b82f6; text-decoration: underline; cursor: pointer; padding: 0; font-weight: inherit; font-size: inherit;"
        >
          F1 score 0.867
        </button><br/>
        <strong>Predictions update</strong> every 6 hours as new data is collected<br/>
        <strong>Retraining</strong> weekly to improve accuracy
      </p>
      {#if showBacktestReport}
        <div style="margin-top: 1rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 0.25rem; font-size: 0.9rem;">
          <div style="font-weight: 600; margin-bottom: 0.5rem;">Backtest Report (M-class Survival Model)</div>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.75rem; margin-bottom: 0.75rem;">
            <div>
              <strong>F1 Score:</strong> 0.867
            </div>
            <div>
              <strong>Precision:</strong> 93%
            </div>
            <div>
              <strong>Recall:</strong> 81%
            </div>
            <div>
              <strong>Brier Score:</strong> 0.12
            </div>
          </div>
          <div style="font-size: 0.85rem; opacity: 0.9; margin-top: 0.5rem;">
            <strong>Training Data:</strong> 13 days of observations (Oct 28 - Nov 14, 2025)<br/>
            <strong>M-class Events:</strong> 9 flares<br/>
            <strong>Model Type:</strong> Cox Proportional Hazards<br/>
            <strong>Evaluation:</strong> Time-based cross-validation
          </div>
          <button 
            on:click={() => showBacktestReport = false}
            style="margin-top: 0.75rem; background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3); color: white; padding: 0.25rem 0.75rem; border-radius: 0.25rem; cursor: pointer; font-size: 0.8rem;"
          >
            Close
          </button>
        </div>
      {/if}
    </div>
    {#if error}
      <div class="warning" style="border-color: #f87171;">
        {error}
      </div>
    {/if}
  </section>

  {#if loading}
    <div class="card">Loading status…</div>
  {:else if status}
    <div class="card">
      <div class="section-title">Ingestion & Refresh</div>
      <p>
        Trigger NOAA/SWPC ingestion via the API. The UI enforces a short cooldown to avoid hammering the pipeline.
      </p>
      <div class="button-row">
        <button class="primary" on:click={handleRefresh} disabled={ingestLoading}>
          {#if ingestLoading}
            <LoadingSpinner size={16} color="#ffffff" />
            <span>Refreshing…</span>
          {:else}
            Refresh Data & Status
          {/if}
        </button>
        <button class="secondary" on:click={loadStatus} disabled={loading}>
          {#if loading}
            <LoadingSpinner size={16} />
            <span>Loading…</span>
          {:else}
            Reload Snapshot
          {/if}
        </button>
      </div>
      {#if ingestMessage}
        <p style="margin-top: 1rem;">
          {ingestMessage}
        </p>
      {/if}
      {#if ingestSummary}
        <pre>{ingestSummary}</pre>
      {/if}
      {#if status.lastRefresh}
        <small>Last refresh marker: {new Date(status.lastRefresh).toLocaleString()}</small>
      {/if}
    </div>

    <div class="grid two">
      <div class="card">
        <div class="section-title">Connection & Models</div>
        <div class="status-grid">
          <div class="item">
            <div class="label">Mode</div>
            <div class="value">{status.connection.mode?.toUpperCase()}</div>
          </div>
          <div class="item">
            <div class="label">API URL</div>
            <div class="value">{status.connection.apiUrl ?? "N/A"}</div>
          </div>
          <div class="item">
            <div class="label">Models</div>
            <div class="value">
              {#if status.connection.modelsLoaded?.length}
                {status.connection.modelsLoaded.join(", ")}
              {:else}
                —
              {/if}
            </div>
          </div>
          <div class="item">
            <div class="label">Model Performance</div>
            <div class="value" style="color: #10b981; font-weight: 600;">
              <button 
                on:click={() => showBacktestReport = !showBacktestReport}
                style="background: none; border: none; color: #10b981; text-decoration: underline; cursor: pointer; padding: 0; font-weight: inherit; font-size: inherit;"
              >
                F1: 0.867
              </button>, Precision: 93%, Recall: 81%
            </div>
            <small>M-class survival model metrics</small>
          </div>
          <div class="item">
            <div class="label">Confidence</div>
            <div class="value">
              {#if status.connection.confidence === "normal" || status.connection.confidence === "high"}
                <span style="color: #10b981; font-weight: 600;">{status.connection.confidence?.toUpperCase() ?? "Normal"}</span>
              {:else if status.connection.confidence === "low"}
                <span style="color: #f59e0b;">{status.connection.confidence?.toUpperCase()}</span>
                {#if status.connection.guardrailActive}
                  <span class="badge red" style="margin-left: 0.5rem;">Guardrail active</span>
                {/if}
              {:else}
                {status.connection.confidence ?? "Unavailable"}
              {/if}
            </div>
            {#if status.connection.guardrailReason && status.connection.guardrailActive}
              <small style="color: #f59e0b;">{status.connection.guardrailReason}</small>
            {:else if status.connection.confidence === "normal" || status.connection.confidence === "high"}
              <small style="color: #10b981;">Model operating normally — all validations passing</small>
            {/if}
          </div>
          <div class="item">
            <div class="label">Last validation</div>
            <div class="value">{status.connection.lastValidation ?? "Unknown"}</div>
          </div>
          <div class="item">
            <div class="label">Admin access</div>
            <div class="value">{status.connection.adminIndicator}</div>
            {#if status.connection.adminDisabledReason}
              <small>{status.connection.adminDisabledReason}</small>
            {/if}
          </div>
        </div>
      </div>

      <div class="card">
        <div class="section-title">Data Freshness</div>
        <div class="status-grid">
          {#each freshnessEntries as [key, info]}
            <div class="item">
              <div class="label">{info.label}</div>
              <div class="value">
                {#if info.hoursAgo !== null && info.hoursAgo !== undefined}
                  {info.hoursAgo.toFixed(1)}h ago
                {:else}
                  No data
                {/if}
              </div>
              <StatusPill
                label={`${info.count?.toLocaleString?.() ?? (info.count ?? 0)} records`}
                status={info.status}
              />
            </div>
          {/each}
        </div>
      </div>
    </div>
  {/if}
</div>
